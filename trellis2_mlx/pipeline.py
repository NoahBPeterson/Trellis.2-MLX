"""Trellis2ImageTo3DPipelineMLX — end-to-end image → 3D mesh.

Supports `pipeline_type` ∈ {"512", "1024"} today. Cascades (`1024_cascade`,
`1536_cascade`) raise NotImplementedError until the VAE upsample primitive lands.

Mirrors upstream/trellis2/pipelines/trellis2_image_to_3d.py with MLX tensors:

  image (PIL)
    → preprocess_image (alpha-crop, premultiply)
    → DinoV3FeatureExtractor (torch-CPU) → cond (1, N, 1024) + zeros neg_cond
    → SS flow DiT @ 16^3 (1.3B, 12 Euler steps, CFG interval)
    → SparseStructureDecoder → 64^3 occupancy → threshold → coords at target res
        * target_res = 32 for pipeline_type="512", 64 for "1024"
    → Shape SLat flow DiT (1.3B, 12 Euler steps, CFG interval)
        * 512-variant @ res=32 for "512", 1024-variant @ res=64 for "1024"
    → de-normalize via shape_slat_normalization (mean/std from pipeline.json)
    → FlexiDualGridVaeDecoder (474M) → dual-grid (vertices, intersected, quad_lerp)
        * VAE resolution = 512 or 1024 to match
    → flexible_dual_grid_to_mesh (CPU) → (V, F)

Texture pipeline (tex flow + tex VAE + UV atlas bake) is deferred.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import mlx.core as mx
import numpy as np
from PIL import Image

from .image_cond import DinoV3FeatureExtractor
from .models.flow_dit import SLatFlowModel, SparseStructureFlowModel
from .models.sparse_vae import FlexiDualGridVaeDecoder
from .models.ss_decoder import SparseStructureDecoder
from .modules.sparse_tensor import SparseTensor
from .postprocess.dual_grid import flexible_dual_grid_to_mesh
from .preprocess import preprocess_image
from .rembg import BiRefNetRembg
from .samplers import FlowEulerGuidanceIntervalSampler


def _strip_unused(cfg_args: dict) -> dict:
    return {k: v for k, v in cfg_args.items() if k not in ("initialization", "dtype")}


def _load_config(path: Path) -> dict:
    return json.loads(path.read_text())


# Prefix stripped on load; kept in sync with scripts/prefix_checkpoints.py
_CKPT_PREFIXES: dict[str, str] = {
    "ss_flow_img_dit_1_3B_64.safetensors":                  "ss_flow",
    "ss_dec_conv3d_16l8.safetensors":                       "ss_dec",
    "slat_flow_img2shape_dit_1_3B_512.safetensors":         "shape_flow_512",
    "slat_flow_img2shape_dit_1_3B_1024.safetensors":        "shape_flow_1024",
    "slat_flow_imgshape2tex_dit_1_3B_512.safetensors":      "tex_flow_512",
    "slat_flow_imgshape2tex_dit_1_3B_1024.safetensors":     "tex_flow_1024",
    "shape_dec_next_dc_f16c32.safetensors":                 "shape_dec",
    "shape_enc_next_dc_f16c32.safetensors":                 "shape_enc",
    "tex_dec_next_dc_f16c32.safetensors":                   "tex_dec",
    "tex_enc_next_dc_f16c32.safetensors":                   "tex_enc",
}


def _load_weights_prefixed(module, path: Path, cast_dtype: Optional["mx.Dtype"] = None) -> None:
    """Load MLX safetensors where every tensor is prefixed `<component>.`; strip the
    prefix so `module.load_weights` sees the bare submodule names it expects.

    Optionally cast every loaded tensor to `cast_dtype` (e.g., to convert bf16
    DiT weights to fp16 for faster Metal SDPA). Cast happens before load, so the
    target dtype lives on-disk-equivalent in memory — no extra copy at use time.
    """
    path = Path(path)
    prefix = _CKPT_PREFIXES.get(path.name)
    if prefix is None:
        if cast_dtype is None:
            module.load_weights(str(path))
        else:
            raw = mx.load(str(path))
            casted = [(k, v.astype(cast_dtype)) for k, v in raw.items()]
            module.load_weights(casted)
        return
    raw = mx.load(str(path))
    pfx = prefix + "."
    if all(k.startswith(pfx) for k in raw):
        items = [(k[len(pfx):], v) for k, v in raw.items()]
    else:
        # Legacy un-prefixed file
        items = list(raw.items())
    if cast_dtype is not None:
        items = [(k, v.astype(cast_dtype)) for k, v in items]
    module.load_weights(items)


# pipeline_type → plan describing which flows/conds/resolutions to use.
# - flow_lr/flow_hr: ckpt stems of the shape flow model(s); equal when non-cascade.
# - ss_target_res: the SS-decoder occupancy is maxpooled from 64^3 down to this.
# - vae_res: final grid-resolution for the VAE + dual-grid mesh extraction.
# - cascade: when True, sample LR first, VAE-upsample active-voxel coords, resample HR.
# - hr_resolution: for cascades, the target hr grid (1024 or 1536).
_PIPELINE_PARAMS: dict[str, dict] = {
    "512": {
        "flow_lr": "slat_flow_img2shape_dit_1_3B_512",
        "flow_hr": "slat_flow_img2shape_dit_1_3B_512",
        "cond_sizes": (512,),
        "ss_target_res": 32, "vae_res": 512, "cascade": False,
    },
    "1024": {
        "flow_lr": "slat_flow_img2shape_dit_1_3B_1024",
        "flow_hr": "slat_flow_img2shape_dit_1_3B_1024",
        "cond_sizes": (1024,),
        "ss_target_res": 64, "vae_res": 1024, "cascade": False,
    },
    "1024_cascade": {
        "flow_lr": "slat_flow_img2shape_dit_1_3B_512",
        "flow_hr": "slat_flow_img2shape_dit_1_3B_1024",
        "cond_sizes": (512, 1024),
        "ss_target_res": 32, "vae_res": 1024, "cascade": True,
        "lr_resolution": 512, "hr_resolution": 1024,
    },
    "1536_cascade": {
        "flow_lr": "slat_flow_img2shape_dit_1_3B_512",
        "flow_hr": "slat_flow_img2shape_dit_1_3B_1024",
        "cond_sizes": (512, 1024),
        "ss_target_res": 32, "vae_res": 1536, "cascade": True,
        "lr_resolution": 512, "hr_resolution": 1536,
    },
}


class Trellis2ImageTo3DPipelineMLX:
    """All-in-memory pipeline. Cascade variants load 2 shape flows (5.2 GB)."""

    def __init__(
        self,
        ckpt_dir: str | Path,
        pipeline_config: dict,
        pipeline_type: str = "512",
        dino_device: str = "cpu",
        rembg_device: str = "auto",
        max_num_tokens: int = 49152,
        dit_compute_dtype: str = "bfloat16",
    ):
        self.ckpt_dir = Path(ckpt_dir)
        self.pipeline_type = pipeline_type
        self.pipeline_config = pipeline_config
        self.max_num_tokens = max_num_tokens
        self.rembg_device = rembg_device
        self._rembg_model_name: str = pipeline_config.get("rembg_model", {}).get("args", {}).get("model_name", "briaai/RMBG-2.0")
        if dit_compute_dtype not in ("bfloat16", "float16"):
            raise ValueError(f"dit_compute_dtype must be 'bfloat16' or 'float16', got {dit_compute_dtype!r}")
        self.dit_compute_dtype = getattr(mx, dit_compute_dtype)
        if pipeline_type not in _PIPELINE_PARAMS:
            raise ValueError(f"unknown pipeline_type {pipeline_type!r}; expected one of {list(_PIPELINE_PARAMS)}")
        p = _PIPELINE_PARAMS[pipeline_type]
        self.ss_target_res: int = p["ss_target_res"]
        self.vae_res: int = p["vae_res"]
        self.is_cascade: bool = p["cascade"]
        self.lr_resolution: Optional[int] = p.get("lr_resolution")
        self.hr_resolution: Optional[int] = p.get("hr_resolution")
        self.cond_sizes: tuple[int, ...] = p["cond_sizes"]

        # Models
        self.ss_flow = self._load_ss_flow()
        self.ss_dec = self._load_ss_dec()
        self.shape_flow_lr = self._load_shape_flow(p["flow_lr"])
        if p["flow_hr"] == p["flow_lr"]:
            self.shape_flow_hr = self.shape_flow_lr
        else:
            self.shape_flow_hr = self._load_shape_flow(p["flow_hr"])
        self.shape_vae = self._load_shape_vae()
        self.dino = DinoV3FeatureExtractor(
            pipeline_config["image_cond_model"]["args"]["model_name"],
            image_size=self.cond_sizes[0],
            device=dino_device,
        )

        # Samplers
        self.sampler = FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)

        # Normalization stats for shape SLat
        self.shape_slat_mean = mx.array(pipeline_config["shape_slat_normalization"]["mean"], dtype=mx.float32)
        self.shape_slat_std = mx.array(pipeline_config["shape_slat_normalization"]["std"], dtype=mx.float32)

    # --- model loaders -----------------------------------------------------

    def _load_ss_flow(self) -> SparseStructureFlowModel:
        cfg = _load_config(self.ckpt_dir / "ss_flow_img_dit_1_3B_64.config.json")
        m = SparseStructureFlowModel(**_strip_unused(cfg["args"]))
        _load_weights_prefixed(m, self.ckpt_dir / "ss_flow_img_dit_1_3B_64.safetensors", cast_dtype=self.dit_compute_dtype)
        return m

    def _load_ss_dec(self) -> SparseStructureDecoder:
        # SS decoder is always fp16 (small, already converted; never bf16).
        cfg = _load_config(self.ckpt_dir / "ss_dec_conv3d_16l8.config.json")
        m = SparseStructureDecoder(**cfg["args"])
        _load_weights_prefixed(m, self.ckpt_dir / "ss_dec_conv3d_16l8.safetensors")
        return m

    def _load_shape_flow(self, stem: str) -> SLatFlowModel:
        cfg = _load_config(self.ckpt_dir / f"{stem}.config.json")
        m = SLatFlowModel(**_strip_unused(cfg["args"]))
        _load_weights_prefixed(m, self.ckpt_dir / f"{stem}.safetensors", cast_dtype=self.dit_compute_dtype)
        return m

    def _load_shape_vae(self) -> FlexiDualGridVaeDecoder:
        # Shape VAE is already fp16 on disk; do not re-cast.
        cfg = _load_config(self.ckpt_dir / "shape_dec_next_dc_f16c32.config.json")
        m = FlexiDualGridVaeDecoder(**cfg["args"])
        _load_weights_prefixed(m, self.ckpt_dir / "shape_dec_next_dc_f16c32.safetensors")
        return m

    # --- core steps --------------------------------------------------------

    def _cond_from_image(self, image: Image.Image) -> Tuple[mx.array, mx.array]:
        cond = self.dino([image])  # (1, N, 1024)
        neg = mx.zeros_like(cond)
        return cond, neg

    def _sample_ss(self, cond: mx.array, neg: mx.array, seed: int) -> mx.array:
        """Run SS flow + decoder -> binary (1, 1, 64, 64, 64) occupancy."""
        params = self.pipeline_config["sparse_structure_sampler"]["params"]
        mx.random.seed(seed)
        noise = mx.random.normal((1, 8, 16, 16, 16))
        z_s = self.sampler.sample(
            self.ss_flow, noise,
            cond=cond, neg_cond=neg,
            steps=params["steps"],
            guidance_strength=params["guidance_strength"],
            guidance_interval=tuple(params["guidance_interval"]),
            guidance_rescale=params["guidance_rescale"],
            rescale_t=params["rescale_t"],
        )["samples"]
        occupancy = self.ss_dec(z_s)  # (1, 1, 64, 64, 64)
        return occupancy

    def _coords_from_occupancy(self, occupancy: mx.array, target_res: int) -> mx.array:
        """occupancy: (1, 1, 64, 64, 64) logits → binary mask → coords (F, 4)."""
        binary = occupancy > 0
        # Target coord res for 512 pipeline is 32 (per upstream ss_res mapping)
        if target_res != 64:
            # Max-pool from 64 to target_res
            ratio = 64 // target_res
            assert 64 % target_res == 0
            # MLX doesn't have max_pool3d with stride > kernel cleanly; do a simple reduction.
            arr = np.asarray(binary)[0, 0]  # (64, 64, 64) bool
            arr = arr.reshape(target_res, ratio, target_res, ratio, target_res, ratio).max(axis=(1, 3, 5))
            binary_np = arr
        else:
            binary_np = np.asarray(binary)[0, 0]
        ys, xs, zs = np.where(binary_np)
        n = ys.size
        coords = np.stack([np.zeros(n, dtype=np.int32), ys.astype(np.int32), xs.astype(np.int32), zs.astype(np.int32)], axis=-1)
        return mx.array(coords)

    def _sample_shape_slat(
        self,
        flow: SLatFlowModel,
        cond: mx.array,
        neg: mx.array,
        coords: mx.array,
        seed_offset: int,
    ) -> SparseTensor:
        """Run `flow` on the given coords → denormalized shape latent."""
        params = self.pipeline_config["shape_slat_sampler"]["params"]
        mx.random.seed(seed_offset)
        F = coords.shape[0]
        spatial = (flow.resolution, flow.resolution, flow.resolution)
        noise_feats = mx.random.normal((F, flow.in_channels))
        noise = SparseTensor(feats=noise_feats, coords=coords, spatial_shape=spatial)
        slat = self.sampler.sample(
            flow, noise,
            cond=cond, neg_cond=neg,
            steps=params["steps"],
            guidance_strength=params["guidance_strength"],
            guidance_interval=tuple(params["guidance_interval"]),
            guidance_rescale=params["guidance_rescale"],
            rescale_t=params["rescale_t"],
        )["samples"]
        # Denormalize
        slat = slat.replace(slat.feats * self.shape_slat_std + self.shape_slat_mean)
        return slat

    def _sample_shape_slat_cascade(
        self,
        cond_lr: mx.array, neg_lr: mx.array,
        cond_hr: mx.array, neg_hr: mx.array,
        coords: mx.array,  # coords at ss_target_res grid
        seed: int,
    ) -> tuple[SparseTensor, int]:
        """Two-stage cascade: sample shape SLat at `lr_resolution`, use VAE.upsample
        to derive coords at `hr_resolution`, then resample at hr. Returns
        `(slat_hr, effective_hr_resolution)` — the hr may be reduced from the
        requested value to stay under `max_num_tokens` (1024 is always stable).
        """
        # LR stage
        slat_lr = self._sample_shape_slat(self.shape_flow_lr, cond_lr, neg_lr, coords, seed + 1)
        mx.eval(slat_lr.feats)

        # VAE upsample → coords at (lr_resolution)^3 grid (16x of input 32^3 = 512^3)
        hr_coords_mx = self.shape_vae.upsample(slat_lr, upsample_times=4)
        mx.eval(hr_coords_mx)
        hr_coords_np = np.asarray(hr_coords_mx)

        # Quantize 512-res coords down to (hr_resolution // 16)^3 grid, with token cap.
        # Upstream: coords = round((hr_coords + 0.5) / lr_resolution * (hr_resolution // 16))
        hr_resolution = self.hr_resolution
        assert hr_resolution is not None
        while True:
            batch_col = hr_coords_np[:, :1]
            scale = (hr_resolution // 16) / self.lr_resolution
            xyz = ((hr_coords_np[:, 1:] + 0.5) * scale).astype(np.int32)
            quant = np.concatenate([batch_col, xyz], axis=1)
            coords_hr_np = np.unique(quant, axis=0)
            num_tokens = coords_hr_np.shape[0]
            if num_tokens < self.max_num_tokens or hr_resolution == 1024:
                if hr_resolution != self.hr_resolution:
                    print(f"      cascade: capped hr_resolution to {hr_resolution} to fit max_num_tokens={self.max_num_tokens}")
                break
            hr_resolution -= 128

        print(f"      cascade hr_resolution={hr_resolution}  tokens={num_tokens}")
        coords_hr = mx.array(coords_hr_np.astype(np.int32))

        # HR stage
        slat_hr = self._sample_shape_slat(self.shape_flow_hr, cond_hr, neg_hr, coords_hr, seed + 2)
        mx.eval(slat_hr.feats)
        return slat_hr, hr_resolution

    def _decode_shape(self, slat: SparseTensor, resolution: int) -> Tuple[np.ndarray, np.ndarray]:
        """Shape VAE decode + dual-grid mesh extraction. Returns (V, F) numpy."""
        self.shape_vae.set_resolution(resolution)
        vertices, intersected, quad_lerp, subs = self.shape_vae.decode(slat, return_subs=True)
        mx.eval(vertices.feats, intersected.feats, quad_lerp.feats)
        coords_np = np.asarray(vertices.coords)[:, 1:]
        dv_np = np.asarray(vertices.feats)
        inter_np = np.asarray(intersected.feats) > 0.5
        quad_np = np.asarray(quad_lerp.feats)
        aabb = (np.array([-0.5, -0.5, -0.5]), np.array([0.5, 0.5, 0.5]))
        V, F = flexible_dual_grid_to_mesh(coords_np, dv_np, inter_np, quad_np, aabb, [resolution] * 3)
        return V, F

    # --- public API --------------------------------------------------------

    def _preprocess_with_rembg(self, image: Image.Image) -> Image.Image:
        """Run preprocess_image, lazily constructing a BiRefNet rembg only when the
        input lacks an alpha channel. Aggressively disposes the rembg weights
        immediately after to free ~1 GB before the DiT inference phase — without
        this gc.collect, lingering torch tensors prevent Metal from satisfying
        MLX allocations and the SS flow stage silently OOMs.
        """
        needs_rembg = image.mode != "RGBA" or (np.asarray(image.convert("RGBA"))[:, :, 3] == 255).all()
        if not needs_rembg:
            return preprocess_image(image, rembg_model=None)
        rembg = BiRefNetRembg(self._rembg_model_name, device=self.rembg_device)
        try:
            out = preprocess_image(image, rembg_model=rembg)
        finally:
            rembg.unload()
            del rembg
            import gc
            gc.collect()
        return out

    def _dino_conds(self, image: Image.Image) -> dict[int, tuple[mx.array, mx.array]]:
        """Compute DINOv3 features at each size in self.cond_sizes. Cached by size."""
        out: dict[int, tuple[mx.array, mx.array]] = {}
        for size in self.cond_sizes:
            self.dino.set_image_size(size)
            cond = self.dino([image])
            neg = mx.zeros_like(cond)
            out[size] = (cond, neg)
        return out

    def run(self, image: Image.Image, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """image (RGBA) → (vertices (V, 3), faces (T, 3))."""
        import time as _time
        t0 = _time.time()
        print("[1/5] preprocess")
        pre = self._preprocess_with_rembg(image)
        t1 = _time.time(); print(f"      +{t1-t0:.1f}s")
        print(f"[2/5] DINOv3 image conditioning @ {list(self.cond_sizes)}")
        conds = self._dino_conds(pre)
        cond_512, neg_512 = conds[self.cond_sizes[0]]  # always the first (lowest) is used for SS flow
        t2 = _time.time(); print(f"      cond shapes: {[tuple(v[0].shape) for v in conds.values()]}   +{t2-t1:.1f}s")

        print("[3/5] sparse structure flow + decode")
        occupancy = self._sample_ss(cond_512, neg_512, seed)
        coords = self._coords_from_occupancy(occupancy, self.ss_target_res)
        t3 = _time.time(); print(f"      active voxels at {self.ss_target_res}^3: {coords.shape[0]}   +{t3-t2:.1f}s")

        print("[4/5] shape SLat flow" + (" (cascade)" if self.is_cascade else ""))
        if self.is_cascade:
            # LR uses 512 cond; HR uses 1024 cond.
            cond_lr, neg_lr = conds[512]
            cond_hr, neg_hr = conds[1024]
            slat, effective_res = self._sample_shape_slat_cascade(
                cond_lr, neg_lr, cond_hr, neg_hr, coords, seed
            )
            # If cascade had to cap hr_resolution (1536 → <1536), follow it down for VAE too.
            vae_res = effective_res
        else:
            # Non-cascade: cond_sizes has exactly one entry; that's what the shape flow expects.
            cond, neg = conds[self.cond_sizes[0]]
            slat = self._sample_shape_slat(self.shape_flow_hr, cond, neg, coords, seed + 1)
            mx.eval(slat.feats)
            vae_res = self.vae_res
        t4 = _time.time(); print(f"      slat feats: {tuple(slat.feats.shape)}   +{t4-t3:.1f}s")

        print(f"[5/5] shape VAE decode + dual-grid mesh extract @ {vae_res}^3")
        V, F = self._decode_shape(slat, vae_res)
        t5 = _time.time(); print(f"      mesh: {V.shape[0]} verts, {F.shape[0]} faces   +{t5-t4:.1f}s")
        return V, F

    @classmethod
    def from_pretrained(
        cls,
        ckpt_dir: str | Path,
        pipeline_json: str | Path | None = None,
        pipeline_type: str = "512",
        dino_device: str = "cpu",
        rembg_device: str = "auto",
        dit_compute_dtype: str = "bfloat16",
    ) -> "Trellis2ImageTo3DPipelineMLX":
        ckpt_dir = Path(ckpt_dir)
        if pipeline_json is None:
            pipeline_json = ckpt_dir.parent / "weights" / "pipeline.json"
            if not pipeline_json.exists():
                pipeline_json = ckpt_dir / "pipeline.json"
        cfg = _load_config(Path(pipeline_json))
        return cls(
            ckpt_dir=ckpt_dir, pipeline_config=cfg["args"],
            pipeline_type=pipeline_type, dino_device=dino_device,
            rembg_device=rembg_device, dit_compute_dtype=dit_compute_dtype,
        )

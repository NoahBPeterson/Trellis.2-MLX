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
from .models.sparse_vae import FlexiDualGridVaeDecoder, SparseUnetVaeDecoder
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


# Per-pipeline-type stage weight tables for the live progress / ETA display.
# Calibrated against measured M1 Pro fp16 runs on T.png. Numbers don't have to
# be exact — the ETA self-corrects as stages complete (we use elapsed/done_frac
# extrapolation, so a stage that takes 2x its budget just stretches the remainder).
_STAGE_WEIGHTS: dict[str, dict[str, float]] = {
    "512":          {"preprocess": 0.02,  "dino": 0.02,  "ss": 0.46, "shape": 0.42, "vae": 0.08},
    "1024":         {"preprocess": 0.01,  "dino": 0.005, "ss": 0.09, "shape": 0.85, "vae": 0.045},
    "1024_cascade": {"preprocess": 0.01,  "dino": 0.006, "ss": 0.10, "shape": 0.82, "vae": 0.064},
    "1536_cascade": {"preprocess": 0.01,  "dino": 0.006, "ss": 0.07, "shape": 0.88, "vae": 0.034},
}

# When `with_pbr=True`, stages 6+7 (texture flow + texture VAE) ~ same scale as
# stages 4+5 since they reuse the shape backbone shapes. Approximation:
# tex_flow ≈ shape_flow / 2 (no CFG for tex), tex_vae ≈ shape_vae.
def _stage_weights_pbr(base: dict[str, float]) -> dict[str, float]:
    # Compute weights so shape stages, plus tex_flow≈shape*0.5 and tex_vae≈vae,
    # all sum to 1.0.
    raw = {**base, "tex_flow": base["shape"] * 0.5, "tex_vae": base["vae"]}
    s = sum(raw.values())
    return {k: v / s for k, v in raw.items()}


def _fmt_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds // 60}m{seconds % 60:02d}s"


class _ProgressTracker:
    """Tracks weighted stage completion and projects ETA from elapsed×inv-fraction."""

    def __init__(self, weights: dict[str, float]):
        self.weights = weights
        self.t0 = None
        self.done_frac = 0.0

    def start(self) -> None:
        import time as _time
        self.t0 = _time.time()

    def stage_done(self, stage: str) -> str:
        """Mark a stage finished and return a `[XX% • elapsed YsZs • ETA M:SS]` suffix."""
        import time as _time
        self.done_frac = min(1.0, self.done_frac + self.weights.get(stage, 0.0))
        elapsed = _time.time() - (self.t0 or _time.time())
        if self.done_frac > 0.001:
            total_est = elapsed / self.done_frac
            eta = max(0.0, total_est - elapsed)
            return f"[{self.done_frac*100:5.1f}% • elapsed {_fmt_duration(elapsed)} • ETA {_fmt_duration(eta)}]"
        return f"[  ?  % • elapsed {_fmt_duration(elapsed)} • ETA ?]"


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
        "tex_flow": "slat_flow_imgshape2tex_dit_1_3B_512",
        "tex_cond_size": 512,
        "cond_sizes": (512,),
        "ss_target_res": 32, "vae_res": 512, "cascade": False,
    },
    "1024": {
        "flow_lr": "slat_flow_img2shape_dit_1_3B_1024",
        "flow_hr": "slat_flow_img2shape_dit_1_3B_1024",
        "tex_flow": "slat_flow_imgshape2tex_dit_1_3B_1024",
        "tex_cond_size": 1024,
        "cond_sizes": (1024,),
        "ss_target_res": 64, "vae_res": 1024, "cascade": False,
    },
    "1024_cascade": {
        "flow_lr": "slat_flow_img2shape_dit_1_3B_512",
        "flow_hr": "slat_flow_img2shape_dit_1_3B_1024",
        "tex_flow": "slat_flow_imgshape2tex_dit_1_3B_1024",
        "tex_cond_size": 1024,
        "cond_sizes": (512, 1024),
        "ss_target_res": 32, "vae_res": 1024, "cascade": True,
        "lr_resolution": 512, "hr_resolution": 1024,
    },
    "1536_cascade": {
        "flow_lr": "slat_flow_img2shape_dit_1_3B_512",
        "flow_hr": "slat_flow_img2shape_dit_1_3B_1024",
        "tex_flow": "slat_flow_imgshape2tex_dit_1_3B_1024",
        "tex_cond_size": 1024,
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
        with_pbr: bool = False,
    ):
        self.ckpt_dir = Path(ckpt_dir)
        self.pipeline_type = pipeline_type
        self.pipeline_config = pipeline_config
        self.max_num_tokens = max_num_tokens
        self.rembg_device = rembg_device
        self.with_pbr = with_pbr
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
        self._tex_cond_size: int = p["tex_cond_size"]

        # Models — shape side (always loaded)
        self.ss_flow = self._load_ss_flow()
        self.ss_dec = self._load_ss_dec()
        self.shape_flow_lr = self._load_shape_flow(p["flow_lr"])
        if p["flow_hr"] == p["flow_lr"]:
            self.shape_flow_hr = self.shape_flow_lr
        else:
            self.shape_flow_hr = self._load_shape_flow(p["flow_hr"])
        self.shape_vae = self._load_shape_vae()

        # Models — texture side (only when with_pbr=True; +3.6 GB at fp16)
        self.tex_flow: Optional[SLatFlowModel] = None
        self.tex_vae: Optional[SparseUnetVaeDecoder] = None
        if with_pbr:
            self.tex_flow = self._load_shape_flow(p["tex_flow"])  # same loader (SLatFlowModel arch)
            self.tex_vae = self._load_tex_vae()
            # Texture flow needs both 512 and 1024 conds depending on pipeline_type;
            # add tex cond size to the cond_sizes set if not already there.
            if self._tex_cond_size not in self.cond_sizes:
                self.cond_sizes = tuple(sorted(set(list(self.cond_sizes) + [self._tex_cond_size])))

        self.dino = DinoV3FeatureExtractor(
            pipeline_config["image_cond_model"]["args"]["model_name"],
            image_size=self.cond_sizes[0],
            device=dino_device,
        )

        # Samplers
        self.sampler = FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)

        # Normalization stats for shape SLat (always) and texture SLat (when PBR is on)
        self.shape_slat_mean = mx.array(pipeline_config["shape_slat_normalization"]["mean"], dtype=mx.float32)
        self.shape_slat_std = mx.array(pipeline_config["shape_slat_normalization"]["std"], dtype=mx.float32)
        if with_pbr:
            self.tex_slat_mean = mx.array(pipeline_config["tex_slat_normalization"]["mean"], dtype=mx.float32)
            self.tex_slat_std = mx.array(pipeline_config["tex_slat_normalization"]["std"], dtype=mx.float32)

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

    def _load_tex_vae(self) -> SparseUnetVaeDecoder:
        # Texture VAE: out_channels=6 (PBR), pred_subdiv=False (uses shape decoder's
        # subdivision masks via guide_subs at decode time). Already fp16 on disk.
        cfg = _load_config(self.ckpt_dir / "tex_dec_next_dc_f16c32.config.json")
        m = SparseUnetVaeDecoder(**cfg["args"])
        _load_weights_prefixed(m, self.ckpt_dir / "tex_dec_next_dc_f16c32.safetensors")
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

    def _sample_tex_slat(
        self,
        cond: mx.array,
        neg: mx.array,
        shape_slat: SparseTensor,
        seed_offset: int,
    ) -> SparseTensor:
        """Sample texture SLat conditioned on shape latent. Mirrors upstream
        sample_tex_slat: re-normalize shape, allocate (F, 32) noise (= in_channels=64
        minus shape_feats=32), sample with concat_cond=shape_slat, denormalize."""
        assert self.tex_flow is not None and self.tex_slat_mean is not None
        params = self.pipeline_config["tex_slat_sampler"]["params"]
        # Re-normalize shape SLat to its training distribution
        shape_normed = shape_slat.replace((shape_slat.feats - self.shape_slat_mean) / self.shape_slat_std)
        F = shape_slat.coords.shape[0]
        noise_dim = self.tex_flow.in_channels - shape_slat.feats.shape[1]
        assert noise_dim == 32, f"expected 32 noise channels for tex flow (in_channels=64, shape=32), got {noise_dim}"
        mx.random.seed(seed_offset)
        noise_feats = mx.random.normal((F, noise_dim))
        spatial = (self.tex_flow.resolution,) * 3
        noise = SparseTensor(feats=noise_feats, coords=shape_slat.coords, spatial_shape=spatial)
        slat = self.sampler.sample(
            self.tex_flow, noise,
            cond=cond, neg_cond=neg,
            steps=params["steps"],
            guidance_strength=params["guidance_strength"],
            guidance_interval=tuple(params["guidance_interval"]),
            guidance_rescale=params["guidance_rescale"],
            rescale_t=params["rescale_t"],
            concat_cond=shape_normed,
        )["samples"]
        # Denormalize texture SLat
        slat = slat.replace(slat.feats * self.tex_slat_std + self.tex_slat_mean)
        return slat

    def _decode_shape(self, slat: SparseTensor, resolution: int):
        """Shape VAE decode + dual-grid mesh extraction.

        Returns (V, F, subs, vertex_voxel_coords) — `subs` are the per-stage
        subdivision masks needed to drive the texture VAE; `vertex_voxel_coords`
        is the (F_vertices, 3) array of voxel coords-per-mesh-vertex used to
        attribute per-voxel PBR back to the mesh.
        """
        self.shape_vae.set_resolution(resolution)
        vertices, intersected, quad_lerp, subs = self.shape_vae.decode(slat, return_subs=True)
        mx.eval(vertices.feats, intersected.feats, quad_lerp.feats)
        coords_np = np.asarray(vertices.coords)[:, 1:]
        dv_np = np.asarray(vertices.feats)
        inter_np = np.asarray(intersected.feats) > 0.5
        quad_np = np.asarray(quad_lerp.feats)
        aabb = (np.array([-0.5, -0.5, -0.5]), np.array([0.5, 0.5, 0.5]))
        V, F = flexible_dual_grid_to_mesh(coords_np, dv_np, inter_np, quad_np, aabb, [resolution] * 3)
        return V, F, subs, coords_np

    def _decode_texture(self, tex_slat: SparseTensor, subs) -> np.ndarray:
        """Run the texture VAE with `subs` as guide_subs (from the shape decoder).
        Returns per-voxel PBR (F_active, 6) numpy in [0, 1] range; channels are
        [0:3]=base_color RGB, [3]=metallic, [4]=roughness, [5]=alpha.
        """
        assert self.tex_vae is not None
        out = self.tex_vae(tex_slat, guide_subs=subs)
        # Upstream rescales decoder output from [-1, 1] to [0, 1]
        attrs = out.feats * 0.5 + 0.5
        mx.eval(attrs)
        return np.asarray(attrs)

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

    def run(self, image: Image.Image, seed: int = 42):
        """image (RGBA or RGB) → mesh.

        With `with_pbr=False` (default): returns `(V, F)` — vertices, faces.
        With `with_pbr=True`: returns `(V, F, vertex_attrs)` where vertex_attrs
        is `(N_v, 6)` per-vertex PBR in [0, 1]: [RGB, metallic, roughness, alpha].
        """
        import time as _time
        weights = _STAGE_WEIGHTS.get(self.pipeline_type, _STAGE_WEIGHTS["512"])
        if self.with_pbr:
            weights = _stage_weights_pbr(weights)
        prog = _ProgressTracker(weights)
        prog.start()

        n_stages = "7" if self.with_pbr else "5"
        t0 = _time.time()
        print(f"[1/{n_stages}] preprocess")
        pre = self._preprocess_with_rembg(image)
        t1 = _time.time(); print(f"      +{t1-t0:.1f}s   {prog.stage_done('preprocess')}")

        print(f"[2/{n_stages}] DINOv3 image conditioning @ {list(self.cond_sizes)}")
        conds = self._dino_conds(pre)
        cond_512, neg_512 = conds[self.cond_sizes[0]]
        t2 = _time.time(); print(f"      cond shapes: {[tuple(v[0].shape) for v in conds.values()]}   +{t2-t1:.1f}s   {prog.stage_done('dino')}")

        print(f"[3/{n_stages}] sparse structure flow + decode")
        occupancy = self._sample_ss(cond_512, neg_512, seed)
        coords = self._coords_from_occupancy(occupancy, self.ss_target_res)
        t3 = _time.time(); print(f"      active voxels at {self.ss_target_res}^3: {coords.shape[0]}   +{t3-t2:.1f}s   {prog.stage_done('ss')}")

        print(f"[4/{n_stages}] shape SLat flow" + (" (cascade)" if self.is_cascade else ""))
        if self.is_cascade:
            cond_lr, neg_lr = conds[512]
            cond_hr, neg_hr = conds[1024]
            slat, effective_res = self._sample_shape_slat_cascade(
                cond_lr, neg_lr, cond_hr, neg_hr, coords, seed
            )
            vae_res = effective_res
        else:
            cond, neg = conds[self.cond_sizes[0]]
            slat = self._sample_shape_slat(self.shape_flow_hr, cond, neg, coords, seed + 1)
            mx.eval(slat.feats)
            vae_res = self.vae_res
        t4 = _time.time(); print(f"      slat feats: {tuple(slat.feats.shape)}   +{t4-t3:.1f}s   {prog.stage_done('shape')}")

        print(f"[5/{n_stages}] shape VAE decode + dual-grid mesh extract @ {vae_res}^3")
        V, F, subs, vertex_voxel_coords = self._decode_shape(slat, vae_res)
        t5 = _time.time(); print(f"      mesh: {V.shape[0]} verts, {F.shape[0]} faces   +{t5-t4:.1f}s   {prog.stage_done('vae')}")
        if not self.with_pbr:
            return V, F

        # Stage 6 — texture SLat flow (1.3B DiT, shape-conditioned)
        print(f"[6/{n_stages}] texture SLat flow")
        tex_cond, tex_neg = conds[self._tex_cond_size]
        tex_slat = self._sample_tex_slat(tex_cond, tex_neg, slat, seed + 17)
        mx.eval(tex_slat.feats)
        t6 = _time.time(); print(f"      tex slat feats: {tuple(tex_slat.feats.shape)}   +{t6-t5:.1f}s   {prog.stage_done('tex_flow')}")

        # Stage 7 — texture VAE decode → per-voxel PBR. Uses `subs` from shape decoder
        # so the upsample structure matches the shape mesh's per-vertex voxel coords.
        print(f"[7/{n_stages}] texture VAE decode → per-vertex PBR")
        per_voxel_attrs = self._decode_texture(tex_slat, subs)  # (N, 6) in [0, 1]
        # Vertex i ↔ voxel i (same row index in the dual-grid output), so this is a
        # direct take. Trim/pad if shapes don't match (defensive — shouldn't happen).
        n = min(per_voxel_attrs.shape[0], V.shape[0])
        vertex_attrs = per_voxel_attrs[:n]
        t7 = _time.time(); print(f"      vertex attrs: {vertex_attrs.shape}  rgb_mean={vertex_attrs[:, :3].mean():.3f}  alpha_mean={vertex_attrs[:, 5].mean():.3f}   +{t7-t6:.1f}s   {prog.stage_done('tex_vae')}")
        return V, F, vertex_attrs

    @classmethod
    def from_pretrained(
        cls,
        ckpt_dir: str | Path,
        pipeline_json: str | Path | None = None,
        pipeline_type: str = "512",
        dino_device: str = "cpu",
        rembg_device: str = "auto",
        dit_compute_dtype: str = "bfloat16",
        with_pbr: bool = False,
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
            with_pbr=with_pbr,
        )

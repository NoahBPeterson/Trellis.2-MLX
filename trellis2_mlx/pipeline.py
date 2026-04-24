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


def _load_weights_prefixed(module, path: Path) -> None:
    """Load MLX safetensors where every tensor is prefixed `<component>.`; strip the
    prefix so `module.load_weights` sees the bare submodule names it expects.

    Works with both prefixed (unified-index) and un-prefixed legacy shards.
    """
    path = Path(path)
    prefix = _CKPT_PREFIXES.get(path.name)
    if prefix is None:
        module.load_weights(str(path))
        return
    raw = mx.load(str(path))
    pfx = prefix + "."
    if all(k.startswith(pfx) for k in raw):
        stripped = [(k[len(pfx):], v) for k, v in raw.items()]
        module.load_weights(stripped)
    else:
        # Legacy un-prefixed file; load directly.
        module.load_weights(str(path))


# pipeline_type → (shape flow ckpt stem, DINOv3 image size, SS target_res, VAE res)
_PIPELINE_PARAMS: dict[str, tuple[str, int, int, int]] = {
    "512":  ("slat_flow_img2shape_dit_1_3B_512",  512,  32,  512),
    "1024": ("slat_flow_img2shape_dit_1_3B_1024", 1024, 64, 1024),
}


class Trellis2ImageTo3DPipelineMLX:
    """All-in-memory pipeline. ~10 GB of MLX weights + ~1 GB torch DINOv3."""

    def __init__(
        self,
        ckpt_dir: str | Path,
        pipeline_config: dict,
        pipeline_type: str = "512",
        dino_device: str = "cpu",
    ):
        self.ckpt_dir = Path(ckpt_dir)
        self.pipeline_type = pipeline_type
        self.pipeline_config = pipeline_config
        if pipeline_type in ("1024_cascade", "1536_cascade"):
            raise NotImplementedError(
                f"pipeline_type={pipeline_type!r} needs the VAE upsample() primitive; not yet wired (Phase 5B)"
            )
        if pipeline_type not in _PIPELINE_PARAMS:
            raise ValueError(f"unknown pipeline_type {pipeline_type!r}; expected one of {list(_PIPELINE_PARAMS)}")
        shape_stem, self.dino_image_size, self.ss_target_res, self.vae_res = _PIPELINE_PARAMS[pipeline_type]

        # Models
        self.ss_flow = self._load_ss_flow()
        self.ss_dec = self._load_ss_dec()
        self.shape_flow = self._load_shape_flow(shape_stem)
        self.shape_vae = self._load_shape_vae()
        self.dino = DinoV3FeatureExtractor(
            pipeline_config["image_cond_model"]["args"]["model_name"],
            image_size=self.dino_image_size,
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
        _load_weights_prefixed(m, self.ckpt_dir / "ss_flow_img_dit_1_3B_64.safetensors")
        return m

    def _load_ss_dec(self) -> SparseStructureDecoder:
        cfg = _load_config(self.ckpt_dir / "ss_dec_conv3d_16l8.config.json")
        m = SparseStructureDecoder(**cfg["args"])
        _load_weights_prefixed(m, self.ckpt_dir / "ss_dec_conv3d_16l8.safetensors")
        return m

    def _load_shape_flow(self, stem: str) -> SLatFlowModel:
        cfg = _load_config(self.ckpt_dir / f"{stem}.config.json")
        m = SLatFlowModel(**_strip_unused(cfg["args"]))
        _load_weights_prefixed(m, self.ckpt_dir / f"{stem}.safetensors")
        return m

    def _load_shape_vae(self) -> FlexiDualGridVaeDecoder:
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

    def _sample_shape_slat(self, cond: mx.array, neg: mx.array, coords: mx.array, seed: int) -> SparseTensor:
        """Run shape SLat flow on the given coords → denormalized shape latent."""
        params = self.pipeline_config["shape_slat_sampler"]["params"]
        mx.random.seed(seed + 1)
        F = coords.shape[0]
        spatial = (self.shape_flow.resolution, self.shape_flow.resolution, self.shape_flow.resolution)
        noise_feats = mx.random.normal((F, self.shape_flow.in_channels))
        noise = SparseTensor(feats=noise_feats, coords=coords, spatial_shape=spatial)
        slat = self.sampler.sample(
            self.shape_flow, noise,
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

    def run(self, image: Image.Image, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """image (RGBA) → (vertices (V, 3), faces (T, 3))."""
        import time as _time
        t0 = _time.time()
        print("[1/5] preprocess")
        pre = preprocess_image(image)
        t1 = _time.time(); print(f"      +{t1-t0:.1f}s")
        print("[2/5] DINOv3 image conditioning")
        cond, neg = self._cond_from_image(pre)
        t2 = _time.time(); print(f"      cond shape: {tuple(cond.shape)}   +{t2-t1:.1f}s")
        print("[3/5] sparse structure flow + decode")
        occupancy = self._sample_ss(cond, neg, seed)
        coords = self._coords_from_occupancy(occupancy, self.ss_target_res)
        t3 = _time.time(); print(f"      active voxels at {self.ss_target_res}^3: {coords.shape[0]}   +{t3-t2:.1f}s")
        print("[4/5] shape SLat flow")
        slat = self._sample_shape_slat(cond, neg, coords, seed)
        mx.eval(slat.feats)
        t4 = _time.time(); print(f"      slat feats: {tuple(slat.feats.shape)}   +{t4-t3:.1f}s")
        print("[5/5] shape VAE decode + dual-grid mesh extract")
        V, F = self._decode_shape(slat, self.vae_res)
        t5 = _time.time(); print(f"      mesh: {V.shape[0]} verts, {F.shape[0]} faces   +{t5-t4:.1f}s")
        return V, F

    @classmethod
    def from_pretrained(
        cls,
        ckpt_dir: str | Path,
        pipeline_json: str | Path | None = None,
        pipeline_type: str = "512",
        dino_device: str = "cpu",
    ) -> "Trellis2ImageTo3DPipelineMLX":
        ckpt_dir = Path(ckpt_dir)
        if pipeline_json is None:
            pipeline_json = ckpt_dir.parent / "weights" / "pipeline.json"
            if not pipeline_json.exists():
                pipeline_json = ckpt_dir / "pipeline.json"
        cfg = _load_config(Path(pipeline_json))
        return cls(ckpt_dir=ckpt_dir, pipeline_config=cfg["args"], pipeline_type=pipeline_type, dino_device=dino_device)

"""Dump the SS decoder's 64^3 binary occupancy as a voxel cube GLB for visual inspection.

If this looks like the input silhouette (e.g. a T), we know the bug is in the
downstream shape VAE / dual-grid; if it's garbage, the bug is earlier.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trellis2_mlx.image_cond import DinoV3FeatureExtractor
from trellis2_mlx.models.flow_dit import SparseStructureFlowModel
from trellis2_mlx.models.ss_decoder import SparseStructureDecoder
from trellis2_mlx.preprocess import preprocess_image
from trellis2_mlx.samplers import FlowEulerGuidanceIntervalSampler


def main() -> int:
    image_path = ROOT / "upstream/assets/example_image/T.png"
    out_path = ROOT / "artifacts/ss_occupancy.glb"
    pipeline_json = ROOT / "weights/pipeline.json"

    pipe_cfg = json.loads(pipeline_json.read_text())["args"]
    print(f"Loading SS models...")
    ss_cfg = json.loads((ROOT / "ckpts/ss_flow_img_dit_1_3B_64.config.json").read_text())
    ss_flow = SparseStructureFlowModel(**{k: v for k, v in ss_cfg["args"].items() if k not in ("initialization", "dtype")})
    ss_flow.load_weights(str(ROOT / "ckpts/ss_flow_img_dit_1_3B_64.safetensors"))
    dec_cfg = json.loads((ROOT / "ckpts/ss_dec_conv3d_16l8.config.json").read_text())
    ss_dec = SparseStructureDecoder(**dec_cfg["args"])
    ss_dec.load_weights(str(ROOT / "ckpts/ss_dec_conv3d_16l8.safetensors"))

    print(f"Loading DINOv3...")
    dino = DinoV3FeatureExtractor(pipe_cfg["image_cond_model"]["args"]["model_name"], image_size=512, device="cpu")

    print(f"Running SS flow + decoder on {image_path.name}...")
    img = Image.open(image_path)
    pre = preprocess_image(img)
    cond = dino([pre])
    neg = mx.zeros_like(cond)
    sampler = FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
    mx.random.seed(42)
    noise = mx.random.normal((1, 8, 16, 16, 16))
    params = pipe_cfg["sparse_structure_sampler"]["params"]
    t0 = time.time()
    z_s = sampler.sample(
        ss_flow, noise, cond=cond, neg_cond=neg,
        steps=params["steps"], guidance_strength=params["guidance_strength"],
        guidance_interval=tuple(params["guidance_interval"]),
        guidance_rescale=params["guidance_rescale"], rescale_t=params["rescale_t"],
    )["samples"]
    occupancy = ss_dec(z_s)
    mx.eval(occupancy)
    t1 = time.time()
    print(f"done in {t1 - t0:.1f}s")

    binary = np.asarray(occupancy > 0)[0, 0]  # (64, 64, 64) bool
    density = binary.mean()
    print(f"occupancy shape: {binary.shape}, density: {density:.3%}, active voxels: {int(binary.sum())}")

    # Build a voxel mesh: one small cube per active voxel
    coords = np.argwhere(binary).astype(np.float32)  # (N, 3) in (x, y, z) — or is it (z, y, x)?
    print(f"coord ranges: axis0=[{coords[:,0].min():.0f}, {coords[:,0].max():.0f}]  axis1=[{coords[:,1].min():.0f}, {coords[:,1].max():.0f}]  axis2=[{coords[:,2].min():.0f}, {coords[:,2].max():.0f}]")
    # Build instanced boxes (one per voxel) for visualization
    size = 1.0 / 64
    positions = coords / 64 - 0.5  # center the grid
    # trimesh: build as vertex+face list to avoid per-voxel mesh creation overhead
    # Each voxel is a unit cube at position; use trimesh.points_to_cube_mesh? No, let's just dump centers with tiny boxes.
    cube_verts = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ], dtype=np.float32) * (size / 2)
    cube_faces = np.array([
        [0, 1, 2], [0, 2, 3],   # bottom
        [4, 5, 6], [4, 6, 7],   # top
        [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7],
    ], dtype=np.int32)
    V = positions[:, None, :] + cube_verts[None]  # (N, 8, 3)
    V = V.reshape(-1, 3)
    F = cube_faces[None] + np.arange(positions.shape[0])[:, None, None] * 8  # (N, 12, 3)
    F = F.reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=V, faces=F)
    mesh.export(out_path)
    print(f"wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

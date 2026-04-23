"""Load the shape VAE decoder weights into MLX and count parameters."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trellis2_mlx.models.sparse_vae import FlexiDualGridVaeDecoder
from trellis2_mlx.modules.sparse_tensor import SparseTensor


def main() -> int:
    weights = ROOT / "ckpts" / "shape_dec_next_dc_f16c32.safetensors"
    cfg = json.loads((ROOT / "ckpts" / "shape_dec_next_dc_f16c32.config.json").read_text())
    print(f"args: { {k: v for k, v in cfg['args'].items() if k != 'block_args'} }")

    # Strip block_args from config (upstream passes {} dicts we don't need)
    model = FlexiDualGridVaeDecoder(**cfg["args"])

    t0 = time.time()
    model.load_weights(str(weights))
    print(f"loaded in {time.time() - t0:.2f}s")

    from mlx.utils import tree_flatten
    total = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"parameter count: {total / 1e6:.1f} M  ({sum(1 for _ in tree_flatten(model.parameters()))} tensors)")

    # Smoke forward pass: 64 latent voxels at 32^3 coord space (for 512 output pipeline)
    F = 64
    rng = np.random.default_rng(0)
    coords = np.concatenate([
        np.zeros((F, 1), dtype=np.int32),
        rng.integers(0, 32, (F, 3), dtype=np.int32),
    ], axis=-1)
    coords = np.unique(coords, axis=0)
    F = coords.shape[0]
    feats = rng.standard_normal((F, 32)).astype(np.float32) * 0.5
    latent = SparseTensor(feats=mx.array(feats), coords=mx.array(coords), spatial_shape=(32, 32, 32))

    print(f"\nForward on {F} latent voxels at 32^3 (expect 4 upsamples -> 512^3)...")
    t0 = time.time()
    vertices, intersected, quad_lerp, subs = model.decode(latent, return_subs=True)
    mx.eval(vertices.feats, intersected.feats, quad_lerp.feats)
    t1 = time.time()
    print(f"decode: {t1 - t0:.2f}s")
    print(f"output voxel count: {vertices.feats.shape[0]}")
    print(f"subs per stage: {[s.feats.shape for s in subs]}")
    print(f"vertices: shape={tuple(vertices.feats.shape)} range=[{float(vertices.feats.min()):.3f}, {float(vertices.feats.max()):.3f}]")
    print(f"intersected: shape={tuple(intersected.feats.shape)} mean={float(intersected.feats.mean()):.3f}")
    print(f"quad_lerp: shape={tuple(quad_lerp.feats.shape)} mean={float(quad_lerp.feats.mean()):.3f}")

    # Dual-grid mesh extraction
    from trellis2_mlx.postprocess.dual_grid import flexible_dual_grid_to_mesh
    coords_np = np.asarray(vertices.coords)[:, 1:]  # drop batch col
    dv_np = np.asarray(vertices.feats)
    inter_np = np.asarray(intersected.feats) > 0.5
    quad_np = np.asarray(quad_lerp.feats)
    grid_size = 512
    aabb = (np.array([-0.5, -0.5, -0.5]), np.array([0.5, 0.5, 0.5]))
    t0 = time.time()
    V, F_ = flexible_dual_grid_to_mesh(coords_np, dv_np, inter_np, quad_np, aabb, [grid_size] * 3)
    t1 = time.time()
    print(f"\ndual-grid extraction: {t1 - t0:.2f}s")
    print(f"mesh: vertices={V.shape[0]}, faces={F_.shape[0]}")
    if V.shape[0] > 0 and F_.shape[0] > 0:
        print(f"vertex world-space range: x=[{V[:,0].min():.3f}, {V[:,0].max():.3f}] y=[{V[:,1].min():.3f}, {V[:,1].max():.3f}] z=[{V[:,2].min():.3f}, {V[:,2].max():.3f}]")
        # Save a random-cond smoke GLB (will be geometric noise, just proves the path works)
        from trellis2_mlx.postprocess.glb_export import export_mesh_glb
        out_path = ROOT / "artifacts" / "smoke_vae_mesh.glb"
        export_mesh_glb(V, F_, out_path)
        print(f"wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

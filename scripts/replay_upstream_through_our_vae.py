"""Replay upstream's tex_slat through our texture VAE to isolate the bug.

The diff between upstream_ref.npz and pbr_intermediates.npz showed:
  - base_color RGB and alpha match upstream within ~0.05 ✓
  - metallic +0.30 too high; roughness −0.18 too low (both have ~half the
    expected variance)

Two suspects: (a) our texture flow / SLat sampling produces a wrong
tex_slat going into the VAE, or (b) our texture VAE itself produces wrong
output from a correct tex_slat.

This script feeds upstream's *known-good* tex_slat into OUR texture VAE and
prints per-channel stats against upstream's voxel_attrs.

  - If output matches upstream → DiT / sampler / normalization is the bug.
  - If output still has biased M/R → texture VAE is the bug.

Needs `subs` (subdivision masks) which come from shape decode. We don't
have those in upstream's npz, so we run upstream's shape_slat through
OUR shape VAE to produce them. If our shape VAE is faithful that's a
no-op; if not, we'll see it in the result.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trellis2_mlx.modules.sparse_tensor import SparseTensor
from trellis2_mlx.pipeline import Trellis2ImageTo3DPipelineMLX


def _print_stats(label: str, arr: np.ndarray) -> None:
    print(f"\n  {label} ({arr.shape}):")
    for c, ch in enumerate(["R", "G", "B", "metallic", "roughness", "alpha"]):
        col = arr[:, c].astype(np.float64)
        print(f"    {ch:9s} mean={col.mean():+.4f}  std={col.std():.4f}  "
              f"min={col.min():+.4f}  max={col.max():+.4f}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", type=Path,
                   default=ROOT / "artifacts" / "upstream_ref.npz")
    p.add_argument("--pipeline-type", type=str, default="512")
    args = p.parse_args()

    print(f"Loading upstream dump from {args.upstream}...")
    up = np.load(args.upstream)
    for k in ("shape_slat_coords", "shape_slat_feats",
              "tex_slat_coords", "tex_slat_feats", "voxel_attrs"):
        assert k in up, f"upstream npz missing {k}"
    n_ss = up["shape_slat_coords"].shape[0]
    n_tx = up["tex_slat_coords"].shape[0]
    print(f"  shape_slat: {n_ss} active voxels, feats {up['shape_slat_feats'].shape}")
    print(f"  tex_slat:   {n_tx} active voxels, feats {up['tex_slat_feats'].shape}")
    print(f"  voxel_attrs (target): {up['voxel_attrs'].shape}")

    print(f"\nLoading our pipeline (pipeline_type={args.pipeline_type})...")
    t0 = time.time()
    pipe = Trellis2ImageTo3DPipelineMLX.from_pretrained(
        ckpt_dir=ROOT / "weights" / "ckpts",
        pipeline_json=ROOT / "weights" / "pipeline.json",
        pipeline_type=args.pipeline_type,
        dino_device="cpu",      # not actually invoked, just satisfies init
        rembg_device="cpu",
        dit_compute_dtype="float16",
        with_pbr=True,
    )
    print(f"  loaded in {time.time()-t0:.1f}s")

    # Build MLX SparseTensors from upstream's tensors. spatial_shape is the
    # latent grid size — for pipeline_type='512' the SLat is at 32^3.
    latent_res = {"512": 32, "1024": 32, "1024_cascade": 32, "1536_cascade": 32}[args.pipeline_type]
    spatial = (latent_res, latent_res, latent_res)
    shape_slat = SparseTensor(
        feats=mx.array(up["shape_slat_feats"].astype(np.float32)),
        coords=mx.array(up["shape_slat_coords"].astype(np.int32)),
        spatial_shape=spatial,
    )
    tex_slat = SparseTensor(
        feats=mx.array(up["tex_slat_feats"].astype(np.float32)),
        coords=mx.array(up["tex_slat_coords"].astype(np.int32)),
        spatial_shape=spatial,
    )

    # Step 1: shape VAE on upstream's shape_slat → produce `subs` for tex VAE.
    res = int(args.pipeline_type) if args.pipeline_type in ("512", "1024") else 512
    print(f"\nStep 1: OUR shape VAE on UPSTREAM's shape_slat (resolution={res})...")
    t0 = time.time()
    pipe.shape_vae.set_resolution(res)
    vertices, intersected, quad_lerp, subs = pipe.shape_vae.decode(shape_slat, return_subs=True)
    mx.eval(vertices.feats, intersected.feats, quad_lerp.feats)
    print(f"  shape decode: {time.time()-t0:.1f}s, "
          f"{len(subs)} subdivision levels, vertices.feats={vertices.feats.shape}")

    # Step 2: texture VAE on upstream's tex_slat with our subs.
    print(f"\nStep 2: OUR texture VAE on UPSTREAM's tex_slat (with our subs)...")
    t0 = time.time()
    out = pipe.tex_vae(tex_slat, guide_subs=subs)
    attrs = out.feats * 0.5 + 0.5
    mx.eval(attrs)
    voxel_attrs = np.asarray(attrs)
    print(f"  texture decode: {time.time()-t0:.1f}s, voxel_attrs={voxel_attrs.shape}")

    # Compare distributions
    print("\n=== RESULTS ===")
    _print_stats("UPSTREAM voxel_attrs (target)", up["voxel_attrs"])
    _print_stats("OURS-from-upstream-slat",       voxel_attrs)

    print("\n  Δ (ours-from-upstream-slat − upstream) per channel mean:")
    for c, ch in enumerate(["R", "G", "B", "metallic", "roughness", "alpha"]):
        d = float(voxel_attrs[:, c].mean() - up["voxel_attrs"][:, c].mean())
        flag = "  <-- DIVERGENCE" if abs(d) > 0.05 else ""
        print(f"    {ch:9s} Δmean={d:+.4f}{flag}")

    print("\n  Interpretation:")
    print("    ALL channels match within 0.05  → our VAEs are correct, the bug is")
    print("                                       in our texture flow / sampler / norm.")
    print("    M/R STILL biased like before    → our texture VAE is the bug.")
    print("    Different bias pattern          → our shape VAE produces wrong subs,")
    print("                                       which corrupts the texture VAE input.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

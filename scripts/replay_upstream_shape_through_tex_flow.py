"""Feed upstream's shape_slat through OUR texture flow + texture VAE.

If our texture flow is correct, this should produce voxel_attrs matching
upstream's. If divergence persists with the known-good shape_slat input,
the bug is in our texture flow DiT (not just shape flow).

Note on RNG: the texture flow's noise is sampled by MLX, so it differs from
upstream's. The point isn't bit-exact equality — it's per-channel
distribution stats. If we average over 1.5M voxels, RNG-induced channel
mean drift is < 0.02 (σ/√N). Anything bigger means a real bug.
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
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pipeline-type", type=str, default="512")
    args = p.parse_args()

    print(f"Loading upstream dump from {args.upstream}...")
    up = np.load(args.upstream)
    print(f"  shape_slat: {up['shape_slat_coords'].shape[0]} active voxels")
    print(f"  cond_512:   {up['cond_512'].shape}")

    print(f"\nLoading our pipeline...")
    t0 = time.time()
    pipe = Trellis2ImageTo3DPipelineMLX.from_pretrained(
        ckpt_dir=ROOT / "weights" / "ckpts",
        pipeline_json=ROOT / "weights" / "pipeline.json",
        pipeline_type=args.pipeline_type,
        dino_device="cpu",
        rembg_device="cpu",
        dit_compute_dtype="bfloat16",
        with_pbr=True,
    )
    print(f"  loaded in {time.time()-t0:.1f}s")

    # Build SparseTensor from upstream's shape_slat (denormalized).
    spatial = (32, 32, 32)
    shape_slat = SparseTensor(
        feats=mx.array(up["shape_slat_feats"].astype(np.float32)),
        coords=mx.array(up["shape_slat_coords"].astype(np.int32)),
        spatial_shape=spatial,
    )

    # We need a cond tensor as conditioning for tex flow. Load upstream's cond_512.
    # Our pipeline's get_cond returns dict with 'cond' and 'neg_cond'. neg_cond is just
    # zeros (per upstream's get_cond, line 182 of trellis2_image_to_3d.py).
    cond_512 = mx.array(up["cond_512"].astype(np.float32))
    neg_512 = mx.zeros_like(cond_512)

    # Now sample tex_slat using upstream's shape_slat as concat_cond.
    print(f"\nSampling our texture flow with UPSTREAM's shape_slat as concat_cond...")
    t0 = time.time()
    tex_slat = pipe._sample_tex_slat(cond_512, neg_512, shape_slat, args.seed + 17)
    mx.eval(tex_slat.feats)
    print(f"  done in {time.time()-t0:.1f}s, tex_slat feats={tex_slat.feats.shape}")

    print(f"\nDecoding tex_slat → voxel_attrs via our shape VAE + tex VAE pipeline...")
    t0 = time.time()
    pipe.shape_vae.set_resolution(512)
    _vertices, _intersected, _quad_lerp, subs = pipe.shape_vae.decode(shape_slat, return_subs=True)
    out = pipe.tex_vae(tex_slat, guide_subs=subs)
    voxel_attrs = np.asarray(out.feats * 0.5 + 0.5)
    print(f"  done in {time.time()-t0:.1f}s, voxel_attrs={voxel_attrs.shape}")

    # Compare distributions.
    print("\n=== RESULTS ===")
    _print_stats("UPSTREAM voxel_attrs (target)", up["voxel_attrs"])
    _print_stats("OURS via tex-flow on upstream-shape", voxel_attrs)

    # Per-channel tex_slat compared to upstream's tex_slat (to isolate flow vs VAE)
    print("\n  tex_slat per-channel diff (ours via upstream-shape − upstream):")
    our_tex = np.asarray(tex_slat.feats)
    if our_tex.shape == up["tex_slat_feats"].shape:
        u64 = up["tex_slat_feats"].astype(np.float64)
        o64 = our_tex.astype(np.float64)
        d_means = o64.mean(axis=0) - u64.mean(axis=0)
        top_k = np.argsort(-np.abs(d_means))[:6]
        print(f"    top-6 most-divergent channels:")
        for c in top_k:
            print(f"      ch{int(c):2d}  upstream=({u64[:,c].mean():+.3f}±{u64[:,c].std():.3f})  "
                  f"ours=({o64[:,c].mean():+.3f}±{o64[:,c].std():.3f})  Δmean={d_means[c]:+.3f}")
    else:
        print(f"    shape mismatch: ours {our_tex.shape}, upstream {up['tex_slat_feats'].shape}")
        print(f"    (different active voxels — coord mismatch from RNG)")

    print("\n  voxel_attrs per-channel diff (ours via tex-flow − upstream):")
    for c, ch in enumerate(["R", "G", "B", "metallic", "roughness", "alpha"]):
        d = float(voxel_attrs[:, c].mean() - up["voxel_attrs"][:, c].mean())
        flag = "  <-- DIVERGENCE" if abs(d) > 0.05 else ""
        print(f"    {ch:9s} Δmean={d:+.4f}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

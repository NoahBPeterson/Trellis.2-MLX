"""Compare upstream CUDA reference dump against our MLX cache.

Reads:
  - Upstream:  the .npz produced by `dump_upstream_intermediates.py` (run on a GPU).
  - Ours:      `artifacts/pbr_intermediates.npz` produced by run_example_pbr.py
               with --cache. Currently has only `V`, `F`, `vertex_attrs`.

Prints per-channel distribution stats side-by-side. The first stage where
they meaningfully disagree tells us where to look for our bug.

Usage:
    python scripts/diff_intermediates.py --upstream /path/to/upstream_ref.npz \\
                                          --ours artifacts/pbr_intermediates.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _stats(x: np.ndarray, axis=None) -> dict:
    return {
        "shape": tuple(x.shape),
        "dtype": str(x.dtype),
        "mean": float(x.astype(np.float64).mean(axis=axis)) if axis is None else x.astype(np.float64).mean(axis=axis),
        "std":  float(x.astype(np.float64).std(axis=axis))  if axis is None else x.astype(np.float64).std(axis=axis),
        "min":  float(x.min()) if axis is None else x.min(axis=axis),
        "max":  float(x.max()) if axis is None else x.max(axis=axis),
    }


def _row(name: str, mean, std, mn, mx) -> str:
    def f(v):
        if isinstance(v, np.ndarray):
            return "[" + ", ".join(f"{x:.3f}" for x in v) + "]"
        return f"{v:.4f}"
    return f"  {name:18s} mean={f(mean):30s} std={f(std):30s} min={f(mn):20s} max={f(mx):20s}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", type=Path, required=True,
                   help="Path to upstream .npz from dump_upstream_intermediates.py")
    p.add_argument("--ours", type=Path, default=Path("artifacts/pbr_intermediates.npz"))
    args = p.parse_args()

    up = np.load(args.upstream)
    our = np.load(args.ours)

    print(f"=== upstream: {args.upstream} ===")
    print("  keys:", list(up.keys()))
    print(f"=== ours:     {args.ours} ===")
    print("  keys:", list(our.keys()))

    # ---- Per-stage comparison ----

    # Shape: V, F
    print("\n--- mesh geometry ---")
    if "mesh_V" in up and "V" in our:
        print(f"  upstream: V={up['mesh_V'].shape} F={up['mesh_F'].shape}")
        print(f"  ours:     V={our['V'].shape} F={our['F'].shape}")
        print(f"  Δ vert count: {our['V'].shape[0] - up['mesh_V'].shape[0]:+d}")
        print(f"  Δ face count: {our['F'].shape[0] - up['mesh_F'].shape[0]:+d}")
        for which, v in [("upstream", up["mesh_V"]), ("ours", our["V"])]:
            s = _stats(v, axis=0)
            print(_row(f"V[{which}] xyz", s["mean"], s["std"], s["min"], s["max"]))

    # Texture volume / vertex attrs (the suspected-divergence stage).
    # Upstream's attribute volume is per-voxel; ours is per-vertex (already
    # sampled from the volume). The DISTRIBUTION should still match if our
    # decoders are correct.
    print("\n--- attribute distribution (per-channel) ---")
    print("  channel layout: [0:3]=base_color RGB, [3]=metallic, [4]=roughness, [5]=alpha")
    if "voxel_attrs" in up:
        va = up["voxel_attrs"]
        print(f"\n  UPSTREAM voxel_attrs ({va.shape}):")
        for c, name in enumerate(["R", "G", "B", "metallic", "roughness", "alpha"]):
            col = va[:, c].astype(np.float64)
            print(f"    {name:9s} mean={col.mean():+.4f}  std={col.std():.4f}  "
                  f"min={col.min():+.4f}  max={col.max():+.4f}")
    if "vertex_attrs" in our:
        va = our["vertex_attrs"]
        print(f"\n  OURS vertex_attrs ({va.shape}):")
        for c, name in enumerate(["R", "G", "B", "metallic", "roughness", "alpha"]):
            col = va[:, c].astype(np.float64)
            print(f"    {name:9s} mean={col.mean():+.4f}  std={col.std():.4f}  "
                  f"min={col.min():+.4f}  max={col.max():+.4f}")

    if "voxel_attrs" in up and "vertex_attrs" in our:
        print("\n  Δ (ours − upstream) per channel mean:")
        for c, name in enumerate(["R", "G", "B", "metallic", "roughness", "alpha"]):
            d = float(our["vertex_attrs"][:, c].mean() - up["voxel_attrs"][:, c].mean())
            flag = "  <-- DIVERGENCE" if abs(d) > 0.1 else ""
            print(f"    {name:9s} Δmean={d:+.4f}{flag}")

    # Latent stages
    print("\n--- latent stages (sparse, coord-based; comparing distribution stats) ---")
    for tag in ["cond_512", "ss_coords", "shape_slat_coords", "shape_slat_feats",
                "tex_slat_coords", "tex_slat_feats"]:
        u = up[tag] if tag in up else None
        o = our[tag] if tag in our else None
        if u is None and o is None:
            continue
        if u is None:
            print(f"\n  {tag}: missing in upstream npz")
            continue
        if o is None:
            print(f"\n  {tag}: missing in our cache (run scripts/run_example_pbr.py --cache to refresh)")
            print(f"    upstream: shape={u.shape} dtype={u.dtype}")
            continue
        u64 = u.astype(np.float64)
        o64 = o.astype(np.float64)
        print(f"\n  {tag}:")
        print(f"    upstream: shape={u.shape} mean={u64.mean():+.4f} std={u64.std():.4f} "
              f"min={u.min():+.4f} max={u.max():+.4f}")
        print(f"    ours:     shape={o.shape} mean={o64.mean():+.4f} std={o64.std():.4f} "
              f"min={o.min():+.4f} max={o.max():+.4f}")
        d_mean = float(o64.mean() - u64.mean())
        d_std = float(o64.std() - u64.std())
        flag = ""
        if tag.endswith("_feats"):
            # Latent feats are normalized to mean≈0 std≈1 before sampling, then
            # denormalized to a wider distribution. A meaningful Δstd > 0.5 or
            # |Δmean| > 0.5 implies a transform / noise / weight bug.
            if abs(d_mean) > 0.5 or abs(d_std) > 0.5:
                flag = "  <-- DIVERGENCE"
        print(f"    Δ:        Δmean={d_mean:+.4f}  Δstd={d_std:+.4f}{flag}")

        # Per-channel breakdown for feats — pinpoints which channels diverge.
        if tag.endswith("_feats") and u.ndim == 2 and o.ndim == 2 and u.shape[1] == o.shape[1]:
            C = u.shape[1]
            up_means = u64.mean(axis=0); up_stds = u64.std(axis=0)
            o_means  = o64.mean(axis=0); o_stds  = o64.std(axis=0)
            d_means = o_means - up_means
            top_k = np.argsort(-np.abs(d_means))[:6]
            print(f"    top-6 most-divergent channels (out of {C}):")
            for c in top_k:
                print(f"      ch{int(c):2d}  upstream=({up_means[c]:+.3f}±{up_stds[c]:.3f})  "
                      f"ours=({o_means[c]:+.3f}±{o_stds[c]:.3f})  Δmean={d_means[c]:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

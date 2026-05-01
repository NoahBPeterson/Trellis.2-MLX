"""Rebake atlas + export GLB from cached pbr_intermediates.npz.

Skips sampling entirely (10+ min saved). Use after editing
trellis2_mlx/postprocess/{atlas.py,glb_export.py} to verify changes
without re-running the whole pipeline.

Usage:
    python scripts/rebake_from_cache.py
    python scripts/rebake_from_cache.py --cache artifacts/pbr_intermediates.npz \\
                                         --out artifacts/sample_pbr_rebake.glb
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trellis2_mlx.postprocess.glb_export import export_pbr_glb


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, default=ROOT / "artifacts/pbr_intermediates.npz")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/sample_pbr_rebake.glb")
    p.add_argument("--target-faces", type=int, default=500_000)
    p.add_argument("--atlas-size", type=int, default=2048)
    p.add_argument("--bake-mode", type=str, default="bvh", choices=["bvh", "vertex"])
    args = p.parse_args()

    print(f"Loading cache from {args.cache}...")
    z = np.load(args.cache)
    required = ["V", "F", "vertex_attrs"]
    missing = [k for k in required if k not in z.files]
    if missing:
        print(f"ERROR: cache missing keys: {missing}")
        return 1
    V, F, va = z["V"], z["F"], z["vertex_attrs"]
    print(f"  V={V.shape}  F={F.shape}  vertex_attrs={va.shape}")
    print(f"  attr stats: R={va[:, 0].mean():.3f}±{va[:, 0].std():.3f}, "
          f"M={va[:, 3].mean():.3f}, R={va[:, 4].mean():.3f}, A={va[:, 5].mean():.3f}")

    print(f"\nRebaking → {args.out}")
    t0 = time.time()
    timings = export_pbr_glb(
        vertices=V,
        faces=F,
        vertex_attrs=va,
        out_path=args.out,
        target_faces=args.target_faces,
        atlas_size=args.atlas_size,
        bake_mode=args.bake_mode,
    )
    print(f"\ndone in {time.time()-t0:.1f}s — wrote {args.out}")
    print(f"  timings: {timings}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

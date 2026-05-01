"""Replay upstream's exact noise tensors through our MLX pipeline.

Isolates RNG-induced divergence from model-induced divergence. Without this,
we can't tell whether a +206-voxel mismatch comes from MLX vs PyTorch noise
(unfixable), from precision differences in the model (fixable), or both.

Strategy (staged):
  Stage A: feed only ss noise. Run SS flow → compare active voxel count.
           If matches upstream's 3548 → SS model is RNG-faithful, divergence
           was purely RNG. If not → SS model has a real precision bug.
  Stage B: if A passed, feed ss + shape noise. Run shape SLat flow →
           compare shape_slat_feats per-channel distribution.
  Stage C: if B passed, feed all three. Run tex SLat → compare tex_slat_feats.

Prereqs:
  artifacts/upstream_ref.npz with noise_ss / noise_shape_slat / noise_tex_slat
  (run dump_upstream_intermediates.py with --save-noise on the pod).

Usage:
    python scripts/replay_upstream_noise.py
    python scripts/replay_upstream_noise.py --pipeline-type 512
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


def _print_distribution(label: str, arr: np.ndarray, n_channels: int = 6):
    """Per-channel distribution stats."""
    print(f"\n  {label} {arr.shape}:")
    if arr.ndim == 2 and arr.shape[1] <= n_channels:
        # Tabular per-channel
        for c in range(arr.shape[1]):
            col = arr[:, c].astype(np.float64)
            print(f"    ch{c}  mean={col.mean():+.4f}  std={col.std():.4f}  "
                  f"min={col.min():+.4f}  max={col.max():+.4f}")
    else:
        # Aggregate stats per channel
        means = arr.mean(axis=tuple(range(arr.ndim - 1)))
        stds = arr.std(axis=tuple(range(arr.ndim - 1)))
        print(f"    mean range: [{means.min():+.4f}, {means.max():+.4f}]")
        print(f"    std  range: [{stds.min():.4f}, {stds.max():.4f}]")


def _top_divergent_channels(ours: np.ndarray, theirs: np.ndarray, k: int = 6):
    """Print the k most-divergent per-channel means."""
    om = ours.mean(axis=0)
    tm = theirs.mean(axis=0)
    os_ = ours.std(axis=0)
    ts_ = theirs.std(axis=0)
    diffs = om - tm
    idx = np.argsort(-np.abs(diffs))[:k]
    print(f"    top-{k} divergent channels (by |Δmean|):")
    for c in idx:
        print(f"      ch{int(c):2d}  upstream=({tm[c]:+.4f}±{ts_[c]:.4f})  "
              f"ours=({om[c]:+.4f}±{os_[c]:.4f})  Δmean={diffs[c]:+.4f}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", type=Path,
                   default=ROOT / "artifacts" / "upstream_ref.npz")
    p.add_argument("--pipeline-type", type=str, default="512")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for shape/tex noise if a stage drops out (we still need a fallback).")
    args = p.parse_args()

    print(f"Loading upstream dump from {args.upstream}...")
    up = np.load(args.upstream)

    required = ["noise_ss", "noise_shape_slat", "noise_tex_slat",
                "ss_coords", "shape_slat_feats", "tex_slat_feats"]
    missing = [k for k in required if k not in up.files]
    if missing:
        print(f"ERROR: missing keys in upstream npz: {missing}")
        print("       Re-run dump_upstream_intermediates.py with --save-noise.")
        return 1

    print(f"  upstream noise tensors: ss{up['noise_ss'].shape}  "
          f"shape{up['noise_shape_slat'].shape}  tex{up['noise_tex_slat'].shape}")
    print(f"  upstream active voxels: {up['ss_coords'].shape[0]}")

    print(f"\nLoading our pipeline (bf16)...")
    t0 = time.time()
    pipe = Trellis2ImageTo3DPipelineMLX.from_pretrained(
        ckpt_dir=ROOT / "ckpts",
        pipeline_json=ROOT / "weights" / "pipeline.json",
        pipeline_type=args.pipeline_type,
        dino_device="cpu",
        rembg_device="cpu",
        dit_compute_dtype="bfloat16",
        with_pbr=True,
    )
    print(f"  loaded in {time.time()-t0:.1f}s")

    # We need cond + neg_cond. Use upstream's bit-exact cond_512 since we know
    # that's already verified (and DINOv3 deterministically gives the same).
    cond_512 = mx.array(up["cond_512"].astype(np.float32))
    neg_512 = mx.zeros_like(cond_512)

    # ─────────────────────────────────────────────────────────────────────
    # Stage A: SS isolation
    # ─────────────────────────────────────────────────────────────────────
    print("\n=== Stage A: SS noise isolation ===")
    pipe._noise_override = {"ss": mx.array(up["noise_ss"].astype(np.float32))}
    t0 = time.time()
    occupancy = pipe._sample_ss(cond_512, neg_512, args.seed)
    coords = pipe._coords_from_occupancy(occupancy, pipe.ss_target_res)
    mx.eval(coords)
    n_ours = int(coords.shape[0])
    n_up = int(up["ss_coords"].shape[0])
    print(f"  SS sampled in {time.time()-t0:.1f}s")
    print(f"  Our active voxels (with upstream noise): {n_ours}")
    print(f"  Upstream active voxels:                  {n_up}")
    print(f"  Δ = {n_ours - n_up:+d}")
    if n_ours == n_up:
        print(f"  ✓ MATCH — SS divergence was purely RNG-induced.")
    else:
        print(f"  ✗ MISMATCH — SS model itself diverges from upstream even with identical noise.")
        print(f"    Continuing to stage B (with our coords) is informative but B/C noise won't fit.")

    # ─────────────────────────────────────────────────────────────────────
    # Stage B: shape DiT isolation — UPSTREAM coords + UPSTREAM noise
    # We bypass our SS coords to test shape DiT in true isolation. This
    # gives a clean answer regardless of stage A's outcome.
    # ─────────────────────────────────────────────────────────────────────
    print("\n=== Stage B: shape DiT isolation (upstream coords + upstream shape noise) ===")
    coords_up = mx.array(up["shape_slat_coords"].astype(np.int32))
    pipe._noise_override = {"shape_slat": mx.array(up["noise_shape_slat"].astype(np.float32))}
    t0 = time.time()
    shape_slat = pipe._sample_shape_slat(
        pipe.shape_flow_hr, cond_512, neg_512, coords_up, args.seed + 1
    )
    mx.eval(shape_slat.feats)
    print(f"  shape sampled in {time.time()-t0:.1f}s")
    ours_shape = np.asarray(shape_slat.feats)
    print(f"  shape_slat shape: {ours_shape.shape}  (matches upstream {up['shape_slat_feats'].shape}: "
          f"{ours_shape.shape == up['shape_slat_feats'].shape})")
    _top_divergent_channels(ours_shape, up["shape_slat_feats"].astype(np.float32))
    shape_max_dmean = float(np.abs(ours_shape.mean(0) - up["shape_slat_feats"].astype(np.float32).mean(0)).max())
    shape_pass = shape_max_dmean < 0.1
    print(f"  max |Δch_mean|: {shape_max_dmean:.4f}  "
          f"({'PASS — shape divergence was RNG' if shape_pass else 'FAIL — shape model has residual bug'})")

    # ─────────────────────────────────────────────────────────────────────
    # Stage C: tex DiT isolation — UPSTREAM coords + UPSTREAM shape_slat + UPSTREAM tex noise
    # ─────────────────────────────────────────────────────────────────────
    print("\n=== Stage C: tex DiT isolation (upstream coords + upstream shape_slat + upstream tex noise) ===")
    upstream_shape_slat = SparseTensor(
        feats=mx.array(up["shape_slat_feats"].astype(np.float32)),
        coords=coords_up,
        spatial_shape=(pipe.tex_flow.resolution,) * 3,
    )
    pipe._noise_override = {"tex_slat": mx.array(up["noise_tex_slat"].astype(np.float32))}
    t0 = time.time()
    tex_slat = pipe._sample_tex_slat(cond_512, neg_512, upstream_shape_slat, args.seed + 17)
    mx.eval(tex_slat.feats)
    print(f"  tex sampled in {time.time()-t0:.1f}s")
    ours_tex = np.asarray(tex_slat.feats)
    print(f"  tex_slat shape: {ours_tex.shape}  (matches upstream {up['tex_slat_feats'].shape}: "
          f"{ours_tex.shape == up['tex_slat_feats'].shape})")
    _top_divergent_channels(ours_tex, up["tex_slat_feats"].astype(np.float32))
    tex_max_dmean = float(np.abs(ours_tex.mean(0) - up["tex_slat_feats"].astype(np.float32).mean(0)).max())
    tex_pass = tex_max_dmean < 0.1
    print(f"  max |Δch_mean|: {tex_max_dmean:.4f}  "
          f"({'PASS — tex divergence was RNG' if tex_pass else 'FAIL — tex model has residual bug'})")

    # ─────────────────────────────────────────────────────────────────────
    # Verdict
    # ─────────────────────────────────────────────────────────────────────
    print("\n=== VERDICT ===")
    print(f"  SS    Δ voxels: {n_ours - n_up:+d}  ({'~bf16 floor' if abs(n_ours - n_up) < 30 else 'real bug'})")
    print(f"  Shape max-|Δch_mean|: {shape_max_dmean:.4f}  ({'PASS' if shape_pass else 'FAIL'})")
    print(f"  Tex   max-|Δch_mean|: {tex_max_dmean:.4f}  ({'PASS' if tex_pass else 'FAIL'})")
    print()
    if shape_pass and tex_pass and abs(n_ours - n_up) < 30:
        print("  >>> All three flows are RNG-faithful. The user-visible bug (metallic +0.27, RGB darkening)")
        print("      must come from a non-DiT path — likely sampler-side (Euler / CFG / std rescale /")
        print("      slat normalization) or downstream of the flows entirely.")
    else:
        failing = []
        if abs(n_ours - n_up) >= 30: failing.append("SS DiT")
        if not shape_pass: failing.append("shape DiT")
        if not tex_pass: failing.append("tex DiT")
        print(f"  >>> Real model bug(s) in: {', '.join(failing)}")
        print(f"      Drill into: attention softmax precision, layer-norm precision, RoPE phases,")
        print(f"      or a specific submodule (qk_rms_norm, adaLN-Zero modulation order).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

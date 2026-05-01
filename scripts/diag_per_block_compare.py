"""Per-block hidden-state comparator: ours (MLX) vs upstream (CUDA).

Both sides run an identical *controlled* forward pass through SS / shape / tex
DiTs:
  - hidden state = zeros (so block input is purely weight + bias driven)
  - timestep      = 1.0 (start of sampling)
  - cond          = upstream's bit-exact cond_512 (already verified in diff)
  - coords (sparse) = upstream's shape_slat coords
  - concat_cond (tex) = zeros (so tex's 64-ch input is just zeros)

After each transformer block, we record per-channel mean, per-channel std,
and absmax. The first block where ours diverges from upstream by more than a
threshold is the location of the shared-DiT-block bug.

Prereqs:
  - artifacts/upstream_ref.npz must contain bd_* keys (run dump_upstream
    with --block-diag).

Usage:
    python scripts/diag_per_block_compare.py
    python scripts/diag_per_block_compare.py --threshold 0.1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trellis2_mlx.modules.sparse_tensor import SparseTensor
from trellis2_mlx.pipeline import Trellis2ImageTo3DPipelineMLX


# ---- Capture helpers --------------------------------------------------------

def _capture_blocks(blocks, h, t_emb, cond, phases):
    """Replay block loop, capture per-channel mean/std/absmax after each block.

    h: (B, L, C) for SS dense, (1, F, C) for sparse — same code path either way.
    Returns three arrays: (num_blocks, C), (num_blocks, C), (num_blocks,)
    """
    means, stds, absmaxes = [], [], []
    for block in blocks:
        h = block(h, t_emb, cond, phases=phases)
        mx.eval(h)
        h32 = np.asarray(h.astype(mx.float32))  # (B, L, C)
        # Average over all but last axis (channels)
        axes = tuple(range(h32.ndim - 1))
        means.append(h32.mean(axis=axes))
        stds.append(h32.std(axis=axes))
        absmaxes.append(float(np.abs(h32).max()))
    return (np.stack(means).astype(np.float32),
            np.stack(stds).astype(np.float32),
            np.asarray(absmaxes, dtype=np.float32))


# ---- Per-flow controlled-input runners (mirror upstream's _run_block_diag) --

def _run_ss(model, cond):
    """SS DiT with zero (1, Cin, R, R, R) input + cond + t=1.0.

    Mirrors flow_dit.SparseStructureFlowModel.__call__ exactly, including the
    bf16 cast before the block loop (matches upstream's manual_cast pattern).
    """
    R = model.resolution
    Cin = model.in_channels
    x = mx.zeros((1, Cin, R, R, R), dtype=mx.float32)
    h = x.reshape(1, Cin, -1).transpose(0, 2, 1)
    h = model.input_layer(h)
    from trellis2_mlx.models.flow_dit import _grid_coords_3d
    coords = _grid_coords_3d(R)
    if model.pe_mode == "ape":
        if model._cached_pos_emb is None:
            model._cached_pos_emb = model._ape(coords)
        h = h + model._cached_pos_emb[None]
    t_emb = model.t_embedder(mx.array([1000.0], dtype=mx.float32))
    if model.share_mod:
        t_emb = model.adaLN_modulation(t_emb)
    # Cast to compute dtype (matches __call__).
    compute_dtype = model.input_layer.weight.dtype
    h = h.astype(compute_dtype)
    t_emb = t_emb.astype(compute_dtype)
    cond = cond.astype(compute_dtype)
    phases = None
    if model.pe_mode == "rope":
        if model._cached_phases is None:
            model._cached_phases = model._rope(coords)
        phases = model._cached_phases[None]
    return _capture_blocks(model.blocks, h, t_emb, cond, phases)


def _run_slat(model, cond, coords4):
    """SLat DiT (shape or tex) with zero feats on the given coords + cond + t=1.0.

    Mirrors flow_dit.SLatFlowModel.__call__ exactly, including the bf16 cast.
    coords4: (F, 4) int32 with [batch, x, y, z].
    """
    Cin = model.in_channels
    F = int(coords4.shape[0])
    x = SparseTensor(
        feats=mx.zeros((F, Cin), dtype=mx.float32),
        coords=coords4,
        spatial_shape=(model.resolution, model.resolution, model.resolution),
    )
    h_feats = model.input_layer(x.feats)
    t_emb = model.t_embedder(mx.array([1000.0], dtype=mx.float32))
    if model.share_mod:
        t_emb = model.adaLN_modulation(t_emb)
    # Cast to compute dtype (matches __call__).
    compute_dtype = model.input_layer.weight.dtype
    h_feats = h_feats.astype(compute_dtype)
    t_emb = t_emb.astype(compute_dtype)
    cond = cond.astype(compute_dtype)
    phases = None
    if model.pe_mode == "rope":
        phases = model._rope(x.coords[:, 1:].astype(mx.float32))
        phases = phases[None]
    elif model.pe_mode == "ape":
        pe = model._ape(x.coords[:, 1:].astype(mx.float32))
        h_feats = h_feats + pe.astype(compute_dtype)
    h = mx.expand_dims(h_feats, axis=0)  # (1, F, C)
    return _capture_blocks(model.blocks, h, t_emb, cond, phases)


# ---- Reporting --------------------------------------------------------------

def _format_block_row(i, ours_m, ups_m, ours_s, ups_s, ours_a, ups_a):
    """One row per block. Show top-divergent channel and absmax stats."""
    dmean = ours_m - ups_m  # (C,)
    top_ch = int(np.argmax(np.abs(dmean)))
    return (f"  blk {i:2d}: "
            f"absmax (ours/up)={ours_a:>9.2f}/{ups_a:>9.2f} (Δ={ours_a - ups_a:+.2f})  "
            f"max|Δch_mean|={np.abs(dmean).max():.4f} (ch{top_ch}: "
            f"{ups_m[top_ch]:+.4f}→{ours_m[top_ch]:+.4f})")


def _compare_flow(label, ours, ups, threshold):
    """ours/ups: (means, stds, absmaxes) tuples."""
    om, os_, oa = ours
    um, us_, ua = ups
    n = min(om.shape[0], um.shape[0])
    if om.shape != um.shape:
        print(f"  {label}: SHAPE MISMATCH ours={om.shape} up={um.shape} — comparing first {n}")
    print(f"\n=== {label} (per-block stats; threshold={threshold:.3f} on max|Δ per-channel mean|) ===")
    first_div = None
    for i in range(n):
        row = _format_block_row(i, om[i], um[i], os_[i], us_[i], oa[i], ua[i])
        flag = ""
        max_dmean = float(np.abs(om[i] - um[i]).max())
        if max_dmean > threshold and first_div is None:
            first_div = i
            flag = "  <-- FIRST DIVERGENCE"
        print(row + flag)

    if first_div is None:
        print(f"\n  {label}: no block exceeds threshold — clean under controlled input.")
    else:
        print(f"\n  {label}: first divergent block = {first_div}")
        # Drill in: which channel diverged worst, what's the relative shift?
        i = first_div
        dmean = om[i] - um[i]
        top5 = np.argsort(-np.abs(dmean))[:5]
        print(f"    Top-5 divergent channels at block {i}:")
        for c in top5:
            print(f"      ch{int(c):3d}  upstream=({um[i, c]:+.4f}±{us_[i, c]:.4f})  "
                  f"ours=({om[i, c]:+.4f}±{os_[i, c]:.4f})  Δmean={dmean[c]:+.4f}")
    return first_div


# ---- Main -------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", type=Path,
                   default=ROOT / "artifacts" / "upstream_ref.npz",
                   help="upstream_ref.npz with bd_* keys (from --block-diag run).")
    p.add_argument("--threshold", type=float, default=0.05,
                   help="max-|Δmean-per-channel| threshold to flag divergence.")
    p.add_argument("--pipeline-type", type=str, default="512")
    args = p.parse_args()

    print(f"Loading upstream dump from {args.upstream}...")
    up = np.load(args.upstream)
    required = ["bd_ss_mean", "bd_shape_mean", "bd_tex_mean",
                "shape_slat_coords", "cond_512"]
    missing = [k for k in required if k not in up.files]
    if missing:
        print(f"ERROR: missing keys in upstream npz: {missing}")
        print("       Re-run dump_upstream_intermediates.py with --block-diag.")
        return 1

    print(f"  upstream block counts: SS={up['bd_ss_mean'].shape[0]}, "
          f"shape={up['bd_shape_mean'].shape[0]}, tex={up['bd_tex_mean'].shape[0]}")

    print("\nLoading our MLX pipeline...")
    pipe = Trellis2ImageTo3DPipelineMLX.from_pretrained(
        ckpt_dir=ROOT / "weights" / "ckpts",
        pipeline_json=ROOT / "weights" / "pipeline.json",
        pipeline_type=args.pipeline_type,
        dino_device="cpu",
        rembg_device="cpu",
        dit_compute_dtype="bfloat16",
        with_pbr=True,
    )

    cond = mx.array(up["cond_512"].astype(np.float32))
    coords4 = mx.array(up["shape_slat_coords"].astype(np.int32))

    print("\nRunning ours: SS DiT (controlled zero input)...")
    ours_ss = _run_ss(pipe.ss_flow, cond)
    print(f"  ours SS: {ours_ss[0].shape[0]} blocks, "
          f"absmax range [{ours_ss[2].min():.2f}, {ours_ss[2].max():.2f}]")

    print("Running ours: shape DiT (controlled zero input)...")
    ours_shape = _run_slat(pipe.shape_flow_hr, cond, coords4)
    print(f"  ours shape: {ours_shape[0].shape[0]} blocks, "
          f"absmax range [{ours_shape[2].min():.2f}, {ours_shape[2].max():.2f}]")

    print("Running ours: tex DiT (controlled zero input)...")
    if pipe.tex_flow is None:
        print("  pipe.tex_flow is None — skipping (load pipeline with with_pbr=True)")
        ours_tex = None
    else:
        ours_tex = _run_slat(pipe.tex_flow, cond, coords4)
        print(f"  ours tex: {ours_tex[0].shape[0]} blocks, "
              f"absmax range [{ours_tex[2].min():.2f}, {ours_tex[2].max():.2f}]")

    # Compare each flow
    ups_ss = (up["bd_ss_mean"], up["bd_ss_std"], up["bd_ss_absmax"])
    ups_shape = (up["bd_shape_mean"], up["bd_shape_std"], up["bd_shape_absmax"])
    fd_ss = _compare_flow("SS DiT", ours_ss, ups_ss, args.threshold)
    fd_shape = _compare_flow("Shape DiT", ours_shape, ups_shape, args.threshold)
    if ours_tex is not None:
        ups_tex = (up["bd_tex_mean"], up["bd_tex_std"], up["bd_tex_absmax"])
        fd_tex = _compare_flow("Tex DiT", ours_tex, ups_tex, args.threshold)
    else:
        fd_tex = None

    print("\n=== SUMMARY ===")
    print(f"  SS    first-divergent block: {fd_ss}")
    print(f"  Shape first-divergent block: {fd_shape}")
    print(f"  Tex   first-divergent block: {fd_tex}")
    if fd_ss is not None and fd_ss == fd_shape == fd_tex:
        print(f"\n  >>> All three flows diverge at block {fd_ss} → strong evidence for "
              f"a shared modulated-DiT block bug at index {fd_ss}.")
        print(f"      Drill into that block's submodules: attn (qk_rms_norm, RoPE, SDPA), "
              f"FFN, adaLN-Zero modulation, residual ordering.")
    elif {fd_ss, fd_shape, fd_tex} == {None}:
        print(f"\n  >>> No flow diverges under controlled input. The bug is in the "
              f"sampling loop (RNG, dt schedule, CFG combination, std rescale, "
              f"or normalization stats). Pivot to sampler-side investigation.")
    else:
        print(f"\n  >>> Different first-divergent blocks per flow → likely a "
              f"per-flow weight or config-loading bug. Check pipeline.json "
              f"normalization stats and per-flow ckpt loaders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

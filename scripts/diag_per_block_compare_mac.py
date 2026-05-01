"""Per-block hidden-state comparator: trellis-mac (PyTorch+MPS) vs upstream (CUDA).

Same controlled forward pass as scripts/diag_per_block_compare.py, but the local
side uses the trellis-mac port (PyTorch+MPS) at ext/trellis-mac/TRELLIS.2/ instead
of our MLX rewrite. This isolates whether per-block divergence vs upstream comes
from MLX-only bugs or from the upstream PyTorch code itself running differently
on MPS.

Both sides run an identical *controlled* forward pass through SS / shape / tex
DiTs:
  - hidden state = zeros (so block input is purely weight + bias driven)
  - timestep      = 1.0 (start of sampling)
  - cond          = upstream's bit-exact cond_512
  - coords (sparse) = upstream's shape_slat / tex_slat coords
  - concat_cond (tex) = zeros (so tex's 64-ch input is just zeros)

After each transformer block, we record per-channel mean, per-channel std, and
absmax. The first block where ours diverges from upstream by more than a
threshold is the location of the shared-DiT-block bug.

Prereqs:
  - artifacts/upstream_ref.npz must contain bd_* keys (run dump_upstream
    with --block-diag).
  - ext/trellis-mac/.venv must exist with torch + the patched TRELLIS.2 code.

Usage:
    ext/trellis-mac/.venv/bin/python scripts/diag_per_block_compare_mac.py
    ext/trellis-mac/.venv/bin/python scripts/diag_per_block_compare_mac.py --threshold 0.1
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# --- Paths and environment must be set BEFORE torch import ------------------
ROOT = Path(__file__).resolve().parents[1]
TRELLIS_MAC = ROOT / "ext" / "trellis-mac"

# Mirror generate.py: choose backends, enable MPS CPU fallback, before torch.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
try:
    sys.path.insert(0, str(TRELLIS_MAC))
    import flex_gemm  # noqa: F401
    os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")
except ImportError:
    os.environ.setdefault("SPARSE_CONV_BACKEND", "none")

sys.path.insert(0, str(TRELLIS_MAC / "TRELLIS.2"))
sys.path.append(str(TRELLIS_MAC / "stubs"))

import numpy as np
import torch
import trellis2.modules.sparse as sp  # type: ignore
from trellis2.modules.utils import manual_cast as _manual_cast  # type: ignore


# ---- Capture helpers (identical math to upstream's dump_upstream_intermediates) ----

def _capture_blocks_dense(model, h, t_emb, cond, phases):
    """Loop SS DiT blocks; record per-channel mean/std/absmax after each block.

    h: (B, L, C) where L=R^3.
    """
    means, stds, absmaxes = [], [], []
    for block in model.blocks:
        h = block(h, t_emb, cond, phases)
        h32 = h.detach().float()
        means.append(h32.mean(dim=(0, 1)).cpu().numpy())
        stds.append(h32.std(dim=(0, 1)).cpu().numpy())
        absmaxes.append(float(h32.abs().max().item()))
    return (np.stack(means).astype(np.float32),
            np.stack(stds).astype(np.float32),
            np.asarray(absmaxes, dtype=np.float32))


def _capture_blocks_sparse(model, h, t_emb, cond):
    """Loop SLat DiT blocks; record per-channel stats."""
    means, stds, absmaxes = [], [], []
    for block in model.blocks:
        h = block(h, t_emb, cond)
        f32 = h.feats.detach().float()  # (F, C)
        means.append(f32.mean(dim=0).cpu().numpy())
        stds.append(f32.std(dim=0).cpu().numpy())
        absmaxes.append(float(f32.abs().max().item()))
    return (np.stack(means).astype(np.float32),
            np.stack(stds).astype(np.float32),
            np.asarray(absmaxes, dtype=np.float32))


# ---- Per-flow controlled-input runners -------------------------------------

def _run_ss(model, cond, device):
    """SS DiT with zero (1, Cin, R, R, R) input + cond + t=1.0."""
    R = model.resolution
    Cin = model.in_channels
    t_step = torch.tensor([1000.0], device=device, dtype=torch.float32)
    with torch.no_grad():
        x = torch.zeros(1, Cin, R, R, R, device=device, dtype=torch.float32)
        h = x.view(*x.shape[:2], -1).permute(0, 2, 1).contiguous()  # (1, R^3, Cin)
        h = model.input_layer(h)
        if model.pe_mode == "ape":
            h = h + model.pos_emb[None]
        t_emb = model.t_embedder(t_step)
        if model.share_mod:
            t_emb = model.adaLN_modulation(t_emb)
        t_emb = _manual_cast(t_emb, model.dtype)
        h = _manual_cast(h, model.dtype)
        cond_d = _manual_cast(cond, model.dtype)
        return _capture_blocks_dense(model, h, t_emb, cond_d, model.rope_phases)


def _run_slat(model, cond, coords, device):
    """SLat DiT (shape or tex) with zero feats on the given coords + cond + t=1.0."""
    Cin = model.in_channels
    t_step = torch.tensor([1000.0], device=device, dtype=torch.float32)
    with torch.no_grad():
        zero_feats = torch.zeros(coords.shape[0], Cin, device=device, dtype=torch.float32)
        x_sp = sp.SparseTensor(feats=zero_feats, coords=coords.to(torch.int32))
        h = model.input_layer(x_sp)
        h = h.replace(_manual_cast(h.feats, model.dtype))
        t_emb = model.t_embedder(t_step)
        if model.share_mod:
            t_emb = model.adaLN_modulation(t_emb)
        t_emb = _manual_cast(t_emb, model.dtype)
        cond_d = _manual_cast(cond, model.dtype)
        if model.pe_mode == "ape":
            pe = model.pos_embedder(h.coords[:, 1:])
            h = h + _manual_cast(pe, model.dtype)
        return _capture_blocks_sparse(model, h, t_emb, cond_d)


# ---- Reporting (verbatim from MLX comparator so output is comparable) -------

def _format_block_row(i, ours_m, ups_m, ours_s, ups_s, ours_a, ups_a):
    dmean = ours_m - ups_m  # (C,)
    top_ch = int(np.argmax(np.abs(dmean)))
    return (f"  blk {i:2d}: "
            f"absmax (ours/up)={ours_a:>9.2f}/{ups_a:>9.2f} (Δ={ours_a - ups_a:+.2f})  "
            f"max|Δch_mean|={np.abs(dmean).max():.4f} (ch{top_ch}: "
            f"{ups_m[top_ch]:+.4f}→{ours_m[top_ch]:+.4f})")


def _compare_flow(label, ours, ups, threshold):
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
    p.add_argument("--device", type=str, default="mps",
                   help="Torch device for forward pass (mps or cpu). Default mps.")
    p.add_argument("--weights", type=str,
                   default=str(ROOT / "weights"),
                   help="Local pipeline.json directory (defaults to repo root weights/).")
    args = p.parse_args()

    print(f"Loading upstream dump from {args.upstream}...")
    up = np.load(args.upstream)
    required = ["bd_ss_mean", "bd_shape_mean", "bd_tex_mean",
                "shape_slat_coords", "tex_slat_coords", "cond_512"]
    missing = [k for k in required if k not in up.files]
    if missing:
        print(f"ERROR: missing keys in upstream npz: {missing}")
        print("       Re-run dump_upstream_intermediates.py with --block-diag.")
        return 1
    print(f"  upstream block counts: SS={up['bd_ss_mean'].shape[0]}, "
          f"shape={up['bd_shape_mean'].shape[0]}, tex={up['bd_tex_mean'].shape[0]}")

    print(f"\nSPARSE_CONV_BACKEND={os.environ.get('SPARSE_CONV_BACKEND')}  "
          f"ATTN_BACKEND={os.environ.get('ATTN_BACKEND')}  "
          f"SPARSE_ATTN_BACKEND={os.environ.get('SPARSE_ATTN_BACKEND')}")
    print(f"\nLoading trellis-mac pipeline from {args.weights}...")
    from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.weights)

    device = torch.device(args.device)
    print(f"Moving DiT models to {device}...")
    for k in ("sparse_structure_flow_model", "shape_slat_flow_model_512",
              "tex_slat_flow_model_512"):
        if k in pipeline.models:
            pipeline.models[k].to(device).eval()
    print("Models on device.")

    cond = torch.from_numpy(up["cond_512"].astype(np.float32)).to(device)
    shape_coords = torch.from_numpy(up["shape_slat_coords"].astype(np.int32)).to(device)
    tex_coords = torch.from_numpy(up["tex_slat_coords"].astype(np.int32)).to(device)

    print("\nRunning ours: SS DiT (controlled zero input)...")
    ours_ss = _run_ss(pipeline.models["sparse_structure_flow_model"], cond, device)
    print(f"  ours SS: {ours_ss[0].shape[0]} blocks, "
          f"absmax range [{ours_ss[2].min():.2f}, {ours_ss[2].max():.2f}]")

    print("Running ours: shape DiT @ 512 (controlled zero input)...")
    ours_shape = _run_slat(pipeline.models["shape_slat_flow_model_512"],
                           cond, shape_coords, device)
    print(f"  ours shape: {ours_shape[0].shape[0]} blocks, "
          f"absmax range [{ours_shape[2].min():.2f}, {ours_shape[2].max():.2f}]")

    print("Running ours: tex DiT @ 512 (controlled zero input)...")
    ours_tex = _run_slat(pipeline.models["tex_slat_flow_model_512"],
                         cond, tex_coords, device)
    print(f"  ours tex: {ours_tex[0].shape[0]} blocks, "
          f"absmax range [{ours_tex[2].min():.2f}, {ours_tex[2].max():.2f}]")

    # Compare each flow
    ups_ss = (up["bd_ss_mean"], up["bd_ss_std"], up["bd_ss_absmax"])
    ups_shape = (up["bd_shape_mean"], up["bd_shape_std"], up["bd_shape_absmax"])
    ups_tex = (up["bd_tex_mean"], up["bd_tex_std"], up["bd_tex_absmax"])
    fd_ss = _compare_flow("SS DiT", ours_ss, ups_ss, args.threshold)
    fd_shape = _compare_flow("Shape DiT", ours_shape, ups_shape, args.threshold)
    fd_tex = _compare_flow("Tex DiT", ours_tex, ups_tex, args.threshold)

    print("\n=== SUMMARY (trellis-mac PyTorch+MPS vs upstream CUDA) ===")
    print(f"  SS    first-divergent block: {fd_ss}")
    print(f"  Shape first-divergent block: {fd_shape}")
    print(f"  Tex   first-divergent block: {fd_tex}")
    if fd_ss is not None and fd_ss == fd_shape == fd_tex:
        print(f"\n  >>> All three flows diverge at block {fd_ss} on MPS too — "
              f"the upstream PyTorch code itself behaves differently on MPS at this block.")
    elif {fd_ss, fd_shape, fd_tex} == {None}:
        print(f"\n  >>> No flow diverges under controlled input on MPS. "
              f"The upstream PyTorch DiT blocks are MPS-equivalent to CUDA — "
              f"any divergence we see in MLX is MLX-specific.")
    else:
        print(f"\n  >>> Different first-divergent blocks per flow on MPS — "
              f"investigate whether MPS-specific fallbacks (e.g. segment_reduce CPU "
              f"fallback, conv_none gather-scatter) explain the per-flow asymmetry.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Rigorous numerical accuracy test: fp16 / int8 / int4 on a real DiT.

Loads slat_flow_img2shape_dit_1_3B_512 (used in the 512 pipeline), runs identical
inputs through three copies of block 15, and reports:
  - Block output drift (the residual-style (B, L, C) tensor handed to the next block)
  - Attention SCORES drift (Q @ K^T * scale — pre-softmax, most sensitive to weight perturbation)
  - Full 30-block stack drift (how errors compound across depth)

Reference is fp16 (our proven lossless-vs-bf16 path), since we already committed to
fp16 as the evaluation target.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_map

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trellis2_mlx.models.flow_dit import SLatFlowModel
from trellis2_mlx.pipeline import _load_weights_prefixed


def load_model(cast_dtype=mx.float16) -> SLatFlowModel:
    cfg = json.loads((ROOT / "ckpts/slat_flow_img2shape_dit_1_3B_512.config.json").read_text())["args"]
    cfg = {k: v for k, v in cfg.items() if k not in ("initialization", "dtype")}
    m = SLatFlowModel(**cfg)
    _load_weights_prefixed(m, ROOT / "ckpts/slat_flow_img2shape_dit_1_3B_512.safetensors", cast_dtype=cast_dtype)
    return m


def block_output(model: SLatFlowModel, x, mod, cond, phases, block_idx: int) -> mx.array:
    """Feed x through a single block; return its output (not the full forward)."""
    block = model.blocks[block_idx]
    return block(x, mod, cond, phases=phases)


def attention_scores(model: SLatFlowModel, x, phases, block_idx: int) -> mx.array:
    """Return the pre-softmax attention scores (Q@K^T * scale) from block `block_idx`.
    This is the most sensitive intermediate to weight perturbation."""
    attn = model.blocks[block_idx].self_attn
    B, L, C = x.shape
    H, D = attn.num_heads, attn.head_dim
    qkv = attn.to_qkv(x).reshape(B, L, 3, H, D)
    q, k = qkv[:, :, 0], qkv[:, :, 1]
    if attn.qk_rms_norm:
        q = attn.q_rms_norm(q)
        k = attn.k_rms_norm(k)
    if attn.use_rope:
        from trellis2_mlx.modules.rope import RotaryPositionEmbedder
        q = RotaryPositionEmbedder.apply_rotary_embedding(q, phases)
        k = RotaryPositionEmbedder.apply_rotary_embedding(k, phases)
    q = q.transpose(0, 2, 1, 3)  # (B, H, L, D)
    k = k.transpose(0, 2, 1, 3)
    scores = mx.matmul(q, k.swapaxes(-1, -2)) * (D ** -0.5)
    return scores  # (B, H, L, L)


def full_forward_dry(model: SLatFlowModel, x_feats, coords, t_emb, cond) -> mx.array:
    """Run the full 30-block stack without the out_layer (to keep comparison at max depth)."""
    from trellis2_mlx.modules.sparse_tensor import SparseTensor
    st = SparseTensor(feats=x_feats, coords=coords, spatial_shape=(32, 32, 32))
    h_feats = model.input_layer(st.feats)
    t_emb_use = t_emb
    if model.share_mod:
        t_emb_use = model.adaLN_modulation(t_emb)
    phases = model._rope(st.coords[:, 1:].astype(mx.float32))[None]
    h = h_feats[None]
    for block in model.blocks:
        h = block(h, t_emb_use, cond, phases=phases)
    return h[0]


def summarize(name: str, ref: mx.array, test: mx.array, max_samples: int = 500_000) -> dict:
    """Compute diff summary statistics via numpy."""
    ref_f32 = np.asarray(ref.astype(mx.float32))
    test_f32 = np.asarray(test.astype(mx.float32))
    diff = np.abs(ref_f32 - test_f32)
    # Significant-value mask: only include magnitudes > 0.1× std of the reference
    sig_mask = np.abs(ref_f32) > 0.1 * ref_f32.std()
    if sig_mask.sum() > max_samples:
        # subsample for speed on full-forward outputs
        idx = np.random.default_rng(0).choice(sig_mask.sum(), max_samples, replace=False)
        rel_sig = diff[sig_mask].flatten()[idx] / np.abs(ref_f32[sig_mask].flatten()[idx])
    else:
        rel_sig = diff[sig_mask] / (np.abs(ref_f32[sig_mask]) + 1e-8)
    print(f"\n  {name}")
    print(f"    ref        : shape={ref_f32.shape}  range=[{ref_f32.min():.3f}, {ref_f32.max():.3f}]  std={ref_f32.std():.3f}")
    print(f"    max abs    : {diff.max():.4e}")
    print(f"    mean abs   : {diff.mean():.4e}")
    print(f"    p99 abs    : {np.percentile(diff, 99):.4e}")
    print(f"    rel (|ref| > 0.1×std):")
    print(f"      mean rel : {rel_sig.mean()*100:.3f}%")
    print(f"      p99 rel  : {np.percentile(rel_sig, 99)*100:.3f}%")
    print(f"      max rel  : {rel_sig.max()*100:.3f}%")
    return {"max": diff.max(), "mean_rel": rel_sig.mean(), "p99_rel": np.percentile(rel_sig, 99)}


def main() -> int:
    BLOCK_IDX = 15
    F, C = 3761, 1536       # Sparse SLat shape at pipeline_type=512
    Lkv = 1029
    H = 12; D = C // H; pairs = D // 2
    mx.random.seed(0)

    # Shared inputs across all three variants
    x_fp32 = mx.random.normal((1, F, C))
    mod_fp32 = mx.random.normal((1, 6 * C))
    cond_fp32 = mx.random.normal((1, Lkv, 1024))
    phases_fp32 = mx.random.normal((1, F, pairs, 2))
    # For full-forward: realistic (F, C_in=32) feats + coords at 32^3
    x_in_fp32 = mx.random.normal((F, 32))
    coords_np = np.concatenate([np.zeros((F, 1), dtype=np.int32),
                                np.random.default_rng(0).integers(0, 32, size=(F, 3), dtype=np.int32)], axis=1)
    coords = mx.array(coords_np)
    t_fp32 = mx.array([500.0], dtype=mx.float32)

    results = {}

    # --- fp16 (reference) ---
    print("=" * 70)
    print("Loading fp16 reference model...")
    m_fp16 = load_model(mx.float16)
    t_emb_fp16 = m_fp16.t_embedder(t_fp32)

    x_fp16 = x_fp32.astype(mx.float16)
    mod_fp16 = mod_fp32.astype(mx.float16)
    cond_fp16 = cond_fp32.astype(mx.float16)

    ref_block = block_output(m_fp16, x_fp16, mod_fp16, cond_fp16, phases_fp32, BLOCK_IDX)
    mx.eval(ref_block)
    ref_scores = attention_scores(m_fp16, x_fp16, phases_fp32, BLOCK_IDX)
    mx.eval(ref_scores)
    ref_full = full_forward_dry(m_fp16, x_in_fp32.astype(mx.float16), coords, t_emb_fp16, cond_fp16)
    mx.eval(ref_full)

    print(f"\nfp16 ref captured:")
    print(f"  block-{BLOCK_IDX} out shape      = {tuple(ref_block.shape)}")
    print(f"  attn scores shape       = {tuple(ref_scores.shape)}  (B, H, L, L)")
    print(f"  full-forward out shape  = {tuple(ref_full.shape)}")

    # Free fp16 model before loading next
    del m_fp16

    # Quant modes to test: (label, bits, group_size, mode)
    MODES = [
        ("affine int8 g64",  8, 64,  "affine"),
        ("affine int4 g64",  4, 64,  "affine"),
        ("affine int4 g128", 4, 128, "affine"),
        ("mxfp4 g32",        4, 32,  "mxfp4"),
        ("mxfp8 g32",        8, 32,  "mxfp8"),
        ("nvfp4 g16",        4, 16,  "nvfp4"),
    ]
    for label, bits, gs, mode in MODES:
        print("\n" + "=" * 70)
        print(f"Loading {label} model (bits={bits}, group_size={gs}, mode={mode})...")
        m = load_model(mx.float16)  # start from fp16
        # Skip Linear layers whose last weight dim isn't divisible by group_size.
        # (input_layer: Linear(32, 1536) has weight (1536, 32) — last dim 32. Tiny.)
        def _q_pred(path, module, _gs=gs):
            if not isinstance(module, nn.Linear):
                return False
            return module.weight.shape[-1] % _gs == 0
        nn.quantize(m, group_size=gs, bits=bits, mode=mode, class_predicate=_q_pred)

        # Must re-fetch t_emb since we're on a fresh module
        t_emb = m.t_embedder(t_fp32)

        out = block_output(m, x_fp16, mod_fp16, cond_fp16, phases_fp32, BLOCK_IDX)
        mx.eval(out)
        scores = attention_scores(m, x_fp16, phases_fp32, BLOCK_IDX)
        mx.eval(scores)
        full = full_forward_dry(m, x_in_fp32.astype(mx.float16), coords, t_emb, cond_fp16)
        mx.eval(full)

        print(f"\n--- {label} vs fp16 reference ---")
        r1 = summarize(f"block-{BLOCK_IDX} output", ref_block, out)
        r2 = summarize(f"attention scores (pre-softmax logits)", ref_scores, scores)
        r3 = summarize(f"full 30-block forward", ref_full, full)
        results[label] = {"block": r1, "scores": r2, "full": r3, "bits": bits}
        del m

    # --- Summary table ---
    print("\n" + "=" * 70)
    print("SUMMARY (relative error vs fp16 reference, where |ref| > 0.1×std)")
    print("=" * 70)
    print(f"{'variant':<20s}  {'bits':>4s}  {'1-block out p99':>16s}  {'attn scores p99':>16s}  {'30-block out p99':>18s}")
    for name, r in results.items():
        print(f"{name:<20s}  {r['bits']:>4d}  {r['block']['p99_rel']*100:>14.2f}%   {r['scores']['p99_rel']*100:>14.2f}%   {r['full']['p99_rel']*100:>16.2f}%")
    print()
    print("Rules of thumb:")
    print("  < 1%  = essentially lossless (like our bf16→fp16 move)")
    print("  1–5%  = usually fine for diffusion sampling, occasional drift")
    print("  5–15% = visible but often acceptable for generative models")
    print("  > 15% = likely quality degradation at the mesh level")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

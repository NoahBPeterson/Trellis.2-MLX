"""Per-op timing inside one DiT block — find where the wall-clock actually goes.

Splits ModulatedTransformerCrossBlock into its sub-ops, times each at production
sizes (3.8k tokens for 512 shape flow, 19k for 1024). Reveals which ops dominate.

Optionally captures a Metal trace (.gputrace) viewable in Xcode.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trellis2_mlx.modules.attention import MultiHeadAttention
from trellis2_mlx.modules.norm import LayerNorm32
from trellis2_mlx.modules.blocks import FeedForwardNet


def _to_bf16(p):
    if isinstance(p, mx.array):
        return p.astype(mx.bfloat16)
    return p


def time_op(name: str, fn, warmup: int = 3, repeat: int = 30):
    for _ in range(warmup):
        out = fn()
        if isinstance(out, mx.array):
            mx.eval(out)
        elif isinstance(out, tuple):
            mx.eval(*out)
    mx.synchronize()
    t0 = time.time()
    for _ in range(repeat):
        out = fn()
        if isinstance(out, mx.array):
            mx.eval(out)
        elif isinstance(out, tuple):
            mx.eval(*out)
    mx.synchronize()
    ms = (time.time() - t0) / repeat * 1000
    print(f"  {name:36s} {ms:7.2f} ms/call")
    return ms


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=3761)
    p.add_argument("--cond-tokens", type=int, default=1029)
    p.add_argument("--channels", type=int, default=1536)
    p.add_argument("--cond-channels", type=int, default=1024)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--mlp-ratio", type=float, default=5.3334)
    p.add_argument("--metal-trace", type=str, default=None,
                   help="Optional path to write a Metal .gputrace file")
    args = p.parse_args()

    F, Lkv, C, H = args.tokens, args.cond_tokens, args.channels, args.heads
    D = C // H

    # Build sub-modules in bf16 (matches production)
    norm1 = LayerNorm32(C, eps=1e-6, affine=False)
    norm2 = LayerNorm32(C, eps=1e-6, affine=True)
    norm3 = LayerNorm32(C, eps=1e-6, affine=False)
    self_attn = MultiHeadAttention(C, H, type="self", use_rope=True, qk_rms_norm=True)
    cross_attn = MultiHeadAttention(C, H, ctx_channels=args.cond_channels, type="cross", qk_rms_norm=True)
    mlp = FeedForwardNet(C, mlp_ratio=args.mlp_ratio)
    for m in (norm2, self_attn, cross_attn, mlp):
        m.update(tree_map(_to_bf16, m.parameters()))

    # Inputs
    x = mx.random.normal((1, F, C)).astype(mx.bfloat16)
    cond = mx.random.normal((1, Lkv, args.cond_channels)).astype(mx.bfloat16)
    pairs = D // 2
    phases = mx.random.normal((1, F, pairs, 2))
    scale = mx.random.normal((1, 1, C)).astype(mx.bfloat16)
    shift = mx.random.normal((1, 1, C)).astype(mx.bfloat16)

    # Pre-normed input for sub-op tests
    x_norm = norm1(x)
    h_self = self_attn(x_norm, phases=phases)
    x_after_self = x + h_self
    x_after_cross = x_after_self + cross_attn(norm2(x_after_self), context=cond)

    print(f"=== DiT block sub-op timing (F={F}, Lkv={Lkv}, C={C}, H={H}) ===\n")

    print("Norms (3 per block):")
    t_norm1 = time_op("LayerNorm32 (no affine)",       lambda: norm1(x))
    t_norm2 = time_op("LayerNorm32 (affine)",          lambda: norm2(x))
    t_norm3 = time_op("LayerNorm32 (no affine)",       lambda: norm3(x))

    print("\nModulation arithmetic:")
    t_mod = time_op("scale*x + shift",                  lambda: x_norm * (1 + scale) + shift)

    print("\nSelf-attention (sub-ops):")
    t_qkv_proj = time_op("to_qkv Linear (C→3C)",        lambda: self_attn.to_qkv(x_norm))
    qkv = self_attn.to_qkv(x_norm).reshape(1, F, 3, H, D)
    q_, k_, v_ = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    t_rope = time_op("apply RoPE (Q+K)",                lambda: (
        self_attn.q_rms_norm(q_) if False else None,
    ))  # placeholder
    from trellis2_mlx.modules.rope import RotaryPositionEmbedder
    t_rope = time_op("apply RoPE (Q only)",             lambda: RotaryPositionEmbedder.apply_rotary_embedding(q_, phases))
    t_qkrms = time_op("MultiHeadRMSNorm (Q only)",      lambda: self_attn.q_rms_norm(q_))
    q_t = q_.transpose(0, 2, 1, 3); k_t = k_.transpose(0, 2, 1, 3); v_t = v_.transpose(0, 2, 1, 3)
    t_sdpa = time_op("scaled_dot_product_attention",    lambda: mx.fast.scaled_dot_product_attention(q_t, k_t, v_t, scale=D**-0.5))
    out_sdpa = mx.fast.scaled_dot_product_attention(q_t, k_t, v_t, scale=D**-0.5).transpose(0, 2, 1, 3).reshape(1, F, C)
    t_to_out = time_op("to_out Linear (C→C)",           lambda: self_attn.to_out(out_sdpa))
    t_self_total = time_op("self_attn FULL forward",    lambda: self_attn(x_norm, phases=phases))

    print("\nCross-attention (sub-ops):")
    t_to_q = time_op("to_q Linear (C→C)",               lambda: cross_attn.to_q(x_norm))
    t_to_kv = time_op("to_kv Linear (Cctx→2C)",         lambda: cross_attn.to_kv(cond))
    q_x = cross_attn.to_q(x_norm).reshape(1, F, H, D).transpose(0, 2, 1, 3)
    kv = cross_attn.to_kv(cond).reshape(1, Lkv, 2, H, D)
    k_x = kv[:, :, 0].transpose(0, 2, 1, 3); v_x = kv[:, :, 1].transpose(0, 2, 1, 3)
    t_csdpa = time_op("cross SDPA (L=F, Lkv)",           lambda: mx.fast.scaled_dot_product_attention(q_x, k_x, v_x, scale=D**-0.5))
    t_cross_total = time_op("cross_attn FULL forward",   lambda: cross_attn(x_norm, context=cond))

    print("\nMLP:")
    t_mlp = time_op("MLP (C→ratio*C→C, GELU)",          lambda: mlp(x_norm))

    # Sums
    self_total_estimate = t_qkv_proj + 2*t_rope + 2*t_qkrms + t_sdpa + t_to_out
    cross_total_estimate = t_to_q + t_to_kv + t_csdpa + 0  # plus a to_out
    block_estimate = sum([t_norm1, t_norm2, t_norm3, 2*t_mod, t_self_total, t_cross_total, t_mlp])

    print(f"\nEstimated block total (sum of pieces): {block_estimate:7.2f} ms")
    print(f"Self-attn sub-op sum check:            {self_total_estimate:7.2f} ms (vs full {t_self_total:7.2f})")

    if args.metal_trace:
        out_path = args.metal_trace
        print(f"\nCapturing Metal trace → {out_path}")
        mx.metal.start_capture(out_path)
        # One block forward, with the big self-attn
        for _ in range(3):
            out = self_attn(x_norm, phases=phases)
            mx.eval(out)
        mx.metal.stop_capture()
        print("Open in Xcode (File → Open) to inspect kernel times.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

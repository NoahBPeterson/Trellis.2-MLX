"""Microbench: time a single ModulatedTransformerCrossBlock forward.

Sizes match the real workloads (sparse SLat shape flow, 1024 cond tokens).
Compares un-compiled, mx.compile, and mx.compile+state-captured variants.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trellis2_mlx.modules.blocks import ModulatedTransformerCrossBlock


def bench(fn, args, warmup: int = 3, repeat: int = 30) -> float:
    for _ in range(warmup):
        out = fn(*args)
        if isinstance(out, mx.array):
            mx.eval(out)
    mx.synchronize()
    t0 = time.time()
    for _ in range(repeat):
        out = fn(*args)
        mx.eval(out)
    mx.synchronize()
    return (time.time() - t0) / repeat * 1000  # ms/call


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=3761, help="self-attn token count")
    p.add_argument("--cond-tokens", type=int, default=1029)
    p.add_argument("--channels", type=int, default=1536)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--mlp-ratio", type=float, default=5.3334)
    p.add_argument("--cond-channels", type=int, default=1024)
    args = p.parse_args()

    print(f"Block bench: F={args.tokens}, Lkv={args.cond_tokens}, C={args.channels}, H={args.heads}, mlp_ratio={args.mlp_ratio}")

    # Build a representative block (matches the DiTs we actually run)
    block = ModulatedTransformerCrossBlock(
        channels=args.channels, ctx_channels=args.cond_channels,
        num_heads=args.heads, mlp_ratio=args.mlp_ratio,
        use_rope=True, qk_rms_norm=True, qk_rms_norm_cross=True,
        share_mod=True,
    )
    # Cast all params to bf16 to match production weights, otherwise we measure
    # the cost of fp32→bf16 auto-cast per call instead of real compute.
    def _to_bf16(p):
        if isinstance(p, mx.array):
            return p.astype(mx.bfloat16)
        return p
    from mlx.utils import tree_map
    block.update(tree_map(_to_bf16, block.parameters()))

    # Inputs (B=1, F tokens for sparse). Phases (1, F, head_dim/2, 2) for RoPE.
    head_dim = args.channels // args.heads
    pairs = head_dim // 2
    x = mx.random.normal((1, args.tokens, args.channels)).astype(mx.bfloat16)
    mod = mx.random.normal((1, 6 * args.channels)).astype(mx.bfloat16)
    cond = mx.random.normal((1, args.cond_tokens, args.cond_channels)).astype(mx.bfloat16)
    phases = mx.random.normal((1, args.tokens, pairs, 2))

    def raw(x, mod, cond, phases):
        return block(x, mod, cond, phases=phases)

    # Compiled variants
    fn_eager = raw
    fn_compiled = mx.compile(raw)

    print()
    t1 = bench(fn_eager,    (x, mod, cond, phases))
    print(f"  eager                 {t1:7.2f} ms/call")
    t2 = bench(fn_compiled, (x, mod, cond, phases))
    print(f"  mx.compile (closure)  {t2:7.2f} ms/call   ({t1/t2:.2f}× vs eager)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

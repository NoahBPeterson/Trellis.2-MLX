"""Compare mx.fast.scaled_dot_product_attention against alternatives.

Suspicion: mx.fast SDPA underperforms at large L (19k). Test:
  - mx.fast.scaled_dot_product_attention (current)
  - Naive matmul+softmax (vanilla)
  - Chunked along L (FlashAttention-style tiling)
  - Different head/batch layouts
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def bench(name: str, fn, q, k, v, scale, warmup=3, repeat=20) -> float:
    for _ in range(warmup):
        out = fn(q, k, v, scale)
        mx.eval(out)
    mx.synchronize()
    t0 = time.time()
    for _ in range(repeat):
        out = fn(q, k, v, scale)
        mx.eval(out)
    mx.synchronize()
    ms = (time.time() - t0) / repeat * 1000
    # FLOPS estimate: 2 × B × H × L × Lkv × D (Q@K^T + probs@V)
    B, H, L, D = q.shape
    Lkv = k.shape[2]
    flops = 2 * B * H * L * Lkv * D * 2
    tflops = flops / ms / 1e9
    print(f"  {name:42s} {ms:8.2f} ms   {tflops:5.2f} TFLOPS")
    return ms


def fast_sdpa(q, k, v, scale):
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)


def naive_sdpa(q, k, v, scale):
    # q, k, v: (B, H, L, D)
    scores = mx.matmul(q, k.swapaxes(-1, -2)) * scale  # (B, H, L, Lkv)
    probs = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
    return mx.matmul(probs, v)


def naive_sdpa_no_upcast(q, k, v, scale):
    scores = mx.matmul(q, k.swapaxes(-1, -2)) * scale
    probs = mx.softmax(scores, axis=-1)
    return mx.matmul(probs, v)


def chunked_sdpa(q, k, v, scale, chunk: int = 4096):
    """Tile Q along L. Compute each chunk's attn against full K/V."""
    B, H, L, D = q.shape
    Lkv = k.shape[2]
    out_chunks = []
    for i in range(0, L, chunk):
        q_c = q[:, :, i:i+chunk, :]
        scores = mx.matmul(q_c, k.swapaxes(-1, -2)) * scale
        probs = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
        out_c = mx.matmul(probs, v)
        out_chunks.append(out_c)
    return mx.concatenate(out_chunks, axis=2)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=19104)
    p.add_argument("--lkv", type=int, default=None, help="Lkv tokens (default = L for self-attn)")
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = p.parse_args()

    L = args.tokens
    Lkv = args.lkv or L
    H, D = args.heads, args.head_dim
    B = 1
    dtype = getattr(mx, args.dtype)
    scale = D ** -0.5

    print(f"=== SDPA bench: B={B}, H={H}, L={L}, Lkv={Lkv}, D={D}, dtype={args.dtype} ===\n")
    q = mx.random.normal((B, H, L, D)).astype(dtype)
    k = mx.random.normal((B, H, Lkv, D)).astype(dtype)
    v = mx.random.normal((B, H, Lkv, D)).astype(dtype)

    bench("mx.fast.scaled_dot_product_attention", fast_sdpa, q, k, v, scale)
    # Naive matmul-softmax materializes a (B,H,L,Lkv) matrix; OOMs above ~10k tokens.
    if B * H * L * Lkv * 4 < 8e9:
        bench("naive matmul+softmax (fp32 softmax)",  naive_sdpa,            q, k, v, scale)
        bench("naive matmul+softmax (no upcast)",     naive_sdpa_no_upcast,  q, k, v, scale)
    else:
        print("  (naive variants skipped — attn matrix > 8 GB)")
    for chunk in (8192, 4096, 2048, 1024):
        if chunk > L:
            continue
        # Each chunk's attn matrix: B*H*chunk*Lkv*4 bytes
        if B * H * chunk * Lkv * 4 > 8e9:
            print(f"  chunked Q (chunk={chunk}) skipped — per-chunk attn > 8 GB")
            continue
        bench(f"chunked Q (chunk={chunk}, fp32 softmax)",
              lambda q, k, v, scale, c=chunk: chunked_sdpa(q, k, v, scale, chunk=c),
              q, k, v, scale)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

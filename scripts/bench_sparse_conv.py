"""Micro-benchmark a single submanifold conv call.

Isolates the `SparseConv3d.__call__` from the full VAE to see how fast a single
forward pass runs once the neighbor map is cached. If this is fast but the
full VAE is slow, the bottleneck is elsewhere (allocation, scheduling) rather
than in the conv's own compute.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trellis2_mlx.modules.sparse import SparseConv3d, build_neighbor_map
from trellis2_mlx.modules.sparse_tensor import SparseTensor


def bench(F: int, Ci: int, Co: int, warmup: int = 3, repeat: int = 10, dtype=mx.float16) -> None:
    rng = np.random.default_rng(0)
    xyz = rng.integers(0, int(F ** (1 / 3) * 2), size=(F, 3), dtype=np.int32)
    xyz = np.unique(xyz, axis=0)
    F_actual = xyz.shape[0]
    coords = np.concatenate([np.zeros((F_actual, 1), dtype=np.int32), xyz], axis=-1)

    feats = mx.array(rng.standard_normal((F_actual, Ci)).astype(np.float16))
    conv = SparseConv3d(Ci, Co, kernel_size=3)
    conv.weight = mx.array(rng.standard_normal((Co, 3, 3, 3, Ci)).astype(np.float16))
    conv.bias = mx.array(rng.standard_normal((Co,)).astype(np.float16))
    st = SparseTensor(feats=feats, coords=mx.array(coords), spatial_shape=(int(F_actual ** (1 / 3) * 2),) * 3)

    # Warm up (builds neighbor map cache + MLX JIT)
    for _ in range(warmup):
        out = conv(st)
        mx.eval(out.feats)

    # Time
    t0 = time.time()
    for _ in range(repeat):
        out = conv(st)
        mx.eval(out.feats)
    dt = (time.time() - t0) / repeat

    # Flops estimate: F * 27 * Ci * Co * 2 (matmul)
    flops = F_actual * 27 * Ci * Co * 2
    gflops = flops / dt / 1e9
    print(f"  F={F_actual:>7d}  Ci={Ci:>4d}  Co={Co:>4d}  dt={dt*1000:6.1f}ms  ~{gflops:6.1f} GFLOPS")


def main() -> int:
    # Approximate the stage shapes we see during VAE decode
    print("Sparse conv micro-benchmark (fp16)")
    print("-" * 60)
    print("Stage 0 (post-from_latent)  F~3k  Ci=Co=1024")
    bench(3000, 1024, 1024)
    print("Stage 1 (after 1 C2S up)   F~50k  Ci=Co=512")
    bench(50000, 512, 512)
    print("Stage 2 (after 2 C2S up)   F~200k  Ci=Co=256")
    bench(200000, 256, 256)
    print("Stage 3 (after 3 C2S up)   F~500k  Ci=Co=128")
    bench(500000, 128, 128)
    print("Stage 4 (after 4 C2S up)   F~1.5M  Ci=Co=64")
    bench(1500000, 64, 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

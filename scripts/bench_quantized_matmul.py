"""Compare fp16 matmul vs quantized (int8/int4) matmul at our DiT shapes.

Tests the three dominant linear shapes in the flow DiT:
  - QKV projection: F x 1536 -> 4608
  - MLP up:         F x 1536 -> 8192  (biggest)
  - MLP down:       F x 8192 -> 1536

Reports ms/call and effective TFLOPS for each dtype/mode.
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


def bench_fp16(x, w, b):
    # Plain Linear: x @ w.T + b (MLX stores Linear.weight as (out, in))
    return x @ w.T + b


def bench_quantized_matmul(x, w_q, scales, biases, bits, group_size, bias):
    return mx.quantized_matmul(x, w_q, scales=scales, biases=biases, bits=bits, group_size=group_size) + bias


def time_it(name, fn, warmup=3, repeat=20) -> float:
    for _ in range(warmup):
        out = fn()
        mx.eval(out)
    mx.synchronize()
    t0 = time.time()
    for _ in range(repeat):
        out = fn()
        mx.eval(out)
    mx.synchronize()
    return (time.time() - t0) / repeat * 1000


def bench_shape(F: int, Cin: int, Cout: int, label: str, group_size: int = 64) -> None:
    print(f"\n{label}  (F={F}, {Cin}→{Cout})")
    x_fp16 = mx.random.normal((F, Cin)).astype(mx.float16)
    w_fp16 = mx.random.normal((Cout, Cin)).astype(mx.float16)
    b_fp16 = mx.random.normal((Cout,)).astype(mx.float16)

    flops = 2 * F * Cin * Cout

    # fp16
    t_fp16 = time_it("fp16",
                     lambda: bench_fp16(x_fp16, w_fp16, b_fp16))
    tflops_fp16 = flops / t_fp16 / 1e9
    print(f"  fp16 matmul            {t_fp16:7.2f} ms   {tflops_fp16:5.2f} TFLOPS   (baseline)")

    # int8
    w_q8, scales_q8, biases_q8 = mx.quantize(w_fp16, group_size=group_size, bits=8)
    t_q8 = time_it("int8",
                   lambda: bench_quantized_matmul(x_fp16, w_q8, scales_q8, biases_q8, 8, group_size, b_fp16))
    print(f"  int8 qmatmul           {t_q8:7.2f} ms   {flops/t_q8/1e9:5.2f} TFLOPS   ({t_fp16/t_q8:.2f}× vs fp16)")

    # int4
    w_q4, scales_q4, biases_q4 = mx.quantize(w_fp16, group_size=group_size, bits=4)
    t_q4 = time_it("int4",
                   lambda: bench_quantized_matmul(x_fp16, w_q4, scales_q4, biases_q4, 4, group_size, b_fp16))
    print(f"  int4 qmatmul           {t_q4:7.2f} ms   {flops/t_q4/1e9:5.2f} TFLOPS   ({t_fp16/t_q4:.2f}× vs fp16)")

    # Memory footprint
    size_fp16 = Cout * Cin * 2
    size_q8 = w_q8.nbytes + scales_q8.nbytes + biases_q8.nbytes
    size_q4 = w_q4.nbytes + scales_q4.nbytes + biases_q4.nbytes
    print(f"  weight size: fp16 {size_fp16/1e6:.1f} MB  int8 {size_q8/1e6:.1f} MB ({size_fp16/size_q8:.2f}×)  int4 {size_q4/1e6:.1f} MB ({size_fp16/size_q4:.2f}×)")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=19104, help="F (defaults to 1024-shape-flow size)")
    args = p.parse_args()

    print(f"=== Quantized matmul bench (fp16 baseline) @ F={args.tokens} ===")
    bench_shape(args.tokens, 1536, 4608, "QKV proj")
    bench_shape(args.tokens, 1536, 8192, "MLP up")
    bench_shape(args.tokens, 8192, 1536, "MLP down")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

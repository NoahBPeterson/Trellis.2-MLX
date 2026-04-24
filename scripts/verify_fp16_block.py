"""bf16 vs fp16 numerical comparison on one DiT block.

Builds an identical block in both dtypes (same params, just cast), feeds the
same inputs, and reports max-abs and relative error. Sanity check before
casting the production weights.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_map

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trellis2_mlx.modules.blocks import ModulatedTransformerCrossBlock


def load_block(channels=1536, ctx_channels=1024, num_heads=12, mlp_ratio=5.3334):
    return ModulatedTransformerCrossBlock(
        channels=channels, ctx_channels=ctx_channels,
        num_heads=num_heads, mlp_ratio=mlp_ratio,
        use_rope=True, qk_rms_norm=True, qk_rms_norm_cross=True,
        share_mod=True,
    )


def cast_params(block, dtype):
    block.update(tree_map(lambda p: p.astype(dtype) if isinstance(p, mx.array) else p, block.parameters()))
    return block


def run_block(block, x, mod, cond, phases):
    return block(x, mod, cond, phases=phases)


def main() -> int:
    # Use real loaded weights from a converted DiT shard so we test on actual values
    from trellis2_mlx.models.flow_dit import SLatFlowModel
    import json
    cfg_path = ROOT / "ckpts/slat_flow_img2shape_dit_1_3B_512.config.json"
    cfg = json.loads(cfg_path.read_text())["args"]
    cfg = {k: v for k, v in cfg.items() if k not in ("initialization", "dtype")}
    model = SLatFlowModel(**cfg)
    from trellis2_mlx.pipeline import _load_weights_prefixed
    _load_weights_prefixed(model, ROOT / "ckpts/slat_flow_img2shape_dit_1_3B_512.safetensors")

    # Take block 15 (middle of the 30-block stack) as a representative test
    real_block = model.blocks[15]
    from mlx.utils import tree_flatten
    flat = tree_flatten(real_block.parameters())
    print(f"Loaded block 15 from shape_flow_512. Sample param dtypes:")
    for name, val in flat[:3]:
        print(f"  {name:30s} {val.dtype}  shape={val.shape}")
    print()

    # Build inputs at production-realistic shapes (3.8k tokens like 512 shape flow)
    F, Lkv, C, H = 3761, 1029, 1536, 12
    pairs = (C // H) // 2
    mx.random.seed(0)
    # Use fp32 inputs as ground-truth-ish (we'll cast per-test)
    x_f32 = mx.random.normal((1, F, C))
    mod_f32 = mx.random.normal((1, 6 * C))
    cond_f32 = mx.random.normal((1, Lkv, 1024))
    phases_f32 = mx.random.normal((1, F, pairs, 2))

    # Run bf16 (the production baseline)
    block_bf16 = cast_params(real_block, mx.bfloat16)
    x_bf16 = x_f32.astype(mx.bfloat16); mod_bf16 = mod_f32.astype(mx.bfloat16)
    cond_bf16 = cond_f32.astype(mx.bfloat16); phases_bf16 = phases_f32  # phases stay fp32
    out_bf16 = run_block(block_bf16, x_bf16, mod_bf16, cond_bf16, phases_bf16)
    out_bf16_f32 = out_bf16.astype(mx.float32); mx.eval(out_bf16_f32)

    # Re-load model fresh, cast to fp16 (the experiment)
    model2 = SLatFlowModel(**cfg)
    _load_weights_prefixed(model2, ROOT / "ckpts/slat_flow_img2shape_dit_1_3B_512.safetensors")
    block_fp16 = cast_params(model2.blocks[15], mx.float16)
    x_fp16 = x_f32.astype(mx.float16); mod_fp16 = mod_f32.astype(mx.float16)
    cond_fp16 = cond_f32.astype(mx.float16); phases_fp16 = phases_f32
    out_fp16 = run_block(block_fp16, x_fp16, mod_fp16, cond_fp16, phases_fp16)
    out_fp16_f32 = out_fp16.astype(mx.float32); mx.eval(out_fp16_f32)

    # Comparison — use numpy for percentile + masked-relative-diff
    import numpy as np
    diff_np = np.asarray(mx.abs(out_bf16_f32 - out_fp16_f32))
    bf16_np = np.asarray(out_bf16_f32)
    fp16_np = np.asarray(out_fp16_f32)

    # Only compute relative error where the reference has non-trivial magnitude,
    # otherwise tiny denominators inflate the metric.
    sig_mask = np.abs(bf16_np) > 0.1 * np.std(bf16_np)
    rel_sig = diff_np[sig_mask] / np.abs(bf16_np[sig_mask])

    # bf16 vs bf16 jitter (run twice — should be 0 if MLX is deterministic)
    bf16_self = run_block(block_bf16, x_bf16, mod_bf16, cond_bf16, phases_bf16).astype(mx.float32)
    mx.eval(bf16_self)
    bf16_jitter = float(mx.max(mx.abs(bf16_self - out_bf16_f32)))

    print("Output comparison (block 15, F=3761):")
    print(f"  out_bf16  range: [{bf16_np.min():.3f}, {bf16_np.max():.3f}]  std={bf16_np.std():.3f}")
    print(f"  out_fp16  range: [{fp16_np.min():.3f}, {fp16_np.max():.3f}]  std={fp16_np.std():.3f}")
    print(f"  bf16 self-jitter (deterministic check): {bf16_jitter:.4e}")
    print(f"  fp16 vs bf16:")
    print(f"    max abs diff       : {diff_np.max():.4e}")
    print(f"    mean abs diff      : {diff_np.mean():.4e}")
    print(f"    p50 abs diff       : {np.median(diff_np):.4e}")
    print(f"    p99 abs diff       : {np.percentile(diff_np, 99):.4e}")
    print(f"    p99.9 abs diff     : {np.percentile(diff_np, 99.9):.4e}")
    print(f"    relative-where-significant (|bf16| > 0.1*std):")
    print(f"      max rel diff     : {rel_sig.max():.4e}")
    print(f"      mean rel diff    : {rel_sig.mean():.4e}")
    print(f"      p99 rel diff     : {np.percentile(rel_sig, 99):.4e}")
    print()
    diff_rel_to_bf16_range = diff_np.max() / (bf16_np.max() - bf16_np.min())
    print(f"  diff vs output range : {diff_rel_to_bf16_range*100:.2f}%  (max abs diff / total range)")
    print()
    print("Interpretation:")
    print("  - bf16 self-jitter > 0  → MLX has nondeterminism, raise our acceptable threshold")
    print("  - p99 abs diff < std    → fp16 noise is within natural output variance")
    print("  - max rel diff < 5%     → fp16 acceptable for downstream sampler steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

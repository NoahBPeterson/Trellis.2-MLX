"""Convert upstream torch safetensors to MLX-consumable safetensors.

Name translations (Sequential children):
  t_embedder.mlp.N.*            -> t_embedder.mlp.layers.N.*
  adaLN_modulation.N.*          -> adaLN_modulation.layers.N.*
  blocks.I.mlp.mlp.N.*          -> blocks.I.mlp.mlp.layers.N.*
  blocks.I.adaLN_modulation.N.* -> blocks.I.adaLN_modulation.layers.N.*

All other names pass through unchanged. Dtypes are preserved
(bf16 for DiTs, fp16 for VAEs).

Usage:
  python scripts/convert_weights.py <input.safetensors> <output.safetensors> [--config <name>.json]

Emits a sibling `<output>.config.json` describing the bijection for traceability.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import mlx.core as mx
from safetensors import safe_open


DTYPE_MAP = {
    "F16": mx.float16,
    "BF16": mx.bfloat16,
    "F32": mx.float32,
}


def translate_key(k: str) -> str:
    """Apply Sequential->layers rewrites.

    MLX's `nn.Sequential` stores children under `.layers[i]` rather than
    indexed directly like torch's `nn.Sequential`. Every upstream path that
    indexes into a Sequential needs `layers` injected.
    """
    # Any `.mlp.<digit>.` (DiT FeedForward's inner Sequential, VAE ConvNeXt's mlp)
    k = re.sub(r"\.mlp\.(\d+)\.", r".mlp.layers.\1.", k)
    # Leading t_embedder.mlp.N.*
    k = re.sub(r"^t_embedder\.mlp\.(\d+)\.", r"t_embedder.mlp.layers.\1.", k)
    # adaLN_modulation.N.*  (root and per-block)
    k = re.sub(r"(^|\.)adaLN_modulation\.(\d+)\.", r"\1adaLN_modulation.layers.\2.", k)
    return k


def convert_shard(src: Path, dst: Path) -> dict:
    tensors = {}
    dtype_histogram: dict[str, int] = {}
    bijection: list[dict] = []
    with safe_open(src, framework="pt") as f:
        src_meta = f.metadata() or {}
        for key in f.keys():
            t = f.get_tensor(key)
            dt = str(t.dtype).replace("torch.", "")
            mlx_dtype = DTYPE_MAP.get(
                {"float16": "F16", "bfloat16": "BF16", "float32": "F32"}.get(dt, dt.upper())
            )
            if mlx_dtype is None:
                raise RuntimeError(f"Unknown torch dtype {t.dtype} for {key}")
            # torch .numpy() fails on bf16 — go through f32 intermediate then cast.
            if dt == "bfloat16":
                arr = mx.array(t.float().numpy()).astype(mx.bfloat16)
            else:
                arr = mx.array(t.numpy())
            new_key = translate_key(key)
            tensors[new_key] = arr
            bijection.append({"src": key, "dst": new_key, "shape": list(t.shape), "dtype": dt})
            dtype_histogram[dt] = dtype_histogram.get(dt, 0) + 1

    dst.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(dst), tensors, metadata={"source": src.name, **src_meta})

    manifest = {
        "source": str(src),
        "output": str(dst),
        "num_tensors": len(tensors),
        "dtype_histogram": dtype_histogram,
        "bijection": bijection,
    }
    return manifest


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("src", type=Path, help="upstream .safetensors file")
    p.add_argument("dst", type=Path, help="output .safetensors file")
    p.add_argument("--config", type=Path, help="copy a JSON config alongside output", default=None)
    args = p.parse_args()

    if not args.src.exists():
        print(f"missing: {args.src}", file=sys.stderr)
        return 1

    manifest = convert_shard(args.src, args.dst)
    man_path = args.dst.with_suffix(".bijection.json")
    man_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {args.dst} ({manifest['num_tensors']} tensors, dtypes {manifest['dtype_histogram']})")
    print(f"wrote {man_path}")

    if args.config is not None and args.config.exists():
        out_cfg = args.dst.with_suffix(".config.json")
        out_cfg.write_text(args.config.read_text())
        print(f"copied {args.config} -> {out_cfg}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Prefix every tensor in every ckpts/*.safetensors with a component name, then
emit a unified `model.safetensors.index.json` so HF renders a single Tensors tab.

Rationale: many component checkpoints share tensor names (e.g. `blocks.0.self_attn.to_qkv.weight`
appears in SS flow, shape flow, and tex flow). An HF unified index requires unique
tensor names across all shards, so we re-save each file with its tensors prefixed by
a component tag.

Safety:
- Writes to `<file>.__rewrite.safetensors` first, then atomic `os.replace`.
- If an upstream shard is somehow lost, re-run `scripts/convert_weights.py` to regenerate from `weights/ckpts/`.
- Idempotent: detects already-prefixed files and skips them.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "ckpts"

# filename → component prefix used in the unified index
PREFIXES: dict[str, str] = {
    "ss_flow_img_dit_1_3B_64.safetensors":                  "ss_flow",
    "ss_dec_conv3d_16l8.safetensors":                       "ss_dec",
    "slat_flow_img2shape_dit_1_3B_512.safetensors":         "shape_flow_512",
    "slat_flow_img2shape_dit_1_3B_1024.safetensors":        "shape_flow_1024",
    "slat_flow_imgshape2tex_dit_1_3B_512.safetensors":      "tex_flow_512",
    "slat_flow_imgshape2tex_dit_1_3B_1024.safetensors":     "tex_flow_1024",
    "shape_dec_next_dc_f16c32.safetensors":                 "shape_dec",
    "shape_enc_next_dc_f16c32.safetensors":                 "shape_enc",
    "tex_dec_next_dc_f16c32.safetensors":                   "tex_dec",
    "tex_enc_next_dc_f16c32.safetensors":                   "tex_enc",
}


def _already_prefixed(weights: dict[str, mx.array], prefix: str) -> bool:
    """Heuristic: if ALL keys start with `<prefix>.`, the file has already been rewritten."""
    pfx = prefix + "."
    return all(k.startswith(pfx) for k in weights)


def main() -> int:
    weight_map: dict[str, str] = {}
    total_size: int = 0

    for fname, prefix in PREFIXES.items():
        path = CKPT / fname
        if not path.exists():
            print(f"SKIP missing: {path}", file=sys.stderr)
            continue

        weights = mx.load(str(path))
        n = len(weights)

        if _already_prefixed(weights, prefix):
            print(f"skip {fname:<55s} already prefixed ({n} tensors)")
        else:
            prefixed = {f"{prefix}.{k}": v for k, v in weights.items()}
            # keep .safetensors extension so mx.load() can sniff the format
            tmp = path.with_suffix(".rewrite.safetensors")
            mx.save_safetensors(str(tmp), prefixed)
            # round-trip verify
            rt = mx.load(str(tmp))
            assert len(rt) == n, f"round-trip count mismatch {fname}: {len(rt)} vs {n}"
            assert all(k.startswith(prefix + ".") for k in rt), f"prefix not applied in {fname}"
            os.replace(tmp, path)
            print(f"rewrote {fname:<52s} ({n} tensors, prefix '{prefix}.')")
            weights = rt

        for k, v in weights.items():
            weight_map[k] = f"ckpts/{fname}"
            total_size += v.nbytes

    index = {
        "metadata": {
            "total_size": total_size,
            "format": "mlx-safetensors-unified",
            "note": "Each component's tensors are prefixed with a component tag. See per-file bijection.json for torch↔MLX name mapping (pre-prefix).",
        },
        "weight_map": dict(sorted(weight_map.items())),
    }
    out = ROOT / "model.safetensors.index.json"
    out.write_text(json.dumps(index, indent=2) + "\n")
    print(f"\nwrote {out} with {len(weight_map)} tensors across {len(PREFIXES)} shards ({total_size/1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

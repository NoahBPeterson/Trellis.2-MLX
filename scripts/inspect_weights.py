"""
Dump layer names, shapes, and dtypes from every safetensors shard in weights/ckpts/.

Produces:
- artifacts/weight_manifest.json  — machine-readable manifest
- stdout summary grouped by shard
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = ROOT / "weights" / "ckpts"
EXTRA_DIRS = [ROOT / "weights-trellis1" / "ckpts"]
OUT = ROOT / "artifacts" / "weight_manifest.json"


def inspect_shard(path: Path) -> dict:
    tensors = {}
    with safe_open(path, framework="pt") as f:
        meta = f.metadata() or {}
        for name in f.keys():
            t = f.get_slice(name)
            tensors[name] = {
                "shape": list(t.get_shape()),
                "dtype": str(t.get_dtype()),
            }
    return {"metadata": meta, "tensors": tensors, "num_tensors": len(tensors)}


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    dirs = [WEIGHTS_DIR, *EXTRA_DIRS]
    shards = []
    for d in dirs:
        if d.exists():
            shards.extend(sorted(d.glob("*.safetensors")))

    if not shards:
        print(f"No safetensors files found under {[str(d) for d in dirs]}", file=sys.stderr)
        return 1

    for shard in shards:
        rel = shard.relative_to(ROOT).as_posix()
        print(f"\n== {rel} ==")
        info = inspect_shard(shard)
        manifest[rel] = info
        print(f"   num_tensors: {info['num_tensors']}")
        if info["metadata"]:
            print(f"   metadata: {info['metadata']}")
        # First few + last few tensor names
        names = list(info["tensors"].keys())
        preview = names[:6] + (["..."] if len(names) > 12 else []) + names[-6:] if len(names) > 12 else names
        dtypes = sorted({t["dtype"] for t in info["tensors"].values()})
        print(f"   dtypes: {dtypes}")
        print(f"   sample names:")
        for n in preview:
            if n == "...":
                print(f"     ...")
            else:
                info_t = info["tensors"][n]
                print(f"     {n}   {info_t['shape']}  {info_t['dtype']}")

    OUT.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {OUT.relative_to(ROOT)} ({len(manifest)} shards)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

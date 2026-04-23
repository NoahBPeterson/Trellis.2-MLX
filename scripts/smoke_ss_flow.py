"""Load real SS flow weights and run a forward pass on 16^3 noise."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trellis2_mlx.models.flow_dit import SparseStructureFlowModel


def main() -> int:
    weights = ROOT / "ckpts" / "ss_flow_img_dit_1_3B_64.safetensors"
    cfg = json.loads((ROOT / "ckpts" / "ss_flow_img_dit_1_3B_64.config.json").read_text())
    args = {k: v for k, v in cfg["args"].items() if k not in ("initialization", "dtype")}
    print(f"args: {args}")

    t0 = time.time()
    model = SparseStructureFlowModel(**args)
    model.load_weights(str(weights))
    t1 = time.time()
    print(f"loaded in {t1 - t0:.2f}s")

    from mlx.utils import tree_flatten
    total = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"parameter count: {total / 1e9:.2f} B")

    # x: (B, 8, 16, 16, 16) noise; cond: DINOv3 features (1089, 1024)
    rng = np.random.default_rng(0)
    x = mx.array(rng.standard_normal((1, 8, 16, 16, 16)).astype(np.float32))
    t = mx.array([500.0], dtype=mx.float32)
    cond = mx.array(rng.standard_normal((1, 1089, 1024)).astype(np.float32))

    print("Running SS flow forward (1.3B bf16, RoPE, 4096 tokens)...")
    t0 = time.time()
    out = model(x, t, cond)
    mx.eval(out)
    t1 = time.time()
    print(f"forward: {t1 - t0:.2f}s  shape={tuple(out.shape)} dtype={out.dtype}")
    print(f"stats: mean={float(out.mean()):.4f} std={float(out.std()):.4f} min={float(out.min()):.2f} max={float(out.max()):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

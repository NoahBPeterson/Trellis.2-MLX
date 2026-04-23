"""Load SS decoder and run a forward pass on a dummy 16^3 latent."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trellis2_mlx.models.ss_decoder import SparseStructureDecoder


def main() -> int:
    weights = ROOT / "ckpts" / "ss_dec_conv3d_16l8.safetensors"
    cfg = json.loads((ROOT / "ckpts" / "ss_dec_conv3d_16l8.config.json").read_text())
    print(f"args: {cfg['args']}")
    model = SparseStructureDecoder(**cfg["args"])

    t0 = time.time()
    model.load_weights(str(weights))
    print(f"loaded in {time.time() - t0:.2f}s")
    from mlx.utils import tree_flatten
    total = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"params: {total / 1e6:.1f} M  ({sum(1 for _ in tree_flatten(model.parameters()))} tensors)")

    rng = np.random.default_rng(0)
    x = mx.array(rng.standard_normal((1, 8, 16, 16, 16)).astype(np.float32))
    t0 = time.time()
    out = model(x)
    mx.eval(out)
    t1 = time.time()
    print(f"forward {t1 - t0:.2f}s  out shape {tuple(out.shape)} dtype {out.dtype}")
    print(f"occupancy logits: mean={float(out.mean()):.3f} std={float(out.std()):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

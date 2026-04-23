"""Load a converted DiT safetensors into its MLX model and run a dummy forward pass."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trellis2_mlx.models.flow_dit import SLatFlowModel, SparseStructureFlowModel
from trellis2_mlx.modules.sparse_tensor import SparseTensor


def _load_config(path: Path) -> dict:
    return json.loads(path.read_text())


def _strip_unused(args: dict) -> dict:
    # Our MLX constructors accept **_unused but keep args clean
    return {k: v for k, v in args.items() if k not in ("initialization", "dtype")}


def load_slat_flow(weights: Path, config: Path) -> SLatFlowModel:
    cfg = _load_config(config)
    model = SLatFlowModel(**_strip_unused(cfg["args"]))
    model.load_weights(str(weights))
    return model


def load_ss_flow(weights: Path, config: Path) -> SparseStructureFlowModel:
    cfg = _load_config(config)
    model = SparseStructureFlowModel(**_strip_unused(cfg["args"]))
    model.load_weights(str(weights))
    return model


def main() -> int:
    weights = ROOT / "ckpts" / "slat_flow_imgshape2tex_dit_1_3B_1024.safetensors"
    config = ROOT / "ckpts" / "slat_flow_imgshape2tex_dit_1_3B_1024.config.json"
    print(f"Loading {weights.name} ...")
    t0 = time.time()
    model = load_slat_flow(weights, config)
    t1 = time.time()
    print(f"loaded in {t1 - t0:.2f}s")

    # Count params
    from mlx.utils import tree_flatten
    params = tree_flatten(model.parameters())
    total = sum(p.size for _, p in params)
    print(f"parameter count: {total / 1e9:.2f} B  ({len(params)} tensors)")

    # Dummy sparse input: 64 active voxels at a 1024³ grid with random feats
    F = 64
    rng = np.random.default_rng(0)
    coords_np = np.stack([
        np.zeros(F, dtype=np.int32),
        rng.integers(0, 1024, F, dtype=np.int32),
        rng.integers(0, 1024, F, dtype=np.int32),
        rng.integers(0, 1024, F, dtype=np.int32),
    ], axis=-1)
    # For tex flow with in_channels=64, but model expects concat_cond at runtime:
    # pass shape_slat feats (32) and noise (32) separately
    shape_feats = rng.standard_normal((F, 32)).astype(np.float32)
    noise_feats = rng.standard_normal((F, 32)).astype(np.float32)
    shape_slat = SparseTensor(feats=mx.array(shape_feats), coords=mx.array(coords_np), spatial_shape=(1024, 1024, 1024))
    noise = SparseTensor(feats=mx.array(noise_feats), coords=mx.array(coords_np), spatial_shape=(1024, 1024, 1024))
    t = mx.array([500.0], dtype=mx.float32)
    # Image cond is (B, N_cond, 1024) — DINOv3 features. Use random placeholder.
    cond = mx.array(rng.standard_normal((1, 1089, 1024)).astype(np.float32))

    print("Running forward pass (this is 1.3B params in bf16)...")
    t0 = time.time()
    out = model(noise, t, cond, concat_cond=shape_slat)
    mx.eval(out.feats)  # force MLX to actually compute
    t1 = time.time()
    print(f"forward pass {t1 - t0:.2f}s; output feats shape {tuple(out.feats.shape)} dtype {out.feats.dtype}")
    print(f"output stats: mean={float(out.feats.mean()):.4f} std={float(out.feats.std()):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

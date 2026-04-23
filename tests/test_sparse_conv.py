"""Validate MLX submanifold 3D conv + S2C/C2S against a naive reference.

No CUDA/torchsparse needed — the reference is a hand-written NumPy implementation
of submanifold convolution and spatial<->channel packing. We just want to ensure
our MLX kernels are self-consistent and match the mathematical spec.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trellis2_mlx.modules.sparse import (
    SparseChannel2Spatial,
    SparseConv3d,
    SparseSpatial2Channel,
)
from trellis2_mlx.modules.sparse_tensor import SparseTensor


def _naive_submconv3d(feats, coords, weight, bias):
    """Reference implementation: submanifold 3D conv, numpy only.
    feats: (F, Ci)
    coords: (F, 4) [b, x, y, z]
    weight: (Co, 3, 3, 3, Ci)
    bias  : (Co,) or None
    """
    F, Ci = feats.shape
    Co = weight.shape[0]
    table = {tuple(row.tolist()): i for i, row in enumerate(coords)}
    out = np.zeros((F, Co), dtype=feats.dtype)
    # kernel offsets in the same order as build_neighbor_map: k = kz*9 + ky*3 + kx
    offs = [(dx, dy, dz) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    for kidx, (dx, dy, dz) in enumerate(offs):
        kx, ky, kz = dx + 1, dy + 1, dz + 1
        wk = weight[:, kz, ky, kx, :]  # (Co, Ci)
        for i, (b, x, y, z) in enumerate(coords.tolist()):
            j = table.get((b, x + dx, y + dy, z + dz))
            if j is not None:
                out[i] += feats[j] @ wk.T
    if bias is not None:
        out = out + bias
    return out


def test_submconv3d_matches_naive():
    rng = np.random.default_rng(0)
    F, Ci, Co = 64, 8, 16
    feats = rng.standard_normal((F, Ci)).astype(np.float32) * 0.3
    # Random coords on a 16^3 grid, batch 0
    xyz = rng.integers(0, 16, size=(F, 3)).astype(np.int32)
    xyz = np.unique(xyz, axis=0)
    F = xyz.shape[0]
    feats = feats[:F]
    coords = np.concatenate([np.zeros((F, 1), dtype=np.int32), xyz], axis=-1)
    weight = rng.standard_normal((Co, 3, 3, 3, Ci)).astype(np.float32) * 0.1
    bias = rng.standard_normal((Co,)).astype(np.float32) * 0.05

    # Naive
    t_out = _naive_submconv3d(feats, coords, weight, bias)

    # MLX
    conv = SparseConv3d(Ci, Co, kernel_size=3)
    conv.weight = mx.array(weight)
    conv.bias = mx.array(bias)
    st = SparseTensor(feats=mx.array(feats), coords=mx.array(coords), spatial_shape=(16, 16, 16))
    m_out = np.asarray(conv(st).feats)

    diff = np.abs(t_out - m_out).max()
    print(f"submconv3d fp32 max-abs diff: {diff:.2e}")
    assert diff < 1e-4, f"diff {diff} too large"


def test_s2c_c2s_roundtrip():
    """S2C then C2S should be identity (up to the cached layout)."""
    rng = np.random.default_rng(1)
    F, Ci = 32, 8
    xyz = rng.integers(0, 8, size=(F, 3)).astype(np.int32)
    xyz = np.unique(xyz, axis=0)
    F = xyz.shape[0]
    coords = np.concatenate([np.zeros((F, 1), dtype=np.int32), xyz], axis=-1)
    feats = rng.standard_normal((F, Ci)).astype(np.float32)

    st = SparseTensor(feats=mx.array(feats), coords=mx.array(coords), spatial_shape=(8, 8, 8))
    s2c = SparseSpatial2Channel(2)
    c2s = SparseChannel2Spatial(2)
    packed = s2c(st)
    print(f"S2C: {F} voxels of {Ci} ch -> {packed.feats.shape[0]} voxels of {packed.feats.shape[1]} ch")
    recovered = c2s(packed)
    # Recovered should equal original (via the S2C->C2S cached layout path)
    diff = np.abs(np.asarray(recovered.feats) - feats).max()
    print(f"S2C→C2S round-trip max-abs diff: {diff:.2e}")
    assert diff == 0, f"round-trip not exact: diff {diff}"
    # Coords should also match (order might differ but set equality)
    orig_set = set(map(tuple, coords.tolist()))
    rec_set = set(map(tuple, np.asarray(recovered.coords).tolist()))
    assert orig_set == rec_set, "coord sets differ after round-trip"


if __name__ == "__main__":
    test_submconv3d_matches_naive()
    test_s2c_c2s_roundtrip()
    print("OK: sparse conv + S2C/C2S")

"""Cross-check MLX submanifold conv against torch dense conv3d.

The upstream sparse weight `(Co, Kd, Kh, Kw, Ci)` is a permuted torch conv3d
weight. If we:
  1. Unpermute back to `(Co, Ci, Kd, Kh, Kw)`
  2. Build a dense 3D volume from the sparse voxels (active voxels + zeros elsewhere)
  3. Apply `torch.nn.functional.conv3d` (padding=1 so output matches input grid)
  4. Extract values at the active-voxel positions

...we should get the same result as my MLX submanifold conv. If they disagree,
my conv has an axis-ordering bug — and the TORCH side is authoritative because
it matches how upstream trained the weights.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trellis2_mlx.modules.sparse import SparseConv3d
from trellis2_mlx.modules.sparse_tensor import SparseTensor


def test_submconv_matches_dense_conv3d():
    rng = np.random.default_rng(42)
    Ci, Co = 6, 10
    R = 8  # small grid

    # Random subset of active voxels
    xyz = rng.integers(0, R, size=(30, 3)).astype(np.int32)
    xyz = np.unique(xyz, axis=0)
    Fn = xyz.shape[0]
    feats = rng.standard_normal((Fn, Ci)).astype(np.float32) * 0.3
    coords = np.concatenate([np.zeros((Fn, 1), dtype=np.int32), xyz], axis=-1)

    # Weights: torch Conv3d layout `(Co, Ci, Kd, Kh, Kw)` → upstream permutes to
    # `(Co, Kd, Kh, Kw, Ci)`. Build in upstream-permuted layout, then unpermute
    # for the torch reference.
    w_upstream = rng.standard_normal((Co, 3, 3, 3, Ci)).astype(np.float32) * 0.1
    b = rng.standard_normal((Co,)).astype(np.float32) * 0.05
    w_torch = np.transpose(w_upstream, (0, 4, 1, 2, 3))  # (Co, Ci, Kd, Kh, Kw)

    # --- Torch dense reference -------------------------------------------------
    # Build a dense (1, Ci, R, R, R) volume with features placed at the active
    # voxel positions. Axis convention MUST match upstream: coords[:, 1:4] is
    # (axis0, axis1, axis2) which in torch's Conv3d on (B, C, D, H, W) are
    # (D, H, W). So coord column 1 → D, column 2 → H, column 3 → W.
    volume = np.zeros((1, Ci, R, R, R), dtype=np.float32)
    for i in range(Fn):
        volume[0, :, xyz[i, 0], xyz[i, 1], xyz[i, 2]] = feats[i]
    out_dense = F.conv3d(
        torch.from_numpy(volume),
        torch.from_numpy(w_torch),
        bias=torch.from_numpy(b),
        padding=1,
    ).numpy()
    # Extract output at the active coords
    out_torch = np.stack([out_dense[0, :, xyz[i, 0], xyz[i, 1], xyz[i, 2]] for i in range(Fn)], axis=0)  # (Fn, Co)

    # --- MLX submanifold conv --------------------------------------------------
    mx_conv = SparseConv3d(Ci, Co, kernel_size=3)
    mx_conv.weight = mx.array(w_upstream)
    mx_conv.bias = mx.array(b)
    st = SparseTensor(feats=mx.array(feats), coords=mx.array(coords), spatial_shape=(R, R, R))
    out_mlx = np.asarray(mx_conv(st).feats)

    diff = np.abs(out_torch - out_mlx).max()
    print(f"submconv vs torch dense conv3d fp32 max-abs diff: {diff:.2e}")
    if diff >= 5e-5:
        print("  first 3 rows of torch_out:", out_torch[:3])
        print("  first 3 rows of mlx_out  :", out_mlx[:3])
    assert diff < 5e-5, f"MLX submanifold conv disagrees with torch dense conv3d ({diff}) — kernel-axis order bug"


def test_s2c_matches_dense_spatial_to_channel():
    """S2C on a sparse tensor with ALL voxels active must equal torch's
    pixel_unshuffle equivalent on the dense volume.

    spatial_to_channel via reshape+permute: (B, C, 2h, 2w, 2d) -> (B, 8C, h, w, d)
    where the 8 sub-slots are ordered (x, y, z) = (axis0, axis1, axis2) with
    axis0 fastest in the flat 8 index — matching upstream's subidx encoding.
    """
    from trellis2_mlx.modules.sparse import SparseSpatial2Channel

    R = 4
    Ci = 3
    rng = np.random.default_rng(7)
    # Build a dense volume and a fully-active sparse counterpart
    dense = rng.standard_normal((1, Ci, R, R, R)).astype(np.float32)
    coords = np.stack(np.meshgrid(np.arange(R), np.arange(R), np.arange(R), indexing="ij"), axis=-1).reshape(-1, 3).astype(np.int32)
    coords = np.concatenate([np.zeros((coords.shape[0], 1), dtype=np.int32), coords], axis=-1)
    feats = np.stack([dense[0, :, c[1], c[2], c[3]] for c in coords])  # (F, Ci)

    # Torch reference: axis0-fastest 8-slot packing. The parent voxel at
    # (a0, a1, a2) = (i, j, k) aggregates children at (2i+dx, 2j+dy, 2k+dz)
    # with dx, dy, dz ∈ {0, 1}. Upstream's subidx = dx + 2*dy + 4*dz.
    R2 = R // 2
    torch_packed = np.zeros((1, 8 * Ci, R2, R2, R2), dtype=np.float32)
    for i in range(R2):
        for j in range(R2):
            for k in range(R2):
                for dx in range(2):
                    for dy in range(2):
                        for dz in range(2):
                            sub = dx + 2 * dy + 4 * dz
                            src = dense[0, :, 2 * i + dx, 2 * j + dy, 2 * k + dz]  # (Ci,)
                            torch_packed[0, sub * Ci : (sub + 1) * Ci, i, j, k] = src

    st = SparseTensor(feats=mx.array(feats), coords=mx.array(coords), spatial_shape=(R, R, R))
    s2c = SparseSpatial2Channel(2)
    packed = s2c(st)
    # Pack mlx output back into dense form for comparison
    mlx_packed = np.zeros_like(torch_packed)
    packed_coords = np.asarray(packed.coords)
    packed_feats = np.asarray(packed.feats)
    for n in range(packed_coords.shape[0]):
        _, i, j, k = packed_coords[n]
        mlx_packed[0, :, i, j, k] = packed_feats[n]

    diff = np.abs(torch_packed - mlx_packed).max()
    print(f"S2C vs dense pack (axis0-fastest) fp32 max-abs diff: {diff:.2e}")
    assert diff < 1e-6, f"S2C subpixel ordering disagrees: {diff}"


if __name__ == "__main__":
    test_submconv_matches_dense_conv3d()
    test_s2c_matches_dense_spatial_to_channel()
    print("OK: submanifold conv + S2C match their dense references")

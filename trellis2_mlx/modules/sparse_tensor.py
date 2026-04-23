"""Minimal SparseTensor for MLX.

Represents a batch of active-voxel feature tables:
- feats: (F, C)          — per-active-voxel features
- coords: (F, 4) int32   — [batch_idx, x, y, z]
- spatial_shape: tuple   — (R, R, R) reference grid (for upsample/downsample bookkeeping)

For B=1 (inference default), this is just a (F, C) feature table with xyz coords.
No submanifold-conv logic here — see sparse.py for that. This module is purely a data holder.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import mlx.core as mx


@dataclass
class SparseTensor:
    feats: mx.array          # (F, C)
    coords: mx.array         # (F, 4) int32: [batch, x, y, z]
    spatial_shape: Tuple[int, int, int]
    _cache: dict = None

    def __post_init__(self):
        if self._cache is None:
            object.__setattr__(self, "_cache", {})

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.feats.shape

    @property
    def dtype(self):
        return self.feats.dtype

    def replace(self, feats: mx.array) -> "SparseTensor":
        st = SparseTensor(feats=feats, coords=self.coords, spatial_shape=self.spatial_shape)
        st._cache = self._cache
        return st

    def astype(self, dtype) -> "SparseTensor":
        return self.replace(self.feats.astype(dtype))

    def __add__(self, other):
        if isinstance(other, SparseTensor):
            return self.replace(self.feats + other.feats)
        return self.replace(self.feats + other)

    def __mul__(self, other):
        if isinstance(other, SparseTensor):
            return self.replace(self.feats * other.feats)
        return self.replace(self.feats * other)

    __radd__ = __add__
    __rmul__ = __mul__

    def batch_size(self) -> int:
        # Assumes batch_idx in coords[:, 0] is in [0, B)
        if self.coords.shape[0] == 0:
            return 0
        return int(self.coords[:, 0].max().item()) + 1

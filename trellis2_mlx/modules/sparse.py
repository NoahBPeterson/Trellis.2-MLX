"""Sparse voxel ops for MLX: submanifold Conv3d, Linear, LayerNorm, S2C/C2S.

Ports the CUDA-free semantics of the upstream sparse module for inference. The
submanifold convolution is the only non-trivial kernel — we build a (F, 27)
neighbor index table once per unique coord set (on CPU in NumPy, via dict
hashing), then do the K^3 gather-matmul on MLX.

Coord convention: `coords` is (F, 4) int32 = `[batch, x, y, z]`.
Batch dim is preserved so multi-batch packings work; inference uses B=1.
"""
from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .norm import LayerNorm32
from .sparse_tensor import SparseTensor


# ------------------------------------------------------------------ helpers ----

def _coord_key(coords_np: np.ndarray) -> str:
    """Stable cache key for a coord set."""
    return f"F={coords_np.shape[0]}|sha={hash(coords_np.tobytes())}"


def _coords_np(st_or_array, cache: dict | None = None) -> np.ndarray:
    """Return a cached int64 numpy view of `coords`. Each `np.asarray` on an
    MLX array forces a GPU→CPU sync; caching avoids doing that every block.
    """
    if cache is not None:
        hit = cache.get("coords_np_int64")
        if hit is not None:
            return hit
    # If the caller already has a numpy copy (e.g. from S2C/C2S construction)
    # we prefer it to avoid np.asarray(mx_array) forcing a GPU sync.
    if isinstance(st_or_array, np.ndarray):
        arr = st_or_array.astype(np.int64, copy=False)
    else:
        arr = np.asarray(st_or_array).astype(np.int64, copy=False)
    if cache is not None:
        cache["coords_np_int64"] = arr
    return arr


def build_neighbor_map(
    coords, dilation: int = 1, kernel: int = 3, cache: dict | None = None
) -> mx.array:
    """For each active voxel, look up the row index of its 27 neighbors.

    Vectorized: packs `(batch, axis0, axis1, axis2)` into a single int64 code,
    sorts once, then does 27 np.searchsorted queries — O(F log F + 27 F log F)
    instead of O(27 F) Python dict lookups. A ~50× speedup at F ≈ 1M voxels.

    Returns an int32 array of shape (F, kernel**3). Missing neighbors are set
    to F (a sentinel) so downstream `mx.take` hits a zero sentinel row.
    """
    assert kernel == 3, "only 3x3x3 supported"
    coords_np = _coords_np(coords, cache)
    F = coords_np.shape[0]
    if F == 0:
        return mx.array(np.zeros((0, 27), dtype=np.int32))

    # Hash each coord into a single int64. Use a grid stride large enough to
    # absorb negative/oversized neighbor probes (+/- dilation on each axis).
    max_coord = int(coords_np[:, 1:].max()) + 2 * dilation + 2
    S = max_coord
    off1 = S
    off2 = S * S
    off3 = S * S * S  # batch stride
    code = (coords_np[:, 0] * off3
            + coords_np[:, 1] * off2
            + coords_np[:, 2] * off1
            + coords_np[:, 3])  # (F,) int64
    sort_idx = np.argsort(code, kind="stable")
    sorted_code = code[sort_idx]
    sort_idx_int32 = sort_idx.astype(np.int32)

    # Upstream weight is a torch Conv3d weight (Co, Ci, Kd, Kh, Kw) permuted to
    # (Co, Kd, Kh, Kw, Ci). Torch convention: kernel axis 0 ↔ coord axis 0,
    # axis 2 ↔ coord axis 2. After reshape (Co, K^3, Ci), axis 2 is innermost
    # in the flat K^3 index. Offset enumeration puts dz (axis 2) innermost.
    offs = np.array(
        [[dx, dy, dz] for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)],
        dtype=np.int64,
    )  # (27, 3)
    d = dilation
    deltas = offs[:, 0] * d * off2 + offs[:, 1] * d * off1 + offs[:, 2] * d  # (27,)
    # One broadcasted searchsorted over all 27 offsets at once — a single
    # numpy call instead of 27, saving function-call overhead at F ≈ 1M.
    targets = code[None, :] + deltas[:, None]  # (27, F)
    pos = np.searchsorted(sorted_code, targets)  # (27, F)
    clipped = np.minimum(pos, F - 1)
    hit = sorted_code[clipped] == targets
    nmap_T = np.where(hit, sort_idx_int32[clipped], F).astype(np.int32)  # (27, F)
    return mx.array(np.ascontiguousarray(nmap_T.T))  # (F, 27)


# ------------------------------------------------------------------- layers ----

class SparseLinear(nn.Linear):
    """nn.Linear that consumes/produces a SparseTensor (weights at top level)."""

    def __call__(self, x):
        if isinstance(x, SparseTensor):
            return x.replace(super().__call__(x.feats))
        return super().__call__(x)


class SparseLayerNorm32(LayerNorm32):
    """LayerNorm32 that accepts SparseTensor (weights at top level)."""

    def __call__(self, x):
        if isinstance(x, SparseTensor):
            return x.replace(super().__call__(x.feats))
        return super().__call__(x)


class SparseConv3d(nn.Module):
    """Submanifold 3D conv (stride=1, padding=none).

    Weight layout is upstream's pre-permuted (Co, Kd, Kh, Kw, Ci). The forward
    gathers 27 neighbor feature rows per voxel (with a zero sentinel for empty
    neighbors), multiplies by the 27 per-offset kernel slices, and accumulates.

    Neighbor maps are cached on the input SparseTensor under the key
    `submconv3d_d{dilation}` so repeated calls at the same stage are free.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        assert kernel_size == 3, "only kernel_size=3 supported"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        K = kernel_size
        # (Co, K, K, K, Ci) matches upstream pre-permuted layout
        self.weight = mx.zeros((out_channels, K, K, K, in_channels))
        if bias:
            self.bias = mx.zeros((out_channels,))
        else:
            self.bias = None

    def __call__(self, x: SparseTensor) -> SparseTensor:
        K = self.kernel_size
        Co = self.out_channels
        feats = x.feats  # (F, Ci)
        F, Ci = feats.shape

        # Get or build neighbor map; cache on the SparseTensor
        cache_key = f"submconv3d_d{self.dilation}_k{K}"
        nmap = x._cache.get(cache_key)
        if nmap is None:
            nmap = build_neighbor_map(x.coords, dilation=self.dilation, kernel=K, cache=x._cache)
            x._cache[cache_key] = nmap

        # Append zero sentinel row so nmap==F gathers zero.
        zero_row = mx.zeros((1, Ci), dtype=feats.dtype)
        feats_padded = mx.concatenate([feats, zero_row], axis=0)  # (F+1, Ci)

        # Per-tap loop: for each of the 27 kernel offsets, gather (F, Ci) and
        # matmul against W[k] → (F, Co); accumulate. Avoids materializing the
        # (F, 27, Ci) intermediate — for F≈1.6M, Ci=64 that's ~5 GB in fp16,
        # large enough to thrash MLX's allocator and stall the GPU.
        w_flat = self.weight.reshape(Co, K * K * K, Ci)  # (Co, K^3, Ci)
        out = None
        for k in range(K * K * K):
            sliced = feats_padded[nmap[:, k]]  # (F, Ci)
            wk = w_flat[:, k, :]  # (Co, Ci)
            partial = sliced @ wk.T  # (F, Co)
            out = partial if out is None else out + partial
        if self.bias is not None:
            out = out + self.bias
        return x.replace(out)


# ---------------------------------------------------------- spatial rearrange --

def _compute_s2c_indices(coords_np: np.ndarray, factor: int = 2):
    """Return (new_coords, idx, subidx, max_shape) for spatial-to-channel.

    idx[i] : the parent-voxel row that child i maps into
    subidx[i] : which of factor**3 slots within that parent
    """
    DIM = coords_np.shape[-1] - 1  # 3 for 3D
    b = coords_np[:, 0]
    xyz = coords_np[:, 1:]
    parent_xyz = xyz // factor
    subidx = (xyz % factor)
    sub_flat = subidx[:, 0] * 1 + subidx[:, 1] * factor + subidx[:, 2] * factor**2
    # stable unique on (b, parent_xyz) via a packed code
    # coarse bound for code: use max coord + 1
    maxc = int(parent_xyz.max()) + 1
    code = b.astype(np.int64) * (maxc**3) + parent_xyz[:, 0] * (maxc**2) + parent_xyz[:, 1] * maxc + parent_xyz[:, 2]
    _, inverse = np.unique(code, return_inverse=True)
    inverse = inverse.astype(np.int32)
    # Build new_coords from the first occurrence of each unique code
    order = np.argsort(code, kind="stable")
    seen = np.zeros(inverse.max() + 1, dtype=bool)
    new_coords = np.empty((inverse.max() + 1, DIM + 1), dtype=np.int32)
    for i in order:
        u = inverse[i]
        if not seen[u]:
            new_coords[u, 0] = b[i]
            new_coords[u, 1:] = parent_xyz[i]
            seen[u] = True
    return new_coords, inverse, sub_flat.astype(np.int32)


class SparseSpatial2Channel(nn.Module):
    """Downsample by `factor` by packing 2x2x2 (or factor^3) into channels.

    Channels go from Ci -> factor**3 * Ci. Empty children contribute zero.
    Slot ordering: `(x%f) + (y%f)*f + (z%f)*f^2` — matches upstream.
    """

    def __init__(self, factor: int = 2):
        super().__init__()
        self.factor = factor

    def __call__(self, x: SparseTensor) -> SparseTensor:
        factor = self.factor
        DIM3 = factor**3

        cache_key = f"spatial2channel_{factor}"
        cached = x._cache.get(cache_key)
        if cached is None:
            coords_np = np.asarray(x.coords).astype(np.int32)
            new_coords_np, idx_np, subidx_np = _compute_s2c_indices(coords_np, factor)
            new_coords = mx.array(new_coords_np)
            idx = mx.array(idx_np)
            subidx = mx.array(subidx_np)
            x._cache[cache_key] = (new_coords, idx, subidx)
        else:
            new_coords, idx, subidx = cached

        F_in, Ci = x.feats.shape
        F_out = new_coords.shape[0]

        # Scatter child feats into (F_out * DIM3, Ci) slots, then reshape to (F_out, DIM3 * Ci)
        flat_slot = idx * DIM3 + subidx  # (F_in,)
        packed = mx.zeros((F_out * DIM3, Ci), dtype=x.feats.dtype)
        # mx doesn't have scatter in nn; use at-like construction by building
        # a target array manually. For inference with F_in small, a Python loop
        # is too slow — use mx.at (if available) or a numpy-assisted path.
        # Cleanest for MLX 0.18: build with numpy then back to MLX.
        packed_np = np.zeros((F_out * DIM3, Ci), dtype=_mx_to_np_dtype(x.feats.dtype))
        packed_np[np.asarray(flat_slot)] = np.asarray(x.feats)
        packed = mx.array(packed_np)
        packed = packed.reshape(F_out, DIM3 * Ci)

        spatial_new = tuple((s + factor - 1) // factor for s in x.spatial_shape)
        out = SparseTensor(feats=packed, coords=new_coords, spatial_shape=spatial_new)
        out._cache[f"channel2spatial_{factor}"] = (x.coords, idx, subidx)
        return out


class SparseChannel2Spatial(nn.Module):
    """Upsample by `factor` — inverse of SparseSpatial2Channel.

    When paired with S2C (cache populated), uses the cached layout. Otherwise
    requires a `subdivision` SparseTensor whose (F, factor^3) feats mark which
    children to materialize per parent.

    Output feature dim = Ci // factor**3.
    """

    def __init__(self, factor: int = 2):
        super().__init__()
        self.factor = factor

    def __call__(
        self, x: SparseTensor, subdivision: Optional[SparseTensor] = None
    ) -> SparseTensor:
        factor = self.factor
        DIM3 = factor**3

        cache_key = f"channel2spatial_{factor}"
        cached = x._cache.get(cache_key)
        if cached is not None:
            new_coords, idx, subidx = cached
        else:
            if subdivision is None:
                raise ValueError("C2S requires either a paired S2C cache or a subdivision mask")
            # Vectorized: a single np.nonzero on the (F, DIM3) mask returns
            # parents + subidx in sorted (parent, subidx) order for all children
            # at once — replaces a Python loop over F parents.
            sub = np.asarray(subdivision.feats).astype(bool)
            parent_repeat, child_subidx = np.nonzero(sub)  # both (N_child,)
            parent_repeat = parent_repeat.astype(np.int32)
            child_subidx = child_subidx.astype(np.int32)

            parent_coords = _coords_np(x.coords, x._cache).astype(np.int32, copy=False)
            repeated = parent_coords[parent_repeat]  # (N_child, 4)
            new_coords_np = np.empty_like(repeated)
            new_coords_np[:, 0] = repeated[:, 0]
            new_coords_np[:, 1] = repeated[:, 1] * factor + (child_subidx % factor)
            new_coords_np[:, 2] = repeated[:, 2] * factor + ((child_subidx // factor) % factor)
            new_coords_np[:, 3] = repeated[:, 3] * factor + (child_subidx // (factor * factor))
            new_coords = mx.array(new_coords_np)
            idx = mx.array(parent_repeat)
            subidx = mx.array(child_subidx)
            _coords_np_int64_precomputed = new_coords_np.astype(np.int64, copy=False)

        F_in, Ci_packed = x.feats.shape
        Ci = Ci_packed // DIM3
        # Reshape to (F_in * DIM3, Ci) — slot layout matches S2C.
        packed = x.feats.reshape(F_in * DIM3, Ci)
        flat_slot = idx * DIM3 + subidx
        new_feats = packed[flat_slot]

        spatial_new = tuple(s * factor for s in x.spatial_shape)
        out = SparseTensor(feats=new_feats, coords=new_coords, spatial_shape=spatial_new)
        # Propagate the numpy int64 coord cache so the next stage's first conv
        # doesn't need a GPU sync to rebuild its neighbor map from scratch.
        if cached is None:
            out._cache["coords_np_int64"] = _coords_np_int64_precomputed
        return out


# ---------------------------------------------------------------------- util ---

def _mx_to_np_dtype(dt):
    import numpy as _np
    m = {
        mx.float32: _np.float32,
        mx.float16: _np.float16,
        mx.bfloat16: _np.float32,  # numpy has no bf16 — upcast for scatter
        mx.int32: _np.int32,
    }
    return m.get(dt, _np.float32)

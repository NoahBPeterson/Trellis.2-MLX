"""Flexible dual-grid mesh extraction (CPU, pure-Python / NumPy).

Ports the inference branch of upstream/o-voxel/o_voxel/convert/flexible_dual_grid.py:flexible_dual_grid_to_mesh.
The upstream relies on a CUDA hashmap for coord→index lookup — we replace that with a
Python dict; the rest is straight tensor arithmetic.

Inputs (NumPy / MLX arrays):
- coords          : (N, 3) int — voxel integer coords at the finest resolution
- dual_vertices   : (N, 3) float in ~[-margin, 1+margin] (already post-sigmoid)
- intersected     : (N, 3) bool — whether each of the 3 dual-grid edges of a voxel
                     is crossed by the surface (x, y, z axis)
- split_weight    : (N, 1) float or None — per-voxel quad-split preference
- aabb            : ((3,), (3,)) — min_xyz, max_xyz of the grid in world units
- grid_size       : (3,) int — grid resolution

Output:
- vertices : (N, 3) float world-space
- faces    : (T, 3) int
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# For each axis k in {0=x, 1=y, 2=z}, the 4 neighbor-voxel offsets that share
# that edge. Matches upstream's edge_neighbor_voxel_offset.
_EDGE_OFFSETS = np.array([
    [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],     # x-axis
    [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],     # y-axis
    [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],     # z-axis
], dtype=np.int32)  # (3, 4, 3)

# Two quad-triangulation splits: diagonal 0-2 and diagonal 1-3.
_QUAD_SPLIT_1 = np.array([0, 1, 2, 0, 2, 3], dtype=np.int64)
_QUAD_SPLIT_2 = np.array([0, 1, 3, 3, 1, 2], dtype=np.int64)


def flexible_dual_grid_to_mesh(
    coords: np.ndarray,
    dual_vertices: np.ndarray,
    intersected: np.ndarray,
    split_weight: Optional[np.ndarray],
    aabb: Tuple[np.ndarray, np.ndarray],
    grid_size: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (vertices (N, 3), faces (T, 3))."""
    coords = np.asarray(coords, dtype=np.int32)
    dual_vertices = np.asarray(dual_vertices, dtype=np.float32)
    intersected = np.asarray(intersected, dtype=bool)
    if split_weight is not None:
        split_weight = np.asarray(split_weight, dtype=np.float32).reshape(-1)
    aabb_min, aabb_max = np.asarray(aabb[0], dtype=np.float32), np.asarray(aabb[1], dtype=np.float32)
    grid_size = np.asarray(grid_size, dtype=np.int32).reshape(3)

    voxel_size = (aabb_max - aabb_min) / grid_size.astype(np.float32)
    N = coords.shape[0]

    # Hashmap: coord tuple -> row index
    coord_to_idx = {tuple(c.tolist()): i for i, c in enumerate(coords)}

    # For each voxel, compute the 3 axes × 4 neighbors offsets
    neighbor_coords = coords[:, None, None, :] + _EDGE_OFFSETS[None]  # (N, 3, 4, 3)

    # Keep only edges that are flagged as intersected
    ax_mask = intersected  # (N, 3)
    # Produce list of (N_i, 4, 3) selected edges
    i_idx, ax_idx = np.where(ax_mask)
    if i_idx.size == 0:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.int32)
    selected = neighbor_coords[i_idx, ax_idx]  # (M, 4, 3)
    M = selected.shape[0]

    # Lookup all 4 neighbor voxel indices
    flat = selected.reshape(M * 4, 3)
    indices = np.full(flat.shape[0], -1, dtype=np.int64)
    for k, c in enumerate(flat):
        idx = coord_to_idx.get(tuple(c.tolist()))
        if idx is not None:
            indices[k] = idx
    indices = indices.reshape(M, 4)
    valid = (indices >= 0).all(axis=1)
    quads = indices[valid]  # (L, 4)
    L = quads.shape[0]
    if L == 0:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.int32)

    # World-space vertices: (coords + dual_vertices) * voxel_size + aabb_min
    mesh_vertices = (coords.astype(np.float32) + dual_vertices) * voxel_size[None] + aabb_min[None]

    # Choose quad split
    if split_weight is None:
        # Pick the split whose two resulting triangles have more-aligned normals.
        t0 = quads[:, _QUAD_SPLIT_1]  # (L, 6)
        t1 = quads[:, _QUAD_SPLIT_2]
        align0 = _normal_alignment(mesh_vertices, t0)
        align1 = _normal_alignment(mesh_vertices, t1)
        choose_1 = align0 > align1
        tri = np.where(choose_1[:, None], t0, t1)
    else:
        sw = split_weight[quads]  # (L, 4)
        s02 = sw[:, 0] * sw[:, 2]
        s13 = sw[:, 1] * sw[:, 3]
        tri = np.where((s02 > s13)[:, None], quads[:, _QUAD_SPLIT_1], quads[:, _QUAD_SPLIT_2])

    faces = tri.reshape(-1, 3).astype(np.int32)
    return mesh_vertices, faces


def _normal_alignment(verts: np.ndarray, tri_pair: np.ndarray) -> np.ndarray:
    """For each row of tri_pair (L, 6) = two triangles (a,b,c, d,e,f),
    return |n1 · n2| where n1 = (b-a)×(c-a), n2 = (c-b)×(f-b)... actually
    matches upstream's pair ordering: for split 1 we compute normal of
    [0,1,2] with (1-0, 2-0) then of [0,2,3] with (2-1, 3-1).
    """
    a = verts[tri_pair[:, 0]]
    b = verts[tri_pair[:, 1]]
    c = verts[tri_pair[:, 2]]
    d = verts[tri_pair[:, 3]]
    e = verts[tri_pair[:, 4]]
    f = verts[tri_pair[:, 5]]
    n1 = np.cross(b - a, c - a)
    n2 = np.cross(c - b, f - b)  # matches upstream's (triangles_0[:, 2] - triangles_0[:, 1], triangles_0[:, 3] - triangles_0[:, 1])
    return np.abs((n1 * n2).sum(axis=-1))

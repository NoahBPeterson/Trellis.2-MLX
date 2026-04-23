"""Smoke tests for the dual-grid → mesh extraction.

Construct a tiny voxel patch, mark one edge intersected, and verify we get a
quad → 2 triangles out. Compares face count and vertex placement against an
independent numpy calculation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trellis2_mlx.postprocess.dual_grid import flexible_dual_grid_to_mesh


def test_empty_intersection():
    coords = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int32)
    dv = np.zeros((2, 3), np.float32)
    inter = np.zeros((2, 3), bool)
    v, f = flexible_dual_grid_to_mesh(
        coords, dv, inter, None, ([0, 0, 0], [1, 1, 1]), [2, 2, 2]
    )
    assert v.shape == (0, 3) and f.shape == (0, 3)


def test_single_quad_from_x_edge():
    """A 2x2x1 block of 4 voxels share an x-axis edge at (0, 0, 0) corner.
    The 4 voxels sharing the x-edge of voxel (0,0,0) are:
    (0,0,0), (0,0,1), (0,1,1), (0,1,0) per _EDGE_OFFSETS[0].
    """
    # The 4 active voxels
    coords = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 1],
        [0, 1, 0],
    ], dtype=np.int32)
    dv = np.array([
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
    ], dtype=np.float32)
    inter = np.zeros((4, 3), bool)
    # Mark the x-edge of voxel 0 as intersected → produces one quad across all 4 voxels
    inter[0, 0] = True
    v, f = flexible_dual_grid_to_mesh(
        coords, dv, inter, None, (np.zeros(3), np.ones(3) * 4), [4, 4, 4]
    )
    assert v.shape == (4, 3), f"expected 4 vertices, got {v.shape}"
    assert f.shape == (2, 3), f"expected 2 faces, got {f.shape}"
    # All 4 vertex indices must be present across the two triangles
    assert set(f.reshape(-1).tolist()) == {0, 1, 2, 3}


def test_returns_world_space_vertices():
    coords = np.array([[1, 2, 3]], dtype=np.int32)
    dv = np.array([[0.25, 0.75, 0.5]], dtype=np.float32)
    inter = np.zeros((1, 3), bool)
    v, _ = flexible_dual_grid_to_mesh(
        coords, dv, inter, None, (np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0])), [4, 4, 4]
    )
    # No intersections → no mesh, but verify the vertex-placement formula on
    # a single intersected edge case.
    coords2 = np.array([
        [1, 2, 3],
        [1, 2, 4],
        [1, 3, 4],
        [1, 3, 3],
    ], dtype=np.int32)
    dv2 = np.tile(np.array([[0.25, 0.75, 0.5]], dtype=np.float32), (4, 1))
    inter2 = np.zeros((4, 3), bool)
    inter2[0, 0] = True
    voxel_size = (1.0 - (-1.0)) / 4  # 0.5
    aabb_min = np.array([-1.0, -1.0, -1.0])
    v2, f2 = flexible_dual_grid_to_mesh(
        coords2, dv2, inter2, None, (aabb_min, np.array([1.0, 1.0, 1.0])), [4, 4, 4]
    )
    # Vertex 0: (1.25, 2.75, 3.5) * 0.5 + (-1) = (-0.375, 0.375, 0.75)
    expected_v0 = np.array([1.25, 2.75, 3.5]) * 0.5 + aabb_min
    assert np.allclose(v2[0], expected_v0, atol=1e-6), f"vertex 0 mismatch: {v2[0]} vs {expected_v0}"


if __name__ == "__main__":
    test_empty_intersection()
    test_single_quad_from_x_edge()
    test_returns_world_space_vertices()
    print("OK: dual-grid mesh extraction")

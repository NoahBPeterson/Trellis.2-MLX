"""Cone-based chart segmentation.

Re-implementation of the segmentation half of CuMesh's `uv_unwrap`. Greedy
region-growing with a normal-cone test: a face joins the current chart iff its
face normal lies within `threshold_rad` of the chart's running average normal.

Chart segmentation is the part of UV unwrap that scales badly for xatlas on
TRELLIS dual-grid output (1000s of small geodesic charts → quadratic blowup).
This algorithm is what upstream `cumesh.CuMesh.uv_unwrap(compute_charts_kwargs=
{"threshold_cone_half_angle_rad": ..., "refine_iterations": ..., ...})` actually
runs (on CUDA upstream; pure-numpy here). Designed for dense voxel-derived
meshes — produces a small number of large charts very quickly.
"""
from __future__ import annotations

import time
from typing import Tuple

import numpy as np


def _face_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Per-face unit normals. Zero-area faces get a placeholder (1, 0, 0)."""
    e1 = verts[faces[:, 1]] - verts[faces[:, 0]]
    e2 = verts[faces[:, 2]] - verts[faces[:, 0]]
    n = np.cross(e1, e2)
    norms = np.linalg.norm(n, axis=1, keepdims=True)
    safe = np.maximum(norms, 1e-12)
    n = n / safe
    n[norms.flatten() < 1e-10] = np.array([1.0, 0.0, 0.0])
    return n.astype(np.float32)


def _build_face_adjacency(faces: np.ndarray, n_verts: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return CSR-style face-face adjacency: `(adj_offsets, adj_neighbors)` such
    that face `f`'s neighbors live in `adj_neighbors[adj_offsets[f]:adj_offsets[f+1]]`.
    Two faces are adjacent iff they share an edge.
    """
    n_faces = len(faces)
    # All directed half-edges as (sorted_v0, sorted_v1, face_idx)
    half = np.concatenate([
        np.stack([faces[:, 0], faces[:, 1]], axis=1),
        np.stack([faces[:, 1], faces[:, 2]], axis=1),
        np.stack([faces[:, 2], faces[:, 0]], axis=1),
    ], axis=0)
    half = np.sort(half, axis=1)
    face_per = np.repeat(np.arange(n_faces, dtype=np.int64), 3)

    # Sort by edge key — adjacent faces share an edge → keys equal
    keys = half[:, 0].astype(np.int64) * (n_verts + 1) + half[:, 1].astype(np.int64)
    order = np.argsort(keys, kind="stable")
    keys_s = keys[order]
    faces_s = face_per[order]

    # For each block of identical keys (same edge), every pair of faces is adjacent.
    # Most edges are shared by exactly 2 faces; non-manifold edges by more. We add
    # all unordered pairs from each block.
    diff = np.diff(keys_s, prepend=keys_s[0] - 1)
    block_starts = np.where(diff != 0)[0]
    block_ends = np.append(block_starts[1:], len(keys_s))

    a_list = []
    b_list = []
    for s, e in zip(block_starts, block_ends):
        if e - s < 2:
            continue
        block = faces_s[s:e]
        # All directed pairs (i, j) where i < j
        for i in range(len(block)):
            for j in range(i + 1, len(block)):
                a_list.append(block[i]); b_list.append(block[j])
                a_list.append(block[j]); b_list.append(block[i])

    if not a_list:
        return np.zeros(n_faces + 1, dtype=np.int64), np.empty(0, dtype=np.int64)

    a = np.array(a_list, dtype=np.int64)
    b = np.array(b_list, dtype=np.int64)
    # Build CSR by sort-and-count
    order2 = np.argsort(a, kind="stable")
    a_sorted = a[order2]
    b_sorted = b[order2]
    counts = np.bincount(a_sorted, minlength=n_faces)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return offsets, b_sorted


def _merge_small_charts(
    chart_id: np.ndarray,
    face_normals: np.ndarray,
    adj_offsets: np.ndarray,
    adj_neighbors: np.ndarray,
    min_chart_size: int = 8,
    verbose: bool = True,
) -> np.ndarray:
    """Merge tiny charts into their largest adjacent neighbor.

    After greedy region-grow, TRELLIS dual-grid meshes leave many singleton/tiny
    charts at staircase boundaries. This step folds them into bigger charts so
    the downstream LSCM + pack steps don't drown in 100k tiny single-face charts.
    """
    n_faces = len(chart_id)
    n_charts = chart_id.max() + 1
    chart_sizes = np.bincount(chart_id, minlength=n_charts)
    small_charts = set(np.where(chart_sizes < min_chart_size)[0].tolist())
    if not small_charts:
        return chart_id
    if verbose:
        print(f"        merge_small_charts: {len(small_charts)} charts < {min_chart_size} faces", flush=True)

    # For each face in a small chart, find its most-common neighbor chart and reassign.
    # Repeat until no changes or all small charts are absorbed.
    chart_id = chart_id.copy()
    for sweep in range(8):  # bounded; usually converges in 2-3 sweeps
        changed = 0
        for f in range(n_faces):
            if chart_sizes[chart_id[f]] >= min_chart_size:
                continue
            # Tally neighbors by chart
            nb_charts = {}
            for k in range(adj_offsets[f], adj_offsets[f + 1]):
                g = adj_neighbors[k]
                cg = int(chart_id[g])
                if cg == chart_id[f]:
                    continue
                nb_charts[cg] = nb_charts.get(cg, 0) + 1
            if not nb_charts:
                continue
            # Reassign to the most-common neighbor chart that's bigger than ours
            best = max(nb_charts.items(), key=lambda kv: (chart_sizes[kv[0]] >= min_chart_size, kv[1]))[0]
            old = chart_id[f]
            chart_id[f] = best
            chart_sizes[old] -= 1
            chart_sizes[best] += 1
            changed += 1
        if verbose and changed:
            print(f"        merge sweep {sweep}: reassigned {changed} faces", flush=True)
        if changed == 0:
            break

    # Compact chart IDs to remove now-empty charts
    used = chart_sizes > 0
    new_id = -np.ones(n_charts, dtype=np.int32)
    new_id[used] = np.arange(used.sum())
    chart_id = new_id[chart_id]
    if verbose:
        print(f"        after merge: {used.sum()} charts (was {n_charts})", flush=True)
    return chart_id


def cone_cluster(
    verts: np.ndarray,
    faces: np.ndarray,
    threshold_rad: float = 0.6,
    min_chart_size: int = 8,
    max_chart_size: int = 4000,
    verbose: bool = True,
) -> np.ndarray:
    """Segment a triangle mesh into charts via cone-clustering region-grow.

    `threshold_rad` is the half-angle cone tolerance (default 0.6 rad ≈ 34°).
    Smaller → more, smaller, flatter charts. Larger → fewer, bigger charts that
    bend more.

    Returns `chart_id[F]` — per-face chart index in [0, n_charts).
    """
    n_faces = len(faces)
    n_verts = len(verts)
    if verbose:
        print(f"      cone_cluster: {n_faces} faces, threshold={threshold_rad:.2f} rad ({np.degrees(threshold_rad):.0f}°)", flush=True)

    t0 = time.time()
    n = _face_normals(verts, faces)
    if verbose:
        print(f"        face normals: {time.time()-t0:.2f}s", flush=True)

    t0 = time.time()
    adj_offsets, adj_neighbors = _build_face_adjacency(faces, n_verts)
    if verbose:
        print(f"        face adjacency ({len(adj_neighbors)} edges): {time.time()-t0:.2f}s", flush=True)

    chart_id = -np.ones(n_faces, dtype=np.int32)
    cos_thresh = np.cos(threshold_rad)

    t0 = time.time()
    # Process faces in arbitrary order. For each unassigned face, BFS-grow a chart
    # from it, accepting neighbors whose normal stays within the cone of the chart's
    # running average normal.
    next_chart = 0
    for seed in range(n_faces):
        if chart_id[seed] != -1:
            continue
        chart_id[seed] = next_chart
        # Running average normal (unit-vector after normalization at end)
        cone_n = n[seed].copy()
        cone_count = 1
        # BFS queue
        queue = [seed]
        while queue:
            if cone_count >= max_chart_size:
                break
            f = queue.pop()
            for k in range(adj_offsets[f], adj_offsets[f + 1]):
                g = adj_neighbors[k]
                if chart_id[g] != -1:
                    continue
                # Test: is g's normal within the chart's cone?
                cone_unit = cone_n / max(np.linalg.norm(cone_n), 1e-12)
                if float(n[g] @ cone_unit) >= cos_thresh:
                    chart_id[g] = next_chart
                    cone_n = cone_n + n[g]
                    cone_count += 1
                    queue.append(g)
                    if cone_count >= max_chart_size:
                        break
        next_chart += 1

    if verbose:
        sizes = np.bincount(chart_id, minlength=next_chart)
        print(f"        region-grow: {next_chart} charts, sizes [{sizes.min()}, {sizes.max()}], "
              f"median {int(np.median(sizes))}, mean {sizes.mean():.0f} ({time.time()-t0:.2f}s)", flush=True)

    if min_chart_size > 1:
        t0 = time.time()
        chart_id = _merge_small_charts(chart_id, n, adj_offsets, adj_neighbors,
                                       min_chart_size=min_chart_size, verbose=verbose)
        if verbose:
            print(f"        merge step total: {time.time()-t0:.2f}s", flush=True)

    return chart_id

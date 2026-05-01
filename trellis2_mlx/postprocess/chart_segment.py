"""CuMesh-faithful chart segmentation for UV unwrap.

Direct numpy port of CuMesh's `compute_charts` (`/tmp/CuMesh/src/atlas.cu`,
referenced upstream as `cumesh.CuMesh.compute_charts`). Three-phase algorithm:

1. **Parallel edge-collapse merging** — start with each face as its own chart,
   merge adjacent charts with mutual-best-match constraint when the merge cost
   stays below `threshold_cone_half_angle_rad`. Cost combines:
       - new merged-cone half-angle
       - area_penalty_weight × new combined area
       - perimeter_area_ratio_weight × (new_perim² / new_area)
2. **Lloyd refinement** — for `refine_iterations` rounds, each face locally
   re-evaluates whether to stay in its chart or join one of its 3 neighbors.
   Score combines geometric similarity (face_normal · chart_axis) and a
   boundary-smoothness term (sum of edge lengths to candidate chart).
3. **Disconnected-component split** — refinement can fragment a chart into
   disconnected pieces; we union-find within each chart to assign new IDs.

The whole loop runs `global_iterations=3` times. The output `chart_id[F]` is
then fed to `xatlas` per-chart for actual UV parameterization (LSCM) and pack.

CuMesh runs all of this on CUDA via cub primitives; we run it in numpy. The
algorithm is identical, just slower (seconds rather than milliseconds, fine
for our pipeline).
"""
from __future__ import annotations

import time
from typing import Tuple

import numpy as np


def _face_normals_and_areas(verts: np.ndarray, faces: np.ndarray
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """Per-face unit normals and triangle areas. Zero-area faces get axis (1,0,0)."""
    e1 = verts[faces[:, 1]] - verts[faces[:, 0]]
    e2 = verts[faces[:, 2]] - verts[faces[:, 0]]
    cross = np.cross(e1, e2)
    norms = np.linalg.norm(cross, axis=1)
    areas = 0.5 * norms
    safe = np.maximum(norms, 1e-12)
    n = cross / safe[:, None]
    n[norms < 1e-10] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return n.astype(np.float32), areas.astype(np.float32)


def _manifold_face_adjacency(verts: np.ndarray, faces: np.ndarray
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """Manifold face-face adjacency. Two faces are adjacent iff they share
    exactly one edge AND that edge is shared by exactly 2 faces (manifold).

    Non-manifold edges (3+ faces) and boundary edges (1 face) are skipped.

    Returns:
      face_adj  (M, 2) int64 — pairs of face indices sharing a manifold edge
      edge_len  (M,)         — length of the shared edge
    """
    n_faces = len(faces)
    # Half-edges: (sorted_v0, sorted_v1, face_idx). The concatenate is column-major
    # (all faces' e01, then all faces' e12, then all faces' e20), so face_per uses np.tile
    # to match — np.repeat would be wrong here.
    half = np.concatenate([
        np.stack([faces[:, 0], faces[:, 1]], axis=1),
        np.stack([faces[:, 1], faces[:, 2]], axis=1),
        np.stack([faces[:, 2], faces[:, 0]], axis=1),
    ], axis=0)
    half = np.sort(half, axis=1)
    face_per = np.tile(np.arange(n_faces, dtype=np.int64), 3)

    # Sort by edge key
    n_verts = int(faces.max()) + 1
    keys = half[:, 0].astype(np.int64) * (n_verts + 1) + half[:, 1].astype(np.int64)
    order = np.argsort(keys, kind="stable")
    keys_s = keys[order]
    faces_s = face_per[order]
    edges_s = half[order]

    # Find blocks of identical keys (same edge); manifold = exactly 2 in block
    diff = np.diff(keys_s, prepend=keys_s[0] - 1)
    block_starts = np.where(diff != 0)[0]
    block_ends = np.append(block_starts[1:], len(keys_s))
    block_lens = block_ends - block_starts

    manifold_blocks = block_starts[block_lens == 2]
    f0 = faces_s[manifold_blocks]
    f1 = faces_s[manifold_blocks + 1]
    v0 = edges_s[manifold_blocks, 0]
    v1 = edges_s[manifold_blocks, 1]

    edge_len = np.linalg.norm(verts[v0] - verts[v1], axis=1).astype(np.float32)
    face_adj = np.stack([f0, f1], axis=1)
    return face_adj, edge_len


def _compute_chart_normal_cones(face_normals: np.ndarray, face_areas: np.ndarray,
                                chart_id: np.ndarray, n_charts: int
                                ) -> np.ndarray:
    """Per-chart (axis, half_angle). axis = normalized mean of face normals
    (CuMesh uses unweighted sum; we do the same). half_angle = max angular
    deviation of any face normal from the axis.

    Returns (n_charts, 4) — [axis_x, axis_y, axis_z, half_angle].
    """
    # Sum normals per chart via bincount (much faster than np.add.at on large arrays)
    axes = np.stack([
        np.bincount(chart_id, weights=face_normals[:, 0], minlength=n_charts),
        np.bincount(chart_id, weights=face_normals[:, 1], minlength=n_charts),
        np.bincount(chart_id, weights=face_normals[:, 2], minlength=n_charts),
    ], axis=1).astype(np.float32)
    norms = np.linalg.norm(axes, axis=1, keepdims=True)
    axes = axes / np.maximum(norms, 1e-12)

    # Max deviation per chart: angle between each face normal and its chart's axis.
    # Use np.maximum.reduceat after sorting by chart_id (vectorized; np.maximum.at is slow).
    cos_dev = (face_normals * axes[chart_id]).sum(axis=1)
    cos_dev = np.clip(cos_dev, -1.0, 1.0)
    dev = np.arccos(cos_dev).astype(np.float32)

    order = np.argsort(chart_id, kind="stable")
    sorted_chart = chart_id[order]
    sorted_dev = dev[order]
    # Find segment boundaries (first occurrence of each unique chart_id in sorted order)
    diff = np.diff(sorted_chart, prepend=sorted_chart[0] - 1)
    seg_starts = np.where(diff != 0)[0]
    seg_max = np.maximum.reduceat(sorted_dev, seg_starts)
    half_angle = np.zeros(n_charts, dtype=np.float32)
    half_angle[sorted_chart[seg_starts]] = seg_max
    return np.concatenate([axes, half_angle[:, None]], axis=1).astype(np.float32)


def _compute_chart_areas(face_areas: np.ndarray, chart_id: np.ndarray, n_charts: int) -> np.ndarray:
    return np.bincount(chart_id, weights=face_areas, minlength=n_charts).astype(np.float32)


def _compute_chart_adjacency(face_adj: np.ndarray, edge_len: np.ndarray,
                             chart_id: np.ndarray, n_charts: int
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build chart adjacency graph from face adjacency.

    Returns:
      ch_pairs       (E, 2) int64 — [c0, c1] with c0 < c1 for each chart-chart edge
      ch_lengths     (E,)         — total shared boundary length between c0 and c1
      chart_perim    (n_charts,)  — sum of all incident chart-chart edge lengths
      chart2edge     (2*E,)       — flat CSR data: for each chart, list of edge indices
      chart2edge_off (n_charts+1,)— CSR offsets
      Note: the same edge index appears twice in chart2edge (once per endpoint chart).
    """
    c0 = chart_id[face_adj[:, 0]]
    c1 = chart_id[face_adj[:, 1]]

    # Drop intra-chart edges (where both faces are in same chart)
    inter = c0 != c1
    c0 = c0[inter]; c1 = c1[inter]; lens = edge_len[inter]
    if len(c0) == 0:
        empty = np.empty(0, dtype=np.int64)
        return (np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.float32),
                np.zeros(n_charts, dtype=np.float32), empty,
                np.zeros(n_charts + 1, dtype=np.int64))

    lo = np.minimum(c0, c1).astype(np.int64)
    hi = np.maximum(c0, c1).astype(np.int64)
    keys = lo * (n_charts + 1) + hi

    # Aggregate duplicate (lo, hi) pairs by summing lengths
    order = np.argsort(keys, kind="stable")
    keys_s = keys[order]
    lens_s = lens[order]
    lo_s = lo[order]
    hi_s = hi[order]
    diff = np.diff(keys_s, prepend=keys_s[0] - 1)
    starts = np.where(diff != 0)[0]
    # cumsum-by-segment trick to sum lens per group
    cs = np.cumsum(lens_s)
    seg_ends = np.append(starts[1:], len(keys_s)) - 1
    seg_starts = starts
    ch_lengths = cs[seg_ends] - np.concatenate([[0.0], cs[seg_ends[:-1]]])
    ch_lengths = ch_lengths.astype(np.float32)
    ch_pairs = np.stack([lo_s[seg_starts], hi_s[seg_starts]], axis=1)

    # Chart perimeter and chart→edge CSR. Build CSR by stacking each edge's two
    # (chart, edge_id) endpoints and sorting by chart.
    n_edges = len(ch_pairs)
    chart_perim = (np.bincount(ch_pairs[:, 0], weights=ch_lengths, minlength=n_charts)
                   + np.bincount(ch_pairs[:, 1], weights=ch_lengths, minlength=n_charts)).astype(np.float32)

    endpoints_chart = np.concatenate([ch_pairs[:, 0], ch_pairs[:, 1]])
    endpoints_edge = np.tile(np.arange(n_edges, dtype=np.int64), 2)
    order = np.argsort(endpoints_chart, kind="stable")
    chart2edge = endpoints_edge[order]
    cnt = np.bincount(endpoints_chart, minlength=n_charts)
    chart2edge_off = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64)

    return ch_pairs, ch_lengths, chart_perim, chart2edge, chart2edge_off


def _merged_cone(cones: np.ndarray, c0: np.ndarray, c1: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized merge of two normal cones. Returns (new_axis (E,3), new_half_angle (E,)).

    Mirrors CuMesh's `compute_chart_adjacency_cost_kernel` math:
      cos_angle = axis0 · axis1
      axis_angle = acos(cos_angle)
      new_low  = min(-half0, axis_angle - half1)
      new_high = max( half0, axis_angle + half1)
      new_half = (new_high - new_low) / 2
      new_axis = rotate axis0 toward axis1 by (new_high + new_low)/2
    """
    a0 = cones[c0, :3]; h0 = cones[c0, 3]
    a1 = cones[c1, :3]; h1 = cones[c1, 3]
    cos_a = np.clip((a0 * a1).sum(axis=1), -1.0, 1.0)
    axis_a = np.arccos(cos_a)
    low = np.minimum(-h0, axis_a - h1)
    high = np.maximum(h0, axis_a + h1)
    new_half = (high - low) * 0.5

    # New axis: axis0 rotated by (high+low)/2 toward axis1 within the plane spanned by both.
    new_axis_angle = (high + low) * 0.5
    # Component of a1 perpendicular to a0
    perp = a1 - a0 * cos_a[:, None]
    perp_norm = np.linalg.norm(perp, axis=1, keepdims=True)
    perp_unit = perp / np.maximum(perp_norm, 1e-12)
    near_zero = (axis_a < 1e-3)[:, None]
    new_axis = np.where(near_zero,
                        a0,
                        a0 * np.cos(new_axis_angle)[:, None] + perp_unit * np.sin(new_axis_angle)[:, None])
    new_axis = new_axis / np.maximum(np.linalg.norm(new_axis, axis=1, keepdims=True), 1e-12)
    return new_axis.astype(np.float32), new_half.astype(np.float32)


def _edge_collapse_pass(chart_id: np.ndarray, n_charts: int,
                        face_adj: np.ndarray, edge_len: np.ndarray,
                        face_areas: np.ndarray, face_normals: np.ndarray,
                        threshold: float, area_penalty: float, perim_ratio: float,
                        verbose: bool = False
                        ) -> Tuple[np.ndarray, int, int]:
    """One pass of (compute connectivity → cost → propagate → collapse).

    Returns `(new_chart_id, new_n_charts, n_collapses)`. If `n_collapses == 0`,
    the segmentation has converged.
    """
    cones = _compute_chart_normal_cones(face_normals, face_areas, chart_id, n_charts)
    areas = _compute_chart_areas(face_areas, chart_id, n_charts)
    ch_pairs, ch_lengths, chart_perim, chart2edge, chart2edge_off = _compute_chart_adjacency(
        face_adj, edge_len, chart_id, n_charts
    )
    if len(ch_pairs) == 0:
        return chart_id, n_charts, 0

    # Cost = new_half_angle + area_penalty * (a0+a1) + perim_ratio * (new_perim² / new_area)
    _, new_half = _merged_cone(cones, ch_pairs[:, 0], ch_pairs[:, 1])
    new_area = areas[ch_pairs[:, 0]] + areas[ch_pairs[:, 1]]
    new_perim = chart_perim[ch_pairs[:, 0]] + chart_perim[ch_pairs[:, 1]] - 2 * ch_lengths
    cost = new_half + area_penalty * new_area + perim_ratio * (new_perim * new_perim / np.maximum(new_area, 1e-12))
    cost = cost.astype(np.float32)

    # Propagate: for each chart, find its cheapest incident edge (eid). Vectorized
    # via lexsort by (chart, cost, eid) → first occurrence per chart is the best.
    n_edges = len(ch_pairs)
    chart_per_entry = np.repeat(np.arange(n_charts, dtype=np.int64),
                                np.diff(chart2edge_off))
    if len(chart_per_entry) == 0:
        return chart_id, n_charts, 0
    per_entry_eid = chart2edge
    per_entry_cost = cost[per_entry_eid]
    order = np.lexsort((per_entry_eid, per_entry_cost, chart_per_entry))
    sorted_chart = chart_per_entry[order]
    sorted_eid = per_entry_eid[order]
    diff = np.diff(sorted_chart, prepend=sorted_chart[0] - 1)
    firsts = np.where(diff != 0)[0]
    chart_best_eid = -np.ones(n_charts, dtype=np.int64)
    chart_best_eid[sorted_chart[firsts]] = sorted_eid[firsts]

    # Collapse: only edges where cost ≤ threshold AND both endpoints' best is this edge
    # (mutual-best-match). Apply via union-find chart_map (c1 → c0).
    accept = (cost <= threshold) & (chart_best_eid[ch_pairs[:, 0]] == np.arange(n_edges)) & \
             (chart_best_eid[ch_pairs[:, 1]] == np.arange(n_edges))
    n_accepted = int(accept.sum())
    if n_accepted == 0:
        return chart_id, n_charts, 0

    chart_map = np.arange(n_charts, dtype=np.int64)
    accepted_pairs = ch_pairs[accept]
    # Each accepted edge folds c1 → c0 (c0 < c1)
    # In one parallel pass this is safe because mutual-best-match guarantees c0 and c1 each
    # appear in at most one accepted edge.
    chart_map[accepted_pairs[:, 1]] = accepted_pairs[:, 0]

    # Apply map (one level of indirection is enough since each chart was matched at most once)
    new_chart_id = chart_map[chart_id]

    # Compact chart IDs
    unique_ids, inverse = np.unique(new_chart_id, return_inverse=True)
    new_n_charts = len(unique_ids)
    return inverse.astype(np.int32), new_n_charts, n_accepted


def _build_face_edges(faces: np.ndarray, face_adj: np.ndarray, edge_len: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each face, list of (≤3) neighbor faces and the edge length to each.

    Returns:
      neighbor_face   (F, 3) int64, -1 for boundary/non-manifold edges
      neighbor_edge_l (F, 3) float32, 0 where neighbor is -1
      n_neighbors     (F,)   int (count of valid neighbors per face)
    """
    n_faces = len(faces)
    # Each manifold edge contributes f0↔f1 — emit two directed (face, neighbor, length)
    # tuples per edge, sort by face, then assign positions 0/1/2 per group.
    src = np.concatenate([face_adj[:, 0], face_adj[:, 1]])
    dst = np.concatenate([face_adj[:, 1], face_adj[:, 0]])
    lens = np.concatenate([edge_len, edge_len])

    order = np.argsort(src, kind="stable")
    src_s = src[order]; dst_s = dst[order]; lens_s = lens[order]

    # Position within each face's group: cumcount via cumsum of "first-of-group" mask
    diff = np.diff(src_s, prepend=src_s[0] - 1)
    is_first = (diff != 0).astype(np.int64)
    group_start_idx = np.cumsum(is_first) - 1   # group index per entry
    starts = np.where(is_first)[0]
    pos_in_group = np.arange(len(src_s)) - starts[group_start_idx]

    # Cap at 3 (drop any 4th+ neighbor — non-manifold defensively skipped earlier already)
    keep = pos_in_group < 3
    src_s = src_s[keep]; dst_s = dst_s[keep]; lens_s = lens_s[keep]; pos_in_group = pos_in_group[keep]

    neighbor_face = -np.ones((n_faces, 3), dtype=np.int64)
    neighbor_edge_l = np.zeros((n_faces, 3), dtype=np.float32)
    neighbor_face[src_s, pos_in_group] = dst_s
    neighbor_edge_l[src_s, pos_in_group] = lens_s
    n_neighbors = (neighbor_face >= 0).sum(axis=1)
    return neighbor_face, neighbor_edge_l, n_neighbors.astype(np.int64)


def _refine_pass(chart_id: np.ndarray, n_charts: int,
                 face_normals: np.ndarray, neighbor_face: np.ndarray,
                 neighbor_edge_l: np.ndarray, smooth_strength: float
                 ) -> np.ndarray:
    """One Lloyd pass: each face re-evaluates self vs neighbor charts.

    score(c) = (face_normal · chart_axis_c) + smooth_strength · sum_of_edge_lengths_to_c
    """
    face_areas_dummy = np.ones(len(face_normals), dtype=np.float32)
    cones = _compute_chart_normal_cones(face_normals, face_areas_dummy, chart_id, n_charts)
    chart_axes = cones[:, :3]

    n_faces = len(face_normals)
    nb_chart = chart_id[neighbor_face.clip(0)]  # -1 → 0; we mask below
    valid = neighbor_face >= 0
    # Candidates per face: (current_chart, nb_chart_0, nb_chart_1, nb_chart_2)
    cand_charts = np.stack([
        chart_id,
        np.where(valid[:, 0], nb_chart[:, 0], chart_id),
        np.where(valid[:, 1], nb_chart[:, 1], chart_id),
        np.where(valid[:, 2], nb_chart[:, 2], chart_id),
    ], axis=1).astype(np.int64)  # (F, 4)

    # Smooth score: for each (face, candidate), sum edge_lengths over neighbors that are in that candidate's chart
    smooth = np.zeros((n_faces, 4), dtype=np.float32)
    for k in range(4):
        match = (nb_chart == cand_charts[:, k:k+1]) & valid  # (F, 3)
        smooth[:, k] = (neighbor_edge_l * match).sum(axis=1)

    # Geo score: face_normal · candidate_chart_axis
    cand_axes = chart_axes[cand_charts]  # (F, 4, 3)
    geo = (face_normals[:, None, :] * cand_axes).sum(axis=-1)  # (F, 4)
    invalid = geo <= 0.0  # CuMesh skips these

    score = (geo + smooth_strength * smooth).astype(np.float32)
    score[invalid] = -np.inf
    # CuMesh's epsilon-dampening: prefer current chart on near-ties (within 1e-5).
    score[:, 0] += 1e-5

    # Tiebreak rule (CuMesh): among candidates within fp tolerance of the max,
    # pick the smallest chart_id. Two-stage: argmax score → mask → argmin chart_id.
    max_score = score.max(axis=1, keepdims=True)
    is_max = score >= (max_score - 1e-9)
    masked_chart = np.where(is_max, cand_charts, np.iinfo(np.int64).max)
    new_chart = masked_chart.min(axis=1).astype(np.int32)
    return new_chart


def _split_disconnected(chart_id: np.ndarray, face_adj: np.ndarray
                        ) -> Tuple[np.ndarray, int]:
    """Union-find faces sharing a manifold edge AND in the same chart. If a chart
    became disconnected during refinement, each component gets a new chart ID.

    Implementation: scipy's connected_components on the face-face graph restricted
    to same-chart edges. Each connected component becomes a new chart ID.

    Returns `(new_chart_id, new_n_charts)`.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    n_faces = len(chart_id)
    same = chart_id[face_adj[:, 0]] == chart_id[face_adj[:, 1]]
    edges = face_adj[same]
    if len(edges) == 0:
        # No same-chart edges → every face is its own component
        return np.arange(n_faces, dtype=np.int32), n_faces
    data = np.ones(len(edges), dtype=np.int8)
    adj = csr_matrix(
        (np.concatenate([data, data]),
         (np.concatenate([edges[:, 0], edges[:, 1]]),
          np.concatenate([edges[:, 1], edges[:, 0]]))),
        shape=(n_faces, n_faces),
    )
    n_comp, labels = connected_components(adj, directed=False)
    return labels.astype(np.int32), int(n_comp)


def cone_cluster(
    verts: np.ndarray,
    faces: np.ndarray,
    threshold_rad: float = np.radians(90),
    refine_iterations: int = 100,
    global_iterations: int = 3,
    smooth_strength: float = 1.0,
    area_penalty_weight: float = 0.1,
    perimeter_area_ratio_weight: float = 1e-4,
    verbose: bool = True,
) -> np.ndarray:
    """CuMesh-faithful chart segmentation.

    Argument names mirror `cumesh.CuMesh.compute_charts(...)`. Returns
    `chart_id[F]` — per-face chart index in [0, n_charts).
    """
    n_faces = len(faces)
    if verbose:
        print(f"      cone_cluster: {n_faces} faces, threshold={threshold_rad:.2f} rad "
              f"({np.degrees(threshold_rad):.0f}°), {global_iterations}×({refine_iterations} refine)", flush=True)

    t0 = time.time()
    face_normals, face_areas = _face_normals_and_areas(verts, faces)
    if verbose:
        print(f"        face normals + areas: {time.time()-t0:.2f}s", flush=True)

    t0 = time.time()
    face_adj, edge_len = _manifold_face_adjacency(verts, faces)
    if verbose:
        print(f"        manifold edges ({len(face_adj)}): {time.time()-t0:.2f}s", flush=True)

    t0 = time.time()
    neighbor_face, neighbor_edge_l, _ = _build_face_edges(faces, face_adj, edge_len)
    if verbose:
        print(f"        face→edge index: {time.time()-t0:.2f}s", flush=True)

    # Init: each face is its own chart
    chart_id = np.arange(n_faces, dtype=np.int32)
    n_charts = n_faces

    for g in range(global_iterations):
        if verbose:
            print(f"        global iteration {g + 1}/{global_iterations}", flush=True)

        # Phase 1: edge collapse until convergence
        t0 = time.time()
        collapse_iters = 0
        total_collapsed = 0
        while True:
            chart_id, n_charts, n_collapsed = _edge_collapse_pass(
                chart_id, n_charts, face_adj, edge_len, face_areas, face_normals,
                threshold_rad, area_penalty_weight, perimeter_area_ratio_weight,
                verbose=False,
            )
            collapse_iters += 1
            total_collapsed += n_collapsed
            if verbose and collapse_iters in (1, 2, 5, 10, 20, 50, 100):
                print(f"          [iter {collapse_iters}] {n_collapsed} merged → {n_charts} charts", flush=True)
            if n_collapsed == 0:
                break
            if collapse_iters > 200:
                if verbose:
                    print("        WARN: collapse not converging, stopping at 200 iters", flush=True)
                break
        if verbose:
            print(f"          collapse: {collapse_iters} iters, {total_collapsed} merges, "
                  f"{n_charts} charts ({time.time()-t0:.2f}s)", flush=True)

        # Phase 2: Lloyd refinement
        t0 = time.time()
        for r in range(refine_iterations):
            old = chart_id.copy()
            new_chart_id = _refine_pass(chart_id, n_charts, face_normals,
                                         neighbor_face, neighbor_edge_l, smooth_strength)
            n_changed = int((new_chart_id != old).sum())
            # Compact chart IDs (refinement may leave some IDs empty)
            unique, inverse = np.unique(new_chart_id, return_inverse=True)
            chart_id = inverse.astype(np.int32)
            n_charts = len(unique)
            if verbose and r in (0, 1, 5, 20, 50, 99):
                print(f"          [refine {r}] {n_changed} faces switched → {n_charts} charts", flush=True)
        if verbose:
            print(f"          refine: {refine_iterations} iters, {n_charts} charts "
                  f"({time.time()-t0:.2f}s)", flush=True)

        # Phase 3: split disconnected components within each chart
        t0 = time.time()
        chart_id, n_charts = _split_disconnected(chart_id, face_adj)
        if verbose:
            print(f"          split disconnected: {n_charts} charts ({time.time()-t0:.2f}s)", flush=True)

    if verbose:
        sizes = np.bincount(chart_id, minlength=n_charts)
        print(f"        final: {n_charts} charts, sizes [{sizes.min()}, {sizes.max()}], "
              f"median {int(np.median(sizes))}, mean {sizes.mean():.0f}", flush=True)

    return chart_id

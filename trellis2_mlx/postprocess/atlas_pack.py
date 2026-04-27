"""Per-chart orthographic UV projection + greedy shelf-pack into an atlas.

After cone-cluster chart segmentation each chart has near-uniform normal, so
projecting its vertices onto the plane orthogonal to the chart's average normal
gives a near-isometric 2D parameterization — no LSCM solve required. Charts
are then packed into the atlas via a simple shelf-packing algorithm.

This replaces xatlas's parametrize+pack with our own deterministic, fast
implementation tuned for the dense-but-flat-charts output of cone clustering.
"""
from __future__ import annotations

import time
from typing import Tuple

import numpy as np


def _chart_basis(verts: np.ndarray, face_normals_chart: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute an orthonormal (u, v) basis on the plane orthogonal to the chart's
    average normal direction. Returns `(u_axis, v_axis)` each shape (3,).
    """
    n = face_normals_chart.mean(axis=0)
    n = n / max(np.linalg.norm(n), 1e-12)
    # Pick an arbitrary helper not parallel to n
    helper = np.array([0, 0, 1], dtype=np.float32) if abs(n[2]) < 0.9 else np.array([1, 0, 0], dtype=np.float32)
    u = np.cross(n, helper); u = u / max(np.linalg.norm(u), 1e-12)
    v = np.cross(n, u); v = v / max(np.linalg.norm(v), 1e-12)
    return u.astype(np.float32), v.astype(np.float32)


def project_charts_orthographic(
    verts: np.ndarray,
    faces: np.ndarray,
    chart_id: np.ndarray,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For each chart, project its vertices onto the plane perpendicular to the
    chart's mean face normal. Vertices shared between charts are duplicated (one
    copy per chart they appear in).

    Returns:
      new_verts:    (V', 3) duplicated vertex positions, one entry per (vertex, chart)
      new_faces:    (F, 3) face indices into new_verts
      new_uvs:      (V', 2) per-new-vertex chart-local UVs (each chart's bbox normalized to [0, 1])
      new_chart:    (V', ) chart id per new vertex
      vmapping:     (V', ) original vertex index for each new vertex (for attr lookup)
    """
    n_faces = len(faces)
    # Per-face normals (re-computed; cheap)
    e1 = verts[faces[:, 1]] - verts[faces[:, 0]]
    e2 = verts[faces[:, 2]] - verts[faces[:, 0]]
    fn = np.cross(e1, e2)
    fn = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)

    n_charts = chart_id.max() + 1
    if verbose:
        print(f"      project_charts_orthographic: {n_charts} charts, {n_faces} faces", flush=True)

    # Per-chart basis (u, v) axes
    t0 = time.time()
    u_axes = np.zeros((n_charts, 3), dtype=np.float32)
    v_axes = np.zeros((n_charts, 3), dtype=np.float32)
    for c in range(n_charts):
        mask = chart_id == c
        u_axes[c], v_axes[c] = _chart_basis(verts, fn[mask])
    if verbose:
        print(f"        chart bases: {time.time()-t0:.2f}s", flush=True)

    # For each face, the 3 vertices live in chart `chart_id[face]`. We need to
    # duplicate vertices that appear in multiple charts. Use (vertex_idx, chart_id)
    # as the key for new vertex IDs.
    t0 = time.time()
    face_chart = chart_id  # alias
    # For each (face, corner) → (vert_idx, chart). Build unique pairs.
    vert_per_corner = faces.reshape(-1)              # (3*F,)
    chart_per_corner = np.repeat(face_chart, 3)       # (3*F,)
    keys = vert_per_corner.astype(np.int64) * (n_charts + 1) + chart_per_corner.astype(np.int64)
    unique_keys, inverse, _ = np.unique(keys, return_inverse=True, return_counts=True)
    # Build vmapping (orig vert idx) and new_chart (chart per new vert)
    vmapping = (unique_keys // (n_charts + 1)).astype(np.int64)
    new_chart = (unique_keys % (n_charts + 1)).astype(np.int32)
    new_verts = verts[vmapping]
    new_faces = inverse.reshape(n_faces, 3).astype(np.int64)
    if verbose:
        print(f"        vertex split: {len(verts)} → {len(new_verts)} ({time.time()-t0:.2f}s)", flush=True)

    # Project each new vertex via its chart basis
    t0 = time.time()
    u_per_v = u_axes[new_chart]
    v_per_v = v_axes[new_chart]
    new_uvs = np.stack([
        (new_verts * u_per_v).sum(axis=1),
        (new_verts * v_per_v).sum(axis=1),
    ], axis=1).astype(np.float32)

    # Normalize each chart's UVs to its own [0, 1] bbox
    for c in range(n_charts):
        mask = new_chart == c
        if not mask.any():
            continue
        block = new_uvs[mask]
        bb_min = block.min(axis=0)
        bb_max = block.max(axis=0)
        size = bb_max - bb_min
        size[size < 1e-9] = 1.0
        new_uvs[mask] = (block - bb_min) / size
    if verbose:
        print(f"        ortho project + chart-local norm: {time.time()-t0:.2f}s", flush=True)

    return new_verts.astype(np.float32), new_faces, new_uvs, new_chart, vmapping


def shelf_pack(
    chart_uvs: np.ndarray,           # (V', 2) chart-local UVs in [0, 1]
    chart_id: np.ndarray,            # (V', ) chart per new vertex
    chart_world_sizes: np.ndarray,   # (n_charts, 2) bbox sizes in world units (for area weighting)
    padding: float = 4 / 2048.0,
    verbose: bool = True,
) -> np.ndarray:
    """Pack each chart's [0, 1] UV box into a single [0, 1]² atlas via iterative
    shelf-packing. Charts are sized proportional to sqrt(world_area), then a
    uniform global scale is auto-tuned so they pack tightly without overflow.

    Returns transformed UVs in [0, 1]² atlas-space (per-vertex).
    """
    n_charts = len(chart_world_sizes)
    # Each chart's "natural" aspect ratio comes from world bbox; relative size
    # comes from sqrt(world_area). We'll auto-scale these together later.
    w_world = np.maximum(chart_world_sizes[:, 0], 1e-9)
    h_world = np.maximum(chart_world_sizes[:, 1], 1e-9)
    chart_aspect_w = np.sqrt(w_world / h_world)
    chart_aspect_h = np.sqrt(h_world / w_world)

    # Iteratively binary-search a uniform scale s such that all charts fit in
    # [0, 1]² with shelf-packing, maximizing s.
    def try_pack(scale: float):
        cw = chart_aspect_w * scale
        ch = chart_aspect_h * scale
        order = np.argsort(-ch)
        placements = np.zeros((n_charts, 4), dtype=np.float32)
        shelf_y = 0.0
        shelf_h = 0.0
        cursor_x = 0.0
        for c in order:
            w, h = float(cw[c]), float(ch[c])
            if cursor_x + w > 1.0 - padding:
                shelf_y += shelf_h + padding
                shelf_h = 0.0
                cursor_x = 0.0
            if shelf_y + h > 1.0:
                return None  # overflow
            placements[c] = [cursor_x, shelf_y, w, h]
            cursor_x += w + padding
            shelf_h = max(shelf_h, h)
        used_y = shelf_y + shelf_h
        return placements, cw, ch, used_y

    # Binary search the scale
    lo, hi = 1e-6, 1.0
    best = None
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        result = try_pack(mid)
        if result is None:
            hi = mid
        else:
            best = (mid, *result)
            lo = mid
    if best is None:
        # Even at the smallest scale, doesn't fit — fallback to uniform tiny
        scale, placements, cw, ch, used_y = 1e-3, *try_pack(1e-3)
    else:
        scale, placements, cw, ch, used_y = best

    if verbose:
        used_area = (cw * ch).sum()
        print(f"        shelf_pack: scale={scale:.4f}, atlas height {used_y:.2f}, "
              f"chart-area-coverage {used_area*100:.1f}%, {n_charts} charts", flush=True)

    out = chart_uvs.copy()
    out[:, 0] = chart_uvs[:, 0] * cw[chart_id] + placements[chart_id, 0]
    out[:, 1] = chart_uvs[:, 1] * ch[chart_id] + placements[chart_id, 1]
    return out


def chart_world_sizes_from_uvs(
    verts: np.ndarray,
    chart_uvs_local: np.ndarray,
    chart_id: np.ndarray,
) -> np.ndarray:
    """Estimate each chart's world-space (width, height) — used to weight atlas
    placement so big surfaces get more atlas pixels. We approximate via the
    chart's vertex bbox in 3D, projected through the local UV span.
    """
    n_charts = chart_id.max() + 1
    sizes = np.zeros((n_charts, 2), dtype=np.float32)
    for c in range(n_charts):
        mask = chart_id == c
        if not mask.any():
            sizes[c] = [1.0, 1.0]
            continue
        v3d = verts[mask]
        # 3D bbox extent
        ext = v3d.max(axis=0) - v3d.min(axis=0)
        # Use top 2 axes by extent as proxy width/height
        top2 = np.sort(ext)[-2:]
        sizes[c] = [max(top2[1], 1e-6), max(top2[0], 1e-6)]
    return sizes

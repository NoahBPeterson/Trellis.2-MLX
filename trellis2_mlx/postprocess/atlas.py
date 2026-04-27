"""UV unwrap (cone-cluster + orthographic projection) + per-chart pack
+ per-vertex attribute bake into a 2D PBR atlas.

This is our pure-numpy replacement for upstream's `cumesh.CuMesh.uv_unwrap`
(CUDA-only). xatlas was tried initially but its geodesic-based segmentation
explodes to thousands of tiny charts on TRELLIS dual-grid topology, taking hours.
Cumesh's cone-cluster approach is what's needed here, and we've reimplemented
it in `chart_segment.cone_cluster` + `atlas_pack.project_charts_orthographic`.

Per-vertex `attrs` channel layout (matches upstream `MeshWithPbrMaterial.layout`):
  [0:3] = base_color RGB
  [3]   = metallic
  [4]   = roughness
  [5]   = alpha
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def decimate(
    verts: np.ndarray,
    faces: np.ndarray,
    vert_attrs: np.ndarray,
    target_faces: int = 1_000_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quadric decimation via trimesh, with attrs re-projected via nearest-neighbor.

    Returns `(verts_dec, faces_dec, attrs_dec)`. If `target_faces >= len(faces)`,
    returns the inputs unchanged.
    """
    if target_faces >= len(faces):
        return verts, faces, vert_attrs

    import trimesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    simplified = mesh.simplify_quadric_decimation(face_count=target_faces)

    # Re-project per-vertex attrs from the original mesh via nearest-neighbor.
    # trimesh's decimation doesn't preserve generic vertex attributes, so we KD-tree
    # the original verts and look up the closest one for each new vertex.
    from scipy.spatial import cKDTree
    tree = cKDTree(verts)
    _, nn_idx = tree.query(np.asarray(simplified.vertices), k=1)
    attrs_dec = vert_attrs[nn_idx]
    return np.asarray(simplified.vertices), np.asarray(simplified.faces), attrs_dec


def repair_mesh(verts: np.ndarray, faces: np.ndarray, vert_attrs: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pre-clean a mesh for xatlas: weld duplicate verts, drop degenerate
    triangles (zero area or duplicate indices), drop unreferenced verts.

    xatlas's parametrize segfaults on TRELLIS dual-grid output without this — the
    dual-grid extraction produces T-junctions, near-duplicate verts, and
    occasional zero-area triangles that xatlas's chart builder can't handle.
    """
    print(f"      repair: input {len(verts)} verts / {len(faces)} faces", flush=True)

    # 1. Drop faces with duplicate vertex indices (degenerate)
    a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
    nondegen = (a != b) & (b != c) & (a != c)
    faces = faces[nondegen]
    print(f"      repair: dropped {(~nondegen).sum()} faces with duplicate indices", flush=True)

    # 2. Weld duplicate vertex positions (within float32 precision).
    # Round to a fine grid and use unique-row indexing — much faster than KD-tree.
    quant = (verts.astype(np.float64) * 1e6).round().astype(np.int64)
    _, unique_idx, inverse = np.unique(quant, axis=0, return_index=True, return_inverse=True)
    if len(unique_idx) < len(verts):
        print(f"      repair: welded {len(verts)} → {len(unique_idx)} unique verts", flush=True)
        verts_w = verts[unique_idx]
        attrs_w = vert_attrs[unique_idx]
        faces = inverse[faces]
        # Drop any faces that became degenerate after welding
        a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
        nondegen2 = (a != b) & (b != c) & (a != c)
        faces = faces[nondegen2]
        print(f"      repair: dropped {(~nondegen2).sum()} more degenerate faces post-weld", flush=True)
    else:
        verts_w, attrs_w = verts, vert_attrs

    # 3. Drop faces with zero or near-zero area (collinear vertices)
    p0 = verts_w[faces[:, 0]]
    p1 = verts_w[faces[:, 1]]
    p2 = verts_w[faces[:, 2]]
    cross = np.cross(p1 - p0, p2 - p0)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    nontrivial = area > 1e-12
    if (~nontrivial).any():
        print(f"      repair: dropped {(~nontrivial).sum()} zero-area faces", flush=True)
        faces = faces[nontrivial]

    # 4. Drop unreferenced verts (compact the index space)
    used = np.zeros(len(verts_w), dtype=bool)
    used[faces.reshape(-1)] = True
    if not used.all():
        old_to_new = -np.ones(len(verts_w), dtype=np.int64)
        old_to_new[used] = np.arange(used.sum())
        verts_w = verts_w[used]
        attrs_w = attrs_w[used]
        faces = old_to_new[faces]
        print(f"      repair: dropped {(~used).sum()} unreferenced verts", flush=True)

    print(f"      repair: output {len(verts_w)} verts / {len(faces)} faces", flush=True)
    return verts_w, faces.astype(np.int64), attrs_w


def unwrap(verts: np.ndarray, faces: np.ndarray, vert_attrs: np.ndarray,
           threshold_deg: float = 75.0, min_chart_size: int = 256,
           max_chart_size: int = 2000,
           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """UV unwrap via cone-cluster chart segmentation + orthographic projection +
    shelf-pack. Replaces xatlas.

    Returns `(new_verts, new_faces, new_uvs, new_attrs)`. Vertices on chart
    boundaries get duplicated (one copy per chart they belong to), so
    `new_verts.shape[0] >= verts.shape[0]`. `new_uvs` are in [0, 1]² atlas-space.
    """
    from .chart_segment import cone_cluster
    from .atlas_pack import project_charts_orthographic, shelf_pack, chart_world_sizes_from_uvs

    verts, faces, vert_attrs = repair_mesh(verts, faces, vert_attrs)
    print(f"      cone-cluster unwrap: {len(verts)} verts, {len(faces)} faces", flush=True)

    chart_id = cone_cluster(verts, faces, threshold_rad=np.radians(threshold_deg),
                            min_chart_size=min_chart_size, max_chart_size=max_chart_size)
    n_charts = int(chart_id.max() + 1)

    new_verts, new_faces, local_uvs, new_chart_per_v, vmapping = project_charts_orthographic(
        verts, faces, chart_id
    )
    new_attrs = vert_attrs[vmapping]

    chart_sizes_world = chart_world_sizes_from_uvs(new_verts, local_uvs, new_chart_per_v)
    atlas_uvs = shelf_pack(local_uvs, new_chart_per_v, chart_sizes_world)

    return new_verts, new_faces, atlas_uvs.astype(np.float32), new_attrs


def bake_atlas(
    faces: np.ndarray,
    uvs: np.ndarray,
    vert_attrs: np.ndarray,
    atlas_size: int = 2048,
    channels: int = 6,
    progress_every: int = 100_000,
) -> np.ndarray:
    """Rasterize per-vertex attrs into a (H, W, channels) atlas via per-face
    scanline + barycentric interpolation. UVs are expected in [0, 1]; (0, 0)
    maps to the top-left corner.

    Pure numpy. Per-face Python overhead dominates at high face counts; expect
    ~100k faces/min on M1 Pro at atlas_size=2048.
    """
    H = W = atlas_size
    atlas = np.zeros((H, W, channels), dtype=np.float32)
    coverage = np.zeros((H, W), dtype=bool)

    # UV → atlas pixel coords (continuous; pixel centers are at .5)
    uv_px = uvs * np.array([W - 1, H - 1], dtype=np.float32)

    n_faces = len(faces)
    skipped = 0
    for fi in range(n_faces):
        if progress_every and fi > 0 and fi % progress_every == 0:
            print(f"      bake_atlas: {fi}/{n_faces} faces ({fi/n_faces*100:.0f}%)", flush=True)
        v0, v1, v2 = faces[fi]
        p0 = uv_px[v0]; p1 = uv_px[v1]; p2 = uv_px[v2]
        a0 = vert_attrs[v0]; a1 = vert_attrs[v1]; a2 = vert_attrs[v2]

        # Triangle bbox in atlas pixels
        x_min = max(0, int(np.floor(min(p0[0], p1[0], p2[0]))))
        x_max = min(W, int(np.ceil(max(p0[0], p1[0], p2[0]))) + 1)
        y_min = max(0, int(np.floor(min(p0[1], p1[1], p2[1]))))
        y_max = min(H, int(np.ceil(max(p0[1], p1[1], p2[1]))) + 1)
        if x_min >= x_max or y_min >= y_max:
            skipped += 1
            continue

        # Pixel center grid in bbox
        ys, xs = np.mgrid[y_min:y_max, x_min:x_max]
        px = xs.astype(np.float32) + 0.5
        py = ys.astype(np.float32) + 0.5

        # Barycentric via unnormalized edge functions, then normalize.
        # Important: orthographic-projected triangles can be CW or CCW depending on
        # which side of the chart's plane the source face was on. We accept either
        # winding by testing same-sign-as-denom rather than positivity.
        denom = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
        if abs(denom) < 1e-12:
            skipped += 1
            continue
        e0 = (p1[1] - p2[1]) * (px - p2[0]) + (p2[0] - p1[0]) * (py - p2[1])
        e1 = (p2[1] - p0[1]) * (px - p2[0]) + (p0[0] - p2[0]) * (py - p2[1])
        e2 = denom - e0 - e1
        if denom > 0:
            inside = (e0 >= -1e-5) & (e1 >= -1e-5) & (e2 >= -1e-5)
        else:
            inside = (e0 <= 1e-5) & (e1 <= 1e-5) & (e2 <= 1e-5)
        l0 = e0 / denom
        l1 = e1 / denom
        l2 = e2 / denom
        if not inside.any():
            continue
        ys_in = ys[inside]
        xs_in = xs[inside]
        l0i = l0[inside][:, None]
        l1i = l1[inside][:, None]
        l2i = l2[inside][:, None]
        # Interpolate per-vertex attrs
        attrs = l0i * a0 + l1i * a1 + l2i * a2  # (n_inside, channels)
        atlas[ys_in, xs_in] = attrs
        coverage[ys_in, xs_in] = True

    print(f"      bake_atlas: done ({n_faces - skipped} faces filled, {skipped} skipped, "
          f"coverage {coverage.mean()*100:.1f}%)", flush=True)
    return atlas


def fill_atlas_seams(atlas: np.ndarray, coverage: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Dilate filled pixels into uncovered neighbors to hide seam bleed at glTF
    sample edges. Cheap O(iterations × atlas_size²) numpy.
    """
    out = atlas.copy()
    cov = coverage.copy()
    for _ in range(iterations):
        # 4-neighbor dilation: copy from any covered neighbor into uncovered pixels
        new_cov = cov.copy()
        new_out = out.copy()
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            shifted_cov = np.roll(cov, shift=(dy, dx), axis=(0, 1))
            shifted_out = np.roll(out, shift=(dy, dx), axis=(0, 1))
            mask = (~new_cov) & shifted_cov
            new_out[mask] = shifted_out[mask]
            new_cov[mask] = True
        out, cov = new_out, new_cov
    return out

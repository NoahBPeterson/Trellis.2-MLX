"""UV unwrap via CuMesh-faithful cone-cluster + per-chart xatlas, plus
per-vertex attribute bake into a 2D PBR atlas.

Two-stage pipeline mirroring `cumesh.CuMesh.uv_unwrap` (CUDA upstream;
pure-numpy here):

  1. **Cone-cluster chart segmentation** (`chart_segment.cone_cluster`) —
     ports CuMesh's parallel edge-collapse + Lloyd refinement to produce a
     small number of well-shaped charts.
  2. **Per-chart xatlas** — feeds each chart's sub-mesh to xatlas separately
     for parameterization (LSCM) and packing. xatlas alone hangs/segfaults on
     TRELLIS dual-grid topology because its own segmentation produces 1000s
     of tiny geodesic charts; pre-clustering with cone-cluster avoids this.

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
           threshold_deg: float = 90.0,
           refine_iterations: int = 0,
           global_iterations: int = 1,
           smooth_strength: float = 1.0,
           area_penalty_weight: float = 0.1,
           perimeter_area_ratio_weight: float = 1e-4,
           xatlas_resolution: int = 2048,
           xatlas_padding: int = 2,
           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """CuMesh-faithful UV unwrap: cone-cluster segmentation + per-chart xatlas.

    Returns `(new_verts, new_faces, new_uvs, new_attrs, atlas_pixels)`. Vertices on
    chart boundaries get duplicated (one copy per chart they belong to), so
    `new_verts.shape[0] >= verts.shape[0]`. `new_uvs` are in [0, 1]² atlas-space.
    `atlas_pixels` is xatlas's chosen atlas resolution — useful so the bake step
    can match it instead of downsampling chart islands into oblivion.

    The `threshold_deg` / `refine_iterations` / etc. arguments mirror
    `cumesh.CuMesh.uv_unwrap(compute_charts_kwargs={...})`.
    """
    import time
    import xatlas
    from .chart_segment import cone_cluster

    verts, faces, vert_attrs = repair_mesh(verts, faces, vert_attrs)
    print(f"      cone-cluster unwrap: {len(verts)} verts, {len(faces)} faces", flush=True)

    chart_id = cone_cluster(
        verts, faces,
        threshold_rad=np.radians(threshold_deg),
        refine_iterations=refine_iterations,
        global_iterations=global_iterations,
        smooth_strength=smooth_strength,
        area_penalty_weight=area_penalty_weight,
        perimeter_area_ratio_weight=perimeter_area_ratio_weight,
    )
    n_charts = int(chart_id.max() + 1)
    print(f"      cone-cluster: {n_charts} charts → handing each to xatlas", flush=True)

    # Per-chart sub-meshes for xatlas (mirror cumesh.CuMesh.uv_unwrap's flow):
    #   for each chart c:
    #     chart_faces[c] : faces with chart_id == c, vertex indices remapped to the chart's local space
    #     chart_verts[c] : the unique vertices used by chart_faces[c]
    #     chart_vmap[c]  : local→original vertex index map (for attr lookup later)
    t0 = time.time()
    atlas = xatlas.Atlas()
    chart_vmaps = []
    for c in range(n_charts):
        face_mask = chart_id == c
        sub_faces = faces[face_mask]
        used_verts, local_faces = np.unique(sub_faces.reshape(-1), return_inverse=True)
        local_faces = local_faces.reshape(-1, 3).astype(np.uint32)
        sub_verts = verts[used_verts].astype(np.float32)
        if len(local_faces) == 0:
            chart_vmaps.append(used_verts.astype(np.int64))
            continue
        atlas.add_mesh(sub_verts, local_faces)
        chart_vmaps.append(used_verts.astype(np.int64))
    print(f"      assembled {n_charts} chart sub-meshes for xatlas ({time.time()-t0:.1f}s)", flush=True)

    # xatlas: each chart is small + well-conditioned (single near-flat region), so its own
    # segmentation collapses to one chart per sub-mesh; the parameterization (LSCM) is what
    # we actually want. Single generate() call packs all charts together.
    t0 = time.time()
    co = xatlas.ChartOptions()
    co.max_iterations = 1  # cone-cluster already segmented; xatlas just parameterizes
    po = xatlas.PackOptions()
    po.resolution = int(xatlas_resolution)
    po.padding = int(xatlas_padding)
    po.bilinear = True
    po.rotate_charts = True
    atlas.generate(co, po, verbose=False)
    print(f"      xatlas generate: {time.time()-t0:.1f}s "
          f"(atlas {atlas.width}x{atlas.height}, util {atlas.utilization})", flush=True)

    # Gather per-chart results, concatenate. xatlas returns:
    #   xrefs  (V', )  — local vertex index in the sub-mesh given to add_mesh
    #   x_faces (F', 3) — face indices into the new V' vertices
    #   x_uvs  (V', 2)  — UV coords (in pixel-space; we normalize to [0, 1])
    t0 = time.time()
    out_verts = []
    out_faces = []
    out_uvs = []
    out_vmap = []  # original vertex idx for each new vertex (for attr lookup)
    cnt = 0
    skipped = 0
    for c in range(atlas.mesh_count):
        try:
            xrefs, x_faces, x_uvs = atlas.get_mesh(c)
        except Exception:
            skipped += 1
            continue
        # xrefs is local→sub-mesh index; chain through chart_vmaps[c] to get original vertex idx
        orig_idx = chart_vmaps[c][xrefs]
        out_verts.append(verts[orig_idx])
        out_faces.append(x_faces.astype(np.int64) + cnt)
        out_uvs.append(x_uvs)
        out_vmap.append(orig_idx)
        cnt += len(xrefs)
    if skipped:
        print(f"      xatlas: skipped {skipped} charts that failed to parameterize", flush=True)

    new_verts = np.concatenate(out_verts, axis=0).astype(np.float32)
    new_faces = np.concatenate(out_faces, axis=0).astype(np.int64)
    new_uvs = np.concatenate(out_uvs, axis=0).astype(np.float32)
    new_vmap = np.concatenate(out_vmap, axis=0).astype(np.int64)
    new_attrs = vert_attrs[new_vmap]
    # xatlas-python returns UVs already in [0, 1]² atlas-space across all sub-meshes.
    print(f"      gather: {len(new_verts)} verts, {len(new_faces)} faces ({time.time()-t0:.1f}s)", flush=True)

    atlas_pixels = max(int(atlas.width), int(atlas.height))
    return new_verts, new_faces, new_uvs.astype(np.float32), new_attrs, atlas_pixels


def bake_atlas(
    faces: np.ndarray,
    uvs: np.ndarray,
    vert_attrs: np.ndarray,
    atlas_size: int = 2048,
    channels: int = 6,
    progress_every: int = 100_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rasterize per-vertex attrs into a (H, W, channels) atlas via per-face
    scanline + barycentric interpolation. UVs are expected in [0, 1]; (0, 0)
    maps to the top-left corner.

    Returns `(atlas, coverage)` where `coverage` is a (H, W) bool mask of
    pixels that were touched by at least one triangle. Callers should
    inpaint uncovered pixels (e.g. via `cv2.inpaint`) — uncovered pixels
    are left at zero, which means mirror-smooth black under PBR sampling.

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
    return atlas, coverage


def _closest_point_on_triangle_batch(
    P: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closest point on each triangle to its corresponding query point. Vectorized.

    Inputs are all (N, 3). Returns:
      closest: (N, 3) closest point on triangle
      bary:    (N, 3) barycentric weights (w_A, w_B, w_C) summing to 1
      d2:      (N,)   squared distance from P to the closest point

    Algorithm: Ericson "Real-Time Collision Detection" §5.1.5 — compute regions
    via signed-area tests, branch by region (vertex / edge / interior).
    """
    AB = B - A; AC = C - A; AP = P - A
    d1 = (AB * AP).sum(axis=1)
    d2_ = (AC * AP).sum(axis=1)
    BP = P - B
    d3 = (AB * BP).sum(axis=1)
    d4 = (AC * BP).sum(axis=1)
    CP = P - C
    d5 = (AB * CP).sum(axis=1)
    d6 = (AC * CP).sum(axis=1)
    vc = d1 * d4 - d3 * d2_
    vb = d5 * d2_ - d1 * d6
    va = d3 * d6 - d5 * d4
    denom = va + vb + vc
    safe_denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)

    # Edge AB: vc <= 0, d1 >= 0, d3 <= 0
    v_ab = np.where((d1 - d3) != 0, d1 / np.where(d1 - d3 == 0, 1.0, d1 - d3), 0.0)
    v_ab = np.clip(v_ab, 0.0, 1.0)
    # Edge AC: vb <= 0, d2 >= 0, d6 <= 0
    v_ac = np.where((d2_ - d6) != 0, d2_ / np.where(d2_ - d6 == 0, 1.0, d2_ - d6), 0.0)
    v_ac = np.clip(v_ac, 0.0, 1.0)
    # Edge BC: va <= 0, (d4 - d3) >= 0, (d5 - d6) >= 0
    v_bc = np.where((d4 - d3 + d5 - d6) != 0,
                    (d4 - d3) / np.where(d4 - d3 + d5 - d6 == 0, 1.0, d4 - d3 + d5 - d6), 0.0)
    v_bc = np.clip(v_bc, 0.0, 1.0)

    # Interior bary
    v_int = vb / safe_denom
    w_int = vc / safe_denom

    # Region masks (each query falls into exactly one region)
    in_A  = (d1 <= 0) & (d2_ <= 0)
    in_B  = (d3 >= 0) & (d4 <= d3)
    in_C  = (d6 >= 0) & (d5 <= d6)
    in_AB = (~in_A) & (~in_B) & (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    in_AC = (~in_A) & (~in_C) & (vb <= 0) & (d2_ >= 0) & (d6 <= 0)
    in_BC = (~in_B) & (~in_C) & (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    in_int = ~(in_A | in_B | in_C | in_AB | in_AC | in_BC)

    bary = np.zeros_like(A)
    bary[in_A]  = [1.0, 0.0, 0.0]
    bary[in_B]  = [0.0, 1.0, 0.0]
    bary[in_C]  = [0.0, 0.0, 1.0]
    bary[in_AB] = np.stack([1.0 - v_ab[in_AB], v_ab[in_AB], np.zeros(in_AB.sum())], axis=1)
    bary[in_AC] = np.stack([1.0 - v_ac[in_AC], np.zeros(in_AC.sum()), v_ac[in_AC]], axis=1)
    bary[in_BC] = np.stack([np.zeros(in_BC.sum()), 1.0 - v_bc[in_BC], v_bc[in_BC]], axis=1)
    bary[in_int] = np.stack([1.0 - v_int[in_int] - w_int[in_int], v_int[in_int], w_int[in_int]], axis=1)

    closest = bary[:, 0:1] * A + bary[:, 1:2] * B + bary[:, 2:3] * C
    d2 = ((closest - P) ** 2).sum(axis=1)
    return closest.astype(np.float32), bary.astype(np.float32), d2.astype(np.float32)


def bake_atlas_bvh(
    faces_uv: np.ndarray,
    uvs: np.ndarray,
    V_uv: np.ndarray,
    V_orig: np.ndarray,
    F_orig: np.ndarray,
    attrs_orig: np.ndarray,
    atlas_size: int = 2048,
    knn_candidates: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-texel bake with BVH-projection back to the original mesh.

    Mirrors upstream `o-voxel/postprocess.py:236-256` minus the volume sampling
    (we don't surface the attribute volume from the pipeline, so we resample
    against the original mesh's per-vertex attrs instead — ~9× the sample
    density of bake_atlas's per-vertex bary on the simplified mesh).

    Pipeline per texel:
      1. UV face id + barycentric (per-face scanline rasterization)
      2. 3D position on simplified surface = V_uv[face] · bary
      3. K-nearest original face centroids via cKDTree
      4. Closest-point-on-triangle for those K candidates → pick min distance
      5. attrs = attrs_orig[F_orig[best_face]] · best_bary

    `V_uv` are the simplified mesh's vertex positions (split at chart seams by
    xatlas; positions match V_simp). `(V_orig, F_orig, attrs_orig)` is the
    pre-decimation mesh.
    """
    import time
    from scipy.spatial import cKDTree

    H = W = atlas_size
    C = attrs_orig.shape[1]
    atlas = np.zeros((H, W, C), dtype=np.float32)
    coverage = np.zeros((H, W), dtype=bool)
    uv_px = uvs * np.array([W - 1, H - 1], dtype=np.float32)

    # 1. Rasterize: for each UV face, collect (atlas_y, atlas_x, 3D-pos) per covered texel.
    t0 = time.time()
    n_faces = len(faces_uv)
    all_y, all_x, all_pos = [], [], []
    skipped = 0
    for fi in range(n_faces):
        v0, v1, v2 = faces_uv[fi]
        p0 = uv_px[v0]; p1 = uv_px[v1]; p2 = uv_px[v2]
        x_min = max(0, int(np.floor(min(p0[0], p1[0], p2[0]))))
        x_max = min(W, int(np.ceil(max(p0[0], p1[0], p2[0]))) + 1)
        y_min = max(0, int(np.floor(min(p0[1], p1[1], p2[1]))))
        y_max = min(H, int(np.ceil(max(p0[1], p1[1], p2[1]))) + 1)
        if x_min >= x_max or y_min >= y_max:
            skipped += 1
            continue
        ys, xs = np.mgrid[y_min:y_max, x_min:x_max]
        px = xs.astype(np.float32) + 0.5
        py = ys.astype(np.float32) + 0.5
        denom = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
        if abs(denom) < 1e-12:
            skipped += 1
            continue
        e0 = (p1[1] - p2[1]) * (px - p2[0]) + (p2[0] - p1[0]) * (py - p2[1])
        e1 = (p2[1] - p0[1]) * (px - p2[0]) + (p0[0] - p2[0]) * (py - p2[1])
        e2 = denom - e0 - e1
        inside = ((e0 >= -1e-5) & (e1 >= -1e-5) & (e2 >= -1e-5)) if denom > 0 else \
                 ((e0 <= 1e-5) & (e1 <= 1e-5) & (e2 <= 1e-5))
        if not inside.any():
            continue
        l0 = (e0 / denom)[inside]
        l1 = (e1 / denom)[inside]
        l2 = (e2 / denom)[inside]
        # 3D pos via simplified-mesh barycentric (V_uv positions are V_simp)
        pos = (l0[:, None] * V_uv[v0] + l1[:, None] * V_uv[v1] + l2[:, None] * V_uv[v2]).astype(np.float32)
        all_y.append(ys[inside])
        all_x.append(xs[inside])
        all_pos.append(pos)
    if not all_pos:
        return atlas, coverage
    Y = np.concatenate(all_y)
    X = np.concatenate(all_x)
    P = np.concatenate(all_pos, axis=0)
    print(f"      bake_bvh: rasterized {len(P)} texels ({skipped} faces skipped) in {time.time()-t0:.1f}s", flush=True)

    # 2. cKDTree on original face centroids.
    t0 = time.time()
    centroids = V_orig[F_orig].mean(axis=1).astype(np.float32)
    tree = cKDTree(centroids)
    print(f"      bake_bvh: built KDTree on {len(centroids)} original faces in {time.time()-t0:.1f}s", flush=True)

    # 3. K-nearest centroids per query, then exact closest-point-on-triangle for those K.
    t0 = time.time()
    # Process in chunks to keep peak memory bounded.
    chunk = 200_000
    best_face = np.empty(len(P), dtype=np.int64)
    best_bary = np.empty((len(P), 3), dtype=np.float32)
    for start in range(0, len(P), chunk):
        end = min(start + chunk, len(P))
        Pc = P[start:end]
        _, knn = tree.query(Pc, k=knn_candidates)  # (n, K)
        # Expand to (n*K, 3) for vectorized closest-point-on-triangle.
        flat_q = np.repeat(Pc, knn_candidates, axis=0)
        flat_face = knn.reshape(-1)
        tri = V_orig[F_orig[flat_face]]  # (n*K, 3, 3)
        _, bary, d2 = _closest_point_on_triangle_batch(flat_q, tri[:, 0], tri[:, 1], tri[:, 2])
        d2 = d2.reshape(end - start, knn_candidates)
        bary = bary.reshape(end - start, knn_candidates, 3)
        knn_re = knn  # (n, K)
        best_idx = np.argmin(d2, axis=1)  # (n,)
        rows = np.arange(end - start)
        best_face[start:end] = knn_re[rows, best_idx]
        best_bary[start:end] = bary[rows, best_idx]
    print(f"      bake_bvh: BVH-projected {len(P)} texels in {time.time()-t0:.1f}s", flush=True)

    # 4. Interpolate original vertex_attrs.
    t0 = time.time()
    face_verts = F_orig[best_face]  # (n, 3)
    attrs_per_texel = (best_bary[:, 0:1] * attrs_orig[face_verts[:, 0]]
                       + best_bary[:, 1:2] * attrs_orig[face_verts[:, 1]]
                       + best_bary[:, 2:3] * attrs_orig[face_verts[:, 2]])
    atlas[Y, X] = attrs_per_texel
    coverage[Y, X] = True
    print(f"      bake_bvh: attr interpolation {time.time()-t0:.1f}s, "
          f"coverage {coverage.mean()*100:.1f}%", flush=True)
    return atlas, coverage


def inpaint_atlas(atlas: np.ndarray, coverage: np.ndarray,
                  base_color_radius: int = 3,
                  scalar_radius: int = 1) -> np.ndarray:
    """Telea-inpaint each PBR channel into uncovered pixels. Mirrors upstream
    `o_voxel.postprocess.to_glb`: radius 3 for base-color RGB, radius 1 for
    metallic / roughness / alpha. Telea is isophote-driven so it extends real
    baked values into the gaps, instead of leaving them at the (broken)
    roughness-0 / black-mirror default.

    `atlas` is float32 in [0, 1] with channel layout
    `[R, G, B, metallic, roughness, alpha]`. Returns a same-shape array.
    """
    import cv2

    H, W, C = atlas.shape
    assert C == 6, f"inpaint_atlas expects 6-channel atlas, got {C}"
    mask_inv = (~coverage).astype(np.uint8)
    if mask_inv.sum() == 0:
        return atlas

    u8 = (atlas * 255.0).clip(0, 255).astype(np.uint8)
    bc = cv2.inpaint(u8[:, :, :3], mask_inv, base_color_radius, cv2.INPAINT_TELEA)
    m  = cv2.inpaint(u8[:, :, 3],  mask_inv, scalar_radius, cv2.INPAINT_TELEA)
    r  = cv2.inpaint(u8[:, :, 4],  mask_inv, scalar_radius, cv2.INPAINT_TELEA)
    a  = cv2.inpaint(u8[:, :, 5],  mask_inv, scalar_radius, cv2.INPAINT_TELEA)
    out = np.empty_like(u8)
    out[:, :, :3] = bc
    out[:, :, 3] = m
    out[:, :, 4] = r
    out[:, :, 5] = a
    return out.astype(np.float32) / 255.0

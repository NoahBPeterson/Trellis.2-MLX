"""Mesh export to GLB via trimesh.

`export_mesh_glb`: untextured geometry or vertex-color GLB (simple).
`export_pbr_glb`:  full PBR — UV unwrap + per-vertex attr bake into a 2-image
                   atlas (base-color RGBA + metallicRoughness), packaged as a
                   glTF PBRMaterial. Mirrors what `o_voxel.postprocess.to_glb`
                   does on CUDA upstream, but on CPU.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def export_mesh_glb(
    vertices: np.ndarray,
    faces: np.ndarray,
    out_path: str | Path,
    vertex_colors: Optional[np.ndarray] = None,
) -> None:
    """Write `(V, F)` as a GLB. `vertex_colors` is optional (V, 3|4) in [0, 1]."""
    import trimesh
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices, dtype=np.float32),
                           faces=np.asarray(faces, dtype=np.int32), process=False)
    if vertex_colors is not None:
        vc = np.asarray(vertex_colors, dtype=np.float32)
        if vc.ndim == 2 and vc.shape[1] == 3:
            alpha = np.ones((vc.shape[0], 1), dtype=np.float32)
            vc = np.concatenate([vc, alpha], axis=1)
        mesh.visual.vertex_colors = (vc * 255).clip(0, 255).astype(np.uint8)
    mesh.export(str(out_path))


def export_pbr_glb(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_attrs: np.ndarray,
    out_path: str | Path,
    target_faces: int = 500_000,
    atlas_size: int = 2048,
    fill_seams: bool = True,
    bake_mode: str = "bvh",
) -> dict:
    """Decimate → UV-unwrap → bake atlas → write textured glTF.

    `vertex_attrs` is `(V, 6)` in [0, 1] with channel layout:
        [0:3] base_color RGB, [3] metallic, [4] roughness, [5] alpha.

    `bake_mode`:
      * `"vertex"` — per-vertex barycentric on the simplified mesh (legacy).
        Each chart's UV island gets a linear gradient between 3 simplified
        vertex colors per face. Loses detail finer than a simplified vertex.
      * `"bvh"` — per-texel BVH-project to the *original* (un-decimated) mesh,
        sample original vertex_attrs there. ~9× the sample density. Mirrors
        upstream o-voxel's per-texel approach (minus the volume sampling we
        don't have access to in cache).

    Returns a dict of stage timings.
    """
    import time
    import trimesh
    from . import atlas as atlas_mod

    timings = {}
    t0 = time.time()

    # 1. Decimate
    print(f"      decimate: {len(faces)} → target {target_faces} faces", flush=True)
    v_d, f_d, a_d = atlas_mod.decimate(vertices, faces, vertex_attrs, target_faces)
    print(f"      decimate: produced {len(f_d)} faces ({time.time()-t0:.1f}s)", flush=True)
    timings["decimate"] = time.time() - t0

    # 2. UV unwrap
    t1 = time.time()
    print(f"      cone-cluster + xatlas unwrap on {len(v_d)} verts...", flush=True)
    v_u, f_u, uvs, a_u, atlas_pixels = atlas_mod.unwrap(v_d, f_d, a_d,
                                                        xatlas_resolution=atlas_size)
    print(f"      unwrap: {len(v_d)} → {len(v_u)} verts after seam splits ({time.time()-t1:.1f}s)", flush=True)
    timings["unwrap"] = time.time() - t1

    # 3. Bake atlas. With many tiny charts, xatlas often grows the atlas above the
    # requested resolution to fit them. Bake at xatlas's chosen size so chart islands
    # get the pixels they need; downsample to `atlas_size` after for shipping.
    t2 = time.time()
    bake_size = max(atlas_size, atlas_pixels)
    print(f"      bake_atlas: {len(f_u)} faces → {bake_size}x{bake_size} atlas "
          f"(xatlas chose {atlas_pixels}, target {atlas_size}, mode={bake_mode})", flush=True)
    if bake_mode == "bvh":
        # `vertices`, `faces`, `vertex_attrs` here are the ORIGINAL pre-decimation
        # mesh; v_u is the simplified mesh's vertex positions (split at chart seams
        # by xatlas). Per-texel rasterize on v_u/f_u, project each texel back to
        # the original mesh for higher attribute density.
        atlas, coverage = atlas_mod.bake_atlas_bvh(
            f_u, uvs, v_u,
            np.asarray(vertices, dtype=np.float32),
            np.asarray(faces, dtype=np.int64),
            np.asarray(vertex_attrs, dtype=np.float32),
            atlas_size=bake_size,
        )
    else:
        atlas, coverage = atlas_mod.bake_atlas(f_u, uvs, a_u, atlas_size=bake_size)
    if fill_seams:
        # Telea inpaint per channel — same primitive upstream uses (cv2.inpaint
        # in `o_voxel.postprocess.to_glb`). Required: without it, uncovered
        # pixels stay at roughness=0 and render as a black mirror.
        atlas = atlas_mod.inpaint_atlas(atlas, coverage)

    if bake_size != atlas_size:
        from PIL import Image
        # Resize each channel via PIL bilinear; PBR has 6 channels so split + recombine.
        chans = [Image.fromarray((atlas[:, :, c] * 255).clip(0, 255).astype(np.uint8))
                 .resize((atlas_size, atlas_size), Image.BILINEAR)
                 for c in range(atlas.shape[-1])]
        atlas = np.stack([np.asarray(im).astype(np.float32) / 255.0 for im in chans], axis=-1)
        print(f"      downsample atlas {bake_size} → {atlas_size}", flush=True)

    timings["bake"] = time.time() - t2
    print(f"      bake_atlas: done ({timings['bake']:.1f}s)")

    # 4. Pack as glTF PBR. Trellis attrs are stored in sRGB-interpretable
    # space already; matching upstream o-voxel we skip the linear→sRGB encode
    # (A/B'd against shivampkumar/trellis-mac's gamma=1/2.2: that variant
    # washed out — see artifacts/sample_pbr_atlas_v11_gamma.glb).
    t3 = time.time()
    base_color = np.clip(atlas[:, :, [0, 1, 2, 5]], 0.0, 1.0)        # RGBA
    metallic = np.clip(atlas[:, :, 3], 0.0, 1.0)                     # B-channel of metallicRoughness
    roughness = np.clip(atlas[:, :, 4], 0.0, 1.0)                    # G-channel
    metallic_roughness = np.zeros((atlas_size, atlas_size, 3), dtype=np.float32)
    metallic_roughness[:, :, 1] = roughness
    metallic_roughness[:, :, 2] = metallic

    from PIL import Image
    bc_img = Image.fromarray((base_color * 255).clip(0, 255).astype(np.uint8))
    mr_img = Image.fromarray((metallic_roughness * 255).clip(0, 255).astype(np.uint8))

    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=bc_img,
        metallicRoughnessTexture=mr_img,
        metallicFactor=1.0,
        roughnessFactor=1.0,
        alphaMode="OPAQUE",
        doubleSided=True,
    )
    # Match upstream o_voxel.postprocess.to_glb's coordinate-system conversion:
    # Y↔Z swap with Y-negation (model coord-frame → glTF Y-up frame), and
    # UV V-flip (image V=0 is top-row, glTF V=0 is bottom of mesh).
    # Without these, the mesh renders rotated 90° and texels map to wrong faces.
    v_out = np.asarray(v_u, dtype=np.float32).copy()
    v_out[:, 1], v_out[:, 2] = v_out[:, 2].copy(), -v_out[:, 1].copy()
    uvs_out = np.asarray(uvs, dtype=np.float32).copy()
    uvs_out[:, 1] = 1.0 - uvs_out[:, 1]

    visual = trimesh.visual.TextureVisuals(uv=uvs_out, material=material)
    mesh = trimesh.Trimesh(vertices=v_out, faces=f_u, visual=visual, process=False)
    mesh.export(str(out_path))
    timings["write"] = time.time() - t3

    timings["total"] = time.time() - t0
    return timings

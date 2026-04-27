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
) -> dict:
    """Decimate → UV-unwrap → bake atlas → write textured glTF.

    `vertex_attrs` is `(V, 6)` in [0, 1] with channel layout:
        [0:3] base_color RGB, [3] metallic, [4] roughness, [5] alpha.
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
    print(f"      xatlas unwrap on {len(v_d)} verts...", flush=True)
    v_u, f_u, uvs, a_u = atlas_mod.unwrap(v_d, f_d, a_d)
    print(f"      xatlas: {len(v_d)} → {len(v_u)} verts after seam splits ({time.time()-t1:.1f}s)", flush=True)
    timings["unwrap"] = time.time() - t1

    # 3. Bake atlas
    t2 = time.time()
    print(f"      bake_atlas: {len(f_u)} faces → {atlas_size}x{atlas_size} atlas", flush=True)
    atlas = atlas_mod.bake_atlas(f_u, uvs, a_u, atlas_size=atlas_size)
    if fill_seams:
        # Compute coverage from non-zero alpha (channel 5) as a proxy
        coverage = atlas[:, :, 5] > 0.0
        atlas = atlas_mod.fill_atlas_seams(atlas, coverage, iterations=2)
    timings["bake"] = time.time() - t2
    print(f"      bake_atlas: done ({timings['bake']:.1f}s)")

    # 4. Pack as glTF PBR
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
        alphaMode="MASK",
        alphaCutoff=0.5,
        doubleSided=True,
    )
    visual = trimesh.visual.TextureVisuals(uv=uvs, material=material)
    mesh = trimesh.Trimesh(vertices=v_u, faces=f_u, visual=visual, process=False)
    mesh.export(str(out_path))
    timings["write"] = time.time() - t3

    timings["total"] = time.time() - t0
    return timings

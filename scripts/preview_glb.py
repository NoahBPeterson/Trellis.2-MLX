"""Render a mesh preview to PNG using trimesh + pyglet headless."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", type=Path, default=Path("artifacts/sample.glb"))
    p.add_argument("--out", type=Path, default=Path("artifacts/sample_preview.png"))
    args = p.parse_args()

    scene_or_mesh = trimesh.load_mesh(args.mesh, force="mesh")
    mesh = scene_or_mesh if isinstance(scene_or_mesh, trimesh.Trimesh) else trimesh.util.concatenate(list(scene_or_mesh.dump().geometry.values()))

    # Compute a camera pose from the mesh extents
    bbox = mesh.bounds  # (2, 3)
    center = mesh.centroid
    extents = mesh.extents
    # Place camera along +x looking toward origin
    distance = float(extents.max()) * 3.0
    cam_pos = np.array([distance, distance * 0.5, distance])
    scene = mesh.scene()
    scene.camera_transform = scene.camera.look_at(points=np.array([bbox]).reshape(-1, 3), rotation=None)

    try:
        png = scene.save_image(resolution=(512, 512), background=(255, 255, 255))
        args.out.write_bytes(png)
        print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")
    except Exception as e:
        print(f"render failed ({e}); saving wireframe stats instead")
        print(f"  verts: {mesh.vertices.shape[0]}, faces: {mesh.faces.shape[0]}")
        print(f"  bbox: {mesh.bounds.tolist()}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

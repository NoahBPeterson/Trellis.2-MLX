"""Basic mesh export to GLB via trimesh.

v1 writes the raw triangle mesh with vertex colors (decimation, UV-atlas, and
material baking are deferred until we have per-voxel PBR attributes wired in
from the texture VAE decoder).
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
    """Write `(V, F)` as a GLB. `vertex_colors` is optional (V, 3|4) in [0, 1].

    Requires `trimesh` (install via `uv sync --extra postprocess`).
    """
    import trimesh

    mesh = trimesh.Trimesh(vertices=np.asarray(vertices, dtype=np.float32), faces=np.asarray(faces, dtype=np.int32))
    if vertex_colors is not None:
        vc = np.asarray(vertex_colors, dtype=np.float32)
        if vc.ndim == 2 and vc.shape[1] == 3:
            alpha = np.ones((vc.shape[0], 1), dtype=np.float32)
            vc = np.concatenate([vc, alpha], axis=1)
        mesh.visual.vertex_colors = (vc * 255).clip(0, 255).astype(np.uint8)
    mesh.export(str(out_path))

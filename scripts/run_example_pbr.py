"""End-to-end image → textured GLB via Trellis2ImageTo3DPipelineMLX.

Runs all 7 stages (preprocess + DINOv3 + SS flow + shape flow + shape VAE +
texture flow + texture VAE) and writes a vertex-colored GLB.

NOTE: This is the v1 PBR exporter. Each vertex carries its own RGBA color
sampled from per-voxel PBR. A proper UV-atlas baker (Phase 6E) is more compact
on decimated meshes but vertex colors are simpler and let us verify the
texture pipeline produces reasonable PBR before the atlas bake lands.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trellis2_mlx.pipeline import Trellis2ImageTo3DPipelineMLX


def export_vertex_colored_glb(verts: np.ndarray, faces: np.ndarray,
                              vertex_attrs: np.ndarray, out_path: Path) -> None:
    """Write a GLB where each vertex carries its sampled PBR base color (+ alpha).
    Metallic and roughness are dropped (vertex-color GLBs only support RGBA).
    """
    rgb = np.clip(vertex_attrs[:, :3], 0.0, 1.0)
    alpha = np.clip(vertex_attrs[:, 5:6], 0.0, 1.0)
    rgba = np.concatenate([rgb, alpha], axis=-1)
    rgba_u8 = (rgba * 255).astype(np.uint8)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=rgba_u8, process=False)
    mesh.export(out_path)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, default=ROOT / "upstream/assets/example_image/T.png")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/sample_pbr.glb")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pipeline-type", type=str, default="512")
    p.add_argument("--dino-device", type=str, default="cpu")
    p.add_argument("--rembg-device", type=str, default="auto")
    p.add_argument("--dit-dtype", type=str, default="float16", choices=["bfloat16", "float16"])
    args = p.parse_args()

    print(f"Loading PBR pipeline ({args.pipeline_type}, dit_dtype={args.dit_dtype})...")
    t0 = time.time()
    pipe = Trellis2ImageTo3DPipelineMLX.from_pretrained(
        ckpt_dir=ROOT / "ckpts",
        pipeline_json=ROOT / "weights" / "pipeline.json",
        pipeline_type=args.pipeline_type,
        dino_device=args.dino_device,
        rembg_device=args.rembg_device,
        dit_compute_dtype=args.dit_dtype,
        with_pbr=True,
    )
    print(f"loaded in {time.time() - t0:.1f}s")

    img = Image.open(args.image)
    print(f"\nGenerating textured mesh from {args.image.name}...")
    t0 = time.time()
    V, F, vertex_attrs = pipe.run(img, seed=args.seed)
    print(f"\ntotal generation: {time.time() - t0:.1f}s")

    print(f"\nVertex attr summary:")
    print(f"  base_color RGB:  min={vertex_attrs[:, :3].min(0).round(3).tolist()}  max={vertex_attrs[:, :3].max(0).round(3).tolist()}  mean={vertex_attrs[:, :3].mean(0).round(3).tolist()}")
    print(f"  metallic     :  min={vertex_attrs[:, 3].min():.3f}  max={vertex_attrs[:, 3].max():.3f}  mean={vertex_attrs[:, 3].mean():.3f}")
    print(f"  roughness    :  min={vertex_attrs[:, 4].min():.3f}  max={vertex_attrs[:, 4].max():.3f}  mean={vertex_attrs[:, 4].mean():.3f}")
    print(f"  alpha        :  min={vertex_attrs[:, 5].min():.3f}  max={vertex_attrs[:, 5].max():.3f}  mean={vertex_attrs[:, 5].mean():.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    export_vertex_colored_glb(V, F, vertex_attrs, args.out)
    print(f"\nwrote {args.out} ({args.out.stat().st_size / 1024 / 1024:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

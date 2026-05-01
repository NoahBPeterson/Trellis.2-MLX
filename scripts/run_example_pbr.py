"""End-to-end image → textured GLB via Trellis2ImageTo3DPipelineMLX.

Runs all 7 stages (preprocess + DINOv3 + SS flow + shape flow + shape VAE +
texture flow + texture VAE) and the o_voxel-replacement post-processing
(decimate → UV unwrap → atlas bake → glTF PBRMaterial GLB).

Use --vertex-colors to skip the atlas bake and ship a quick vertex-colored
GLB instead (faster to produce, larger to ship, but renders fine in viewers
that support vertex colors).
"""
from __future__ import annotations

import argparse
import faulthandler
import sys
import time
from pathlib import Path

# Surface C-level crashes (xatlas segfaults, native lib aborts) with a Python traceback
faulthandler.enable()

import numpy as np
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trellis2_mlx.pipeline import Trellis2ImageTo3DPipelineMLX


def export_vertex_colored_glb(verts: np.ndarray, faces: np.ndarray,
                              vertex_attrs: np.ndarray, out_path: Path) -> None:
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
    p.add_argument("--dit-dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"],
                   help="bfloat16 matches upstream's training/inference dtype (default). "
                        "float16 has a finer grid step than bf16 — produces sharper-precision "
                        "intermediates that diverge from upstream's bf16-trained behavior, "
                        "flipping ~6%% of borderline-occupancy voxels in SS sampling.")
    # Default is the full atlas path. --vertex-colors opts out and ships
    # a vertex-colored GLB instead (skips decimate+UV unwrap+bake).
    p.add_argument("--vertex-colors", action="store_true",
                   help="Skip UV unwrap + atlas bake; write a vertex-colored GLB instead.")
    p.add_argument("--target-faces", type=int, default=500_000,
                   help="Decimation target for --atlas (fast_simplification preserves topology, "
                        "actual count may be much higher).")
    p.add_argument("--atlas-size", type=int, default=2048,
                   help="Atlas resolution for --atlas. Bake time scales ~linearly.")
    p.add_argument("--cache", type=Path, default=None,
                   help="Save (V, F, vertex_attrs) to this .npz after DiT pipeline (skips post-processing).")
    p.add_argument("--load-cache", type=Path, default=None,
                   help="Load (V, F, vertex_attrs) from .npz, skip DiT pipeline. Useful for iterating on atlas bake.")
    args = p.parse_args()

    if args.load_cache is not None and args.load_cache.exists():
        print(f"Loading cached (V, F, vertex_attrs) from {args.load_cache}")
        npz = np.load(args.load_cache)
        V, F, vertex_attrs = npz["V"], npz["F"], npz["vertex_attrs"]
        print(f"loaded V={V.shape} F={F.shape} attrs={vertex_attrs.shape}")
    else:
        print(f"Loading PBR pipeline ({args.pipeline_type}, dit_dtype={args.dit_dtype})...")
        t0 = time.time()
        pipe = Trellis2ImageTo3DPipelineMLX.from_pretrained(
            ckpt_dir=ROOT / "weights" / "ckpts",
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

        if args.cache is not None:
            args.cache.parent.mkdir(parents=True, exist_ok=True)
            extras = getattr(pipe, "_last_intermediates", {}) or {}
            np.savez_compressed(args.cache, V=V, F=F, vertex_attrs=vertex_attrs, **extras)
            print(f"cached intermediates to {args.cache}  "
                  f"(keys: {sorted(['V', 'F', 'vertex_attrs'] + list(extras.keys()))})")

    print(f"\nVertex attr summary:")
    print(f"  base_color RGB:  min={vertex_attrs[:, :3].min(0).round(3).tolist()}  max={vertex_attrs[:, :3].max(0).round(3).tolist()}  mean={vertex_attrs[:, :3].mean(0).round(3).tolist()}")
    print(f"  metallic     :  min={vertex_attrs[:, 3].min():.3f}  max={vertex_attrs[:, 3].max():.3f}  mean={vertex_attrs[:, 3].mean():.3f}")
    print(f"  roughness    :  min={vertex_attrs[:, 4].min():.3f}  max={vertex_attrs[:, 4].max():.3f}  mean={vertex_attrs[:, 4].mean():.3f}")
    print(f"  alpha        :  min={vertex_attrs[:, 5].min():.3f}  max={vertex_attrs[:, 5].max():.3f}  mean={vertex_attrs[:, 5].mean():.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.vertex_colors:
        print(f"\n[8/8] vertex-color GLB write")
        t0 = time.time()
        export_vertex_colored_glb(V, F, vertex_attrs, args.out)
        print(f"      +{time.time()-t0:.1f}s")
    else:
        from trellis2_mlx.postprocess.glb_export import export_pbr_glb
        print(f"\n[8/8] decimate → UV unwrap → atlas bake → PBR GLB")
        timings = export_pbr_glb(
            V, F, vertex_attrs, args.out,
            target_faces=args.target_faces,
            atlas_size=args.atlas_size,
        )
        print(f"      decimate {timings['decimate']:.1f}s · unwrap {timings['unwrap']:.1f}s · "
              f"bake {timings['bake']:.1f}s · write {timings['write']:.1f}s · "
              f"total post {timings['total']:.1f}s")
    print(f"\nwrote {args.out} ({args.out.stat().st_size / 1024 / 1024:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

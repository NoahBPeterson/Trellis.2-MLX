"""End-to-end: image → GLB via Trellis2ImageTo3DPipelineMLX.

Usage:
    uv run python scripts/run_example.py [--image <path>] [--out <path>] [--seed <int>]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trellis2_mlx.pipeline import Trellis2ImageTo3DPipelineMLX
from trellis2_mlx.postprocess.glb_export import export_mesh_glb


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, default=ROOT / "upstream/assets/example_image/T.png")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/sample.glb")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pipeline-type", type=str, default="512")
    p.add_argument("--dino-device", type=str, default="cpu")
    p.add_argument("--dit-dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"],
                   help="Compute dtype for the flow DiT weights. fp16 is ~25%% faster but introduces numerical drift; verify mesh quality before relying on it.")
    args = p.parse_args()

    print(f"Loading pipeline from {ROOT / 'ckpts'} ({args.pipeline_type}, dit_dtype={args.dit_dtype})...")
    t0 = time.time()
    pipe = Trellis2ImageTo3DPipelineMLX.from_pretrained(
        ckpt_dir=ROOT / "ckpts",
        pipeline_json=ROOT / "weights" / "pipeline.json",
        pipeline_type=args.pipeline_type,
        dino_device=args.dino_device,
        dit_compute_dtype=args.dit_dtype,
    )
    print(f"loaded in {time.time() - t0:.1f}s")

    img = Image.open(args.image)
    print(f"\nGenerating from {args.image.name}...")
    t0 = time.time()
    V, F = pipe.run(img, seed=args.seed)
    print(f"\ntotal generation: {time.time() - t0:.1f}s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    export_mesh_glb(V, F, args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Render 3 orthogonal max-projections of the SS occupancy to a single PNG grid.

The SS decoder output has 3 spatial axes we've been calling (axis0, axis1, axis2).
Max-projection along each gives a 2D silhouette along that axis — for a letter T
one of these should clearly show the T shape.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trellis2_mlx.image_cond import DinoV3FeatureExtractor
from trellis2_mlx.models.flow_dit import SparseStructureFlowModel
from trellis2_mlx.models.ss_decoder import SparseStructureDecoder
from trellis2_mlx.preprocess import preprocess_image
from trellis2_mlx.samplers import FlowEulerGuidanceIntervalSampler


def main() -> int:
    image_path = ROOT / "upstream/assets/example_image/T.png"
    out_path = ROOT / "artifacts/ss_projections.png"
    pipe_cfg = json.loads((ROOT / "weights/pipeline.json").read_text())["args"]

    ss_cfg = json.loads((ROOT / "ckpts/ss_flow_img_dit_1_3B_64.config.json").read_text())
    ss_flow = SparseStructureFlowModel(**{k: v for k, v in ss_cfg["args"].items() if k not in ("initialization", "dtype")})
    ss_flow.load_weights(str(ROOT / "ckpts/ss_flow_img_dit_1_3B_64.safetensors"))
    dec_cfg = json.loads((ROOT / "ckpts/ss_dec_conv3d_16l8.config.json").read_text())
    ss_dec = SparseStructureDecoder(**dec_cfg["args"])
    ss_dec.load_weights(str(ROOT / "ckpts/ss_dec_conv3d_16l8.safetensors"))
    dino = DinoV3FeatureExtractor(pipe_cfg["image_cond_model"]["args"]["model_name"], image_size=512, device="cpu")

    img = Image.open(image_path)
    pre = preprocess_image(img)
    cond = dino([pre])
    neg = mx.zeros_like(cond)
    sampler = FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
    mx.random.seed(42)
    noise = mx.random.normal((1, 8, 16, 16, 16))
    p = pipe_cfg["sparse_structure_sampler"]["params"]
    z_s = sampler.sample(ss_flow, noise, cond=cond, neg_cond=neg,
                         steps=p["steps"], guidance_strength=p["guidance_strength"],
                         guidance_interval=tuple(p["guidance_interval"]),
                         guidance_rescale=p["guidance_rescale"], rescale_t=p["rescale_t"])["samples"]
    occ = ss_dec(z_s)
    mx.eval(occ)
    binary = np.asarray(occ > 0)[0, 0].astype(np.uint8)  # (64, 64, 64)
    print(f"occupancy density {binary.mean():.3%}, active {int(binary.sum())}")

    # Three max-projections
    proj0 = binary.max(axis=0)  # view along axis0 → (64, 64) in (axis1, axis2)
    proj1 = binary.max(axis=1)  # view along axis1 → (64, 64) in (axis0, axis2)
    proj2 = binary.max(axis=2)  # view along axis2 → (64, 64) in (axis0, axis1)

    # Also show the input image (resized) for comparison
    pre_small = np.array(pre.resize((64, 64), Image.LANCZOS).convert("L"))

    def up(img):
        im = Image.fromarray(img * 255)
        return np.array(im.resize((256, 256), Image.NEAREST))

    def up_gray(img):
        im = Image.fromarray(img)
        return np.array(im.resize((256, 256), Image.LANCZOS))

    grid = np.concatenate([
        np.concatenate([up_gray(pre_small), up(proj0)], axis=1),
        np.concatenate([up(proj1), up(proj2)], axis=1),
    ], axis=0)
    Image.fromarray(grid).save(out_path)
    print(f"wrote {out_path} (top-left: input, top-right: proj along axis0, bot-left: proj along axis1, bot-right: proj along axis2)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

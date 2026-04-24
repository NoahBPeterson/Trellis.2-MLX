"""Profile the shape VAE decoder to find CPU bottlenecks.

Runs the VAE on a realistic shape_slat input (sampled from the real flow on T.png)
with per-stage timing plus a per-function breakdown via cProfile.
"""
from __future__ import annotations

import cProfile
import json
import pstats
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trellis2_mlx.image_cond import DinoV3FeatureExtractor
from trellis2_mlx.models.flow_dit import SLatFlowModel, SparseStructureFlowModel
from trellis2_mlx.models.sparse_vae import FlexiDualGridVaeDecoder
from trellis2_mlx.models.ss_decoder import SparseStructureDecoder
from trellis2_mlx.modules.sparse_tensor import SparseTensor
from trellis2_mlx.preprocess import preprocess_image
from trellis2_mlx.samplers import FlowEulerGuidanceIntervalSampler


def _build_shape_slat():
    """Reproduce the T.png pipeline up through shape SLat sampling, return the slat."""
    pipe_cfg = json.loads((ROOT / "weights/pipeline.json").read_text())["args"]
    # SS flow + decoder
    ss_cfg = json.loads((ROOT / "ckpts/ss_flow_img_dit_1_3B_64.config.json").read_text())
    ss_flow = SparseStructureFlowModel(**{k: v for k, v in ss_cfg["args"].items() if k not in ("initialization", "dtype")})
    ss_flow.load_weights(str(ROOT / "ckpts/ss_flow_img_dit_1_3B_64.safetensors"))
    dec_cfg = json.loads((ROOT / "ckpts/ss_dec_conv3d_16l8.config.json").read_text())
    ss_dec = SparseStructureDecoder(**dec_cfg["args"])
    ss_dec.load_weights(str(ROOT / "ckpts/ss_dec_conv3d_16l8.safetensors"))
    sl_cfg = json.loads((ROOT / "ckpts/slat_flow_img2shape_dit_1_3B_512.config.json").read_text())
    sl_flow = SLatFlowModel(**{k: v for k, v in sl_cfg["args"].items() if k not in ("initialization", "dtype")})
    sl_flow.load_weights(str(ROOT / "ckpts/slat_flow_img2shape_dit_1_3B_512.safetensors"))
    dino = DinoV3FeatureExtractor(pipe_cfg["image_cond_model"]["args"]["model_name"], image_size=512, device="cpu")

    img = Image.open(ROOT / "upstream/assets/example_image/T.png")
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
    binary = np.asarray(occ > 0)[0, 0]
    arr = binary.reshape(32, 2, 32, 2, 32, 2).max(axis=(1, 3, 5))
    ys, xs, zs = np.where(arr)
    n = ys.size
    coords = mx.array(np.stack([np.zeros(n, dtype=np.int32), ys.astype(np.int32), xs.astype(np.int32), zs.astype(np.int32)], axis=-1))

    mx.random.seed(43)
    noise_feats = mx.random.normal((n, sl_flow.in_channels))
    noise_t = SparseTensor(feats=noise_feats, coords=coords, spatial_shape=(32, 32, 32))
    p = pipe_cfg["shape_slat_sampler"]["params"]
    slat = sampler.sample(sl_flow, noise_t, cond=cond, neg_cond=neg,
                          steps=p["steps"], guidance_strength=p["guidance_strength"],
                          guidance_interval=tuple(p["guidance_interval"]),
                          guidance_rescale=p["guidance_rescale"], rescale_t=p["rescale_t"])["samples"]
    mean = mx.array(pipe_cfg["shape_slat_normalization"]["mean"], dtype=mx.float32)
    std = mx.array(pipe_cfg["shape_slat_normalization"]["std"], dtype=mx.float32)
    slat = slat.replace(slat.feats * std + mean)
    mx.eval(slat.feats)
    return slat


def main() -> int:
    print("Building shape SLat (real pipeline up through shape flow)...")
    t0 = time.time()
    slat = _build_shape_slat()
    print(f"  shape_slat feats {tuple(slat.feats.shape)}  built in {time.time() - t0:.1f}s")

    dc_cfg = json.loads((ROOT / "ckpts/shape_dec_next_dc_f16c32.config.json").read_text())
    vae = FlexiDualGridVaeDecoder(**dc_cfg["args"])
    vae.load_weights(str(ROOT / "ckpts/shape_dec_next_dc_f16c32.safetensors"))
    vae.set_resolution(512)

    print("\nProfiling VAE decode (cProfile, top 20 cumulative)...")
    pr = cProfile.Profile()
    pr.enable()
    t0 = time.time()
    v, i, q, subs = vae.decode(slat, return_subs=True)
    mx.eval(v.feats, i.feats, q.feats)
    total = time.time() - t0
    pr.disable()
    print(f"VAE decode total: {total:.2f}s")
    print(f"  output voxels: {v.feats.shape[0]}")

    stats = pstats.Stats(pr).strip_dirs().sort_stats("cumulative")
    stats.print_stats(25)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

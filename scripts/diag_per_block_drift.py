"""Per-block per-channel drift diagnostic.

Loads our shape DiT, feeds it upstream's shape_slat as input (denormalized
then re-normalized to the model's expected input space), and runs ONE
forward pass with t=1.0 (start-of-sampling). Logs per-channel mean of the
hidden state after each transformer block.

If a single block introduces a sharp per-channel bias jump, the bug is in
that block's code (or its weights). If drift accumulates uniformly,
the bug is in the per-step DiT math.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trellis2_mlx.modules.sparse_tensor import SparseTensor
from trellis2_mlx.pipeline import Trellis2ImageTo3DPipelineMLX


def main() -> int:
    print("Loading our pipeline...")
    pipe = Trellis2ImageTo3DPipelineMLX.from_pretrained(
        ckpt_dir=ROOT / "weights" / "ckpts",
        pipeline_json=ROOT / "weights" / "pipeline.json",
        pipeline_type="512",
        dino_device="cpu",
        rembg_device="cpu",
        dit_compute_dtype="bfloat16",
        with_pbr=True,
    )
    model = pipe.shape_flow_hr  # SLatFlowModel for shape

    up = np.load(ROOT / "artifacts" / "upstream_ref.npz")
    cond_512 = mx.array(up["cond_512"].astype(np.float32))

    # Build noise input matching upstream's shape_slat coords. We can't reproduce
    # upstream's exact noise but we use deterministic zeros so per-block drift
    # is purely from weights and biases (not RNG).
    coords = mx.array(up["shape_slat_coords"].astype(np.int32))
    F = coords.shape[0]
    in_C = model.in_channels  # 32 for shape, 64 for tex
    x = SparseTensor(
        feats=mx.zeros((F, in_C), dtype=mx.float32),
        coords=coords,
        spatial_shape=(32, 32, 32),
    )
    t = mx.array([1000.0], dtype=mx.float32)

    # Reproduce model's __call__ inline so we can sample after each block.
    h_feats = model.input_layer(x.feats)
    t_emb = model.t_embedder(t)
    if model.share_mod:
        t_emb = model.adaLN_modulation(t_emb)
    phases = None
    if model.pe_mode == "rope":
        phases = model._rope(x.coords[:, 1:].astype(mx.float32))
        phases = phases[None]
    elif model.pe_mode == "ape":
        pe = model._ape(x.coords[:, 1:].astype(mx.float32))
        h_feats = h_feats + pe
    h = mx.expand_dims(h_feats, axis=0)  # (1, F, C)

    print(f"\nForward pass — zero input, cond_512 from upstream, t=1.0")
    print(f"  h after input_layer (B,F,C={tuple(h.shape)}):")
    h_np = np.asarray(h.astype(mx.float32))
    print(f"    mean per-channel: min={h_np.mean(axis=(0,1)).min():.4f}  "
          f"max={h_np.mean(axis=(0,1)).max():.4f}  "
          f"absmax={abs(h_np.mean(axis=(0,1))).max():.4f}")

    drift_history = []
    for i, block in enumerate(model.blocks):
        h = block(h, t_emb, cond_512, phases=phases)
        mx.eval(h)
        h_np = np.asarray(h.astype(mx.float32))
        per_ch_means = h_np.mean(axis=(0, 1))  # (C,)
        absmax = float(np.abs(per_ch_means).max())
        argmax = int(np.argmax(np.abs(per_ch_means)))
        drift_history.append((i, absmax, argmax, per_ch_means[argmax]))
        if i in (0, 1, 2, 5, 10, 15, 20, 25, 29):
            top5 = np.argsort(-np.abs(per_ch_means))[:5]
            print(f"  after block {i:2d}: absmax_per_ch_mean={absmax:.3f} "
                  f"(ch{argmax})  top5: {[(int(c), float(per_ch_means[c])) for c in top5]}")

    # Final layer norm + out
    from trellis2_mlx.models.flow_dit import _layer_norm_last
    h = _layer_norm_last(h)
    h = model.out_layer(h)
    mx.eval(h)
    h_np = np.asarray(h.astype(mx.float32))
    print(f"\nAfter final LN + out_layer (shape {h.shape}):")
    pcm = h_np.mean(axis=(0, 1))
    print(f"  per-channel means: min={pcm.min():+.3f}  max={pcm.max():+.3f}")
    print(f"  full per-channel:  {pcm.round(3).tolist()}")
    print(f"\n(For reference, upstream voxel_attrs after VAE → shape_slat passthrough has channel-mean range ~ -3 to +4 across the 32 channels)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

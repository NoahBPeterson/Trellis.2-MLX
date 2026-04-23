"""Validate MLX SparseStructureFlowModel vs upstream torch-CPU on a small config.

A full 1.3B DiT run on CPU is too slow for tests, so we test a 3-block / 256-ch
tiny config that exercises the same code paths (RoPE/APE, share_mod, qk_rms_norm).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

os.environ["ATTN_BACKEND"] = "naive"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "upstream"))

from trellis2.models.sparse_structure_flow import SparseStructureFlowModel as TorchSS
from trellis2_mlx.models.flow_dit import SparseStructureFlowModel as MlxSS


def _copy_ss_weights(t_model: TorchSS, m_model: MlxSS) -> None:
    sd = t_model.state_dict()

    def set_mx(owner, attr_path, tensor):
        obj = owner
        parts = attr_path.split(".")
        for p in parts[:-1]:
            if p.isdigit():
                obj = obj.layers[int(p)]
            else:
                obj = getattr(obj, p)
        last = parts[-1]
        if last.isdigit():
            obj.layers[int(last)] = mx.array(tensor.numpy())
        else:
            setattr(obj, last, mx.array(tensor.numpy()))

    # Timestep embedder
    m_model.t_embedder.mlp.layers[0].weight = mx.array(sd["t_embedder.mlp.0.weight"].numpy())
    m_model.t_embedder.mlp.layers[0].bias = mx.array(sd["t_embedder.mlp.0.bias"].numpy())
    m_model.t_embedder.mlp.layers[2].weight = mx.array(sd["t_embedder.mlp.2.weight"].numpy())
    m_model.t_embedder.mlp.layers[2].bias = mx.array(sd["t_embedder.mlp.2.bias"].numpy())

    # Shared adaLN
    if m_model.share_mod:
        m_model.adaLN_modulation.layers[1].weight = mx.array(sd["adaLN_modulation.1.weight"].numpy())
        m_model.adaLN_modulation.layers[1].bias = mx.array(sd["adaLN_modulation.1.bias"].numpy())

    # Input / output
    m_model.input_layer.weight = mx.array(sd["input_layer.weight"].numpy())
    m_model.input_layer.bias = mx.array(sd["input_layer.bias"].numpy())
    m_model.out_layer.weight = mx.array(sd["out_layer.weight"].numpy())
    m_model.out_layer.bias = mx.array(sd["out_layer.bias"].numpy())

    # Per-block
    for i, mb in enumerate(m_model.blocks):
        prefix = f"blocks.{i}."
        if mb.norm2.affine:
            mb.norm2.weight = mx.array(sd[prefix + "norm2.weight"].numpy())
            mb.norm2.bias = mx.array(sd[prefix + "norm2.bias"].numpy())
        mb.self_attn.to_qkv.weight = mx.array(sd[prefix + "self_attn.to_qkv.weight"].numpy())
        mb.self_attn.to_qkv.bias = mx.array(sd[prefix + "self_attn.to_qkv.bias"].numpy())
        if mb.self_attn.qk_rms_norm:
            mb.self_attn.q_rms_norm.gamma = mx.array(sd[prefix + "self_attn.q_rms_norm.gamma"].numpy())
            mb.self_attn.k_rms_norm.gamma = mx.array(sd[prefix + "self_attn.k_rms_norm.gamma"].numpy())
        mb.self_attn.to_out.weight = mx.array(sd[prefix + "self_attn.to_out.weight"].numpy())
        mb.self_attn.to_out.bias = mx.array(sd[prefix + "self_attn.to_out.bias"].numpy())
        mb.cross_attn.to_q.weight = mx.array(sd[prefix + "cross_attn.to_q.weight"].numpy())
        mb.cross_attn.to_q.bias = mx.array(sd[prefix + "cross_attn.to_q.bias"].numpy())
        mb.cross_attn.to_kv.weight = mx.array(sd[prefix + "cross_attn.to_kv.weight"].numpy())
        mb.cross_attn.to_kv.bias = mx.array(sd[prefix + "cross_attn.to_kv.bias"].numpy())
        if mb.cross_attn.qk_rms_norm:
            mb.cross_attn.q_rms_norm.gamma = mx.array(sd[prefix + "cross_attn.q_rms_norm.gamma"].numpy())
            mb.cross_attn.k_rms_norm.gamma = mx.array(sd[prefix + "cross_attn.k_rms_norm.gamma"].numpy())
        mb.cross_attn.to_out.weight = mx.array(sd[prefix + "cross_attn.to_out.weight"].numpy())
        mb.cross_attn.to_out.bias = mx.array(sd[prefix + "cross_attn.to_out.bias"].numpy())
        mb.mlp.mlp.layers[0].weight = mx.array(sd[prefix + "mlp.mlp.0.weight"].numpy())
        mb.mlp.mlp.layers[0].bias = mx.array(sd[prefix + "mlp.mlp.0.bias"].numpy())
        mb.mlp.mlp.layers[2].weight = mx.array(sd[prefix + "mlp.mlp.2.weight"].numpy())
        mb.mlp.mlp.layers[2].bias = mx.array(sd[prefix + "mlp.mlp.2.bias"].numpy())
        if mb.share_mod:
            mb.modulation = mx.array(sd[prefix + "modulation"].numpy())
        else:
            mb.adaLN_modulation.layers[1].weight = mx.array(sd[prefix + "adaLN_modulation.1.weight"].numpy())
            mb.adaLN_modulation.layers[1].bias = mx.array(sd[prefix + "adaLN_modulation.1.bias"].numpy())


def test_ss_flow_small_matches_torch():
    """Tiny 3-block / 256-ch / RoPE / qk_rms / share_mod config — same shape as real thing."""
    cfg = dict(
        resolution=4, in_channels=8, out_channels=8,
        model_channels=256, cond_channels=192,
        num_blocks=3, num_heads=4, mlp_ratio=4.0,
        pe_mode="rope", share_mod=True,
        qk_rms_norm=True, qk_rms_norm_cross=True,
        initialization="scaled",
        dtype="float32",
    )
    torch.manual_seed(0)
    t_model = TorchSS(**cfg)
    t_model.eval()
    # 'scaled' init zeroes out_layer and adaLN tails — re-randomize so forward is non-trivial
    with torch.no_grad():
        torch.manual_seed(42)
        for m in (t_model.out_layer, t_model.adaLN_modulation[-1]):
            m.weight.normal_(mean=0, std=0.1)
            m.bias.normal_(mean=0, std=0.1)
        for blk in t_model.blocks:
            # Re-randomize modulation so the block actually modulates
            blk.modulation.normal_(mean=0, std=0.05)

    m_model = MlxSS(**{k: v for k, v in cfg.items() if k not in ("initialization", "dtype")})
    _copy_ss_weights(t_model, m_model)

    rng = np.random.default_rng(0)
    B = 1
    x = rng.standard_normal((B, 8, 4, 4, 4)).astype(np.float32)
    t = np.array([100.0], dtype=np.float32)
    cond = rng.standard_normal((B, 20, 192)).astype(np.float32)

    with torch.no_grad():
        t_out = t_model(torch.from_numpy(x), torch.from_numpy(t), torch.from_numpy(cond)).numpy()
    m_out = np.asarray(m_model(mx.array(x), mx.array(t), mx.array(cond)))

    diff = np.abs(t_out - m_out).max()
    rel = diff / (np.abs(t_out).max() + 1e-8)
    print(f"SS flow (3 blocks, 256ch) fp32 max-abs diff: {diff:.2e}  rel: {rel:.2e}")
    print(f"Output shape torch={t_out.shape} mlx={m_out.shape}")
    assert diff < 1e-3, f"SS flow max diff {diff}"


if __name__ == "__main__":
    test_ss_flow_small_matches_torch()
    print("OK: SparseStructureFlowModel matches torch")

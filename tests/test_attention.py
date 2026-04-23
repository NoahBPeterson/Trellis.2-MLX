"""Numerics check: MLX MultiHeadAttention vs upstream torch MultiHeadAttention.

Validates all four relevant configurations:
  1. self-attention, qk_rms_norm=False, use_rope=False
  2. self-attention, qk_rms_norm=True,  use_rope=False
  3. self-attention, qk_rms_norm=True,  use_rope=True   (SS flow config)
  4. cross-attention, qk_rms_norm_cross=True           (image cond cross-attn)

For each config: same random weights on both sides, compare forward outputs.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "upstream"))

# Force naive SDPA backend in upstream so no flash-attn/xformers dependency
os.environ["ATTN_BACKEND"] = "naive"

from trellis2.modules.attention.modules import MultiHeadAttention as TorchMHA
from trellis2.modules.attention.rope import RotaryPositionEmbedder as TorchRoPE
from trellis2_mlx.modules.attention import MultiHeadAttention as MlxMHA
from trellis2_mlx.modules.rope import RotaryPositionEmbedder as MlxRoPE


def _copy_mha_weights(torch_mha: TorchMHA, mlx_mha: MlxMHA) -> None:
    """Copy torch MHA weights into the MLX module (in-place)."""
    sd = torch_mha.state_dict()
    if mlx_mha._type == "self":
        mlx_mha.to_qkv.weight = mx.array(sd["to_qkv.weight"].numpy())
        mlx_mha.to_qkv.bias = mx.array(sd["to_qkv.bias"].numpy())
    else:
        mlx_mha.to_q.weight = mx.array(sd["to_q.weight"].numpy())
        mlx_mha.to_q.bias = mx.array(sd["to_q.bias"].numpy())
        mlx_mha.to_kv.weight = mx.array(sd["to_kv.weight"].numpy())
        mlx_mha.to_kv.bias = mx.array(sd["to_kv.bias"].numpy())
    if mlx_mha.qk_rms_norm:
        mlx_mha.q_rms_norm.gamma = mx.array(sd["q_rms_norm.gamma"].numpy())
        mlx_mha.k_rms_norm.gamma = mx.array(sd["k_rms_norm.gamma"].numpy())
    mlx_mha.to_out.weight = mx.array(sd["to_out.weight"].numpy())
    mlx_mha.to_out.bias = mx.array(sd["to_out.bias"].numpy())


def _run_case(B, L, C, H, Ctx=None, qk_rms=False, rope=False, cross=False):
    torch.manual_seed(0)
    kwargs = dict(
        channels=C, num_heads=H,
        qk_rms_norm=qk_rms, qkv_bias=True,
        ctx_channels=Ctx,
    )
    if cross:
        kwargs.update(type="cross")
    else:
        kwargs.update(type="self", use_rope=rope)
    t_mha = TorchMHA(**kwargs, attn_mode="full")
    t_mha.eval()
    m_mha = MlxMHA(
        channels=C, num_heads=H,
        ctx_channels=Ctx,
        type="cross" if cross else "self",
        attn_mode="full",
        qkv_bias=True,
        use_rope=(rope and not cross),
        qk_rms_norm=qk_rms,
    )
    _copy_mha_weights(t_mha, m_mha)

    x = np.random.default_rng(0).standard_normal((B, L, C)).astype(np.float32)
    if cross:
        ctx = np.random.default_rng(1).standard_normal((B, L * 2, Ctx)).astype(np.float32)
    else:
        ctx = None

    phases_m = phases_t = None
    if rope:
        head_dim = C // H
        coords = np.random.default_rng(2).integers(0, 16, size=(L, 3)).astype(np.float32)
        t_rope = TorchRoPE(head_dim=head_dim, dim=3)
        phases_t = t_rope(torch.from_numpy(coords)).unsqueeze(0)  # (1, L, pairs)
        m_rope = MlxRoPE(head_dim=head_dim, dim=3)
        phases_m = mx.expand_dims(m_rope(mx.array(coords)), axis=0)

    # Torch forward
    with torch.no_grad():
        if cross:
            t_out = t_mha(torch.from_numpy(x), torch.from_numpy(ctx)).numpy()
        else:
            t_out = t_mha(torch.from_numpy(x), phases=phases_t).numpy()

    # MLX forward
    if cross:
        m_out = np.asarray(m_mha(mx.array(x), context=mx.array(ctx)))
    else:
        m_out = np.asarray(m_mha(mx.array(x), phases=phases_m))

    diff = np.abs(t_out - m_out).max()
    return diff, t_out, m_out


def test_self_plain():
    diff, _, _ = _run_case(B=2, L=32, C=256, H=4, qk_rms=False, rope=False)
    assert diff < 2e-4, f"self-attn plain max diff {diff}"
    print(f"self-attn plain fp32 max-abs diff: {diff:.2e}")


def test_self_qk_rms():
    diff, _, _ = _run_case(B=2, L=32, C=256, H=4, qk_rms=True, rope=False)
    assert diff < 2e-4, f"self-attn qk_rms max diff {diff}"
    print(f"self-attn qk_rms fp32 max-abs diff: {diff:.2e}")


def test_self_qk_rms_rope():
    diff, _, _ = _run_case(B=2, L=32, C=256, H=4, qk_rms=True, rope=True)
    assert diff < 2e-4, f"self-attn qk_rms+rope max diff {diff}"
    print(f"self-attn qk_rms+RoPE fp32 max-abs diff: {diff:.2e}")


def test_cross_qk_rms():
    diff, _, _ = _run_case(B=2, L=32, C=256, H=4, Ctx=192, qk_rms=True, cross=True)
    assert diff < 2e-4, f"cross-attn qk_rms max diff {diff}"
    print(f"cross-attn qk_rms fp32 max-abs diff: {diff:.2e}")


if __name__ == "__main__":
    test_self_plain()
    test_self_qk_rms()
    test_self_qk_rms_rope()
    test_cross_qk_rms()
    print("OK: MHA matches torch across all 4 configs")

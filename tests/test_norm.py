"""Validate MLX LayerNorm32 + MultiHeadRMSNorm against torch equivalents."""
from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trellis2_mlx.modules.norm import LayerNorm32, MultiHeadRMSNorm


def test_layernorm32_matches_torch():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 17, 64)).astype(np.float32)
    w = rng.standard_normal((64,)).astype(np.float32) * 0.1 + 1.0
    b = rng.standard_normal((64,)).astype(np.float32) * 0.01

    # torch reference
    t_out = F.layer_norm(torch.from_numpy(x), (64,), torch.from_numpy(w), torch.from_numpy(b), eps=1e-6).numpy()

    # MLX
    ln = LayerNorm32(64, eps=1e-6, affine=True)
    ln.weight = mx.array(w)
    ln.bias = mx.array(b)
    m_out = np.asarray(ln(mx.array(x)))

    diff = np.abs(t_out - m_out).max()
    assert diff < 1e-5, f"LayerNorm32 fp32 max diff {diff}"
    print(f"LayerNorm32 fp32 max-abs diff: {diff:.2e}")


def test_multihead_rms_norm_matches_torch():
    """Matches upstream MultiHeadRMSNorm forward:
        (F.normalize(x.float(), dim=-1) * gamma * sqrt(D)).to(x.dtype)
    """
    rng = np.random.default_rng(1)
    H, D = 12, 128
    x = rng.standard_normal((2, 17, H, D)).astype(np.float32)
    gamma = rng.standard_normal((H, D)).astype(np.float32) * 0.1 + 1.0

    # torch reference (literal upstream formula)
    t_x = torch.from_numpy(x).float()
    t_g = torch.from_numpy(gamma)
    t_out = (F.normalize(t_x, dim=-1) * t_g * (D**0.5)).numpy()

    rn = MultiHeadRMSNorm(D, H, eps=1e-6)
    rn.gamma = mx.array(gamma)
    m_out = np.asarray(rn(mx.array(x)))

    diff = np.abs(t_out - m_out).max()
    assert diff < 1e-4, f"MultiHeadRMSNorm max diff {diff}"
    print(f"MultiHeadRMSNorm fp32 max-abs diff: {diff:.2e}")


if __name__ == "__main__":
    test_layernorm32_matches_torch()
    test_multihead_rms_norm_matches_torch()
    print("OK: norms match torch")

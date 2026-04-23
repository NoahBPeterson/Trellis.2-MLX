"""Validate MLX RotaryPositionEmbedder against upstream torch implementation."""
from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "upstream"))

from trellis2.modules.attention.rope import RotaryPositionEmbedder as TorchRoPE
from trellis2_mlx.modules.rope import RotaryPositionEmbedder as MlxRoPE


def test_rope_3d_matches_torch():
    head_dim = 128
    H, N = 4, 32
    rng = np.random.default_rng(42)

    # 3D coordinates in [0, 16)
    coords = rng.integers(0, 16, size=(N, 3)).astype(np.float32)
    x = rng.standard_normal((1, N, H, head_dim)).astype(np.float32)

    # torch reference
    t_rope = TorchRoPE(head_dim=head_dim, dim=3, rope_freq=(1.0, 10000.0))
    t_phases = t_rope(torch.from_numpy(coords))  # (N, pairs) complex
    t_out = TorchRoPE.apply_rotary_embedding(torch.from_numpy(x), t_phases.unsqueeze(0)).numpy()

    # MLX
    m_rope = MlxRoPE(head_dim=head_dim, dim=3, rope_freq=(1.0, 10000.0))
    m_phases = m_rope(mx.array(coords))  # (N, pairs, 2)
    # Broadcast to (1, N, pairs, 2) to match batch of x
    m_phases_batched = mx.expand_dims(m_phases, axis=0)
    m_out = np.asarray(MlxRoPE.apply_rotary_embedding(mx.array(x), m_phases_batched))

    diff = np.abs(t_out - m_out).max()
    assert diff < 1e-4, f"RoPE max-abs diff {diff}"
    print(f"RoPE fp32 max-abs diff: {diff:.2e}")
    print(f"RoPE output shape torch={t_out.shape} mlx={m_out.shape}")


def test_rope_1d_matches_torch():
    """1D RoPE (like a normal text transformer)."""
    head_dim = 64
    N = 16
    rng = np.random.default_rng(1)
    coords = np.arange(N, dtype=np.float32).reshape(N, 1)
    x = rng.standard_normal((1, N, 2, head_dim)).astype(np.float32)

    t_rope = TorchRoPE(head_dim=head_dim, dim=1, rope_freq=(1.0, 10000.0))
    t_phases = t_rope(torch.from_numpy(coords))
    t_out = TorchRoPE.apply_rotary_embedding(torch.from_numpy(x), t_phases.unsqueeze(0)).numpy()

    m_rope = MlxRoPE(head_dim=head_dim, dim=1, rope_freq=(1.0, 10000.0))
    m_phases = m_rope(mx.array(coords))
    m_phases_batched = mx.expand_dims(m_phases, axis=0)
    m_out = np.asarray(MlxRoPE.apply_rotary_embedding(mx.array(x), m_phases_batched))

    diff = np.abs(t_out - m_out).max()
    assert diff < 1e-4, f"1D RoPE max-abs diff {diff}"
    print(f"1D RoPE fp32 max-abs diff: {diff:.2e}")


if __name__ == "__main__":
    test_rope_1d_matches_torch()
    test_rope_3d_matches_torch()
    print("OK: RoPE matches torch")

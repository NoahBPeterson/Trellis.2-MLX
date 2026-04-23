"""Absolute sinusoidal position embedding + timestep embedding.

Ports:
- AbsolutePositionEmbedder : upstream/trellis2/modules/transformer/blocks.py:8
- TimestepEmbedder         : upstream/trellis2/models/sparse_structure_flow.py:12
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class AbsolutePositionEmbedder(nn.Module):
    """Sinusoidal APE. Output: (N, channels).

    For a (N, in_channels) input, each input dim gets `freq_dim = channels // in_channels // 2`
    frequencies, then sin and cos are concatenated, and the output is zero-padded to `channels`.

    Matches upstream exactly for concat order: [sin(all dims flat), cos(all dims flat)].
    """

    def __init__(self, channels: int, in_channels: int = 3):
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.freq_dim = channels // in_channels // 2
        freqs = mx.arange(self.freq_dim, dtype=mx.float32) / self.freq_dim
        freqs = 1.0 / (10000**freqs)
        self.freqs = freqs

    def __call__(self, x: mx.array) -> mx.array:
        """x: (N, in_channels) -> (N, channels)."""
        assert x.shape[-1] == self.in_channels
        N = x.shape[0]
        flat = x.reshape(-1).astype(mx.float32)  # (N * in_channels,)
        out = flat[:, None] * self.freqs[None, :]  # (N * in_channels, freq_dim)
        out = mx.concatenate([mx.sin(out), mx.cos(out)], axis=-1)  # (N*IC, 2*freq_dim)
        out = out.reshape(N, -1)  # (N, in_channels * 2 * freq_dim)
        if out.shape[1] < self.channels:
            pad = self.channels - out.shape[1]
            out = mx.concatenate([out, mx.zeros((N, pad), dtype=mx.float32)], axis=-1)
        return out


class TimestepEmbedder(nn.Module):
    """Sinusoidal embedding -> MLP.

    Stored weights: mlp.0.{weight,bias}, mlp.2.{weight,bias}.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: mx.array, dim: int, max_period: int = 10000) -> mx.array:
        """t: (N,) -> (N, dim). Matches openai/glide sinusoidal embedding."""
        half = dim // 2
        freqs = mx.exp(
            -mx.log(mx.array(float(max_period))) * mx.arange(0, half, dtype=mx.float32) / half
        )  # (half,)
        args = t.astype(mx.float32)[:, None] * freqs[None, :]  # (N, half)
        emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        if dim % 2:
            emb = mx.concatenate([emb, mx.zeros((emb.shape[0], 1))], axis=-1)
        return emb

    def __call__(self, t: mx.array) -> mx.array:
        freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(freq)

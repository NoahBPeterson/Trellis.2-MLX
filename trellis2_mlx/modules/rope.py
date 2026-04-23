"""3D Rotary Position Embedding for MLX.

Ports upstream/trellis2/modules/attention/rope.py. MLX has no `view_as_complex`,
so phases are stored as stacked (cos, sin) real pairs and rotation is applied
as an explicit 2x2 matmul over consecutive feature pairs.

Shape conventions:
- `indices`: (..., N, dim) integer/float spatial positions (dim=3 for TRELLIS)
- `phases` : (..., N, head_dim // 2, 2)  where last axis = (cos, sin)
- features x passed to `apply_rotary_embedding` have shape (..., N, H, head_dim)
  and the last axis is interpreted as head_dim//2 pairs of (even, odd).

Matches the torch upstream's `view_as_complex(x.reshape(..., -1, 2))` pairing:
pairs are (x[..., 0], x[..., 1]), (x[..., 2], x[..., 3]), ...
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class RotaryPositionEmbedder(nn.Module):
    def __init__(
        self,
        head_dim: int,
        dim: int = 3,
        rope_freq: tuple[float, float] = (1.0, 10000.0),
    ):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even"
        self.head_dim = head_dim
        self.dim = dim
        self.rope_freq = rope_freq
        self.freq_dim = head_dim // 2 // dim
        # freqs: (freq_dim,) — rope_freq[0] * rope_freq[1] ** -(i/freq_dim)
        freqs = mx.arange(self.freq_dim, dtype=mx.float32) / self.freq_dim
        freqs = rope_freq[0] / (rope_freq[1] ** freqs)
        self.freqs = freqs  # non-trainable buffer

    def get_phases(self, indices: mx.array) -> mx.array:
        """indices: (..., N, dim) -> phases (..., N, head_dim//2, 2) as (cos,sin)."""
        assert indices.shape[-1] == self.dim, f"last dim must be {self.dim}, got {indices.shape[-1]}"
        # Flatten dim axis into the element axis: (..., N*dim) * freqs -> (..., N*dim, freq_dim)
        shape_prefix = indices.shape[:-1]  # (..., N)
        flat = indices.reshape(-1).astype(mx.float32)  # (prod*dim,)
        # Outer product -> (prod*dim, freq_dim)
        angles = flat[:, None] * self.freqs[None, :]
        # Reshape back: (..., N, dim * freq_dim)
        angles = angles.reshape(*shape_prefix, self.dim * self.freq_dim)
        pairs = self.head_dim // 2
        # Pad with zero-angles if dim * freq_dim < pairs
        if angles.shape[-1] < pairs:
            pad = pairs - angles.shape[-1]
            angles = mx.concatenate(
                [angles, mx.zeros((*shape_prefix, pad), dtype=mx.float32)],
                axis=-1,
            )
        # (..., N, pairs, 2) with (cos, sin)
        phases = mx.stack([mx.cos(angles), mx.sin(angles)], axis=-1)
        return phases

    def __call__(self, indices: mx.array) -> mx.array:
        return self.get_phases(indices)

    @staticmethod
    def apply_rotary_embedding(x: mx.array, phases: mx.array) -> mx.array:
        """Apply complex rotation to pairs of features.

        Args:
            x: (..., N, H, head_dim)  — features with final axis split into pairs
            phases: (..., N, head_dim // 2, 2) — (cos, sin) for each pair
        Returns:
            Same shape as x, same dtype.
        """
        orig = x.dtype
        x32 = x.astype(mx.float32)
        *prefix, N, H, D = x32.shape
        pairs = D // 2
        # Reshape to expose pairs: (..., N, H, pairs, 2) with (real, imag)
        xp = x32.reshape(*prefix, N, H, pairs, 2)
        # phases is (..., N, pairs, 2); broadcast heads axis
        cos = phases[..., 0:1]  # (..., N, pairs, 1)
        sin = phases[..., 1:2]
        cos = mx.expand_dims(cos, axis=-3)  # (..., N, 1, pairs, 1)
        sin = mx.expand_dims(sin, axis=-3)
        x_real = xp[..., 0:1]
        x_imag = xp[..., 1:2]
        # (a + ib) * (cos + i*sin) = (a*cos - b*sin) + i*(a*sin + b*cos)
        out_real = x_real * cos - x_imag * sin
        out_imag = x_real * sin + x_imag * cos
        out = mx.concatenate([out_real, out_imag], axis=-1)  # (..., N, H, pairs, 2)
        out = out.reshape(*prefix, N, H, D)
        return out.astype(orig)

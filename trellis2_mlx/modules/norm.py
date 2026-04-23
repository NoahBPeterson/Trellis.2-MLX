"""LayerNorm + per-head RMSNorm.

Upstream uses LayerNorm32 (compute in fp32) everywhere; MLX's nn.LayerNorm
already promotes to fp32 internally so nn.LayerNorm is the direct equivalent.

MultiHeadRMSNorm matches upstream/trellis2/modules/attention/modules.py:9 —
per-head gamma of shape (heads, head_dim), multiplied by a fixed sqrt(head_dim)
scale after F.normalize.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class LayerNorm32(nn.Module):
    """LayerNorm that always computes in fp32, matching upstream convention."""

    def __init__(self, dims: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.dims = dims
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = mx.ones((dims,))
            self.bias = mx.zeros((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        orig = x.dtype
        x32 = x.astype(mx.float32)
        mean = x32.mean(axis=-1, keepdims=True)
        var = x32.var(axis=-1, keepdims=True)
        x32 = (x32 - mean) * mx.rsqrt(var + self.eps)
        if self.affine:
            x32 = x32 * self.weight.astype(mx.float32) + self.bias.astype(mx.float32)
        return x32.astype(orig)


class MultiHeadRMSNorm(nn.Module):
    """Per-head RMSNorm over the last (head_dim) axis.

    Mirrors `MultiHeadRMSNorm` at upstream/trellis2/modules/attention/modules.py:9.
    Expects input shape (..., H, D) and gamma of shape (H, D).
    """

    def __init__(self, dim: int, heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.eps = eps
        self.scale = dim**0.5
        self.gamma = mx.ones((heads, dim))

    def __call__(self, x: mx.array) -> mx.array:
        orig = x.dtype
        x32 = x.astype(mx.float32)
        # L2-normalize along last axis (like torch.nn.functional.normalize)
        norm = mx.sqrt((x32 * x32).sum(axis=-1, keepdims=True) + self.eps)
        x32 = x32 / norm
        x32 = x32 * self.gamma.astype(mx.float32) * self.scale
        return x32.astype(orig)

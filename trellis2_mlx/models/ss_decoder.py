"""Sparse-structure decoder (dense 3D conv).

Ports `upstream-trellis1/trellis/models/sparse_structure_vae.py:SparseStructureDecoder`
(the microsoft/TRELLIS-image-large v1 decoder). Takes a 16³ latent and upsamples
to a 64³ binary occupancy mask via two 2× upsamples through pixel-shuffle.

MLX uses NDHWC channel-last layout; this module assumes inputs are (B, C, D, H, W)
matching upstream's torch convention and transposes at the boundaries.
"""
from __future__ import annotations

from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn


def pixel_shuffle_3d(x: mx.array, scale: int) -> mx.array:
    """Channels-last (NDHWC) 3D pixel shuffle.

    Input : (B, D, H, W, C * scale^3)
    Output: (B, D*scale, H*scale, W*scale, C)

    Channel sub-dims are split as (C, s_d, s_h, s_w) with C outermost — matching
    the torch `reshape(..., C_, s, s, s, H, W, D).permute(...)` ordering once
    the channels-last transpose is accounted for.
    """
    B, D, H, W, C_full = x.shape
    s = scale
    C = C_full // (s * s * s)
    x = x.reshape(B, D, H, W, C, s, s, s)
    x = x.transpose(0, 1, 5, 2, 6, 3, 7, 4)
    x = x.reshape(B, D * s, H * s, W * s, C)
    return x


class ChannelLayerNorm32(nn.Module):
    """LayerNorm over channel axis, fp32 compute.

    MLX inputs are NDHWC already, so this is just a last-axis LN (unlike the
    torch NCDHW version which permutes back and forth).
    """

    def __init__(self, dims: int, eps: float = 1e-5, affine: bool = True):
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
        out = (x32 - mean) * mx.rsqrt(var + self.eps)
        if self.affine:
            out = out * self.weight.astype(mx.float32) + self.bias.astype(mx.float32)
        return out.astype(orig)


class ResBlock3d(nn.Module):
    def __init__(self, channels: int, out_channels: Optional[int] = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.norm1 = ChannelLayerNorm32(channels)
        self.norm2 = ChannelLayerNorm32(self.out_channels)
        self.conv1 = nn.Conv3d(channels, self.out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        if channels != self.out_channels:
            self.skip_connection = nn.Conv3d(channels, self.out_channels, kernel_size=1)
        else:
            self.skip_connection = None

    def __call__(self, x: mx.array) -> mx.array:
        h = self.norm1(x)
        h = nn.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nn.silu(h)
        h = self.conv2(h)
        skip = x if self.skip_connection is None else self.skip_connection(x)
        return h + skip


class UpsampleBlock3d(nn.Module):
    """Conv(Ci -> Co * 8) + pixel shuffle ×2."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv3d(in_channels, out_channels * 8, kernel_size=3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv(x)
        return pixel_shuffle_3d(x, 2)


class SparseStructureDecoder(nn.Module):
    """Dense 3D conv decoder: (B, 8, 16, 16, 16) latent → (B, 1, 64, 64, 64) occupancy."""

    def __init__(
        self,
        out_channels: int,
        latent_channels: int,
        num_res_blocks: int,
        channels: List[int],
        num_res_blocks_middle: int = 2,
        use_fp16: bool = False,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.channels = channels
        self.use_fp16 = use_fp16

        self.input_layer = nn.Conv3d(latent_channels, channels[0], kernel_size=3, padding=1)
        self.middle_block = nn.Sequential(*[
            ResBlock3d(channels[0], channels[0]) for _ in range(num_res_blocks_middle)
        ])
        # Flat list matching upstream's nn.ModuleList.extend pattern:
        # for each stage: `num_res_blocks` ResBlocks, then one UpsampleBlock if not last.
        blocks: List[nn.Module] = []
        for i, ch in enumerate(channels):
            for _ in range(num_res_blocks):
                blocks.append(ResBlock3d(ch, ch))
            if i < len(channels) - 1:
                blocks.append(UpsampleBlock3d(ch, channels[i + 1]))
        self.blocks = blocks

        self.out_layer = nn.Sequential(
            ChannelLayerNorm32(channels[-1]),
            nn.SiLU(),
            nn.Conv3d(channels[-1], out_channels, kernel_size=3, padding=1),
        )

    def __call__(self, x: mx.array) -> mx.array:
        """x: (B, Cin, D, H, W) torch-style; internal: NDHWC."""
        # Transpose once at the entry (B, C, D, H, W) -> (B, D, H, W, C)
        x_in_dtype = x.dtype
        x = x.transpose(0, 2, 3, 4, 1)
        h = self.input_layer(x)
        if self.use_fp16:
            h = h.astype(mx.float16)
        h = self.middle_block(h)
        for blk in self.blocks:
            h = blk(h)
        h = h.astype(x_in_dtype)
        h = self.out_layer(h)
        # (B, D, H, W, Cout) -> (B, Cout, D, H, W)
        return h.transpose(0, 4, 1, 2, 3)

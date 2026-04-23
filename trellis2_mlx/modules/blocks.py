"""DiT transformer blocks with adaLN modulation.

Ports upstream/trellis2/modules/transformer/{blocks.py, modulated.py}.
Used by both dense `SparseStructureFlowModel` and sparse `SLatFlowModel` — the
block math is identical; sparse wrappers just operate on feature tables (F, C)
instead of dense (B, L, C). This module implements the dense (B, L, C) path;
the sparse path reuses it by treating (F, C) as (1, F, C).
"""
from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .attention import MultiHeadAttention
from .norm import LayerNorm32


class FeedForwardNet(nn.Module):
    """MLP with GELU(tanh). Matches upstream naming: mlp.0 / mlp.2."""

    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=True),
            nn.GELU(approx="tanh"),
            nn.Linear(hidden, channels, bias=True),
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.mlp(x)


class ModulatedTransformerCrossBlock(nn.Module):
    """DiT block with self-attn + cross-attn + MLP, all adaLN-modulated.

    Matches upstream/trellis2/modules/transformer/modulated.py:ModulatedTransformerCrossBlock:
      - norm1: LN no-affine        (pre self-attn)
      - norm2: LN WITH affine      (pre cross-attn) — upstream has this one affine
      - norm3: LN no-affine        (pre MLP)

    When `share_mod=True`, the block carries a learnable `modulation` parameter
    of shape (6 * channels,) that is added to the modulation signal computed
    once at the model root (via `adaLN_modulation`). When False, the block owns
    its own `adaLN_modulation` sequential (SiLU + Linear(C -> 6C)).
    """

    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
    ):
        super().__init__()
        self.share_mod = share_mod
        self.channels = channels

        self.norm1 = LayerNorm32(channels, eps=1e-6, affine=False)
        self.norm2 = LayerNorm32(channels, eps=1e-6, affine=True)
        self.norm3 = LayerNorm32(channels, eps=1e-6, affine=False)

        self.self_attn = MultiHeadAttention(
            channels, num_heads,
            type="self", attn_mode="full",
            qkv_bias=qkv_bias, use_rope=use_rope, qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = MultiHeadAttention(
            channels, num_heads,
            ctx_channels=ctx_channels,
            type="cross", attn_mode="full",
            qkv_bias=qkv_bias, qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = FeedForwardNet(channels, mlp_ratio=mlp_ratio)

        if share_mod:
            # Per-block learnable bias added to shared modulation signal
            self.modulation = mx.zeros((6 * channels,))
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True),
            )

    def __call__(
        self,
        x: mx.array,
        mod: mx.array,
        context: mx.array,
        phases: Optional[mx.array] = None,
    ) -> mx.array:
        """x: (B, L, C); mod: (B, 6C) if share_mod else (B, C); context: (B, Lkv, Cctx)."""
        if self.share_mod:
            m = (self.modulation + mod).astype(mod.dtype)
        else:
            m = self.adaLN_modulation(mod)
        # chunk into 6 along last axis
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mx.split(m, 6, axis=-1)

        # Self-attention
        h = self.norm1(x)
        h = h * (1 + mx.expand_dims(scale_msa, axis=1)) + mx.expand_dims(shift_msa, axis=1)
        h = self.self_attn(h, phases=phases)
        h = h * mx.expand_dims(gate_msa, axis=1)
        x = x + h

        # Cross-attention (no modulation in upstream beyond the affine LN)
        h = self.norm2(x)
        h = self.cross_attn(h, context=context)
        x = x + h

        # MLP
        h = self.norm3(x)
        h = h * (1 + mx.expand_dims(scale_mlp, axis=1)) + mx.expand_dims(shift_mlp, axis=1)
        h = self.mlp(h)
        h = h * mx.expand_dims(gate_mlp, axis=1)
        x = x + h
        return x

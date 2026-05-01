"""Multi-head (self/cross) attention in MLX, mirroring upstream behavior.

Port of upstream/trellis2/modules/attention/modules.py:MultiHeadAttention.

Key behaviors preserved:
- self vs cross attention projection layouts
  - self: single `to_qkv` Linear(C -> 3C)
  - cross: `to_q` Linear(C -> C) + `to_kv` Linear(Ctx -> 2C)
- `qk_rms_norm`: optional per-head RMSNorm on Q and K before attention
- `use_rope`: optional rotary embedding on Q and K (self-attention only)
- Output projection via `to_out` Linear(C -> C)
"""
from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.core.fast import scaled_dot_product_attention as _mx_sdpa

from .norm import MultiHeadRMSNorm
from .rope import RotaryPositionEmbedder


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int] = None,
        type: str = "self",
        attn_mode: str = "full",
        qkv_bias: bool = True,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        assert channels % num_heads == 0
        assert type in ("self", "cross")
        assert attn_mode == "full", "only full attention supported in MLX port"
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self._type = type
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        if type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)

        if qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)

        self.to_out = nn.Linear(channels, channels)

        # Cross-attn cache: id(context) -> (k, v) in SDPA layout (B, H, Lkv, D),
        # post-k_rms_norm. Cross-attn K,V are constant across all sampler steps
        # for a given cond, so we compute once per (block, cond) pair instead of
        # 12× per pair. Cleared by the flow model's clear_caches().
        self._cross_kv_cache: dict = {}

    def clear_cross_kv_cache(self) -> None:
        self._cross_kv_cache.clear()

    def __call__(
        self,
        x: mx.array,
        context: Optional[mx.array] = None,
        phases: Optional[mx.array] = None,
    ) -> mx.array:
        """x: (B, L, C); context: (B, Lkv, Ctx) for cross. phases: (B, L, pairs, 2)."""
        B, L, C = x.shape
        H, D = self.num_heads, self.head_dim
        if self._type == "self":
            qkv = self.to_qkv(x).reshape(B, L, 3, H, D)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # each (B, L, H, D)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k = self.k_rms_norm(k)
            if self.use_rope:
                assert phases is not None, "RoPE requires phases"
                q = RotaryPositionEmbedder.apply_rotary_embedding(q, phases)
                k = RotaryPositionEmbedder.apply_rotary_embedding(k, phases)
            # SDPA layout (B, H, L, D)
            q = q.transpose(0, 2, 1, 3)
            k = k.transpose(0, 2, 1, 3)
            v = v.transpose(0, 2, 1, 3)
        else:
            assert context is not None, "cross attention needs context"
            q = self.to_q(x).reshape(B, L, H, D)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
            q = q.transpose(0, 2, 1, 3)

            cache_key = id(context)
            cached = self._cross_kv_cache.get(cache_key)
            if cached is not None:
                k, v = cached
            else:
                Lkv = context.shape[1]
                kv = self.to_kv(context).reshape(B, Lkv, 2, H, D)
                k, v = kv[:, :, 0], kv[:, :, 1]
                if self.qk_rms_norm:
                    k = self.k_rms_norm(k)
                k = k.transpose(0, 2, 1, 3)
                v = v.transpose(0, 2, 1, 3)
                self._cross_kv_cache[cache_key] = (k, v)

        out = _mx_sdpa(q, k, v, scale=D**-0.5)  # (B, H, L, D)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, C)
        return self.to_out(out)

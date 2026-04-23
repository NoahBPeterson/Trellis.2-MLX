"""Flow-matching DiT models (dense + sparse).

Ports:
- SparseStructureFlowModel : upstream/trellis2/models/sparse_structure_flow.py:56
  Dense DiT operating on (B, C, R, R, R) inputs with RoPE (3D grid).
- SLatFlowModel            : upstream/trellis2/models/structured_latent_flow.py:15
  Sparse DiT operating on SparseTensor inputs with APE over voxel coords.
  Supports optional concat-conditioning (tex flow concats shape SLat to input).

Both share ModulatedTransformerCrossBlock. Weight names match upstream exactly so
converted safetensors load directly.
"""
from __future__ import annotations

from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from ..modules.blocks import ModulatedTransformerCrossBlock
from ..modules.pos_embed import AbsolutePositionEmbedder, TimestepEmbedder
from ..modules.rope import RotaryPositionEmbedder
from ..modules.sparse_tensor import SparseTensor


def _layer_norm_last(x: mx.array, eps: float = 1e-5) -> mx.array:
    """Affine-less LayerNorm over the last axis (F.layer_norm(h, h.shape[-1:]))."""
    orig = x.dtype
    x32 = x.astype(mx.float32)
    mean = x32.mean(axis=-1, keepdims=True)
    var = x32.var(axis=-1, keepdims=True)
    out = (x32 - mean) * mx.rsqrt(var + eps)
    return out.astype(orig)


class SparseStructureFlowModel(nn.Module):
    """Dense DiT for the coarse 16³ sparse-structure latent."""

    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4.0,
        pe_mode: str = "ape",
        rope_freq: tuple = (1.0, 10000.0),
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        **_unused,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or (model_channels // num_head_channels)
        self.pe_mode = pe_mode
        self.share_mod = share_mod

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True),
            )

        # Position embedding: build the embedder (not the buffer) so parameters
        # stay clean. Buffers are computed lazily inside __call__.
        head_dim = model_channels // self.num_heads
        if pe_mode == "ape":
            self._ape = AbsolutePositionEmbedder(model_channels, 3)
            self._rope = None
        elif pe_mode == "rope":
            self._rope = RotaryPositionEmbedder(head_dim, 3, rope_freq)
            self._ape = None
        else:
            raise ValueError(f"Unknown pe_mode {pe_mode}")
        # Cache slot; populated on first forward
        self._cached_phases = None
        self._cached_pos_emb = None

        self.input_layer = nn.Linear(in_channels, model_channels)
        self.blocks = [
            ModulatedTransformerCrossBlock(
                model_channels, cond_channels,
                num_heads=self.num_heads, mlp_ratio=mlp_ratio,
                use_rope=(pe_mode == "rope"),
                qk_rms_norm=qk_rms_norm, qk_rms_norm_cross=qk_rms_norm_cross,
                share_mod=share_mod,
            )
            for _ in range(num_blocks)
        ]
        self.out_layer = nn.Linear(model_channels, out_channels)

    def __call__(self, x: mx.array, t: mx.array, cond: mx.array) -> mx.array:
        """x: (B, C, R, R, R); t: (B,); cond: (B, N_cond, cond_channels)."""
        B, C, R1, R2, R3 = x.shape
        assert R1 == R2 == R3 == self.resolution, f"Expected {self.resolution}^3, got {R1}x{R2}x{R3}"
        # (B, C, R, R, R) -> (B, R*R*R, C)
        h = x.reshape(B, C, -1).transpose(0, 2, 1)
        h = self.input_layer(h)
        # Lazily build position buffers on first forward
        coords = _grid_coords_3d(self.resolution)
        if self.pe_mode == "ape":
            if self._cached_pos_emb is None:
                self._cached_pos_emb = self._ape(coords)
            h = h + self._cached_pos_emb[None]
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        phases = None
        if self.pe_mode == "rope":
            if self._cached_phases is None:
                self._cached_phases = self._rope(coords)
            phases = self._cached_phases[None]  # (1, L, pairs, 2)
        for block in self.blocks:
            h = block(h, t_emb, cond, phases=phases)
        h = _layer_norm_last(h)
        h = self.out_layer(h)
        # (B, L, Cout) -> (B, Cout, R, R, R)
        return h.transpose(0, 2, 1).reshape(B, self.out_channels, R1, R2, R3)


class SLatFlowModel(nn.Module):
    """Sparse DiT over active voxels. Inference path: B=1.

    Unlike dense SS flow (which precomputes RoPE phases over a regular grid),
    sparse flow computes RoPE phases at runtime from per-voxel coordinates.
    """

    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4.0,
        pe_mode: str = "rope",
        rope_freq: tuple = (1.0, 10000.0),
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        **_unused,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or (model_channels // num_head_channels)
        self.pe_mode = pe_mode
        self.share_mod = share_mod

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True),
            )

        head_dim = model_channels // self.num_heads
        if pe_mode == "rope":
            self._rope = RotaryPositionEmbedder(head_dim, 3, rope_freq)
            self._ape = None
        elif pe_mode == "ape":
            self._ape = AbsolutePositionEmbedder(model_channels, 3)
            self._rope = None
        else:
            raise ValueError(f"Unknown pe_mode {pe_mode}")

        # SparseLinear is just nn.Linear on feats; same key names as upstream
        self.input_layer = nn.Linear(in_channels, model_channels)
        self.blocks = [
            ModulatedTransformerCrossBlock(
                model_channels, cond_channels,
                num_heads=self.num_heads, mlp_ratio=mlp_ratio,
                use_rope=(pe_mode == "rope"),
                qk_rms_norm=qk_rms_norm, qk_rms_norm_cross=qk_rms_norm_cross,
                share_mod=share_mod,
            )
            for _ in range(num_blocks)
        ]
        self.out_layer = nn.Linear(model_channels, out_channels)

    def __call__(
        self,
        x: SparseTensor,
        t: mx.array,
        cond: mx.array,
        concat_cond: Optional[SparseTensor] = None,
    ) -> SparseTensor:
        """x: SparseTensor with feats (F, in_channels); t: (B,); cond: (B, N_cond, cond_channels).

        For B=1 inference, feats is (F, C) and self-attention is dense over F tokens.
        """
        if concat_cond is not None:
            x = x.replace(mx.concatenate([x.feats, concat_cond.feats], axis=-1))

        h_feats = self.input_layer(x.feats)  # (F, C)
        t_emb = self.t_embedder(t)            # (B, C)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)  # (B, 6C)

        # Position encoding
        phases = None
        if self.pe_mode == "rope":
            # coords is (F, 4) with [batch, x, y, z]; RoPE expects (F, 3)
            phases = self._rope(x.coords[:, 1:].astype(mx.float32))  # (F, pairs, 2)
            phases = phases[None]  # (1, F, pairs, 2) for B=1
        elif self.pe_mode == "ape":
            pe = self._ape(x.coords[:, 1:].astype(mx.float32))  # (F, C)
            h_feats = h_feats + pe

        # For B=1, reshape (F, C) -> (1, F, C) for the dense block
        assert t_emb.shape[0] == 1, "Sparse DiT currently supports B=1 inference only"
        h = h_feats[None]  # (1, F, C)
        for block in self.blocks:
            h = block(h, t_emb, cond, phases=phases)
        h = _layer_norm_last(h)
        h = self.out_layer(h)

        out_feats = h[0]  # (F, Cout)
        return x.replace(out_feats)


def _grid_coords_3d(R: int) -> mx.array:
    """Returns (R**3, 3) float32 tensor of [x, y, z] in row-major ('ij') order."""
    idx = mx.arange(R, dtype=mx.float32)
    xx, yy, zz = mx.meshgrid(idx, idx, idx, indexing="ij")
    return mx.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], axis=-1)

"""Sparse UNet VAE (decoder-only for inference).

Ports the inference path of:
- upstream/trellis2/models/sc_vaes/sparse_unet_vae.py (SparseUnetVaeDecoder)
- upstream/trellis2/models/sc_vaes/fdg_vae.py        (FlexiDualGridVaeDecoder)

Block structure (matches upstream's weight names):
  SparseConvNeXtBlock3d:  conv (3x3x3) → norm (affine) → mlp (Lin,SiLU,Lin) + x
  SparseResBlockC2S3d:    norm1 (affine) → silu → conv1 → C2S(2) → norm2 (no-affine)
                          → silu → conv2 → + skip_connection(C2S(x))
                          + to_subdiv linear head predicting 8-way subdivision mask
"""
from __future__ import annotations

from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from ..modules.norm import LayerNorm32
from ..modules.sparse import (
    SparseChannel2Spatial,
    SparseConv3d,
    SparseLayerNorm32,
    SparseLinear,
)
from ..modules.sparse_tensor import SparseTensor


class SparseConvNeXtBlock3d(nn.Module):
    """conv3x3x3 → LN(affine) → Linear(ratio*C) → SiLU → Linear(C), residual."""

    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.channels = channels
        hidden = int(channels * mlp_ratio)
        self.conv = SparseConv3d(channels, channels, kernel_size=3)
        self.norm = LayerNorm32(channels, eps=1e-6, affine=True)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=True),
            nn.SiLU(),
            nn.Linear(hidden, channels, bias=True),
        )

    def __call__(self, x: SparseTensor) -> SparseTensor:
        h = self.conv(x)
        h = h.replace(self.norm(h.feats))
        h = h.replace(self.mlp(h.feats))
        return h + x


class SparseResBlockC2S3d(nn.Module):
    """Upsample-by-2 residual block: LN→SiLU→conv1→C2S→LN→SiLU→conv2 + skip.

    The `to_subdiv` head predicts (per parent voxel) an 8-way mask over child
    positions to materialize after C2S. During inference with `pred_subdiv=True`,
    the block returns both the upsampled SparseTensor and the subdiv logits so
    the pipeline can drive the next stage's layout.
    """

    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        pred_subdiv: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.pred_subdiv = pred_subdiv
        self.norm1 = LayerNorm32(channels, eps=1e-6, affine=True)
        self.norm2 = LayerNorm32(self.out_channels, eps=1e-6, affine=False)
        # conv1: Ci -> Co * 8 (pre-expand for C2S packing)
        self.conv1 = SparseConv3d(channels, self.out_channels * 8, kernel_size=3)
        self.conv2 = SparseConv3d(self.out_channels, self.out_channels, kernel_size=3)
        if pred_subdiv:
            self.to_subdiv = SparseLinear(channels, 8)
        self.updown = SparseChannel2Spatial(2)

    def __call__(self, x: SparseTensor, subdiv: Optional[SparseTensor] = None):
        if self.pred_subdiv:
            subdiv = self.to_subdiv(x)
        h = x.replace(self.norm1(x.feats))
        h = h.replace(nn.silu(h.feats))
        h = self.conv1(h)
        subdiv_bin = subdiv.replace(subdiv.feats > 0) if subdiv is not None else None
        h = self.updown(h, subdiv_bin)
        x_up = self.updown(x, subdiv_bin)
        h = h.replace(self.norm2(h.feats))
        h = h.replace(nn.silu(h.feats))
        h = self.conv2(h)
        # skip_connection: repeat each of (channels // 8) post-C2S channels `rep` times consecutively.
        # Matches upstream: lambda x: x.feats.repeat_interleave(out_channels // (channels // 8), dim=1)
        rep = self.out_channels // (self.channels // 8)
        cx = x_up.feats.shape[1]
        skip_feats = mx.repeat(x_up.feats[:, :, None], rep, axis=2).reshape(x_up.feats.shape[0], cx * rep)
        skip = x_up.replace(skip_feats)
        out = h + skip
        if self.pred_subdiv:
            return out, subdiv
        return out


class SparseUnetVaeDecoder(nn.Module):
    """Multi-stage UNet decoder: `[ConvNeXt]*n + C2S` per stage, 4 total upsamples."""

    def __init__(
        self,
        out_channels: int,
        model_channels: List[int],
        latent_channels: int,
        num_blocks: List[int],
        block_type: List[str],
        up_block_type: List[str],
        block_args: List[dict],
        pred_subdiv: bool = True,
        use_fp16: bool = False,
    ):
        super().__init__()
        assert all(bt == "SparseConvNeXtBlock3d" for bt in block_type), block_type
        assert all(ubt == "SparseResBlockC2S3d" for ubt in up_block_type), up_block_type
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.pred_subdiv = pred_subdiv
        self.use_fp16 = use_fp16

        self.from_latent = SparseLinear(latent_channels, model_channels[0])
        self.output_layer = SparseLinear(model_channels[-1], out_channels)

        # Nested list: one inner list per stage. The last element of each non-final
        # stage is an upsample block; interior elements are ConvNeXt blocks.
        self.blocks: List[List[nn.Module]] = []
        for i in range(len(num_blocks)):
            stage: List[nn.Module] = []
            for _ in range(num_blocks[i]):
                stage.append(SparseConvNeXtBlock3d(model_channels[i], **block_args[i]))
            if i < len(num_blocks) - 1:
                stage.append(
                    SparseResBlockC2S3d(
                        model_channels[i],
                        model_channels[i + 1],
                        pred_subdiv=pred_subdiv,
                        **{k: v for k, v in block_args[i].items() if k in ()},
                    )
                )
            self.blocks.append(stage)

    def __call__(
        self,
        x: SparseTensor,
        guide_subs: Optional[List[SparseTensor]] = None,
        return_subs: bool = False,
    ):
        assert guide_subs is None or not self.pred_subdiv, "guide_subs only valid when pred_subdiv=False"
        assert not return_subs or self.pred_subdiv, "return_subs requires pred_subdiv=True"

        h = self.from_latent(x)
        if self.use_fp16:
            h = h.astype(mx.float16)

        import os
        import time
        timing = os.environ.get("TRELLIS_VAE_TIMING") == "1"

        subs: List[SparseTensor] = []
        for i, stage in enumerate(self.blocks):
            for j, block in enumerate(stage):
                is_last = j == len(stage) - 1
                if timing:
                    mx.eval(h.feats)
                    t0 = time.time()
                if i < len(self.blocks) - 1 and is_last:
                    if self.pred_subdiv:
                        h, sub = block(h)
                        subs.append(sub)
                    else:
                        h = block(h, subdiv=guide_subs[i] if guide_subs is not None else None)
                else:
                    h = block(h)
                if timing:
                    mx.eval(h.feats)
                    dt = time.time() - t0
                    kind = type(block).__name__
                    F = h.feats.shape[0]
                    print(f"  stage {i} block {j:2d} {kind:22s} F={F:>7d} Co={h.feats.shape[1]:>5d} {dt*1000:7.1f}ms")
                else:
                    mx.eval(h.feats)

        h = h.astype(x.dtype)
        # Final affine-less LN, then output projection
        feats = h.feats
        feats32 = feats.astype(mx.float32)
        mean = feats32.mean(axis=-1, keepdims=True)
        var = feats32.var(axis=-1, keepdims=True)
        feats = ((feats32 - mean) * mx.rsqrt(var + 1e-5)).astype(feats.dtype)
        h = h.replace(feats)
        h = self.output_layer(h)
        if return_subs:
            return h, subs
        return h


class FlexiDualGridVaeDecoder(SparseUnetVaeDecoder):
    """Thin wrapper that fixes out_channels=7 (vertex xyz + 3 intersections + quad_lerp).

    Matches `upstream/trellis2/models/sc_vaes/fdg_vae.py:FlexiDualGridVaeDecoder` for
    the inference branch: apply sigmoid to xyz, return intersection logits as bools,
    softplus the quad_lerp coefficient. Mesh extraction is handled externally.
    """

    def __init__(
        self,
        resolution: int,
        model_channels: List[int],
        latent_channels: int,
        num_blocks: List[int],
        block_type: List[str],
        up_block_type: List[str],
        block_args: List[dict],
        voxel_margin: float = 0.5,
        use_fp16: bool = False,
    ):
        super().__init__(
            out_channels=7,
            model_channels=model_channels,
            latent_channels=latent_channels,
            num_blocks=num_blocks,
            block_type=block_type,
            up_block_type=up_block_type,
            block_args=block_args,
            pred_subdiv=True,
            use_fp16=use_fp16,
        )
        self.resolution = resolution
        self.voxel_margin = voxel_margin

    def set_resolution(self, resolution: int) -> None:
        self.resolution = resolution

    def decode(self, x: SparseTensor, return_subs: bool = True):
        """Forward + post-activation for dual-grid mesh extraction.

        Returns `(vertices_st, intersected_bool_st, quad_lerp_st, subs)` where
        `vertices_st.feats` is in [-voxel_margin, 1+voxel_margin] and represents
        per-voxel sub-cell offsets for the three dual-grid vertex degrees of freedom.
        """
        out = super().__call__(x, return_subs=return_subs)
        if return_subs:
            h, subs = out
        else:
            h, subs = out, None
        feats = h.feats
        xyz = (1 + 2 * self.voxel_margin) * mx.sigmoid(feats[:, 0:3]) - self.voxel_margin
        intersected = feats[:, 3:6] > 0  # boolean activations on the 3 dual-grid edges
        quad_lerp = mx.logaddexp(mx.zeros_like(feats[:, 6:7]), feats[:, 6:7])  # softplus
        vertices = h.replace(xyz)
        inter = h.replace(intersected.astype(feats.dtype))
        quad = h.replace(quad_lerp)
        return vertices, inter, quad, subs

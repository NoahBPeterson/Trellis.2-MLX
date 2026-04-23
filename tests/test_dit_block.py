"""End-to-end DiT block numerics: MLX vs upstream torch.

Matches the TRELLIS.2-4B flow-DiT config: 1536 channels, 12 heads, mlp_ratio 5.3334,
qk_rms_norm=True, qk_rms_norm_cross=True, share_mod=True, use_rope=True.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

os.environ["ATTN_BACKEND"] = "naive"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "upstream"))

from trellis2.modules.transformer.modulated import ModulatedTransformerCrossBlock as TorchBlock
from trellis2.modules.attention.rope import RotaryPositionEmbedder as TorchRoPE
from trellis2_mlx.modules.blocks import ModulatedTransformerCrossBlock as MlxBlock
from trellis2_mlx.modules.rope import RotaryPositionEmbedder as MlxRoPE


def _copy_block_weights(tb: TorchBlock, mb: MlxBlock) -> None:
    sd = tb.state_dict()
    # Norms
    if mb.norm2.affine:
        mb.norm2.weight = mx.array(sd["norm2.weight"].numpy())
        mb.norm2.bias = mx.array(sd["norm2.bias"].numpy())
    # Self-attn
    mb.self_attn.to_qkv.weight = mx.array(sd["self_attn.to_qkv.weight"].numpy())
    mb.self_attn.to_qkv.bias = mx.array(sd["self_attn.to_qkv.bias"].numpy())
    if mb.self_attn.qk_rms_norm:
        mb.self_attn.q_rms_norm.gamma = mx.array(sd["self_attn.q_rms_norm.gamma"].numpy())
        mb.self_attn.k_rms_norm.gamma = mx.array(sd["self_attn.k_rms_norm.gamma"].numpy())
    mb.self_attn.to_out.weight = mx.array(sd["self_attn.to_out.weight"].numpy())
    mb.self_attn.to_out.bias = mx.array(sd["self_attn.to_out.bias"].numpy())
    # Cross-attn
    mb.cross_attn.to_q.weight = mx.array(sd["cross_attn.to_q.weight"].numpy())
    mb.cross_attn.to_q.bias = mx.array(sd["cross_attn.to_q.bias"].numpy())
    mb.cross_attn.to_kv.weight = mx.array(sd["cross_attn.to_kv.weight"].numpy())
    mb.cross_attn.to_kv.bias = mx.array(sd["cross_attn.to_kv.bias"].numpy())
    if mb.cross_attn.qk_rms_norm:
        mb.cross_attn.q_rms_norm.gamma = mx.array(sd["cross_attn.q_rms_norm.gamma"].numpy())
        mb.cross_attn.k_rms_norm.gamma = mx.array(sd["cross_attn.k_rms_norm.gamma"].numpy())
    mb.cross_attn.to_out.weight = mx.array(sd["cross_attn.to_out.weight"].numpy())
    mb.cross_attn.to_out.bias = mx.array(sd["cross_attn.to_out.bias"].numpy())
    # MLP (Sequential -> mlp.mlp.0, mlp.mlp.2 on both sides)
    mb.mlp.mlp.layers[0].weight = mx.array(sd["mlp.mlp.0.weight"].numpy())
    mb.mlp.mlp.layers[0].bias = mx.array(sd["mlp.mlp.0.bias"].numpy())
    mb.mlp.mlp.layers[2].weight = mx.array(sd["mlp.mlp.2.weight"].numpy())
    mb.mlp.mlp.layers[2].bias = mx.array(sd["mlp.mlp.2.bias"].numpy())
    # Modulation
    if mb.share_mod:
        mb.modulation = mx.array(sd["modulation"].numpy())
    else:
        mb.adaLN_modulation.layers[1].weight = mx.array(sd["adaLN_modulation.1.weight"].numpy())
        mb.adaLN_modulation.layers[1].bias = mx.array(sd["adaLN_modulation.1.bias"].numpy())


def test_dit_block_matches_torch():
    # TRELLIS.2 flow-DiT hyperparams
    C, H, MLP = 384, 6, 4.0  # smaller for speed; same config shape as the real thing
    CTX = 256
    B, L, Lkv = 2, 16, 12

    torch.manual_seed(0)
    tb = TorchBlock(
        channels=C, ctx_channels=CTX, num_heads=H, mlp_ratio=MLP,
        use_rope=True, qk_rms_norm=True, qk_rms_norm_cross=True,
        share_mod=True,
    )
    tb.eval()
    mb = MlxBlock(
        channels=C, ctx_channels=CTX, num_heads=H, mlp_ratio=MLP,
        use_rope=True, qk_rms_norm=True, qk_rms_norm_cross=True,
        share_mod=True,
    )
    _copy_block_weights(tb, mb)

    rng = np.random.default_rng(0)
    x = rng.standard_normal((B, L, C)).astype(np.float32)
    mod = rng.standard_normal((B, 6 * C)).astype(np.float32) * 0.1
    context = rng.standard_normal((B, Lkv, CTX)).astype(np.float32)
    coords = rng.integers(0, 16, size=(L, 3)).astype(np.float32)

    head_dim = C // H
    t_rope = TorchRoPE(head_dim=head_dim, dim=3)
    phases_t = t_rope(torch.from_numpy(coords)).unsqueeze(0)  # (1, L, pairs) complex
    m_rope = MlxRoPE(head_dim=head_dim, dim=3)
    phases_m = mx.expand_dims(m_rope(mx.array(coords)), axis=0)

    with torch.no_grad():
        t_out = tb(torch.from_numpy(x), torch.from_numpy(mod), torch.from_numpy(context), phases_t).numpy()
    m_out = np.asarray(mb(mx.array(x), mx.array(mod), mx.array(context), phases=phases_m))

    diff = np.abs(t_out - m_out).max()
    rel = diff / (np.abs(t_out).max() + 1e-8)
    print(f"DiT block fp32 max-abs diff: {diff:.2e}  rel: {rel:.2e}")
    assert diff < 5e-4, f"DiT block max diff {diff} too large"


if __name__ == "__main__":
    test_dit_block_matches_torch()
    print("OK: DiT block matches torch")

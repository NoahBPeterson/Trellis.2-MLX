"""Validate MLX FlowEulerGuidanceIntervalSampler against upstream sampler math.

Since upstream's sampler is pure torch math we can compare step-by-step without
any CUDA dependency (flash_attn is only pulled in if we run a real flow model).
We supply a dummy 'model' function that returns a deterministic `pred_v` given
(x_t, t, cond) so the sampler trajectory is fully determined.
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

from trellis2.pipelines.samplers.flow_euler import FlowEulerGuidanceIntervalSampler as TSampler
from trellis2_mlx.samplers import FlowEulerGuidanceIntervalSampler as MSampler


def test_sampler_trajectory_matches_torch():
    """Compare Euler trajectories with CFG + interval + rescale against torch."""
    rng = np.random.default_rng(0)
    B, C = 1, 4
    noise = rng.standard_normal((B, C)).astype(np.float32)
    cond_np = rng.standard_normal((B, 8, 16)).astype(np.float32)
    neg_np = np.zeros_like(cond_np)

    # Deterministic 'model': pred_v = Lx_t + Mcond.mean + b
    L = rng.standard_normal((C, C)).astype(np.float32) * 0.1
    M = rng.standard_normal((16, C)).astype(np.float32) * 0.1
    b = rng.standard_normal((C,)).astype(np.float32) * 0.01

    def torch_model(x_t, t, cond, **kw):
        x = torch.as_tensor(x_t, dtype=torch.float32)
        c = torch.as_tensor(cond, dtype=torch.float32)
        return x @ torch.from_numpy(L) + c.mean(dim=1) @ torch.from_numpy(M) + torch.from_numpy(b)

    def mlx_model(x_t, t, cond, **kw):
        return x_t @ mx.array(L) + cond.mean(axis=1) @ mx.array(M) + mx.array(b)

    # Torch sampler
    t_sampler = TSampler(sigma_min=1e-5)
    t_out = t_sampler.sample(
        torch_model,
        torch.from_numpy(noise),
        cond=torch.from_numpy(cond_np),
        neg_cond=torch.from_numpy(neg_np),
        steps=8,
        rescale_t=3.0,
        guidance_strength=5.0,
        guidance_interval=(0.4, 0.9),
        verbose=False,
    )

    # MLX sampler
    m_sampler = MSampler(sigma_min=1e-5)
    m_out = m_sampler.sample(
        mlx_model,
        mx.array(noise),
        cond=mx.array(cond_np),
        neg_cond=mx.array(neg_np),
        steps=8,
        rescale_t=3.0,
        guidance_strength=5.0,
        guidance_interval=(0.4, 0.9),
        verbose=False,
    )

    diff = np.abs(np.asarray(m_out["samples"]) - t_out["samples"].numpy()).max()
    print(f"Sampler (CFG g=5, interval [0.4, 0.9], 8 steps) fp32 max-abs diff: {diff:.2e}")
    assert diff < 1e-4, f"sampler diff too large: {diff}"


def test_sampler_with_rescale():
    rng = np.random.default_rng(1)
    B, C = 1, 4
    noise = rng.standard_normal((B, C)).astype(np.float32)
    cond_np = rng.standard_normal((B, 8, 16)).astype(np.float32)
    neg_np = np.zeros_like(cond_np)

    L = rng.standard_normal((C, C)).astype(np.float32) * 0.1
    M = rng.standard_normal((16, C)).astype(np.float32) * 0.1
    b = rng.standard_normal((C,)).astype(np.float32) * 0.01

    def torch_model(x_t, t, cond, **kw):
        x = torch.as_tensor(x_t, dtype=torch.float32)
        c = torch.as_tensor(cond, dtype=torch.float32)
        return x @ torch.from_numpy(L) + c.mean(dim=1) @ torch.from_numpy(M) + torch.from_numpy(b)

    def mlx_model(x_t, t, cond, **kw):
        return x_t @ mx.array(L) + cond.mean(axis=1) @ mx.array(M) + mx.array(b)

    t_sampler = TSampler(sigma_min=1e-5)
    t_out = t_sampler.sample(
        torch_model,
        torch.from_numpy(noise),
        cond=torch.from_numpy(cond_np),
        neg_cond=torch.from_numpy(neg_np),
        steps=8, rescale_t=3.0,
        guidance_strength=5.0, guidance_interval=(0.0, 1.0),
        guidance_rescale=0.5,
        verbose=False,
    )
    m_sampler = MSampler(sigma_min=1e-5)
    m_out = m_sampler.sample(
        mlx_model,
        mx.array(noise),
        cond=mx.array(cond_np), neg_cond=mx.array(neg_np),
        steps=8, rescale_t=3.0,
        guidance_strength=5.0, guidance_interval=(0.0, 1.0),
        guidance_rescale=0.5,
        verbose=False,
    )
    diff = np.abs(np.asarray(m_out["samples"]) - t_out["samples"].numpy()).max()
    print(f"Sampler (CFG rescale=0.5) fp32 max-abs diff: {diff:.2e}")
    assert diff < 1e-4, f"diff {diff}"


if __name__ == "__main__":
    test_sampler_trajectory_matches_torch()
    test_sampler_with_rescale()
    print("OK: FlowEulerGuidanceIntervalSampler matches torch")

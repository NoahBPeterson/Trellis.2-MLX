"""Flow-matching Euler sampler with classifier-free guidance + interval + rescale.

Ports `upstream/trellis2/pipelines/samplers/{flow_euler,classifier_free_guidance_mixin,guidance_interval_mixin}.py`
with MLX tensors. The model forward accepts (x_t, t, cond, **kwargs) and returns
a velocity estimate; CFG combines positive and negative conditions, optional
standard-deviation rescale keeps the sample variance close to the positive-cond pass.

For sparse inference, `x_t` can be a `SparseTensor` — its `feats` are updated and
`coords` preserved across the sampling trajectory.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import mlx.core as mx
import numpy as np

from .modules.sparse_tensor import SparseTensor


def _std_all_but_batch(x: mx.array) -> mx.array:
    """std over all non-batch dims, keepdims=True. Scalar-per-batch."""
    axes = tuple(range(1, x.ndim))
    return x.std(axis=axes, keepdims=True) if axes else x


def _std_sparse_global(x: mx.array) -> mx.array:
    """For a (F, C) sparse feats tensor, std over all entries (scalar)."""
    return x.std()


class FlowEulerGuidanceIntervalSampler:
    """Euler sampler with CFG, guidance interval, and CFG rescale.

    All three torch-side mixins collapsed into one class for clarity.
    """

    def __init__(self, sigma_min: float = 1e-5):
        self.sigma_min = sigma_min

    # --- Flow parameterization helpers (match upstream math) ---------------

    def _pred_to_xstart(self, x_t, t: float, pred):
        a = (1.0 - self.sigma_min) * x_t
        b = (self.sigma_min + (1.0 - self.sigma_min) * t) * pred
        return _sub(a, b)

    def _xstart_to_pred(self, x_t, t: float, x_0):
        num = _sub(_mul(1.0 - self.sigma_min, x_t), x_0)
        denom = self.sigma_min + (1.0 - self.sigma_min) * t
        return _div(num, denom)

    def _v_to_xstart(self, x_t, t: float, v):
        return _sub(_mul(1.0 - self.sigma_min, x_t), _mul(self.sigma_min + (1.0 - self.sigma_min) * t, v))

    # --- Inference wrapper ------------------------------------------------

    def _inference_model(
        self,
        model: Callable,
        x_t,
        t: float,
        cond: mx.array,
        neg_cond: Optional[mx.array],
        guidance_strength: float,
        guidance_interval: Tuple[float, float],
        guidance_rescale: float = 0.0,
        **kwargs,
    ):
        in_interval = guidance_interval[0] <= t <= guidance_interval[1]
        effective_g = guidance_strength if in_interval else 1.0

        t_vec = mx.array([1000.0 * t], dtype=mx.float32)
        if effective_g == 1.0 or neg_cond is None:
            return model(x_t, t_vec, cond, **kwargs)
        if effective_g == 0.0:
            return model(x_t, t_vec, neg_cond, **kwargs)

        pred_pos = model(x_t, t_vec, cond, **kwargs)
        pred_neg = model(x_t, t_vec, neg_cond, **kwargs)
        pred = _add(_mul(effective_g, pred_pos), _mul(1.0 - effective_g, pred_neg))

        if guidance_rescale > 0:
            # Rescale the CFG sample's std toward the positive-only sample's std.
            x0_pos = self._pred_to_xstart(x_t, t, pred_pos)
            x0_cfg = self._pred_to_xstart(x_t, t, pred)
            if isinstance(x0_pos, SparseTensor):
                s_pos = x0_pos.feats.std()
                s_cfg = x0_cfg.feats.std()
                scale = s_pos / s_cfg
                rescaled = x0_cfg.replace(x0_cfg.feats * scale)
            else:
                s_pos = _std_all_but_batch(x0_pos)
                s_cfg = _std_all_but_batch(x0_cfg)
                rescaled = x0_cfg * (s_pos / s_cfg)
            x0 = _add(_mul(guidance_rescale, rescaled), _mul(1.0 - guidance_rescale, x0_cfg))
            pred = self._xstart_to_pred(x_t, t, x0)
        return pred

    # --- Main loop --------------------------------------------------------

    def sample(
        self,
        model: Callable,
        noise,
        cond: mx.array,
        neg_cond: Optional[mx.array] = None,
        steps: int = 12,
        rescale_t: float = 1.0,
        guidance_strength: float = 7.5,
        guidance_interval: Tuple[float, float] = (0.0, 1.0),
        guidance_rescale: float = 0.0,
        verbose: bool = True,
        **kwargs,
    ):
        """Returns a dict with 'samples' (final x_0 estimate) and per-step history."""
        sample = noise
        t_seq = np.linspace(1.0, 0.0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_seq = t_seq.tolist()
        pairs = list(zip(t_seq[:-1], t_seq[1:]))

        iterator = pairs
        if verbose:
            try:
                from tqdm import tqdm
                iterator = tqdm(pairs, desc="Sampling")
            except ImportError:
                pass

        hist_x_t = []
        hist_x_0 = []
        for t_now, t_prev in iterator:
            pred_v = self._inference_model(
                model, sample, t_now, cond, neg_cond,
                guidance_strength=guidance_strength,
                guidance_interval=guidance_interval,
                guidance_rescale=guidance_rescale,
                **kwargs,
            )
            # Euler step: x_{t_prev} = x_t - (t - t_prev) * v
            dt = t_now - t_prev
            sample = _sub(sample, _mul(dt, pred_v))
            # Force materialization at each step. Without this, MLX queues all 12
            # steps lazily and tqdm fills instantly while the user waits 2+ minutes
            # on the next mx.eval(). Each step depends on the previous (Euler integration),
            # so per-step eval doesn't cost any cross-step fusion.
            if isinstance(sample, SparseTensor):
                mx.eval(sample.feats)
            else:
                mx.eval(sample)
            hist_x_t.append(sample)
            hist_x_0.append(self._v_to_xstart(sample, t_prev, pred_v))
        # Release ~250 MB of cached cross-attn K/V tensors before the next stage.
        if hasattr(model, "clear_caches"):
            model.clear_caches()
        return {"samples": sample, "pred_x_t": hist_x_t, "pred_x_0": hist_x_0}


# -------------------- Sparse/dense-aware arithmetic --------------------------

def _add(a, b):
    if isinstance(a, SparseTensor) and isinstance(b, SparseTensor):
        return a.replace(a.feats + b.feats)
    if isinstance(a, SparseTensor):
        return a.replace(a.feats + b)
    if isinstance(b, SparseTensor):
        return b.replace(a + b.feats)
    return a + b


def _sub(a, b):
    return _add(a, _mul(-1, b))


def _mul(scale, x):
    if isinstance(x, SparseTensor):
        return x.replace(x.feats * scale)
    if isinstance(scale, SparseTensor):
        return scale.replace(scale.feats * x)
    return x * scale


def _div(x, scale):
    if isinstance(x, SparseTensor):
        return x.replace(x.feats / scale)
    return x / scale

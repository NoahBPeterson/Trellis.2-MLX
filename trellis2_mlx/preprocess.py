"""Image preprocessing: alpha-aware crop to bbox, resize to max 1024.

Ports `upstream/trellis2/pipelines/trellis2_image_to_3d.py:preprocess_image`.
Detects the alpha channel; if missing or empty, calls an optional `rembg_model`
callable (BiRefNet / RMBG-2.0) to produce one. Then bbox-crops and premultiplies.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from PIL import Image


def preprocess_image(
    img: Image.Image,
    rembg_model: Optional[Callable[[Image.Image], Image.Image]] = None,
) -> Image.Image:
    """Crop to alpha bbox, premultiply, return an RGB Image.

    Args:
        img: input PIL Image (RGB or RGBA).
        rembg_model: optional callable `image → RGBA`. Invoked when `img` has no
            usable alpha channel. Pass a `BiRefNetRembg` instance, or any equivalent.
            If `None`, a missing alpha raises ValueError.
    """
    # Resize to max 1024 (matches upstream) BEFORE rembg — rembg itself resizes internally
    # but keeping this first avoids shipping a 4K image through the mask network
    # just to throw most of it away.
    max_size = max(img.size)
    scale = min(1.0, 1024 / max_size)
    if scale < 1.0:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)

    has_alpha = False
    if img.mode == "RGBA":
        alpha = np.asarray(img)[:, :, 3]
        if not (alpha == 255).all():
            has_alpha = True

    if not has_alpha:
        if rembg_model is None:
            raise ValueError(
                "Input image has no meaningful alpha channel. Pass a `rembg_model` "
                "(e.g. `BiRefNetRembg(...)`) to preprocess_image, or supply an RGBA image."
            )
        img = rembg_model(img.convert("RGB"))
        if img.mode != "RGBA":
            raise RuntimeError(f"rembg_model returned mode={img.mode}, expected RGBA")

    arr = np.asarray(img)
    alpha = arr[:, :, 3]
    mask = alpha > int(0.8 * 255)
    ys, xs = np.where(mask)
    if ys.size == 0:
        raise ValueError("No foreground pixels found after rembg — subject mask is empty")
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    size = max(x1 - x0, y1 - y0)
    half = int(size // 2)
    bbox = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))

    cropped = img.crop(bbox)
    out = np.asarray(cropped).astype(np.float32) / 255.0
    # Premultiply RGB by alpha
    rgb_premult = out[:, :, :3] * out[:, :, 3:4]
    return Image.fromarray((rgb_premult * 255).astype(np.uint8))

"""Image preprocessing: alpha-aware crop to bbox, resize to max 1024.

Ports the alpha-path of `upstream/trellis2/pipelines/trellis2_image_to_3d.py:preprocess_image`.
If the input has a meaningful alpha channel, crop and premultiply; otherwise
we'd need BiRefNet rembg (deferred — for v1, use images with alpha).
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def preprocess_image(img: Image.Image, require_alpha: bool = True) -> Image.Image:
    """Crop to alpha bbox, premultiply, return an RGB Image.

    Args:
        img: input PIL Image (RGB or RGBA)
        require_alpha: if True, raise when the input lacks a meaningful alpha channel.
            Set False to pass-through the input unchanged (useful when upstream rembg
            has already produced RGBA elsewhere).
    """
    if img.mode != "RGBA":
        if require_alpha:
            raise ValueError("Input image needs an alpha channel (BiRefNet rembg not ported yet)")
        return img.convert("RGB")

    max_size = max(img.size)
    scale = min(1.0, 1024 / max_size)
    if scale < 1.0:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)

    arr = np.array(img)
    alpha = arr[:, :, 3]
    if (alpha == 255).all():
        raise ValueError("Alpha is uniformly opaque — image has no foreground mask (BiRefNet rembg not ported yet)")

    mask = alpha > int(0.8 * 255)
    ys, xs = np.where(mask)
    if ys.size == 0:
        raise ValueError("No foreground pixels in alpha channel")
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    size = max(x1 - x0, y1 - y0)
    half = int(size // 2)
    bbox = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))

    cropped = img.crop(bbox)
    out = np.array(cropped).astype(np.float32) / 255.0
    # Premultiply RGB by alpha
    rgb_premult = out[:, :, :3] * out[:, :, 3:4]
    return Image.fromarray((rgb_premult * 255).astype(np.uint8))

"""BiRefNet background-removal wrapper.

Ports `upstream/trellis2/pipelines/rembg/BiRefNet.py` to our MLX layout. The
model itself stays in torch (runs once per input image, ~200M params) — same
strategy as our DINOv3 wrapper, since porting another ViT-ish model for a
one-shot preprocess step doesn't pay for itself.

Upstream defaults to `ZhengPeng7/BiRefNet`, but our `weights/pipeline.json`
uses `briaai/RMBG-2.0`; both load via `AutoModelForImageSegmentation` with
`trust_remote_code=True` and share the BiRefNet forward signature.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageSegmentation


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class BiRefNetRembg:
    """One-shot background remover. Load, call on one image, free via `.unload()`."""

    def __init__(self, model_name: str = "briaai/RMBG-2.0", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = AutoModelForImageSegmentation.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()
        self.model.to(device)

    def to(self, device: str) -> "BiRefNetRembg":
        self.device = device
        self.model.to(device)
        return self

    def unload(self) -> None:
        """Free the underlying torch model so its memory is returned before DiT inference.

        Sets the model attribute to None *and* forces gc.collect() + clears torch
        device caches. The aggressive cleanup matters: without it, ~800 MB of torch
        tensors linger in the Python heap and Metal can't satisfy MLX allocations
        for the DiT compute that follows, silently OOM-killing the process on
        memory-tight systems.
        """
        import gc
        if self.model is not None:
            try:
                self.model.cpu()
            except Exception:
                pass
            del self.model
            self.model = None
        gc.collect()
        # Best-effort: release any cached MPS buffers if torch was using MPS
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    @torch.no_grad()
    def __call__(self, image: Image.Image) -> Image.Image:
        """Return an RGBA copy of `image` with alpha = BiRefNet's foreground mask."""
        if self.model is None:
            raise RuntimeError("BiRefNetRembg has been unloaded; instantiate a new one to run again")
        rgb = image.convert("RGB")
        orig_size = rgb.size  # (W, H)
        # Transform: resize → to tensor → ImageNet-normalize
        img1024 = rgb.resize((1024, 1024), Image.Resampling.BILINEAR)
        arr = np.asarray(img1024, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        tensor = (tensor - _IMAGENET_MEAN) / _IMAGENET_STD
        tensor = tensor.to(self.device)

        # Forward → returns a list of progressively refined logits; take the last.
        preds = self.model(tensor)[-1].sigmoid().detach().cpu()
        mask_1024 = preds[0, 0].numpy()  # (1024, 1024) in [0, 1]
        mask_img = Image.fromarray((mask_1024 * 255).astype(np.uint8)).resize(orig_size, Image.Resampling.BILINEAR)

        rgba = rgb.convert("RGBA")
        rgba.putalpha(mask_img)
        return rgba

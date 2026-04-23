"""Image conditioner: DINOv3 ViT-L/16 via torch, exporting mx.array features.

For v1 we run the image encoder in torch (CPU or MPS) and convert the output
patch features to MLX before feeding into the flow DiTs' cross-attention. The
encoder is small (~300M) relative to the 4B DiTs, and this keeps us away from
porting DINOv3's exact numerics while still producing the same conditioning
vector upstream expects.

Mirrors `upstream/trellis2/modules/image_feature_extractor.py:DinoV3FeatureExtractor`.
"""
from __future__ import annotations

from typing import List, Union

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import DINOv3ViTModel

# ImageNet normalization (same as upstream's torchvision Normalize call)
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class DinoV3FeatureExtractor:
    """Returns (B, N_patches, 1024) mx.array conditioning features."""

    def __init__(self, model_name: str, image_size: int = 512, device: str = "cpu"):
        self.model_name = model_name
        self.image_size = image_size
        self.device = device
        self.model = DINOv3ViTModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(device)

    def to(self, device: str) -> "DinoV3FeatureExtractor":
        self.device = device
        self.model.to(device)
        return self

    def _extract_features(self, image: torch.Tensor) -> torch.Tensor:
        """Manual layer stack (matches upstream extract_features) for the last
        pre-norm hidden states, then an affine-less final LayerNorm."""
        image = image.to(self.model.embeddings.patch_embeddings.weight.dtype)
        hidden_states = self.model.embeddings(image, bool_masked_pos=None)
        position_embeddings = self.model.rope_embeddings(image)
        for layer_module in self.model.layer:
            hidden_states = layer_module(hidden_states, position_embeddings=position_embeddings)
        return F.layer_norm(hidden_states, hidden_states.shape[-1:])

    @torch.no_grad()
    def __call__(self, image: Union[torch.Tensor, List[Image.Image]]) -> mx.array:
        if isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image)
            rs = [i.resize((self.image_size, self.image_size), Image.LANCZOS) for i in image]
            arrs = [np.array(r.convert("RGB")).astype(np.float32) / 255.0 for r in rs]
            batch = np.stack([a.transpose(2, 0, 1) for a in arrs], axis=0)
            image = torch.from_numpy(batch)
        assert image.ndim == 4, f"expected (B, 3, H, W), got {image.shape}"
        image = image.to(self.device)
        image = (image - _IMAGENET_MEAN.to(self.device)) / _IMAGENET_STD.to(self.device)
        features = self._extract_features(image)  # (B, N, 1024)
        return mx.array(features.float().cpu().numpy())

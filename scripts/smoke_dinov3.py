"""Download DINOv3 ViT-L/16 and extract features from the TRELLIS.2 example image."""
from __future__ import annotations

import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trellis2_mlx.image_cond import DinoV3FeatureExtractor
from trellis2_mlx.preprocess import preprocess_image


def main() -> int:
    img = Image.open(ROOT / "upstream/assets/example_image/T.png")
    pre = preprocess_image(img)
    print(f"preprocessed image: {pre.size}")

    t0 = time.time()
    extractor = DinoV3FeatureExtractor("facebook/dinov3-vitl16-pretrain-lvd1689m", image_size=512, device="cpu")
    t1 = time.time()
    print(f"loaded DINOv3 in {t1 - t0:.1f}s")

    t0 = time.time()
    cond = extractor([pre])
    t1 = time.time()
    print(f"feature extraction: {t1 - t0:.2f}s  shape={tuple(cond.shape)} dtype={cond.dtype}")
    import numpy as np
    arr = np.asarray(cond)
    print(f"cond stats: mean={arr.mean():.4f} std={arr.std():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

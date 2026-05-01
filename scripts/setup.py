"""Download upstream weights for first-time setup.

Idempotent: re-running is a no-op if files already cached.

Downloads:
  - microsoft/TRELLIS.2-4B          (~10 GB, MIT)   → weights/
  - microsoft/TRELLIS-image-large   (~150 MB, MIT)  → weights/ckpts/ss_dec_*
    (TRELLIS.2 reuses TRELLIS-v1's sparse-structure decoder verbatim)

Verifies DINOv3 access (gated; cannot be auto-downloaded). BiRefNet is fetched
lazily by the rembg path on the first non-RGBA input — no action needed here.

Usage:
    hf auth login
    python scripts/setup.py
    python scripts/run_example_pbr.py --image assets/T.png --pipeline-type 512
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"

# TRELLIS.2-4B: skip encoders (we only run decoders at inference) to save ~1.4 GB.
TRELLIS2_PATTERNS = [
    "pipeline.json",
    "ckpts/ss_flow_img_dit_1_3B_64_bf16.*",
    "ckpts/slat_flow_img2shape_dit_1_3B_*_bf16.*",
    "ckpts/slat_flow_imgshape2tex_dit_1_3B_*_bf16.*",
    "ckpts/shape_dec_next_dc_f16c32_fp16.*",
    "ckpts/tex_dec_next_dc_f16c32_fp16.*",
]

DINOV3_REPO = "facebook/dinov3-vitl16-pretrain-lvd1689m"


def _download_trellis2() -> None:
    from huggingface_hub import snapshot_download
    print(f"[1/3] microsoft/TRELLIS.2-4B  →  {WEIGHTS}")
    snapshot_download(
        repo_id="microsoft/TRELLIS.2-4B",
        local_dir=str(WEIGHTS),
        allow_patterns=TRELLIS2_PATTERNS,
    )


def _download_ss_dec() -> None:
    from huggingface_hub import hf_hub_download
    print(f"[2/3] microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16  →  {WEIGHTS / 'ckpts'}")
    for ext in ("safetensors", "json"):
        hf_hub_download(
            repo_id="microsoft/TRELLIS-image-large",
            filename=f"ckpts/ss_dec_conv3d_16l8_fp16.{ext}",
            local_dir=str(WEIGHTS),
        )


def _verify_dinov3() -> bool:
    """Probe the gated DINOv3 repo. Returns True if accessible."""
    print(f"[3/3] verifying access to {DINOV3_REPO}")
    try:
        from huggingface_hub import HfApi
        HfApi().repo_info(DINOV3_REPO)
    except Exception as e:
        print(f"      ✗ cannot access {DINOV3_REPO}: {e}")
        print(f"")
        print(f"      DINOv3 is gated. To grant access:")
        print(f"        1. Visit https://huggingface.co/{DINOV3_REPO}")
        print(f"        2. Click 'Agree and access' (review usually takes <1 min)")
        print(f"        3. Run: hf auth login    (if you haven't yet)")
        print(f"        4. Re-run this script")
        return False
    print(f"      ✓ DINOv3 accessible")
    return True


def main() -> int:
    WEIGHTS.mkdir(exist_ok=True)
    _download_trellis2()
    _download_ss_dec()
    if not _verify_dinov3():
        return 1
    print()
    print("Setup complete. Try:")
    print("  python scripts/run_example_pbr.py --image assets/T.png --pipeline-type 512")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

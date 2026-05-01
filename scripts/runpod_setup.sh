#!/usr/bin/env bash
#
# Idempotent setup for capturing an upstream TRELLIS.2 reference dump on
# RunPod (or any CUDA box). Run as `bash runpod_setup.sh` after scp'ing
# `dump_upstream_intermediates.py` and `T.png` into /workspace.
#
# Lessons baked in (each one cost us a debug round on the 4090 attempt):
#   1. RTX 4090 (24 GB) cannot fit the texture VAE decode at 512 resolution
#      — the `subs` from the shape VAE alone are ~12-14 GB. **Use A100 40GB
#      minimum.** A100 80GB / H100 80GB are roomier.
#   2. Container disk: deploy with 100 GB. /workspace volume: 100 GB.
#   3. Pre-set env vars in the pod template (otherwise re-export every reconnect):
#         HF_HOME=/workspace/hf-cache
#         HF_HUB_CACHE=/workspace/hf-cache/hub
#         TMPDIR=/workspace/tmp
#         PIP_CACHE_DIR=/workspace/pip-cache
#         UV_CACHE_DIR=/workspace/uv-cache
#   4. PyTorch 2.6.0 + CUDA 12.4 *devel* image is what upstream targets.
#      The CUDA 12.8 driver typical on RunPod 4090s is forward-compatible
#      but the pre-built wheels' nvidia-cudnn-cu13 conflicts with torch's
#      bundled cu12 cuDNN, causing CUDNN_STATUS_NOT_INITIALIZED. We avoid
#      this by always installing torch+torchvision *with* deps (so the
#      correct cu12 nvidia sidecars stay pinned).
#   5. uv's resolver is more aggressive than pip: it'll upgrade torch to
#      2.11+cu130 the moment any package allows it. We pin torch in front
#      of FlexGEMM and re-pin again afterwards.
#   6. transformers 5.x DINOv3 moved layer modules from `model.layer` to
#      `model.model.layer`. Patched in-place via sed.
#   7. flash-attn's setup.py imports torch — must use --no-build-isolation
#      so it sees the system torch.
#   8. transformers.from_pretrained does NOT read ~/.cache/huggingface/token
#      on its own — only HF_TOKEN env var. Logging in via `hf auth login`
#      is necessary but not sufficient. We export HF_TOKEN explicitly here.
#   9. upstream's o_voxel/postprocess.to_glb has a `.cpu().numpy()` call on
#      a tensor with requires_grad=True, which RuntimeErrors. We sed-patch
#      it to `.detach().cpu().numpy()` (idempotent).
#  10. tmux isn't preinstalled in RunPod's pytorch image — apt-install it
#      so users don't lose their session on disconnect.
#
# Usage:
#   bash runpod_setup.sh                    # full setup + run
#   bash runpod_setup.sh --skip-run         # setup only, run manually
#   SKIP_INSTALL=1 bash runpod_setup.sh     # already set up; just re-run dump
#   DUMP_ARGS="--block-diag" bash runpod_setup.sh   # extra flags to dump script
#   NO_TIMESTAMP=1 bash runpod_setup.sh     # disable per-line timestamp prefix

set -euo pipefail

# -- Timestamp every line of output -------------------------------------------
# Re-exec self through `ts` (from moreutils). Sentinel env var prevents infinite
# recursion. Without this, you can't tell if a step has been hung for 30 sec or
# 5 min — installs and CUDA builds are silent for ages.
if [[ -z "${TIMESTAMPED:-}" ]] && [[ -z "${NO_TIMESTAMP:-}" ]]; then
    if ! command -v ts &>/dev/null; then
        apt-get update -qq 2>/dev/null && apt-get install -y -qq moreutils 2>/dev/null || true
    fi
    if command -v ts &>/dev/null; then
        export TIMESTAMPED=1
        exec bash "$0" "$@" 2>&1 | ts '[%H:%M:%S]'
    fi
    # If ts install failed for any reason, fall through and run un-prefixed.
fi

# -- Config -------------------------------------------------------------------
EXT_DIR="/workspace/ext"
TRELLIS_DIR="$EXT_DIR/TRELLIS.2"
DUMP_SCRIPT="/workspace/dump_upstream_intermediates.py"
INPUT_IMAGE="/workspace/T.png"
OUT_NPZ="/workspace/upstream_ref.npz"
OUT_GLB="/workspace/upstream_ref.glb"

TORCH_VERSION="2.6.0"
TORCHVISION_VERSION="0.21.0"
TORCH_INDEX="https://download.pytorch.org/whl/cu124"

# -- Helpers ------------------------------------------------------------------
log()  { echo "[setup] $*"; }
have() { command -v "$1" &>/dev/null; }

# -- Sanity checks -----------------------------------------------------------
log "GPU & toolchain:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
log "If GPU < 40 GB VRAM, you'll OOM in tex VAE decode. Use A100 40GB+."

[[ -f "$DUMP_SCRIPT" ]] || { echo "missing $DUMP_SCRIPT — scp it first"; exit 1; }
[[ -f "$INPUT_IMAGE" ]] || { echo "missing $INPUT_IMAGE — scp it first"; exit 1; }

# -- tmux (so disconnects don't kill the run) --------------------------------
if ! have tmux; then
    log "Installing tmux (so a dropped ssh doesn't kill the run)..."
    apt-get update -qq && apt-get install -y -qq tmux
fi

# -- Env (in case the pod template didn't set these) -------------------------
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/workspace/hf-cache/hub}"
export TMPDIR="${TMPDIR:-/workspace/tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/pip-cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/uv-cache}"
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TMPDIR" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$EXT_DIR"

# -- uv install --------------------------------------------------------------
if ! have uv; then
    log "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="/root/.local/bin:$PATH"
fi
have uv || { echo "uv install failed"; exit 1; }
UV="uv pip install --system"

if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then

# -- Step 1: pin torch + torchvision (WITH deps so cu12 cuDNN sidecars come along)
log "Installing torch $TORCH_VERSION + torchvision $TORCHVISION_VERSION (cu124, with deps)..."
$UV --force-reinstall torch=="$TORCH_VERSION" torchvision=="$TORCHVISION_VERSION" \
    --index-url "$TORCH_INDEX"

# Sanity
python -c "
import torch
assert torch.__version__.startswith('$TORCH_VERSION'), f'wrong torch: {torch.__version__}'
assert torch.version.cuda == '12.4', f'wrong cuda: {torch.version.cuda}'
print('torch ok:', torch.__version__, torch.version.cuda)
print('cuDNN:', torch.backends.cudnn.version())
"

# -- Step 2: basic deps (NOT flash-attn — it needs --no-build-isolation)
log "Installing basic Python deps..."
$UV \
    imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja \
    trimesh tensorboard pandas lpips zstandard kornia timm \
    safetensors xatlas fast-simplification setuptools wheel pybind11 \
    git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# Re-pin torch in case a transitive bumped it
$UV --force-reinstall --no-deps torch=="$TORCH_VERSION" torchvision=="$TORCHVISION_VERSION" \
    --index-url "$TORCH_INDEX"

# -- Step 3: flash-attn (no-build-isolation since its setup.py imports torch)
log "Building flash-attn 2.7.3 (5-10 min on first build)..."
$UV --no-build-isolation flash-attn==2.7.3

# -- Step 4: clone CUDA extensions (idempotent)
log "Cloning CUDA extensions into $EXT_DIR..."
cd "$EXT_DIR"
[[ -d nvdiffrast ]] || git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git
[[ -d CuMesh    ]] || git clone --recursive https://github.com/JeffreyXiang/CuMesh.git
[[ -d FlexGEMM  ]] || git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git
[[ -d TRELLIS.2 ]] || git clone --recursive https://github.com/microsoft/TRELLIS.2.git

# -- Step 5: build CUDA extensions in the right order with torch re-pins
log "Building nvdiffrast..."
$UV --no-build-isolation ./nvdiffrast

log "Building CuMesh (5 min compile)..."
$UV --no-build-isolation ./CuMesh

log "Pre-pinning torch before FlexGEMM (which would otherwise pull torch 2.11)..."
$UV --force-reinstall --no-deps torch=="$TORCH_VERSION" torchvision=="$TORCHVISION_VERSION" \
    --index-url "$TORCH_INDEX"

log "Building FlexGEMM..."
$UV --no-build-isolation ./FlexGEMM

log "Re-pinning torch after FlexGEMM..."
$UV --force-reinstall --no-deps torch=="$TORCH_VERSION" --index-url "$TORCH_INDEX"

log "Building o-voxel..."
$UV --no-build-isolation ./TRELLIS.2/o-voxel

# -- Step 6: transformers 5.x for DINOv3 (with the runtime deps it needs)
log "Installing transformers 5.x..."
$UV --no-deps transformers==5.7.0 huggingface-hub==1.12.2 tokenizers==0.22.2
# Runtime deps that --no-deps skipped:
$UV regex packaging requests

# -- Step 7: patch upstream's DINOv3 layer access (model.layer → model.model.layer)
log "Patching DINOv3 layer access for transformers 5.x..."
PATCH_FILE="$TRELLIS_DIR/trellis2/modules/image_feature_extractor.py"
if ! grep -q "_layers = getattr" "$PATCH_FILE"; then
    sed -i 's|        for i, layer_module in enumerate(self.model.layer):|        _layers = getattr(self.model, "layer", None) or self.model.model.layer\n        for i, layer_module in enumerate(_layers):|' \
        "$PATCH_FILE"
fi
grep -n "_layers\|model\.layer" "$PATCH_FILE" | head -3

# -- Step 7b: patch upstream's o_voxel.postprocess.to_glb (.numpy() on grad tensor)
log "Patching o_voxel.postprocess for .detach() before .numpy()..."
OVOXEL_POSTPROCESS=$(python -c "import o_voxel.postprocess as m; print(m.__file__)")
if grep -q '\.cpu()\.numpy()' "$OVOXEL_POSTPROCESS" && ! grep -q '\.detach()\.cpu()\.numpy()' "$OVOXEL_POSTPROCESS"; then
    sed -i 's|\.cpu()\.numpy()|.detach().cpu().numpy()|g' "$OVOXEL_POSTPROCESS"
    log "  patched $OVOXEL_POSTPROCESS"
fi

fi  # SKIP_INSTALL guard

# -- Step 8: HF auth — must run EVERY time, even with SKIP_INSTALL=1.
# transformers.from_pretrained ignores ~/.cache/huggingface/token and ONLY
# checks the HF_TOKEN env var, so we must export it explicitly.
log "HF auth check..."
if [[ -z "${HF_TOKEN:-}" ]] && [[ -f /root/.cache/huggingface/token ]]; then
    export HF_TOKEN="$(cat /root/.cache/huggingface/token)"
    log "  exported HF_TOKEN from cached login"
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
    log "  No HF_TOKEN set and no cached login. Run 'hf auth login' first."
    [[ "${1:-}" != "--skip-run" ]] && exit 1
fi
# Verify access to the gated DINOv3 repo before kicking off the long dump.
if ! HF_TOKEN="$HF_TOKEN" python -c "
from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ['HF_TOKEN'])
api.repo_info('facebook/dinov3-vitl16-pretrain-lvd1689m')
print('  DINOv3 access OK')
" 2>&1; then
    log "  Cannot access facebook/dinov3-vitl16-pretrain-lvd1689m."
    log "  Visit https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m"
    log "  while logged in, click 'Agree and access repository'."
    [[ "${1:-}" != "--skip-run" ]] && exit 1
fi

# -- Step 9: kick off the dump (foreground; wrap in nohup yourself if needed)
# Pass extra args via DUMP_ARGS env var, e.g. DUMP_ARGS="--block-diag" bash runpod_setup.sh
if [[ "${1:-}" != "--skip-run" ]]; then
    log "Running dump → $OUT_NPZ + $OUT_GLB ${DUMP_ARGS:+(extra: $DUMP_ARGS)}"
    cd "$TRELLIS_DIR"
    PYTHONPATH=. python -u "$DUMP_SCRIPT" \
        --image "$INPUT_IMAGE" \
        --seed 42 \
        --pipeline-type 512 \
        --out "$OUT_NPZ" \
        --glb-out "$OUT_GLB" \
        ${DUMP_ARGS:-}
    log "Done. Pull results with:"
    log "  scp -P <port> -i <key> 'root@<ip>:/workspace/upstream_ref.{npz,glb}' ./artifacts/"
fi

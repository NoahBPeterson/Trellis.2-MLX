# RunPod runbook: capture upstream TRELLIS.2 reference dump

Goal: run `microsoft/TRELLIS.2-4B` on `T.png` with seed=42 on a CUDA GPU,
save stage-by-stage intermediates, and bring the .npz back to our laptop
so we can numerically compare against `artifacts/pbr_intermediates.npz`.

## 1. Provision the pod

[runpod.io](https://runpod.io) → **Deploy** → **GPU Pods**.

- **GPU**: A100 80GB (works), H100 80GB (faster), or RTX A6000 48GB (cheapest, fits).
- **Template**: `runpod/pytorch:2.6.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
  (or any "PyTorch 2.6.0 + CUDA 12.4 devel" template — the **devel** image
  matters; `nvcc` is needed to build the CUDA extensions).
- **Disk**: 50GB+ container, 30GB+ volume (model weights are ~15GB).
- Click **Deploy** and wait ~1 minute for it to come up. Click **Connect** →
  copy the SSH command (looks like `ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519`).

## 2. Copy script + image to the pod

From this local machine:

```bash
# replace <pod-ip> and <pod-port> with what RunPod gave you.
SSH="ssh -p <pod-port> root@<pod-ip>"
SCP="scp -P <pod-port>"

$SCP scripts/dump_upstream_intermediates.py root@<pod-ip>:/workspace/
$SCP upstream/assets/example_image/T.png root@<pod-ip>:/workspace/
```

## 3. Build the CUDA stack on the pod

SSH in, then run:

```bash
cd /workspace
git clone --recursive https://github.com/microsoft/TRELLIS.2.git
cd TRELLIS.2

# Match upstream's setup.sh — install everything we need.
# The Python in the runpod pytorch:2.6 image is the right one already.
pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja \
            trimesh transformers tensorboard pandas lpips zstandard \
            kornia timm huggingface_hub safetensors xatlas fast-simplification

pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

pip install flash-attn==2.7.3

# CUDA extensions (each takes 3-8 min to compile)
mkdir -p /tmp/extensions
git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
pip install /tmp/extensions/nvdiffrast --no-build-isolation

git clone --recursive https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh
pip install /tmp/extensions/CuMesh --no-build-isolation

git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM
pip install /tmp/extensions/FlexGEMM --no-build-isolation

# o-voxel ships in the TRELLIS.2 repo
pip install /workspace/TRELLIS.2/o-voxel --no-build-isolation

# The TRELLIS.2 repo itself (so `from trellis2.pipelines import ...` works)
cd /workspace/TRELLIS.2 && pip install -e .

# HF auth — needs access to facebook/dinov3-vitl16-pretrain-lvd1689m, briaai/RMBG-2.0
huggingface-cli login   # paste a token with read access
```

## 4. Run the dump

Still on the pod:

```bash
cd /workspace
python dump_upstream_intermediates.py \
    --image T.png \
    --seed 42 \
    --pipeline-type 512 \
    --out /workspace/upstream_ref.npz \
    --glb-out /workspace/upstream_ref.glb
```

Expected timing on A100: ~5-10 minutes total (pipeline load ~1 min,
generation ~3-5 min, GLB export ~1 min). Output:

```
=== upstream dump on NVIDIA A100-SXM4-80GB ===
  pipeline loaded (XXs)
  preprocess: ...
  cond_512: ...
  ss_coords: ...
  shape_slat: ...
  tex_slat: ...
  mesh: ...
  voxel: ...

wrote /workspace/upstream_ref.npz (XX.X MB)

=== stats summary ===
  preprocessed_image: ...
  cond_512: shape=... mean=... std=...
  ...
```

## 5. Pull results back to local

```bash
$SCP root@<pod-ip>:/workspace/upstream_ref.npz   ./artifacts/
$SCP root@<pod-ip>:/workspace/upstream_ref.glb   ./artifacts/
```

## 6. Stop the pod

RunPod web UI → **Stop** (saves volume; resume later) or **Terminate**
(deletes everything; cheaper if one-shot).

## 7. Compare

```bash
python scripts/diff_intermediates.py \
    --upstream artifacts/upstream_ref.npz \
    --ours artifacts/pbr_intermediates.npz
```

The output prints per-channel attribute distributions for both runs side
by side, with `<-- DIVERGENCE` flags on any channel that differs by more
than 0.1 in mean. The first divergent stage tells us which part of our
MLX port is wrong.

## Troubleshooting

- **`nvcc not found`**: you picked a runtime image, not devel. Re-deploy
  with a `*-devel-*` template.
- **`flash-attn` build errors**: `pip install ninja` first, then retry.
  Make sure CUDA arch matches GPU (`TORCH_CUDA_ARCH_LIST=8.0` for A100,
  `9.0` for H100).
- **HF model download hangs**: confirm `huggingface-cli whoami` works and
  you've requested access at
  https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m and
  https://huggingface.co/briaai/RMBG-2.0.
- **OOM on GPU**: pipeline loads with `low_vram=True` by default, should
  fit on 24GB+. If you're on a smaller GPU (12-16GB), reduce
  `--pipeline-type` to `512` (already the default in this script).

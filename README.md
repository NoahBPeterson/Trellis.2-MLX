---
license: mit
pipeline_tag: image-to-3d
library_name: trellis2-mlx
base_model: microsoft/TRELLIS.2-4B
language:
- en
tags:
- mlx
- apple-silicon
- trellis2
- image-to-3d
- flow-matching
- sparse-voxel
- 3d-generation
---

# TRELLIS.2-4B — MLX port for Apple Silicon

An MLX-native re-implementation of Microsoft's [TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B) image-to-3D model, running on a MacBook (Apple Silicon, unified memory). The compute-heavy transformer backbones (~4B params across three flow-matching DiTs) and the sparse 3D VAE run on-device via MLX; CUDA-bound post-processing (mesh extraction, UV atlas) is replaced with CPU / pure-Python equivalents.

Weights are consumed directly from the upstream `microsoft/TRELLIS.2-4B` torch safetensors at load time — no separate conversion artifacts. The transforms (key rename for `nn.Sequential` indexing, dense Conv3d axis permute, bf16 round-trip) happen inline in [`pipeline.py`](./trellis2_mlx/pipeline.py), keeping a clean `git clone` → `setup.py` → run loop with no re-uploaded weights.

**Status:** single-image → fully textured PBR GLB working end-to-end at `pipeline_type` ∈ {`512`, `1024`, `1024_cascade`}. `1536_cascade` is wired but needs more RAM than most Macs have. The PBR pipeline (cone-cluster UV unwrap + per-texel BVH bake) produces output visually equivalent to upstream CUDA on the reference T.png. Single-seed bit-exact parity isn't achievable because MLX RNG ≠ PyTorch RNG, but model code and post-processing are bit-faithful within bf16 precision floor (verified via per-block hidden-state diff against a CUDA reference dump; see [`scripts/diag_per_block_compare.py`](./scripts/diag_per_block_compare.py)). Pass `--vertex-colors` to opt out of UV atlas baking for a smaller vertex-colored GLB.

---

## Architecture

```
┌─────────────┐
│  Input PNG  │
└──────┬──────┘
       │ alpha-crop, premultiply
       ▼
┌──────────────────────┐   ~300M params, facebook/dinov3-vitl16
│ DINOv3 image encoder │   (torch, CPU/MPS) → (1, N, 1024) patch features
└──────┬───────────────┘
       │ cond — feeds every downstream DiT via cross-attention
       ▼   ═══════════════ STAGE 1: where is the object? ═══════════════
┌──────────────────────────┐  1.3B bf16 · 30-block DiT · 12 Euler steps
│  SS Flow DiT (dense)     │  operates on 16³ = 4096 dense tokens with 3D RoPE
│  output: 16³ × 8 latent  │
└──────────┬───────────────┘
           │
           ▼
┌─────────────────────────────┐  small dense conv3d stack (~50M, fp16)
│  SS Decoder                 │  16³ × 8 latent → 64³ binary occupancy
│                             │  → maxpool to 32³ → active voxel coords
└──────────┬──────────────────┘
           │  (3k–50k active voxels)
           ▼   ═══════ STAGE 2: what SHAPE is the object? ═══════════════
┌─────────────────────────────────┐  1.3B bf16 · sparse DiT · 12 Euler steps
│  Shape SLat Flow DiT (sparse)   │  one 32-dim latent per active voxel
│                                 │  cross-attends to image cond, 3D RoPE on coords
│  output: (F, 32) shape latent   │
└──────────┬──────────────────────┘
           │
           ▼
┌───────────────────────────────────┐  fp16 · 5-stage sparse UNet
│  Shape VAE decoder                │  [1024,512,256,128,64] channels,
│                                   │  submanifold 3×3×3 conv, 16× spatial up
│  output: per-voxel (xyz, edges,   │
│          quad_lerp) at 512³       │
└──────────┬────────────────────────┘
           │
           ▼
┌──────────────────────────────────┐  pure-Python dual-contouring
│ flexible_dual_grid_to_mesh (CPU) │  voxel data → triangle mesh (V, F)
└──────────┬───────────────────────┘
           │
           ▼
     ┌─────────┐
     │ mesh.glb│  (trimesh export)
     └─────────┘
```

Texture (PBR) pipeline adds a parallel **Stage 3**: a 1.3B sparse DiT (`tex_flow`) cross-attending to image cond and concat-conditioning on the shape latent, plus a 2nd sparse UNet VAE that decodes a `(F, 32)` texture latent to per-voxel PBR attributes (RGB, metallic, roughness, alpha). Both run end-to-end via [`scripts/run_example_pbr.py`](./scripts/run_example_pbr.py). See the component manifest in [`config.json`](./config.json).

---

## Requirements

- **Hardware:** Apple Silicon (M1/M2/M3/M4). Shape pipeline at 512 peaks around 12 GB unified memory. The 1536 cascade will need ≥64 GB.
- **OS:** macOS 13+.
- **Python:** 3.11 or 3.12.
- **MLX:** 0.18.0+.

`torch` is installed for two reasons: the DINOv3 image encoder (runs once per image via `transformers`), and optional validation tests against CPU reference implementations.

---

## Install

This repo uses [`uv`](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone https://github.com/NoahBPeterson/Trellis.2-MLX.git trellis-mlx
cd trellis-mlx
uv sync --extra postprocess --extra image-cond --extra rembg

# Request access to the gated DINOv3 model first:
#   https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m
hf auth login
uv run python scripts/setup.py    # downloads ~10 GB of weights, idempotent
```

`scripts/setup.py` fetches the upstream torch safetensors from `microsoft/TRELLIS.2-4B` plus the SS decoder from `microsoft/TRELLIS-image-large` (TRELLIS.2 reuses TRELLIS-v1's SS decoder verbatim) into `weights/ckpts/`. No conversion step — weights are consumed in their upstream torch layout and transformed at load time.

The `rembg` extra installs `kornia`, `timm`, `torchvision` (BiRefNet's runtime deps) and is only needed for inputs without an alpha channel — if you always pass pre-composited RGBA images you can drop it. BiRefNet (`briaai/RMBG-2.0`) is downloaded automatically on first non-RGBA input.

---

## Quickstart

**Geometry only (no PBR, ~5 stages, fastest):**

```bash
uv run python scripts/run_example.py \
    --image assets/T.png \
    --out  artifacts/sample.glb \
    --pipeline-type 512
```

**Geometry + PBR (7 stages, full UV atlas; ~10 min total on M-series MacBook):**

```bash
uv run python scripts/run_example_pbr.py \
    --image assets/T.png \
    --out  artifacts/sample_pbr.glb \
    --pipeline-type 512
```

Default output is a UV-atlas-textured GLB with full PBR materials (base color + metallic-roughness + alpha, 2k atlas). Pass `--vertex-colors` to skip UV unwrap + atlas bake for a smaller vertex-colored GLB (faster but lower fidelity, suitable for quick-look comparisons).

`--dit-dtype` defaults to `bfloat16` to match upstream's training/inference dtype. Pass `--dit-dtype float16` for ~32% faster sampling at sub-pixel mesh deviation; the float16 grid is finer than upstream's bf16, so single-seed parity diverges measurably even though visual quality is essentially identical (see [Performance](#performance) below).

Or from Python:

```python
from PIL import Image
from trellis2_mlx.pipeline import Trellis2ImageTo3DPipelineMLX
from trellis2_mlx.postprocess.glb_export import export_mesh_glb

pipe = Trellis2ImageTo3DPipelineMLX.from_pretrained(
    ckpt_dir="weights/ckpts",
    pipeline_type="512",
)

V, F = pipe.run(Image.open("path/to/image.png"), seed=42)
export_mesh_glb(V, F, "out.mesh.glb")
```

---

## Performance (M-series MacBook, reference run)

Measured on a reference Apple Silicon MacBook running `scripts/run_example.py` on `upstream/assets/example_image/T.png`. Your mileage will vary by chip tier and unified-memory size.

### `pipeline_type="512"`

| Stage                               | Time   |
| ----------------------------------- | -----: |
| Preprocess (alpha-crop, premul)     |   0.1s |
| DINOv3 image conditioning (torch)   |   4.4s |
| SS flow (dense DiT) + decoder       | 263.3s |
| Shape SLat flow (sparse DiT)        | 120.5s |
| Shape VAE decode + dual-grid mesh   |  21.7s |
| **Total**                           | **~410s** |

Mesh output: 1,730,326 verts / 3,701,394 faces.

### `pipeline_type="1024"`

| Stage                                     | Time    |
| ----------------------------------------- | ------: |
| Preprocess                                |    0.1s |
| DINOv3 @ 1024px (4101 patch tokens)       |    6.7s |
| SS flow (dense DiT) + decoder             |  129.2s |
| Shape SLat flow (sparse DiT, ~19k tokens) | 1204.2s |
| Shape VAE decode + dual-grid mesh         |   97.6s |
| **Total**                                 | **~24 min** |

Mesh output: 6,080,680 verts / 12,001,866 faces (3.8× the 512 mesh, same bbox, higher fidelity).

### `pipeline_type="1024_cascade"`

| Stage                                               | Time    |
| --------------------------------------------------- | ------: |
| Preprocess                                          |    0.1s |
| DINOv3 @ 512 + 1024 (both conds)                    |    8.3s |
| SS flow + decoder                                   |  143.3s |
| Shape SLat cascade (LR @512 + VAE-upsample + HR @1024, 18902 tokens) | 1552.7s |
| Shape VAE decode + dual-grid mesh                   |  104.1s |
| **Total**                                           | **~30 min** |

Mesh output: 6,946,823 verts / 14,091,496 faces. ~7% faster than direct `1024` because the low-res pass constrains the active-voxel set before the expensive HR flow. The cascade tokens end up very close to direct `1024` (18902 vs 19104) so on T.png the speedup is modest; on images whose low-res structure is much sparser than the high-res, the gap widens.

### `pipeline_type="512"` + full PBR + UV atlas

| Stage                                       | Time   |
| ------------------------------------------- | -----: |
| Preprocess                                  |   0.1s |
| DINOv3 image conditioning                   |   3.2s |
| SS flow (dense DiT) + decoder               | 127.5s |
| Shape SLat flow (sparse DiT)                | 112.6s |
| Shape VAE decode + dual-grid mesh           |  24.0s |
| Tex SLat flow (sparse DiT)                  |  65.8s |
| Tex VAE decode → per-vertex PBR             |  19.6s |
| **Sampling + decode subtotal**              | **~5m53s** |
| Decimate (3.4M → 500k faces)                |   3.7s |
| UV unwrap (cone-cluster + per-chart xatlas) | 206.7s |
| Atlas bake (BVH per-texel, 2k atlas)        |  62.2s |
| GLB write                                   |   1.3s |
| **Post-processing subtotal**                | **~4m34s** |
| **Total**                                   | **~10m27s** |

Output: 1.7M-vert dual-grid mesh decimated to 500k faces; 2048×2048 atlas with base-color RGBA + metallic-roughness; ~32 MB GLB.

UV unwrap is the largest single line item — most of it is the per-chart xatlas calls inside cone-cluster (~150k charts on the steampunk T-shape). For applications that don't need texture atlases, `--vertex-colors` skips post-processing entirely (write 1.3s) and shipping the per-vertex PBR directly.

### `pipeline_type="1024"` + full PBR + UV atlas

| Stage                                       | Time     |
| ------------------------------------------- | -------: |
| Preprocess                                  |    0.1s  |
| DINOv3 image conditioning @ 1024            |    6.7s  |
| SS flow (dense DiT) + decoder               |  129.2s  |
| Shape SLat flow (sparse DiT, ~19k tokens)   | 1204.2s  |
| Shape VAE decode + dual-grid mesh @ 1024³   |   97.6s  |
| Tex SLat flow (sparse DiT, ~19k tokens)     |  548.7s  |
| Tex VAE decode → per-vertex PBR             |   72.3s  |
| **Sampling + decode subtotal**              | **~34m18s** |
| Decimate (12M → 545k faces)                 |   29.2s  |
| UV unwrap (cone-cluster + per-chart xatlas) |  212.9s  |
| Atlas bake (BVH per-texel, 2k atlas)        |   61.4s  |
| GLB write                                   |    1.0s  |
| **Post-processing subtotal**                | **~5m05s** |
| **Total**                                   | **~39m20s** |

Output: 6.08M-vert dual-grid mesh decimated to 545k faces; 2048×2048 atlas (downsampled from xatlas's chosen 4813²); ~38 MB GLB.

Material distribution at 1024 is closer to upstream than at 512 — metallic-mean drops from ~0.98 (512) to ~0.81 (1024), tracking upstream CUDA's 0.71 more closely. The higher-resolution shape feature space yields a more diverse PBR distribution.

The flow DiTs currently dominate wall-clock; the sparse VAE has been tuned (batched `np.searchsorted` neighbor maps, per-kernel conv fused via gather, vectorized C2S subdivision) and micro-benchmarks in the 900–3000 GFLOPS range per submanifold conv. The 1024 shape-flow cost grows roughly with the square of the token count (self-attention), so the difference between 512 and 1024 is mostly attention compute. Further DiT-side optimization is planned — see [Roadmap](#roadmap).

### Optional: `--dit-dtype float16` (opt-in, ~25–32% faster)

Apple's Metal SDPA and matmul kernels run ~1.3× faster on fp16 inputs than on bf16. Our upstream DiT weights ship as bf16, but can be cast to fp16 at load time via `--dit-dtype float16`.

> **Numbers below are stale**, captured before the bf16-compute fix in [`d616db3`](https://github.com/NoahBPeterson/Trellis.2-MLX/commit/d616db3) — that fix changed the bf16 path's active-voxel count (3548 → 3754), so absolute vert/face values are no longer accurate. The relative ~32% speedup of fp16 over bf16 still holds (it's a hardware-level matmul throughput ratio). Re-measurement is on the to-do list.

|             | bf16 (default) | fp16 (opt-in) |
| ----------- | -------------: | ------------: |
| Total       |          410s  |          278s  (**–32%**) |
| Verts       |       1651404  |       1651540  (+0.008%) |
| Faces       |       3501928  |       3501882  (–0.001%) |
| Bbox        |      identical |      identical (4 decimals) |
| Median vertex displacement vs bf16 | — | **1e-6**  (sub-pixel) |
| Max vertex displacement            | — | 0.0019 (0.19% of extent) |

Essentially lossless. Off by default for strict numerical parity with upstream; opt in via the CLI flag or `dit_compute_dtype="float16"` to `from_pretrained`.

For comparison, upstream on an NVIDIA H100 reports ~3s at 512³ and ~17s at 1024³. We are not trying to match that; the objective is "works on a MacBook."

---

## Repo layout

```
trellis-mlx/
├── README.md                            ← this file
├── pyproject.toml                       ← uv-managed project
├── assets/T.png                         ← reference input image (committed, ~1.7 MB)
├── weights/
│   ├── pipeline.json                    ← sampler params + normalization stats (committed)
│   └── ckpts/                           ← downloaded by `scripts/setup.py` (gitignored, ~10 GB)
│       ├── ss_flow_img_dit_1_3B_64_bf16.{safetensors,json}            (2.4 GB)
│       ├── slat_flow_img2shape_dit_1_3B_{512,1024}_bf16.*             (2.4 GB ea.)
│       ├── slat_flow_imgshape2tex_dit_1_3B_{512,1024}_bf16.*          (2.4 GB ea.)
│       ├── shape_dec_next_dc_f16c32_fp16.{safetensors,json}           (905M)
│       ├── tex_dec_next_dc_f16c32_fp16.{safetensors,json}             (905M)
│       └── ss_dec_conv3d_16l8_fp16.{safetensors,json}                 (141M, from TRELLIS-image-large)
├── trellis2_mlx/
│   ├── pipeline.py                      ← Trellis2ImageTo3DPipelineMLX + load-time torch→MLX
│   ├── samplers.py                      ← FlowEulerGuidanceIntervalSampler (pure MLX)
│   ├── image_cond.py                    ← DinoV3FeatureExtractor (torch wrapper)
│   ├── preprocess.py                    ← alpha-aware image preprocess
│   ├── rembg.py                         ← BiRefNet wrapper (lazy, RGB-only inputs)
│   ├── models/
│   │   ├── flow_dit.py                  ← SparseStructureFlowModel + SLatFlowModel
│   │   ├── sparse_vae.py                ← FlexiDualGridVaeDecoder + SparseUnetVaeDecoder
│   │   └── ss_decoder.py                ← dense conv3d SS decoder
│   ├── modules/
│   │   ├── attention.py, rope.py, norm.py, blocks.py, pos_embed.py
│   │   ├── sparse.py                    ← submanifold conv + S2C/C2S + neighbor maps
│   │   └── sparse_tensor.py
│   └── postprocess/
│       ├── dual_grid.py                 ← pure-Python `flexible_dual_grid_to_mesh`
│       ├── atlas.py                     ← cone-cluster UV unwrap + BVH atlas bake
│       └── glb_export.py
├── scripts/
│   ├── setup.py                         ← one-shot weight download (HF snapshot_download)
│   ├── run_example.py                   ← image → GLB CLI (geometry only)
│   ├── run_example_pbr.py               ← image → textured GLB CLI (full PBR)
│   ├── rebake_from_cache.py             ← skip-sampling rebake from cached intermediates
│   ├── dump_upstream_intermediates.py   ← capture CUDA reference dump (for diagnostics)
│   ├── diff_intermediates.py            ← distribution diff vs cached intermediates
│   └── diag_*.py, replay_*.py           ← per-block / RNG isolation diagnostics
└── tests/                               ← attention / DiT / sparse-conv / dual-grid numerics
```

**Loading note.** This repo is *not* loadable via `transformers.AutoModel.from_pretrained` — the `transformers` library has no MLX backend. Use:

```python
from trellis2_mlx.pipeline import Trellis2ImageTo3DPipelineMLX
pipe = Trellis2ImageTo3DPipelineMLX.from_pretrained(
    ckpt_dir="weights/ckpts",
    pipeline_type="512",
)
```

---

## Components (summary)

| Component | Class | Dtype | File (under `weights/ckpts/`) | Inference |
| --- | --- | ---: | --- | :---: |
| SS Flow DiT | `SparseStructureFlowModel` | bf16 | `ss_flow_img_dit_1_3B_64_bf16.safetensors` | ✅ |
| SS Decoder | `SparseStructureDecoder` | fp16 | `ss_dec_conv3d_16l8_fp16.safetensors` | ✅ |
| Shape SLat DiT (512) | `SLatFlowModel` | bf16 | `slat_flow_img2shape_dit_1_3B_512_bf16.safetensors` | ✅ |
| Shape SLat DiT (1024) | `SLatFlowModel` | bf16 | `slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors` | ✅ |
| Shape VAE decoder | `FlexiDualGridVaeDecoder` | fp16 | `shape_dec_next_dc_f16c32_fp16.safetensors` | ✅ |
| Tex SLat DiT (512) | `SLatFlowModel` | bf16 | `slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors` | ✅ |
| Tex SLat DiT (1024) | `SLatFlowModel` | bf16 | `slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors` | ✅ |
| Tex VAE decoder | `SparseUnetVaeDecoder` | fp16 | `tex_dec_next_dc_f16c32_fp16.safetensors` | ✅ |
| Image encoder | DINOv3 ViT-L/16 (torch) | — | external: `facebook/dinov3-vitl16-pretrain-lvd1689m` | ✅ |
| Background removal | BiRefNet (torch) | — | external: `briaai/RMBG-2.0` | ✅ on-demand for non-RGBA inputs |

Each safetensors ships with a sibling `<stem>.json` describing the model's constructor args. `setup.py` skips the (unused-at-inference) shape/tex VAE encoders.

---

## Load-time tensor transforms

Upstream torch tensors are streamed through three transforms in [`pipeline.py`](./trellis2_mlx/pipeline.py)'s `_convert_torch_tensor`:

- **DiT Sequential indexing.** Torch `nn.Sequential` indexes children directly (`mlp.0.weight`); MLX stores them under `.layers[i]` (`mlp.layers.0.weight`). A regex rewrites `mlp.<N>`, `adaLN_modulation.<N>`, `middle_block.<N>`, `out_layer.<N>`.
- **Dense Conv3d axis order.** Torch `Conv3d` weights are `(Co, Ci, Kd, Kh, Kw)`; MLX expects channels-last `(Co, Kd, Kh, Kw, Ci)`. Detected by shape pattern (sparse conv weights upstream are already in MLX layout — no permute).
- **bf16 dtype handling.** Numpy can't represent bf16, so bf16 tensors round-trip through fp32 during the torch → MLX hand-off, then cast back to `mx.bfloat16`. ~50s on first load; first-call only since Metal caches the loaded weights.

---

## Validation

```bash
uv run pytest tests/
```

Checks:

- `tests/test_attention.py` — MLX MHA vs upstream's `_naive_sdpa`, max-abs ~1e-4 fp32 / ~5e-3 bf16.
- `tests/test_dit_block.py` — one full `ModulatedTransformerCrossBlock` forward pass vs torch-CPU upstream.
- `tests/test_flow_dit.py` — end-to-end `SparseStructureFlowModel` numerics.
- `tests/test_sparse_conv.py`, `tests/test_sparse_vs_dense.py` — submanifold-conv correctness. The `vs_dense` variant cross-checks against a dense torch `Conv3d` and was the authoritative test that caught the first-pass kernel axis-ordering bug.
- `tests/test_dual_grid.py` — dual-grid mesh extraction fixture parity.
- `tests/test_rope.py`, `tests/test_norm.py`, `tests/test_sampler.py` — module-level unit tests.

---

## Roadmap

**Immediate:**
- **DiT performance.** Sampling dominates wall-clock (~6 min of the ~10-min 512 PBR run; ~34 min of the ~39-min 1024 PBR run). Likely wins: graph-level fusion, attention kernel tuning for our specific (1, F, C) sparse shapes, KV-cache for fixed-cond CFG passes.
- **Mesh quality.** Polygon faceting visible on small features (gears, rivets) compared to upstream — likely from `fast-simplification`'s topology-agnostic decimation vs CuMesh's curvature-aware decimator. Worth investigating a port of CuMesh's decimator.
- **Native-MLX DINOv3 port.** Removes the last torch dependency at inference time (saves the ~3–4s torch-CPU cost and the gated-model headache for new contributors).
- **`pipeline_type="1536_cascade"`** — code path is wired (same as `1024_cascade` with `hr_resolution=1536`), but token count blows past the 49k cap on most inputs. Cap-downgrade fallback is implemented; needs a machine with 64+ GB to verify.

**v2:**
- 1024-resolution shape + tex DiTs at full quality (currently shape@1024 takes ~32 min, tex@1024 isn't smoke-tested).
- Int8/Int4 weight quantization (deferred after numerical investigation — int4 errors compound to 268% p99 across 30 blocks; int8 is borderline at 12% drift; neither offers speed gains on M1 Pro for our matmul shapes; see `scripts/verify_quant_numerics.py`).

**Diagnostic / verification tooling** (in tree, see [`scripts/`](./scripts)):
- `dump_upstream_intermediates.py` + `runpod_setup.sh` — capture upstream CUDA reference dump on a RunPod A100. Per-stage tensors, per-block hidden states, optional per-stage noise capture.
- `diff_intermediates.py` — distribution diff vs cached `pbr_intermediates.npz`.
- `diag_per_block_compare.py` — ours vs upstream per-block hidden-state comparator under controlled zero input. Used to verify bf16 compute path matches upstream within precision floor.
- `replay_upstream_noise.py` — RNG isolation: feeds upstream's noise into our pipeline to separate RNG-driven variance from model bugs.
- `replay_upstream_through_our_vae.py` — VAE bit-exact verifier.
- `rebake_from_cache.py` — skip-sampling rebake for fast iteration on atlas/export changes.

---

## Provenance & license

This is a faithful port — no re-training, no distillation. All weights are loaded directly from `microsoft/TRELLIS.2-4B` (and the SS decoder from `microsoft/TRELLIS-image-large`) under the upstream MIT license. This port and the MLX modules are released under the same MIT license.

Upstream authors: Jianfeng Xiang, Xiaoxue Chen, Sicheng Xu, Ruicheng Wang, Zelong Lv, Yu Deng, Hongyuan Zhu, Yue Dong, Hao Zhao, Nicholas Jing Yuan, Jiaolong Yang.

```bibtex
@article{xiang2025trellis2,
  title   = {Native and Compact Structured Latents for 3D Generation},
  author  = {Xiang, Jianfeng and Chen, Xiaoxue and Xu, Sicheng and Wang, Ruicheng
             and Lv, Zelong and Deng, Yu and Zhu, Hongyuan and Dong, Yue
             and Zhao, Hao and Yuan, Nicholas Jing and Yang, Jiaolong},
  journal = {Tech report},
  year    = {2025}
}
```

If you publish work built on this specific MLX port, a note pointing back to this repo is appreciated but not required.

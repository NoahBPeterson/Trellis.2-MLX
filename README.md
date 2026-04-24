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

Weights were converted from the upstream `microsoft/TRELLIS.2-4B` safetensors shards — no re-training, same numerics intent, per-tensor bijection recorded alongside each checkpoint.

**Status:** single-image → triangle-mesh GLB working end-to-end at `pipeline_type` ∈ {`512`, `1024`, `1024_cascade`}. `1536_cascade` is wired but needs more RAM than most Macs have. The PBR texture pipeline is scoped but deferred (see [Roadmap](#roadmap)).

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

Texture (PBR) pipeline uses a parallel Stage 3 sparse DiT + 2nd sparse UNet VAE; weights are converted and shipped but not yet wired into the runtime pipeline. See the component manifest in [`config.json`](./config.json).

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
git clone https://huggingface.co/<user>/trellis-mlx
cd trellis-mlx
uv sync --extra postprocess --extra image-cond
```

DINOv3 is an access-gated model; request access at
[facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
and `huggingface-cli login` before first run.

---

## Quickstart

```bash
uv run python scripts/run_example.py \
    --image upstream/assets/example_image/T.png \
    --out  artifacts/sample.glb \
    --pipeline-type 512
```

Or from Python:

```python
from PIL import Image
from trellis2_mlx.pipeline import Trellis2ImageTo3DPipelineMLX
from trellis2_mlx.postprocess.glb_export import export_mesh_glb

pipe = Trellis2ImageTo3DPipelineMLX.from_pretrained(
    ckpt_dir="ckpts",
    pipeline_json="weights/pipeline.json",
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

Mesh output: 1,651,404 verts / 3,501,928 faces.

### `pipeline_type="1024"`

| Stage                               | Time    |
| ----------------------------------- | ------: |
| Preprocess                          |    0.0s |
| DINOv3 @ 1024px (4101 patch tokens) |    6.4s |
| SS flow (dense DiT) + decoder       |  158.2s |
| Shape SLat flow (sparse DiT, ~19k tokens) | 1680.0s |
| Shape VAE decode + dual-grid mesh   |  100.6s |
| **Total**                           | **~32 min** |

Mesh output: 6,772,966 verts / 13,554,918 faces (4.1× the 512 mesh, same bbox, higher fidelity).

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

The flow DiTs currently dominate wall-clock; the sparse VAE has been tuned (batched `np.searchsorted` neighbor maps, per-kernel conv fused via gather, vectorized C2S subdivision) and micro-benchmarks in the 900–3000 GFLOPS range per submanifold conv. The 1024 shape-flow cost grows roughly with the square of the token count (self-attention), so the difference between 512 and 1024 is mostly attention compute. Further DiT-side optimization is planned — see [Roadmap](#roadmap).

For comparison, upstream on an NVIDIA H100 reports ~3s at 512³ and ~17s at 1024³. We are not trying to match that; the objective is "works on a MacBook."

---

## Repo layout

```
trellis-mlx/
├── config.json                          ← top-level HF metadata (see Components below)
├── README.md                            ← this file
├── pyproject.toml                       ← uv-managed project
├── weights/pipeline.json                ← sampler params + normalization stats (from upstream)
├── ckpts/
│   ├── ss_flow_img_dit_1_3B_64.safetensors            (bf16, 2.4 GB)
│   ├── ss_flow_img_dit_1_3B_64.config.json
│   ├── ss_flow_img_dit_1_3B_64.bijection.json         ← torch ↔ MLX tensor-name map
│   ├── slat_flow_img2shape_dit_1_3B_{512,1024}.*      (bf16, 2.4 GB ea.)
│   ├── slat_flow_imgshape2tex_dit_1_3B_{512,1024}.*   (bf16, 2.4 GB ea., texture — deferred)
│   ├── shape_{enc,dec}_next_dc_f16c32.*               (fp16, 676M / 905M)
│   ├── tex_{enc,dec}_next_dc_f16c32.*                 (fp16, deferred)
│   └── ss_dec_conv3d_16l8.*                           (fp16, 141M, from TRELLIS-image-large)
├── trellis2_mlx/
│   ├── pipeline.py                      ← Trellis2ImageTo3DPipelineMLX
│   ├── samplers.py                      ← FlowEulerGuidanceIntervalSampler (pure MLX)
│   ├── image_cond.py                    ← DinoV3FeatureExtractor (torch wrapper)
│   ├── preprocess.py                    ← alpha-aware image preprocess
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
│       └── glb_export.py
├── scripts/
│   ├── convert_weights.py               ← torch safetensors → MLX safetensors
│   ├── inspect_weights.py               ← dumps {name: (shape, dtype)} manifest
│   ├── run_example.py                   ← image → GLB CLI
│   ├── bench_sparse_conv.py             ← per-conv GFLOPS micro-bench
│   └── profile_vae.py                   ← cProfile on shape VAE decode
├── tests/                               ← attention / DiT / sparse-conv / dual-grid numerics
└── upstream/                            ← microsoft/TRELLIS.2 clone (for reference + assets)
```

**Unified Tensors tab.** Every one of the 4418 tensors across all 10 shards is listed under a single `model.safetensors.index.json`. Because several tensor names (e.g. `blocks.0.self_attn.to_qkv.weight`) naturally repeat across the three 1.3B DiTs, each shard's tensors are prefixed with a component tag so the index keeps a flat, collision-free namespace. Example entries:

```
ss_flow.blocks.0.self_attn.to_qkv.weight        → ckpts/ss_flow_img_dit_1_3B_64.safetensors
shape_flow_512.blocks.0.self_attn.to_qkv.weight → ckpts/slat_flow_img2shape_dit_1_3B_512.safetensors
tex_flow_512.blocks.0.self_attn.to_qkv.weight   → ckpts/slat_flow_imgshape2tex_dit_1_3B_512.safetensors
shape_dec.blocks.3.1.to_subdiv.weight           → ckpts/shape_dec_next_dc_f16c32.safetensors
```

The Python pipeline strips the component prefix on load (see `_load_weights_prefixed` in `trellis2_mlx/pipeline.py`); code still sees the original upstream-compatible layer names. The per-file `*.bijection.json` records the pre-prefix torch↔MLX name mapping for each shard, so external tooling that wants to match against upstream `microsoft/TRELLIS.2-4B` can recover the original names too.

**Loading note.** This repo is *not* loadable via `transformers.AutoModel.from_pretrained` — the `transformers` library has no MLX backend. Use:

```python
from trellis2_mlx.pipeline import Trellis2ImageTo3DPipelineMLX
pipe = Trellis2ImageTo3DPipelineMLX.from_pretrained(
    ckpt_dir="ckpts",
    pipeline_json="weights/pipeline.json",
    pipeline_type="512",
)
```

---

## Components (summary)

| Component | Class | Dtype | File | Inference |
| --- | --- | ---: | --- | :---: |
| SS Flow DiT | `SparseStructureFlowModel` | bf16 | `ckpts/ss_flow_img_dit_1_3B_64.safetensors` | ✅ |
| SS Decoder | `SparseStructureDecoder` | fp16 | `ckpts/ss_dec_conv3d_16l8.safetensors` | ✅ |
| Shape SLat DiT (512) | `SLatFlowModel` | bf16 | `ckpts/slat_flow_img2shape_dit_1_3B_512.safetensors` | ✅ |
| Shape SLat DiT (1024) | `SLatFlowModel` | bf16 | `ckpts/slat_flow_img2shape_dit_1_3B_1024.safetensors` | ⏳ |
| Shape VAE decoder | `FlexiDualGridVaeDecoder` | fp16 | `ckpts/shape_dec_next_dc_f16c32.safetensors` | ✅ |
| Shape VAE encoder | `FlexiDualGridVaeEncoder` | fp16 | `ckpts/shape_enc_next_dc_f16c32.safetensors` | — *(training)* |
| Tex SLat DiT (512/1024) | `SLatFlowModel` | bf16 | `ckpts/slat_flow_imgshape2tex_dit_1_3B_{512,1024}.safetensors` | ⏳ |
| Tex VAE decoder/encoder | `SparseUnetVaeDecoder`/`Encoder` | fp16 | `ckpts/tex_{dec,enc}_next_dc_f16c32.safetensors` | ⏳ |
| Image encoder | DINOv3 ViT-L/16 (torch) | — | external: `facebook/dinov3-vitl16-pretrain-lvd1689m` | ✅ |
| Background removal | BiRefNet (torch) | — | external: `briaai/RMBG-2.0` | — *(bypassed for pre-composited RGBA)* |

Full per-component hyperparameters and torch ↔ MLX bijection map live in [`config.json`](./config.json) and each `ckpts/*.config.json`.

---

## Conversion notes

The conversion script is [`scripts/convert_weights.py`](./scripts/convert_weights.py). Non-trivial remappings:

- **DiT Sequential wrapping.** Upstream torch uses `nn.Sequential(Linear, SiLU, Linear)` and stores children as `mlp.0`/`mlp.2`. MLX `nn.Sequential` stores children under `.layers[i]` — the converter rewrites `mlp.<N>` → `mlp.layers.<N>` and similar for `adaLN_modulation`, `middle_block`, etc.
- **Sparse conv weight layout.** Upstream already stores sparse conv kernels as `(Co, Kd, Kh, Kw, Ci)` per [`conv_flex_gemm.py:34`](./upstream/trellis2/modules/sparse/conv/conv_flex_gemm.py). MLX keeps the same layout; no permute needed.
- **Dense conv3d axis order.** SS decoder uses PyTorch `Conv3d` weight layout `(Co, Ci, Kd, Kh, Kw)`; MLX expects channels-last `(Co, Kd, Kh, Kw, Ci)`. The converter permutes dense (non-sparse) conv kernels.
- **qk_rms_norm gamma shape.** `(num_heads, head_dim)` per-head RMSNorm; initialized to 1, unchanged by conversion but called out because it's uncommon.

Every converted file has a sibling `*.bijection.json` recording `{mlx_name: original_torch_name}` for each tensor, for traceability.

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
- Flow-DiT perf pass — currently ~87% of wall-clock on `512`, ~86% on `1024`. Attention-level tuning, possibly a kernel for the shared `adaLN_modulation` projection, and reducing unnecessary `mx.eval` barriers.
- `pipeline_type="1536_cascade"` smoke — the code path is wired (same as `1024_cascade` with `hr_resolution=1536`), but the token count is expected to blow past the 49k cap on most inputs. Upstream's cap-downgrade fallback is implemented; needs a machine with 64+ GB to verify.

**v2:**
- Texture pipeline (`Trellis2ImageToTexturedGLBPipelineMLX`). Backbone + VAE arch is identical to shape; concat-condition on shape latent; weights are already converted in `ckpts/*tex*.safetensors`.
- UV atlas baking via `xatlas-python` + `trimesh`; PBR material bake per voxel.

**v3:**
- `pipeline_type="1536_cascade"`. Needs 64+ GB unified memory.
- Native-MLX port of DINOv3 to remove the final torch dependency at inference time.

---

## Provenance & license

This is a faithful port — no re-training, no distillation. All weights are derived from `microsoft/TRELLIS.2-4B` under the upstream MIT license. This port, the conversion scripts, and the MLX modules are released under the same MIT license.

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

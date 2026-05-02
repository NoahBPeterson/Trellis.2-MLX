# TRELLIS.2 — pipeline architecture

End-to-end flow for `pipeline_type="512"` on `assets/T.png` (the canonical
reference input). Numbers below are post-optimization measurements as of
2026-05-02 on M1 Pro 16 GB. SHA256 of the output GLB is
`af78d41f91b704be1a4d2429c24a0ecf601f9f0190e4c29cd550c835b4db34fb`.

```
                  Input Image  (PNG/JPG, 1024×1024)
                         │
         ┌───────────────▼────────────────┐
         │  rembg  (BiRefNet / RMBG-2.0)  │   background removal + center crop
         └───────────────┬────────────────┘
                         ▼  (922, 922, RGBA)
         ┌────────────────────────────────┐
         │  DINOv3  (vitl16-pretrain)     │   image → patch tokens
         └───────────────┬────────────────┘
                         ▼  cond_512: (1, 1029, 1024)
                         │
                         │   ┌─────────────────────────────────┐
                         │   │  Three transformer DiTs         │
                         │   │  • 1.3B params each             │
                         │   │  • 30 modulated blocks each     │
                         │   │  • bf16 compute throughout      │
                         │   │  • 12-step Euler flow matching  │
                         │   │  • CFG + guidance interval      │
                         │   │  • Sparse self-attn + img-cond  │
                         │   │    cross-attn (DINOv3 features) │
                         │   │  • Cross-attn K,V cached across │
                         │   │    sampler steps (1 compute_kv  │
                         │   │    per cond per block, not 12×) │
                         │   └─────────────────────────────────┘
                         │
         ┌───────────────▼────────────────┐
         │  ① Sparse-Structure Flow DiT   │   dense 16³ → upsample → 32³ occupancy
         └───────────────┬────────────────┘
                         ▼  ss_coords: (3754, 4)   [batch, x, y, z]
         ┌───────────────▼────────────────┐
         │  ② Shape SLat Flow DiT         │   sparse, 32-ch latent per active voxel
         │     "what's geometry inside?"  │
         └───────────────┬────────────────┘
                         ▼  shape_slat.feats: (3754, 32)
         ┌───────────────▼────────────────┐
         │  ③ Texture SLat Flow DiT       │   sparse, 32-ch latent per voxel
         │     "what's color/PBR inside?" │   (in=64: 32 noise + 32 shape concat)
         └───────────────┬────────────────┘
                         ▼  tex_slat.feats: (3754, 32)
                         │
                         │   ┌─────────────────────────────────┐
                         │   │  Two VAE decoders (no encoders) │
                         │   │  • ~470M params each, fp16      │
                         │   │  • verified bit-exact w/upstream│
                         │   └─────────────────────────────────┘
                         │
         ┌───────────────▼────────────────┐
         │  ④ Shape VAE  (FlexiDualGrid)  │   sparse latent → continuous mesh
         │     marching-cubes-style       │   + subdivision masks (`subs`)
         └───────────────┬────────────────┘
                         ▼  V=(1,730,326, 3)  F=(3,701,394, 3)  subs=...
         ┌───────────────▼────────────────┐
         │  ⑤ Texture VAE  (SparseUnet)   │   tex_slat + subs → per-voxel PBR
         │     guided by ④'s subs masks   │   [RGB | metallic | roughness | α]
         └───────────────┬────────────────┘
                         ▼  voxel_attrs: (1,730,326, 6)
                         │
                         │   ┌─────────────────────────────────┐
                         │   │  Post-processing (CPU/numpy)    │
                         │   │  • mesh repair + decimation     │
                         │   │  • UV unwrap + bake to atlas    │
                         │   └─────────────────────────────────┘
                         │
         ┌───────────────▼────────────────┐
         │  Quadric decimate (fast-simp)  │   3.7M faces → 500k faces
         │  (was CuMesh CUDA; ours numpy) │
         └───────────────┬────────────────┘
                         ▼  V=(179,645, 3)  F=(500,000, 3)
         ┌────────────────────────────────┐
         │  Cone-cluster chart segmenter  │   group faces by curvature → 152,585
         │                                │   charts (median 1, mean 3 faces/chart)
         └───────────────┬────────────────┘
                         ▼
         ┌────────────────────────────────┐
         │  xatlas UV unwrap + pack       │   xatlas chooses 5063×5063 atlas;
         │  (per-chart, sorted-indexed)   │   downsampled to 2048×2048 for shipping
         └───────────────┬────────────────┘
                         ▼  per-vertex UVs (844,279 verts after seam splits)
         ┌────────────────────────────────┐
         │  BVH per-texel bake            │   for each texel: nearest-triangle →
         │  voxel_attrs → 4 atlases       │   barycentric → sample voxel_attrs.
         │  (basecolor, M, R, alpha)      │   Rasterize loop is numba-JIT'd;
         │                                │   BVH project chunks run in parallel.
         └───────────────┬────────────────┘
                         ▼  3.6M texels covered (14% of 5063²)
         ┌────────────────────────────────┐
         │  cv2.inpaint Telea seam fill   │   smooth chart borders so seams
         │  (4 channels in parallel)      │   don't render as black lines
         └───────────────┬────────────────┘
                         ▼
         ┌────────────────────────────────┐
         │  trimesh.export → glTF/GLB     │   vertices + faces + UVs + atlases
         └───────────────┬────────────────┘
                         ▼
                    output.glb  (~32 MB at 512 PBR)

  Key model sizes (all loaded into ~16 GB RAM total on Apple Silicon):
  - DINOv3: 0.3B
  - 3× DiTs: 3 × 1.3B = 3.9B
  - 2× VAE decoders: 2 × 0.47B = 0.94B
  - Total: ~5.1B params (mostly bf16)

  Validation status (post-optimization, 2026-05-02):
  - DINOv3 / cond_512   → bit-exact vs upstream torch ✅
  - 3× DiTs             → bf16 manual_cast pattern matches upstream ✅
  - 2× VAEs             → bit-exact ✅
  - Post-processing     → byte-identical GLB across all 6 optimization commits ✅
  - End-to-end shasum   → af78d41f91b704be1a4d2429c24a0ecf601f9f0190e4c29cd550c835b4db34fb

  Architectural insight: TRELLIS.2 cleanly separates structure (where are
  voxels?), shape (what's the geometry?), and texture (what's the appearance?)
  into three sequential flows. Each flow conditions on the previous one's
  output. The same `ModulatedTransformerCrossBlock` is used in all three
  DiTs — a per-block bug therefore propagates through every stage.
```

---

## Stage-by-stage timing (M1 Pro 16 GB, T.png 512 PBR, 2026-05-02)

| Stage                                       | Time   | % of total |
| ------------------------------------------- | -----: | ---------: |
| Pipeline load (one-time per process)        |  23.4s |       6%   |
| Preprocess + DINOv3                         |   5.1s |       1%   |
| ① SS Flow DiT + decoder                     | 123.6s |      30%   |
| ② Shape SLat Flow DiT                       |  97.5s |      23%   |
| ④ Shape VAE decode + dual-grid mesh         |  24.7s |       6%   |
| ③ Texture SLat Flow DiT                     |  62.8s |      15%   |
| ⑤ Texture VAE decode → per-vertex PBR       |  21.2s |       5%   |
| Decimate (3.7M → 500k faces)                |   3.3s |       1%   |
| UV unwrap (cone-cluster + xatlas)           |  34.4s |       8%   |
| BVH atlas bake (numba rasterize + threaded) |  23.4s |       6%   |
| GLB write (trimesh)                         |   1.2s |      <1%   |
| **Total**                                   | **~7m01s** | |

Sampling (the 3 DiTs + 2 VAEs) is ~80% of wall-clock. The remaining ~20% is
post-processing — UV unwrap dominates, then atlas bake.

## What's compute-heavy vs trivial to port

For a future browser / ONNX port, the cost surfaces split cleanly:

**GPU-bound (must run on accelerated backend)**
- DINOv3 ViT-L/16 — already runs in `transformers.js` upstream
- 3× DiT forwards — the dominant cost; needs ONNX export of the
  `ModulatedTransformerCrossBlock` attention + MLP path
- 2× VAE decoders — sparse 3D conv; ONNX representation may need custom ops

**Pure CPU / numpy / vanilla JS**
- Preprocess (alpha crop, premultiply)
- SS-decoder thresholding → coords
- Dual-grid mesh extraction (`flexible_dual_grid_to_mesh`)
- Decimation (fast-simplification has a JS port: `meshopt-simplifier`)
- Cone-cluster segmenter (pure-numpy port of CuMesh; portable to JS)
- xatlas (has a WASM build: `xatlas-web`)
- BVH per-texel bake (KDTree + closest-point-on-triangle; portable)
- cv2.inpaint Telea (no direct JS port; would need rewrite or substitute)

**Python-only stack today**
- `torch` (DINOv3 wrapper) → can drop, transformers.js handles it
- `mlx` → must replace with ONNX runtime + WebGPU/WebGL
- `numba` → not available in browser; rasterize loop would need vanilla
  JS or WebAssembly
- `scipy.spatial.cKDTree` → portable but slower in JS; consider three.js BVH

## Key file map

| File | Role |
|---|---|
| `trellis2_mlx/pipeline.py` | end-to-end orchestration (`Trellis2ImageTo3DPipelineMLX`) |
| `trellis2_mlx/models/flow_dit.py` | `SparseStructureFlowModel` + `SLatFlowModel` |
| `trellis2_mlx/models/sparse_vae.py` | `FlexiDualGridVaeDecoder` + `SparseUnetVaeDecoder` |
| `trellis2_mlx/models/ss_decoder.py` | dense conv3d SS decoder |
| `trellis2_mlx/modules/attention.py` | `MultiHeadAttention` (self + cross) |
| `trellis2_mlx/modules/blocks.py` | `ModulatedTransformerCrossBlock` |
| `trellis2_mlx/modules/sparse.py` | submanifold conv + S2C/C2S |
| `trellis2_mlx/modules/sparse_tensor.py` | `SparseTensor` data holder |
| `trellis2_mlx/postprocess/dual_grid.py` | `flexible_dual_grid_to_mesh` |
| `trellis2_mlx/postprocess/atlas.py` | cone-cluster + xatlas + BVH bake |
| `trellis2_mlx/postprocess/glb_export.py` | trimesh GLB packager |
| `trellis2_mlx/samplers.py` | `FlowEulerGuidanceIntervalSampler` |
| `scripts/run_example_pbr.py` | CLI entry point for textured GLB |
| `scripts/setup.py` | one-shot weight download via HF |

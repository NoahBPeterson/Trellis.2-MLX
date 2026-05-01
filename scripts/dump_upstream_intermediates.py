"""Run upstream microsoft/TRELLIS.2 with explicit stage-by-stage dumps.

Designed to be the CUDA "ground truth" for comparing our MLX port. Run on a
GPU (we're using RunPod A100/H100 in our workflow) and copy the resulting
.npz back to compare against `artifacts/pbr_intermediates.npz`.

Usage on a fresh CUDA machine (after running upstream/setup.sh with
--basic --flash-attn --nvdiffrast --cumesh --o-voxel --flexgemm):

    python scripts/dump_upstream_intermediates.py \\
        --image upstream/assets/example_image/T.png \\
        --seed 42 \\
        --pipeline-type 512 \\
        --out /tmp/upstream_ref.npz \\
        --glb-out /tmp/upstream_ref.glb

Saves a single .npz with stage-by-stage tensors:
  - preprocessed_image (H, W, 3) uint8 RGB after rembg+crop
  - cond_512           (1, T, D) DINOv3 conditioning
  - ss_coords          (N, 4)    int32 sparse-structure coords (B, x, y, z)
  - shape_slat_coords  (N, 4)    int32
  - shape_slat_feats   (N, C)    float32 (denormalized)
  - tex_slat_coords    (N, 4)    int32
  - tex_slat_feats     (N, C)    float32 (denormalized)
  - mesh_V             (V, 3)    float32
  - mesh_F             (F, 3)    int64
  - voxel_coords       (Nv, 3)   int32   per-voxel attribute coords
  - voxel_attrs        (Nv, C)   float32 attribute volume (RGB+M+R+A)

And a stats summary printed to stdout for direct comparison with
`artifacts/pbr_intermediates.npz`.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
from PIL import Image
import torch


def _summarize(name: str, x: np.ndarray) -> str:
    if x.dtype == bool or np.issubdtype(x.dtype, np.integer):
        return f"  {name}: shape={x.shape} dtype={x.dtype} min={x.min()} max={x.max()}"
    return (f"  {name}: shape={x.shape} dtype={x.dtype} "
            f"mean={x.mean():.4f} std={x.std():.4f} min={x.min():.4f} max={x.max():.4f}")


def _to_np(t: "torch.Tensor") -> np.ndarray:
    return t.detach().cpu().numpy()


class _NoiseCapture:
    """Monkey-patch torch.randn / torch.randn_like to capture noise tensors.

    Used as a context manager around each sampling call so we can pair captured
    tensors with the stage that produced them. The first call within the block
    is taken as the "initial noise" (sampler trajectories are deterministic
    after that for Euler flow matching).
    """

    def __init__(self):
        import torch
        self._torch = torch
        self._captured: list = []
        self._orig_randn = None
        self._orig_randn_like = None

    def __enter__(self):
        self._orig_randn = self._torch.randn
        self._orig_randn_like = self._torch.randn_like
        captured = self._captured

        def randn(*args, **kwargs):
            out = self._orig_randn(*args, **kwargs)
            captured.append(out.detach().cpu().numpy().copy())
            return out

        def randn_like(*args, **kwargs):
            out = self._orig_randn_like(*args, **kwargs)
            captured.append(out.detach().cpu().numpy().copy())
            return out

        self._torch.randn = randn
        self._torch.randn_like = randn_like
        return self

    def __exit__(self, *args):
        self._torch.randn = self._orig_randn
        self._torch.randn_like = self._orig_randn_like

    def first(self):
        """The first captured noise tensor (the sampler's initial noise)."""
        if not self._captured:
            return None
        return self._captured[0]

    def all(self):
        return list(self._captured)


def _capture_blocks_dense(model, h, t_emb, cond, phases):
    """Loop SS DiT blocks; record per-channel mean/std/absmax of hidden state after each block.

    h: (B, L, C) where L=R^3.  Returns three numpy arrays:
       (num_blocks, C), (num_blocks, C), (num_blocks,)
    """
    means, stds, absmaxes = [], [], []
    for block in model.blocks:
        h = block(h, t_emb, cond, phases)
        h32 = h.detach().float()
        means.append(h32.mean(dim=(0, 1)).cpu().numpy())
        stds.append(h32.std(dim=(0, 1)).cpu().numpy())
        absmaxes.append(float(h32.abs().max().item()))
    return (np.stack(means).astype(np.float32),
            np.stack(stds).astype(np.float32),
            np.asarray(absmaxes, dtype=np.float32))


def _capture_blocks_sparse(model, h, t_emb, cond):
    """Loop SLat DiT blocks; record per-channel stats.

    h: SparseTensor with feats (F, C).  Block returns updated SparseTensor.
    """
    means, stds, absmaxes = [], [], []
    for block in model.blocks:
        h = block(h, t_emb, cond)
        f32 = h.feats.detach().float()  # (F, C)
        means.append(f32.mean(dim=0).cpu().numpy())
        stds.append(f32.std(dim=0).cpu().numpy())
        absmaxes.append(float(f32.abs().max().item()))
    return (np.stack(means).astype(np.float32),
            np.stack(stds).astype(np.float32),
            np.asarray(absmaxes, dtype=np.float32))


def _run_block_diag(pipeline, *, cond, shape_coords, tex_coords, shape_slat_normed):
    """Run controlled zero-input forward through SS / shape_512 / tex_512 DiTs.

    Inputs are deterministic (zero hidden state, t=1.0, upstream's bit-exact cond),
    so any per-block stat divergence between upstream and ours is attributable to
    the DiT block code or its weights, NOT to RNG / sampler / normalization.

    Returns dict with keys:
      bd_{ss,shape,tex}_{mean,std,absmax}  per-block stats
      bd_meta_*                            scalar metadata for the local comparator
    """
    import torch
    import trellis2.modules.sparse as sp  # type: ignore  (only on pod)
    from trellis2.modules.utils import manual_cast as _manual_cast  # type: ignore

    out = {}
    device = cond.device
    # t=1.0 in [0,1] sampling space corresponds to t=1000 fed to t_embedder
    t_step = torch.tensor([1000.0], device=device, dtype=torch.float32)

    # Pipeline.sample_*() and decode_*() may offload models back to CPU after
    # use. Move them all back to the same device as cond before manual forward.
    for k in ("sparse_structure_flow_model", "shape_slat_flow_model_512",
              "tex_slat_flow_model_512"):
        if k in pipeline.models:
            pipeline.models[k].to(device).eval()

    # ---- SS DiT (dense) ------------------------------------------------------
    ss_model = pipeline.models["sparse_structure_flow_model"]
    ss_model.eval()
    R = ss_model.resolution
    Cin_ss = ss_model.in_channels
    with torch.no_grad():
        x = torch.zeros(1, Cin_ss, R, R, R, device=device, dtype=torch.float32)
        h = x.view(*x.shape[:2], -1).permute(0, 2, 1).contiguous()  # (1, R^3, Cin)
        h = ss_model.input_layer(h)
        if ss_model.pe_mode == "ape":
            h = h + ss_model.pos_emb[None]
        t_emb = ss_model.t_embedder(t_step)
        if ss_model.share_mod:
            t_emb = ss_model.adaLN_modulation(t_emb)
        # Match upstream's manual_cast to model dtype (the cast actually happens in real fwd)
        t_emb = _manual_cast(t_emb, ss_model.dtype)
        h = _manual_cast(h, ss_model.dtype)
        cond_d = _manual_cast(cond, ss_model.dtype)
        means, stds, amax = _capture_blocks_dense(ss_model, h, t_emb, cond_d, ss_model.rope_phases)
    out["bd_ss_mean"] = means
    out["bd_ss_std"] = stds
    out["bd_ss_absmax"] = amax
    print(f"  SS DiT: {len(means)} blocks, C={means.shape[1]}, "
          f"absmax range [{amax.min():.2f}, {amax.max():.2f}]")

    # ---- Shape DiT (sparse @ 512) -------------------------------------------
    # Build a SparseTensor with zero feats on upstream's actual shape_slat coords.
    # Coords carry the [batch, x, y, z] layout from upstream's sample_sparse_structure
    # output, so RoPE phases will match exactly between upstream and our local replay.
    shape_model = pipeline.models["shape_slat_flow_model_512"]
    shape_model.eval()
    Cin_shape = shape_model.in_channels
    with torch.no_grad():
        zero_feats = torch.zeros(shape_coords.shape[0], Cin_shape, device=device, dtype=torch.float32)
        x_sp = sp.SparseTensor(feats=zero_feats, coords=shape_coords.to(torch.int32))
        h = shape_model.input_layer(x_sp)
        h = h.replace(_manual_cast(h.feats, shape_model.dtype))
        t_emb = shape_model.t_embedder(t_step)
        if shape_model.share_mod:
            t_emb = shape_model.adaLN_modulation(t_emb)
        t_emb = _manual_cast(t_emb, shape_model.dtype)
        cond_d = _manual_cast(cond, shape_model.dtype)
        if shape_model.pe_mode == "ape":
            pe = shape_model.pos_embedder(h.coords[:, 1:])
            h = h + _manual_cast(pe, shape_model.dtype)
        means, stds, amax = _capture_blocks_sparse(shape_model, h, t_emb, cond_d)
    out["bd_shape_mean"] = means
    out["bd_shape_std"] = stds
    out["bd_shape_absmax"] = amax
    print(f"  Shape DiT: {len(means)} blocks, C={means.shape[1]}, "
          f"absmax range [{amax.min():.2f}, {amax.max():.2f}]")

    # ---- Tex DiT (sparse @ 512) ---------------------------------------------
    # Most-controlled input: zero everywhere (both noise AND concat-cond zeroed).
    # in_channels=64 = 32 noise + 32 concat-cond; we feed (F, 64) zeros directly,
    # bypassing the concat path entirely.
    tex_model = pipeline.models["tex_slat_flow_model_512"]
    tex_model.eval()
    Cin_tex = tex_model.in_channels
    with torch.no_grad():
        zero_feats = torch.zeros(tex_coords.shape[0], Cin_tex, device=device, dtype=torch.float32)
        x_sp = sp.SparseTensor(feats=zero_feats, coords=tex_coords.to(torch.int32))
        h = tex_model.input_layer(x_sp)
        h = h.replace(_manual_cast(h.feats, tex_model.dtype))
        t_emb = tex_model.t_embedder(t_step)
        if tex_model.share_mod:
            t_emb = tex_model.adaLN_modulation(t_emb)
        t_emb = _manual_cast(t_emb, tex_model.dtype)
        cond_d = _manual_cast(cond, tex_model.dtype)
        if tex_model.pe_mode == "ape":
            pe = tex_model.pos_embedder(h.coords[:, 1:])
            h = h + _manual_cast(pe, tex_model.dtype)
        means, stds, amax = _capture_blocks_sparse(tex_model, h, t_emb, cond_d)
    out["bd_tex_mean"] = means
    out["bd_tex_std"] = stds
    out["bd_tex_absmax"] = amax
    print(f"  Tex DiT: {len(means)} blocks, C={means.shape[1]}, "
          f"absmax range [{amax.min():.2f}, {amax.max():.2f}]")

    # Metadata so the local comparator can sanity-check it's matching the same setup
    out["bd_meta_t_step"] = np.float32(1000.0)
    out["bd_meta_ss_resolution"] = np.int32(R)
    out["bd_meta_ss_in_channels"] = np.int32(Cin_ss)
    out["bd_meta_shape_in_channels"] = np.int32(Cin_shape)
    out["bd_meta_tex_in_channels"] = np.int32(Cin_tex)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, required=True,
                   help="Input image (use upstream/assets/example_image/T.png for parity).")
    p.add_argument("--weights", type=str, default="microsoft/TRELLIS.2-4B",
                   help="HF repo id or local path to a directory containing pipeline.json.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pipeline-type", type=str, default="512",
                   choices=["512", "1024", "1024_cascade", "1536_cascade"],
                   help="Match what trellis-mlx ran. Our cache was made at 512.")
    p.add_argument("--out", type=Path, required=True, help="Output .npz path.")
    p.add_argument("--glb-out", type=Path, default=None,
                   help="Optional: also write a reference GLB via o_voxel.postprocess.to_glb.")
    p.add_argument("--decimation-target", type=int, default=1_000_000,
                   help="Passed to o_voxel.postprocess.to_glb. Match example.py default.")
    p.add_argument("--texture-size", type=int, default=2048,
                   help="Atlas size. Default 2048 to match our trellis-mlx default.")
    p.add_argument("--block-diag", action="store_true",
                   help="After main dump, run controlled zero-input forward through SS/shape/tex "
                        "DiTs and dump per-block per-channel mean/std/absmax. Used by "
                        "scripts/diag_per_block_compare.py to pinpoint the divergent block.")
    p.add_argument("--save-noise", action="store_true",
                   help="Capture upstream's torch.randn output during sample_sparse_structure / "
                        "sample_shape_slat / sample_tex_slat and dump as noise_{ss,shape_slat,tex_slat}. "
                        "Lets us replay the exact same noise through our MLX pipeline to isolate "
                        "RNG-driven divergence from model-driven divergence.")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: no CUDA device — this script needs a GPU.", file=sys.stderr)
        return 1

    print(f"=== upstream dump on {torch.cuda.get_device_name(0)} ===")
    print(f"image={args.image} seed={args.seed} pipeline_type={args.pipeline_type}")

    # 1. Load pipeline
    t0 = time.time()
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.weights)
    pipeline.cuda()
    print(f"  pipeline loaded ({time.time()-t0:.1f}s)")

    # 2. Inputs
    image = Image.open(args.image).convert("RGBA" if Image.open(args.image).mode == "RGBA" else "RGB")
    print(f"  input image: {image.size} mode={image.mode}")

    dump = {}

    # 3. Preprocess (rembg + crop)
    t0 = time.time()
    image_pp = pipeline.preprocess_image(image)
    dump["preprocessed_image"] = np.asarray(image_pp).astype(np.uint8)
    print(f"  preprocess: {image_pp.size}  ({time.time()-t0:.1f}s)")

    # 4. Seed RNG and condition
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    t0 = time.time()
    cond_dict = pipeline.get_cond([image_pp], 512)
    cond_512 = cond_dict["cond"]
    dump["cond_512"] = _to_np(cond_512)
    print(f"  cond_512: {tuple(cond_512.shape)}  ({time.time()-t0:.1f}s)")

    # 5. Sample sparse structure
    ss_res = {"512": 32, "1024": 64, "1024_cascade": 32, "1536_cascade": 32}[args.pipeline_type]
    t0 = time.time()
    with _NoiseCapture() as cap_ss:
        coords = pipeline.sample_sparse_structure(
            cond_dict, ss_res, num_samples=1,
            sampler_params=pipeline.sparse_structure_sampler_params,
        )
    if args.save_noise and cap_ss.first() is not None:
        dump["noise_ss"] = cap_ss.first().astype(np.float32)
        print(f"  captured noise_ss: {dump['noise_ss'].shape}  ({len(cap_ss.all())} randn calls during SS)")
    dump["ss_coords"] = _to_np(coords).astype(np.int32)
    print(f"  ss_coords: {tuple(coords.shape)}  ({time.time()-t0:.1f}s)")

    # 6. Shape SLat (12 sampling steps)
    t0 = time.time()
    with _NoiseCapture() as cap_shape:
        shape_slat = pipeline.sample_shape_slat(
            cond_dict, pipeline.models["shape_slat_flow_model_512"],
            coords, pipeline.shape_slat_sampler_params,
        )
    if args.save_noise and cap_shape.first() is not None:
        dump["noise_shape_slat"] = cap_shape.first().astype(np.float32)
        print(f"  captured noise_shape_slat: {dump['noise_shape_slat'].shape}  "
              f"({len(cap_shape.all())} randn calls during shape SLat)")
    dump["shape_slat_coords"] = _to_np(shape_slat.coords).astype(np.int32)
    dump["shape_slat_feats"] = _to_np(shape_slat.feats)
    print(f"  shape_slat: coords={tuple(shape_slat.coords.shape)} feats={tuple(shape_slat.feats.shape)}  "
          f"({time.time()-t0:.1f}s)")

    # 7. Texture SLat (12 sampling steps)
    t0 = time.time()
    with _NoiseCapture() as cap_tex:
        tex_slat = pipeline.sample_tex_slat(
            cond_dict, pipeline.models["tex_slat_flow_model_512"],
            shape_slat, pipeline.tex_slat_sampler_params,
        )
    if args.save_noise and cap_tex.first() is not None:
        dump["noise_tex_slat"] = cap_tex.first().astype(np.float32)
        print(f"  captured noise_tex_slat: {dump['noise_tex_slat'].shape}  "
              f"({len(cap_tex.all())} randn calls during tex SLat)")
    dump["tex_slat_coords"] = _to_np(tex_slat.coords).astype(np.int32)
    dump["tex_slat_feats"] = _to_np(tex_slat.feats)
    print(f"  tex_slat: coords={tuple(tex_slat.coords.shape)} feats={tuple(tex_slat.feats.shape)}  "
          f"({time.time()-t0:.1f}s)")

    # 8. Decode shape SLat → mesh; Decode tex SLat → attribute volume
    t0 = time.time()
    meshes, subs = pipeline.decode_shape_slat(shape_slat, 512)
    tex_voxels = pipeline.decode_tex_slat(tex_slat, subs)
    mesh = meshes[0]
    # decode_tex_slat returns a SparseTensor (possibly batched). For num_samples=1,
    # all coords have batch_id=0 so we can read .coords/.feats directly.
    voxel = tex_voxels
    mesh.fill_holes()
    dump["mesh_V"] = _to_np(mesh.vertices).astype(np.float32)
    dump["mesh_F"] = _to_np(mesh.faces).astype(np.int64)
    dump["voxel_coords"] = _to_np(voxel.coords[:, 1:]).astype(np.int32)  # drop batch dim
    dump["voxel_attrs"] = _to_np(voxel.feats).astype(np.float32)
    print(f"  mesh: V={tuple(mesh.vertices.shape)} F={tuple(mesh.faces.shape)}  "
          f"voxel: coords={tuple(voxel.coords.shape)} attrs={tuple(voxel.feats.shape)}  "
          f"({time.time()-t0:.1f}s)")

    # 9. Save the basic dump FIRST, so a block-diag crash can never lose it.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **dump)
    print(f"\nwrote {args.out} ({args.out.stat().st_size / 1024 / 1024:.1f} MB) [main dump]")

    # 10. Optional: per-block hidden-state diagnostic (controlled zero input).
    # On success, re-save the npz with bd_* keys appended.
    if args.block_diag:
        print("\n=== --block-diag: controlled zero-input forward through DiT blocks ===")
        try:
            bd = _run_block_diag(
                pipeline,
                cond=cond_512,
                shape_coords=shape_slat.coords,    # (F, 4) sparse coords with batch dim
                tex_coords=shape_slat.coords,      # tex operates on same active voxels
                shape_slat_normed=None,            # zero concat-cond (most controlled)
            )
            dump.update(bd)
            print(f"  captured stats for {sum(1 for k in bd if k.endswith('_absmax'))} flow(s)")
            np.savez_compressed(args.out, **dump)
            print(f"  re-wrote {args.out} ({args.out.stat().st_size / 1024 / 1024:.1f} MB) [+bd_*]")
        except Exception as e:
            print(f"  block-diag FAILED: {type(e).__name__}: {e}")
            print(f"  basic dump at {args.out} is still intact.")
            import traceback
            traceback.print_exc()

    # 11. Stats summary
    print("\n=== stats summary (compare against trellis-mlx pbr_intermediates.npz) ===")
    for k, v in dump.items():
        print(_summarize(k, v))

    # 11. Optional: also produce a reference GLB so we have a visual baseline.
    if args.glb_out is not None:
        print(f"\n=== writing reference GLB to {args.glb_out} (this is the upstream-quality target) ===")
        from trellis2.representations import MeshWithVoxel
        m_with_voxel = MeshWithVoxel(
            mesh.vertices, mesh.faces,
            origin=[-0.5, -0.5, -0.5], voxel_size=1.0 / 512,
            coords=voxel.coords[:, 1:], attrs=voxel.feats,
            voxel_shape=torch.Size([*voxel.shape, *voxel.spatial_shape]),
            layout=pipeline.pbr_attr_layout,
        )
        import o_voxel  # noqa
        glb = o_voxel.postprocess.to_glb(
            vertices=m_with_voxel.vertices,
            faces=m_with_voxel.faces,
            attr_volume=m_with_voxel.attrs,
            coords=m_with_voxel.coords,
            attr_layout=m_with_voxel.layout,
            voxel_size=m_with_voxel.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=args.decimation_target,
            texture_size=args.texture_size,
            verbose=True,
        )
        args.glb_out.parent.mkdir(parents=True, exist_ok=True)
        glb.export(str(args.glb_out))
        print(f"wrote {args.glb_out} ({args.glb_out.stat().st_size / 1024 / 1024:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

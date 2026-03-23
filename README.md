# GeoAware-SC-lessvram

This repository is a practical fork of GeoAware-SC focused on:

- running on modern GPUs (including RTX 5090 / Blackwell),
- lowering VRAM usage for 12GB-class cards (RTX 3080Ti / RTX 4070),
- keeping feature extraction and keypoint matching behavior compatible with the original pipeline.

## Upstream (original project)

- Original repository: https://github.com/Junyi42/GeoAware-SC
- Paper: https://arxiv.org/abs/2311.17034

Please cite the original paper/repository for research use.

## What changed in this fork

### 1) Less-VRAM inference path

- Added FP16 switch to keypoint matching: `--fp16`
- Added SD resolution switch: `--sd-size`
- Added robust dtype handling (autocast + explicit cast points) to avoid FP16 dtype mismatch crashes
- Kept default behavior unchanged (if you do **not** pass `--fp16`, pipeline runs in FP32)

### 2) 5090 / CUDA 12.8 compatibility work

- Migrated runtime to PyTorch CUDA 12.8 stack
- Kept detectron2/ODISE/Mask2Former install path working on newer Python/toolchain
- Patched compatibility breakpoints caused by newer PyTorch/Lightning/setuptools APIs

### 3) Documentation + reproducible usage

- Updated usage docs around `keypoints/kp_match.py`
- Added practical presets for 24GB and 12GB cards

## Quick start (low VRAM keypoint matching)

### Environment

> `mamba` and `conda` (≥ 23.10 with libmamba solver) are interchangeable below.
> Use `mamba run` / `conda run` instead of `conda activate` on systems where
> shell init is not loaded.

```bash
# ── 1. Create env ──────────────────────────────────────────────
conda create -n geo-aware-5090 python=3.10 -y

# ── 2. Install PyTorch CUDA 12.8 ──────────────────────────────
conda run -n geo-aware-5090 pip install \
  torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128

# ── 3. Install conda-side CUDA toolkit (for C++ extension compilation) ──
#    Required when the system-wide CUDA version differs from 12.8.
conda install -n geo-aware-5090 \
  -c nvidia/label/cuda-12.8.0 cuda-nvcc=12.8 cuda-cudart-dev=12.8 \
  cuda-libraries-dev=12.8 -y

# ── 4. Install this fork + third_party deps ───────────────────
#    IMPORTANT: --no-build-isolation is required because setup.py
#    does `import torch` at the top level.
#    CUDA_HOME must point to the conda env so nvcc 12.8 is used.
export CUDA_HOME=$CONDA_PREFIX   # or /path/to/miniconda3/envs/geo-aware-5090

conda run -n geo-aware-5090 pip install -e . --no-deps --no-build-isolation
CUDA_HOME=$CUDA_HOME conda run -n geo-aware-5090 pip install \
  -e third_party/Mask2Former --no-deps --no-build-isolation
CUDA_HOME=$CUDA_HOME conda run -n geo-aware-5090 pip install \
  -e third_party/ODISE --no-deps --no-build-isolation

# ── 5. Install runtime dependencies ──────────────────────────
conda run -n geo-aware-5090 pip install \
  "loguru>=0.5.3" "faiss-cpu==1.7.1.post3" "matplotlib>=3.4.2" \
  "tqdm>=4.61.2" "numpy>=1.23" "pillow>=9.5.0" "ipykernel==6.29.5" \
  "gdown>=4.6.0" "wandb>=0.16.0" "pytorch-lightning>=1.8" "torchmetrics>=0.9"

# ── 6. Install ODISE runtime dependencies ─────────────────────
conda run -n geo-aware-5090 pip install \
  diffdist==0.1 "einops>=0.3.0" "nltk>=3.6.2" omegaconf==2.1.1 \
  open-clip-torch==2.0.2 opencv-python stable-diffusion-sdkit==2.1.3 timm==0.6.11

# ── 7. Install detectron2 (needs CUDA_HOME) ──────────────────
CUDA_HOME=$CUDA_HOME conda run -n geo-aware-5090 pip install \
  "detectron2 @ https://github.com/facebookresearch/detectron2/archive/v0.6.zip" \
  --no-build-isolation

# ── 8. Post-install fixes ─────────────────────────────────────
# 8a) sdkit downgrades pytorch-lightning & torchmetrics — force them back
conda run -n geo-aware-5090 pip install \
  "pytorch-lightning>=1.8" "torchmetrics>=0.9" --force-reinstall --no-deps

# 8b) opencv-python 4.6 is numpy-1.x only; upgrade to numpy-2.x-compatible
conda run -n geo-aware-5090 pip install --upgrade opencv-python opencv-python-headless

# 8c) Patch fvcore for PyTorch weights_only default change
#     In  <env>/lib/python3.10/site-packages/fvcore/common/checkpoint.py
#     change:  torch.load(f, map_location=torch.device("cpu"))
#     to:      torch.load(f, map_location=torch.device("cpu"), weights_only=False)

# 8d) Patch detectron2 for Pillow ≥ 10  (only needed for v0.6 tag)
#     In  <env>/lib/python3.10/site-packages/detectron2/data/transforms/transform.py
#     change:  interp=Image.LINEAR
#     to:      interp=Image.BILINEAR
```

> **Note on Mask2Former CUDA source patch:**
> The file `third_party/Mask2Former/.../ms_deform_attn_cuda.cu` has already been
> patched in this repo (`value.type()` → `value.scalar_type()`,
> `.type().is_cuda()` → `.is_cuda()`). If you are starting from a fresh
> upstream clone, you will need to apply these changes manually — see
> `setup.py` comment block [D-4] for details.

### Run matching

```bash
# Default (FP32)
mamba run -n geo-aware-5090 python keypoints/kp_match.py \
  --src path/to/source.png \
  --kps path/to/source_kps.json \
  --tgt path/to/target.png

# Low-VRAM mode (recommended on 12GB cards)
mamba run -n geo-aware-5090 python keypoints/kp_match.py \
  --src path/to/source.png \
  --kps path/to/source_kps.json \
  --tgt path/to/target.png \
  --fp16

# If still OOM, lower SD input resolution
mamba run -n geo-aware-5090 python keypoints/kp_match.py \
  --src path/to/source.png \
  --kps path/to/source_kps.json \
  --tgt path/to/target.png \
  --fp16 --sd-size 768
```

## `--sd-size` usage and impact

- Controls SD backbone input resolution before SD feature extraction.
- Default: `960` (`NUM_PATCHES=60`, so `60*16=960`).
- Lower value => lower memory/compute, but possible quality drop on fine details.

Suggested values:

- `960`: best quality, highest cost
- `768`: good balance for 12GB cards
- `640/512`: emergency low-memory fallback

## Practical notes

- `xformers` was tested but is **not** enabled by default in this fork’s recommended setup, because version/device compatibility may cause instability on some GPUs/toolchains.
- The current tested path for this fork is “no xformers + FP16 option in kp_match”.

## License / Citation

This fork builds on the original GeoAware-SC project. Please follow the original license and citation requirements from upstream.

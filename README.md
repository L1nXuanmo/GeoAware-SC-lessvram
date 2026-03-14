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

> Recommended: use `mamba run -n <env> ...` instead of `conda activate` on systems where shell init is not loaded.

```bash
mamba create -n geo-aware-5090 python=3.10 -y

# Install PyTorch CUDA 12.8
mamba run -n geo-aware-5090 pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# Install this fork and local third_party deps
mamba run -n geo-aware-5090 pip install -e . --no-deps
mamba run -n geo-aware-5090 pip install -e third_party/Mask2Former --no-deps
mamba run -n geo-aware-5090 pip install -e third_party/ODISE --no-deps
```

### Run matching

```bash
# Default (FP32)
mamba run -n geo-aware-5090 python keypoints/kp_match.py \
  --src data/01_fix_dual/test/source/image.png \
  --kps data/01_fix_dual/test/source/image_kps.json \
  --tgt data/01_fix_dual/test/target/image.png

# Low-VRAM mode (recommended on 12GB cards)
mamba run -n geo-aware-5090 python keypoints/kp_match.py \
  --src data/01_fix_dual/test/source/image.png \
  --kps data/01_fix_dual/test/source/image_kps.json \
  --tgt data/01_fix_dual/test/target/image.png \
  --fp16

# If still OOM, lower SD input resolution
mamba run -n geo-aware-5090 python keypoints/kp_match.py \
  --src data/01_fix_dual/test/source/image.png \
  --kps data/01_fix_dual/test/source/image_kps.json \
  --tgt data/01_fix_dual/test/target/image.png \
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

## Acknowledgement

Our code is largely based on the following open-source projects: [A Tale of Two Features](https://github.com/Junyi42/sd-dino), [Diffusion Hyperfeatures](https://github.com/diffusion-hyperfeatures/diffusion_hyperfeatures), [DIFT](https://github.com/Tsingularity/dift), [DenseMatching](https://github.com/PruneTruong/DenseMatching), and [SFNet](https://github.com/cvlab-yonsei/SFNet). Our heartfelt gratitude goes to the developers of these resources!

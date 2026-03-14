import os
from setuptools import setup
import torch

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 8], "Requires PyTorch >= 1.8"

# -----------------------------------------------------------------------------
# Compatibility notes for this fork (GeoAware-SC-lessvram)
# -----------------------------------------------------------------------------
# This setup.py intentionally documents compatibility decisions so users can
# install in modern GPU environments (especially RTX 5090 / Blackwell) without
# repeating the full debug process.
#
# [A] CUDA / GPU architecture target
# - Recommended runtime for this fork: PyTorch + CUDA 12.8 wheel stack.
# - Rationale: Blackwell (sm_120) support is not available in older CUDA 11.x
#   pipelines used by the upstream environment recipe.
# - IMPORTANT: CUDA version is selected when installing torch/torchvision
#   (e.g. --index-url https://download.pytorch.org/whl/cu128), not by
#   install_requires pins in setup.py.
#
# [B] Low-VRAM support strategy implemented in code (history summary)
# - Added FP16 path in keypoints/kp_match.py via --fp16.
# - Added SD resolution switch via --sd-size.
# - Added dtype guards/autocast in model_utils/extractor_sd.py and DINO/ODISE
#   call-sites to avoid FP16 matmul dtype mismatch errors.
# - Default behavior remains FP32 unless --fp16 is explicitly provided.
#
# [C] Dependency pin relaxations and why
# - numpy: relaxed to >=1.23 to allow modern OpenCV wheels in Python 3.10+.
# - pillow: relaxed to >=9.5.0 for compatibility with newer detectron2 usage.
# - pytorch-lightning / torchmetrics: relaxed to modern versions because legacy
#   versions conflict with newer setuptools/runtime stacks.
# - Mask2Former / ODISE are installed from local third_party paths.
#
# [D] Manual compatibility patches required in this repository/environment
# (recorded here for reproducibility)
# - third_party/ODISE/odise/model_zoo/model_zoo.py:
#     replaced pkg_resources resource lookup with os.path-based lookup.
# - third_party/ODISE/odise/modeling/meta_arch/ldm.py:
#     updated dtype handling for FP16 path and modern lightning import path.
# - environment site-packages patch (runtime-level):
#     fvcore/common/checkpoint.py torch.load(..., weights_only=False) for
#     checkpoints serialized with objects not allowed by default in newer
#     PyTorch weights_only mode.
#
# [E] xformers note
# - xformers is NOT pinned here on purpose. Across torch/cuda/gpu-arch combos,
#   mismatched wheels can either force an incompatible torch upgrade or fail at
#   runtime on unsupported kernel images.
# -----------------------------------------------------------------------------

setup(
    name="geo-aware",
    author="Junyi Zhang",
    description="Telling Left from Right: Identifying Geometry-Aware Semantic Correspondence",
    python_requires=">=3.8",
    py_modules=[],
    install_requires=[
        "loguru>=0.5.3",
        "faiss-cpu==1.7.1.post3",
        "matplotlib>=3.4.2",
        "tqdm>=4.61.2",
        # numpy==1.23.5 was the original pin; relaxed to >=1.23 because
        # numpy 2.x is required for opencv-python>=4.8 on Python 3.10+.
        "numpy>=1.23",
        # pillow==9.5.0 was the original pin to avoid PIL.Image.LINEAR removal.
        # detectron2 main branch (post-Jul 2023) uses Image.BILINEAR, so any
        # modern Pillow works fine.
        "pillow>=9.5.0",
        "ipykernel==6.29.5",
        "gdown>=4.6.0",
        "wandb>=0.16.0",
        # pytorch-lightning>=1.8 and torchmetrics>=0.9 required because
        # older versions (1.4.2 / 0.6.0) import pkg_resources which is no
        # longer available at top-level in setuptools>=82 (Python 3.10+ envs).
        # NOTE: stable-diffusion-sdkit will report a dependency conflict here
        # (it pins pytorch-lightning==1.4.2 / torchmetrics==0.6.0), but this
        # is safe to ignore: sdkit only provides the `ldm` module (SD model
        # code). We only call model.forward() for feature extraction — the
        # pytorch-lightning Trainer API (affected by the version change) is
        # never used. Training SD itself is out of scope for this project.
        "pytorch-lightning>=1.8",
        "torchmetrics>=0.9",
        f"mask2former @ file://localhost/{os.getcwd()}/third_party/Mask2Former/",
        f"odise @ file://localhost/{os.getcwd()}/third_party/ODISE/"
    ],
    include_package_data=True,
)
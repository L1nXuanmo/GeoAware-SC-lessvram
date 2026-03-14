"""
keypoints/kp_match.py — Semantic keypoint matching using GeoAware-SC backbone.

Given a source image with pre-selected keypoints and a target image,
finds the semantically corresponding pixel locations on the target.

Usage
-----
    python keypoints/kp_match.py \\
        --src  data/images/bowl.jpg \\
        --kps  data/images/bowl_kps.json \\
        --tgt  data/images/target.jpg \\
        [--ckpt results_spair/best_856.PTH] \\
        [--out  results/match_result.json] \\
        [--no-vis] \\
        [--fp16] \\
        [--sd-size 960]

Recommended presets
-------------------
  * 24GB+ GPU:
      python keypoints/kp_match.py ...
  * 12GB GPU (e.g. RTX 3080Ti / RTX 4070):
      python keypoints/kp_match.py ... --fp16
  * If still OOM on 12GB cards:
      python keypoints/kp_match.py ... --fp16 --sd-size 768
      python keypoints/kp_match.py ... --fp16 --sd-size 640
      python keypoints/kp_match.py ... --fp16 --sd-size 512

`--sd-size` meaning and impact
------------------------------
  * Controls SD backbone input resolution before feature extraction.
  * Default is 960 (because NUM_PATCHES=60, and 60*16=960).
  * Smaller values reduce memory/compute in SD forward, but may reduce matching quality.
  * Typical values:
      - 960: best quality, highest cost
      - 768: good balance
      - 640/512: lowest cost, quality may drop on fine details
  * Keep it as multiples of 64 for stable behavior.

Output
------
  <out>.json   — matched keypoint coords in target image space
  <out>.png    — side-by-side visualisation (optional)

JSON output format
------------------
{
  "src_image":  "...",
  "tgt_image":  "...",
  "matches": [
    {
      "id": 0,
      "name": "Grasp",
      "src_xy": [428, 234],        # original-space pixel in source
      "tgt_xy": [512, 310]         # original-space pixel in target
    },
    ...
  ]
}

Memory strategy
---------------
* SD model and DINOv2 model are loaded once, used for both images, then
  deleted and CUDA cache cleared before running correspondence.
* All intermediate tensors are explicitly deleted after use.
* Cosine-similarity map is computed on GPU but immediately moved to CPU.
* Final descriptors (1×768×60×60 ≈ 11 MB each) are kept on CPU between steps.
"""

import argparse
import gc
import json
import os
import sys
import threading
import time

# ── STEP 1: set env var before ANY torch/cuda import ─────────────────────────
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

# ── STEP 2: import torch and immediately init CUDA context ───────────────────
import torch
import torch.nn.functional as F

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.zeros(1, device='cuda')   # forces cuDNN context creation
    torch.cuda.synchronize()
    # benchmark=True: cuDNN searches fastest algorithm (requires sufficient VRAM)
    torch.backends.cudnn.benchmark    = True
    torch.backends.cudnn.deterministic = False
    # TF32 for faster matmul on Ampere+ (no accuracy impact for feature matching)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

# ── STEP 3: now safe to import project modules that touch CUDA/detectron2 ────
import numpy as np
from PIL import Image
from types import SimpleNamespace

# ── project root on path ──────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.utils_correspondence import resize, kpts_to_patch_idx, calculate_keypoint_transformation
from model_utils.extractor_sd import load_model, process_features_and_mask
from model_utils.extractor_dino import ViTExtractor
from model_utils.projection_network import AggregationNetwork
from preprocess_map import set_seed

# ─── constants (must match training config) ───────────────────────────────────
NUM_PATCHES   = 60        # spatial resolution of descriptor map
IMG_SIZE      = 480       # display/correspondence resolution (padded square)
DEFAULT_CKPT  = os.path.join(ROOT, 'results_spair', 'best_856.PTH')


# ─── feature extraction (verbatim from get_processed_feat.ipynb) ─────────────

def get_processed_features(sd_model, sd_aug, aggre_net, extractor_vit, num_patches, img, sd_size=None):
    """
    Extract normalized descriptor map for one image.
    Copied verbatim from get_processed_feat.ipynb / README demo.

    sd_size: SD input resolution override (default: num_patches*16=960). Pass a smaller
             value (e.g. 512) to further reduce peak VRAM during the SD forward pass.
    Returns: CUDA tensor (1, 768, num_patches, num_patches), normalized.
    """
    if sd_size is None:
        sd_size = num_patches * 16
    with torch.no_grad():
        img_sd_input = resize(img, target_res=sd_size, resize=True, to_pil=True)
        features_sd = process_features_and_mask(sd_model, sd_aug, img_sd_input, mask=False, raw=True)
        del features_sd['s2']

        img_dino_input = resize(img, target_res=num_patches * 14, resize=True, to_pil=True)
        img_batch = extractor_vit.preprocess_pil(img_dino_input)
        # Cast input to match DINOv2 model dtype (supports FP16 mode)
        dino_dtype = next(extractor_vit.model.parameters()).dtype
        features_dino = extractor_vit.extract_descriptors(
            img_batch.to(dtype=dino_dtype).cuda(), layer=11, facet='token'
        ).permute(0, 1, 3, 2).reshape(1, -1, num_patches, num_patches)

        desc_gathered = torch.cat([
            features_sd['s3'].float(),
            F.interpolate(features_sd['s4'], size=(num_patches, num_patches),
                          mode='bilinear', align_corners=False).float(),
            F.interpolate(features_sd['s5'], size=(num_patches, num_patches),
                          mode='bilinear', align_corners=False).float(),
            features_dino.float(),
        ], dim=1)

        desc = aggre_net(desc_gathered)  # (1, 768, num_patches, num_patches)
        norms_desc = torch.linalg.norm(desc, dim=1, keepdim=True)
        desc = desc / (norms_desc + 1e-8)
    return desc


# ─── GPU memory monitor ───────────────────────────────────────────────────────

class _VramMonitor:
    """
    Background thread that prints GPU VRAM usage once per second.
    Usage:
        with _VramMonitor("Loading models"):
            load_model(...)
    """
    def __init__(self, label: str = '', interval: float = 1.0):
        self.label    = label
        self.interval = interval
        self._stop    = threading.Event()
        self._thread  = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(self.interval):
            if torch.cuda.is_available():
                used  = torch.cuda.memory_allocated() / 1024**3
                resv  = torch.cuda.memory_reserved()  / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                tag   = f'[VRAM] {self.label:30s}  used={used:.2f}GB  reserved={resv:.2f}GB  total={total:.2f}GB'
                print(tag, flush=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=2)


# ─── coordinate helpers ───────────────────────────────────────────────────────

def _resize_params(orig_w: int, orig_h: int, target_res: int):
    """Return (scale, offset_x, offset_y) for resize() with zero-pad."""
    if orig_h <= orig_w:          # landscape / square
        scale    = target_res / orig_w
        new_h    = int(round(target_res * orig_h / orig_w))
        offset_x = 0
        offset_y = (target_res - new_h) // 2
    else:                          # portrait
        scale    = target_res / orig_h
        new_w    = int(round(target_res * orig_w / orig_h))
        offset_x = (target_res - new_w) // 2
        offset_y = 0
    return scale, offset_x, offset_y


def orig_to_padded(xy, orig_w, orig_h, target_res=IMG_SIZE):
    """Map pixel (x,y) from original image space → padded square space."""
    scale, ox, oy = _resize_params(orig_w, orig_h, target_res)
    x = int(round(xy[0] * scale + ox))
    y = int(round(xy[1] * scale + oy))
    x = max(0, min(target_res - 1, x))
    y = max(0, min(target_res - 1, y))
    return x, y


def padded_to_orig(xy, orig_w, orig_h, target_res=IMG_SIZE):
    """Map pixel (x,y) from padded square space → original image space."""
    scale, ox, oy = _resize_params(orig_w, orig_h, target_res)
    x = int(round((xy[0] - ox) / scale))
    y = int(round((xy[1] - oy) / scale))
    x = max(0, min(orig_w - 1, x))
    y = max(0, min(orig_h - 1, y))
    return x, y


# ─── correspondence ───────────────────────────────────────────────────────────

def find_correspondences(feat_src: torch.Tensor,
                         feat_tgt: torch.Tensor,
                         src_kps_padded: list[tuple[int, int]],
                         img_size: int = IMG_SIZE,
                         num_patches: int = NUM_PATCHES) -> list[tuple[int, int]]:
    """
    For each source keypoint (x,y) in padded-space, find the best-matching
    pixel in target padded-space.

    Uses kpts_to_patch_idx + calculate_keypoint_transformation from
    utils/utils_correspondence.py (the same functions used in pck_train.py).

    feat_src / feat_tgt: CUDA tensors (1, 768, num_patches, num_patches)
    Returns: list of (x, y) in target padded-space (img_size pixels).
    """
    # reshape spatial map (1, C, P, P) → patch list (1, P*P, C)
    desc1 = feat_src.reshape(1, -1, num_patches ** 2).permute(0, 2, 1)  # (1, 3600, 768)
    desc2 = feat_tgt.reshape(1, -1, num_patches ** 2).permute(0, 2, 1)

    # kpts_to_patch_idx expects a torch FloatTensor (N, ≥2) with col-0=x col-1=y in ANNO_SIZE space
    kps_tensor = torch.tensor([[x, y, 1] for x, y in src_kps_padded], dtype=torch.float32)
    args = SimpleNamespace(ANNO_SIZE=img_size, SOFT_EVAL=False)
    patch_idx = kpts_to_patch_idx(args, kps_tensor, num_patches)  # (N,) int patch indices

    # calculate_keypoint_transformation returns (N, 2) tensor: (x, y) in ANNO_SIZE pixels
    kps_1_to_2 = calculate_keypoint_transformation(args, desc1, desc2, patch_idx, num_patches)

    return [(int(k[0].item()), int(k[1].item())) for k in kps_1_to_2]


# ─── visualisation ────────────────────────────────────────────────────────────
_COLOURS = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
]

def _colour(i):
    return _COLOURS[i % len(_COLOURS)]


def save_vis(src_img, tgt_img, src_kps, tgt_kps, names, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax in axes:
        ax.axis('off')
    axes[0].imshow(np.array(src_img))
    axes[0].set_title('Source', fontsize=11)
    axes[1].imshow(np.array(tgt_img))
    axes[1].set_title('Target', fontsize=11)

    for i, ((sx, sy), (tx, ty), name) in enumerate(zip(src_kps, tgt_kps, names)):
        c = _colour(i)
        axes[0].scatter(sx, sy, s=150, c=c, zorder=5, edgecolors='white', linewidths=1.5)
        axes[0].annotate(f' {i}:{name}', (sx, sy), fontsize=8, color='white', fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.2', fc=c, alpha=0.8, ec='none'))
        axes[1].scatter(tx, ty, s=150, c=c, zorder=5, edgecolors='white', linewidths=1.5)
        axes[1].annotate(f' {i}:{name}', (tx, ty), fontsize=8, color='white', fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.2', fc=c, alpha=0.8, ec='none'))
        # draw matching line between images using figure coords is complex −
        # annotate with same colour is enough for clear correspondence

    patches = [mpatches.Patch(color=_colour(i), label=f'{i}: {n}') for i, n in enumerate(names)]
    fig.legend(handles=patches, loc='lower center', ncol=len(names), fontsize=9, framealpha=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[kp_match] Saved vis  → {out_path}')


# ─── main ─────────────────────────────────────────────────────────────────────

def match(src_path: str,
          kps_path: str,
          tgt_path: str,
          ckpt_path: str = DEFAULT_CKPT,
          out_path: str | None = None,
          save_vis_flag: bool = True,
          device: str = 'cuda',
          half_precision: bool = False,
          sd_size: int = NUM_PATCHES * 16) -> dict:
    """
    Full matching pipeline. Returns the result dict (same as saved JSON).
    Can be imported and called programmatically.

    half_precision: use FP16 for SD + DINOv2 weights, reducing VRAM from ~10 GB to ~5 GB.
                    Recommended for GPUs with < 16 GB VRAM (RTX 3080/4080 etc.).
    sd_size:        SD input resolution (default 960). Use 512 to further reduce peak VRAM
                    during SD forward pass (~25% less activation memory). Slightly lower quality.
    """
    set_seed(42)

    # ── load images ──────────────────────────────────────────────────────────
    src_img = Image.open(src_path).convert('RGB')
    tgt_img = Image.open(tgt_path).convert('RGB')
    src_w, src_h = src_img.size
    tgt_w, tgt_h = tgt_img.size

    # ── load keypoints ───────────────────────────────────────────────────────
    with open(kps_path) as f:
        kps_data = json.load(f)

    kps_raw: list[dict] = []
    for i, xyv in enumerate(kps_data['keypoints_xyv']):
        kps_raw.append({
            'id': i,
            'name': kps_data.get('keypoint_names', ['kp'])[i] if i < len(kps_data.get('keypoint_names', [])) else f'kp{i}',
            'src_xy': (int(xyv[0]), int(xyv[1])),
        })

    # map source KPs to padded-square space for feature lookup
    src_kps_padded = [
        orig_to_padded(kp['src_xy'], src_w, src_h, IMG_SIZE)
        for kp in kps_raw
    ]

    # ── load aggregation network (tiny, ~10 MB) ──────────────────────────────
    print('[kp_match] Loading AggregationNetwork ...')
    aggre_net = AggregationNetwork(
        feature_dims=[640, 1280, 1280, 768],
        projection_dim=768,
        device=device,
    )
    aggre_net.load_pretrained_weights(torch.load(ckpt_path, map_location=device))
    aggre_net.eval()

    # ── load backbone models ─────────────────────────────────────────────────
    print(f'[kp_match] Loading SD + DINOv2 (half_precision={half_precision}, sd_size={sd_size}) ...')
    with _VramMonitor('Loading SD+DINOv2'):
        sd_model, sd_aug = load_model(
            diffusion_ver='v1-5',
            image_size=sd_size,
            num_timesteps=50,
            block_indices=[2, 5, 8, 11],
            half_precision=half_precision,
        )
        extractor_vit = ViTExtractor('dinov2_vitb14', stride=14, device=device,
                                     half_precision=half_precision)

    # ── extract features (using get_processed_features from the project demo) ─
    print('[kp_match] Extracting features: source ...')
    with _VramMonitor('Feature extract: source'):
        feat_src = get_processed_features(sd_model, sd_aug, aggre_net, extractor_vit, NUM_PATCHES,
                                          img=src_img, sd_size=sd_size)

    print('[kp_match] Extracting features: target ...')
    with _VramMonitor('Feature extract: target'):
        feat_tgt = get_processed_features(sd_model, sd_aug, aggre_net, extractor_vit, NUM_PATCHES,
                                          img=tgt_img, sd_size=sd_size)

    # ── free backbone models — done with them ─────────────────────────────────
    print('[kp_match] Releasing backbone models from GPU ...')
    del sd_model, sd_aug, extractor_vit, aggre_net
    gc.collect()
    torch.cuda.empty_cache()

    # ── correspondence ───────────────────────────────────────────────────────
    print('[kp_match] Running correspondence ...')
    tgt_kps_padded = find_correspondences(feat_src, feat_tgt, src_kps_padded, IMG_SIZE)

    del feat_src, feat_tgt
    gc.collect()
    torch.cuda.empty_cache()

    # ── map target KPs back to original image space ──────────────────────────
    matches = []
    for kp, tgt_pad in zip(kps_raw, tgt_kps_padded):
        tgt_orig = padded_to_orig(tgt_pad, tgt_w, tgt_h, IMG_SIZE)
        matches.append({
            'id':     kp['id'],
            'name':   kp['name'],
            'src_xy': list(kp['src_xy']),
            'tgt_xy': list(tgt_orig),
        })
        print(f'  [{kp["id"]}] {kp["name"]:10s}  src={kp["src_xy"]}  →  tgt={tgt_orig}')

    result = {
        'src_image': os.path.abspath(src_path),
        'tgt_image': os.path.abspath(tgt_path),
        'matches':   matches,
    }

    # ── save JSON ─────────────────────────────────────────────────────────────
    if out_path is None:
        stem = os.path.splitext(tgt_path)[0]
        out_path = f'{stem}_match.json'

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'[kp_match] Saved JSON → {out_path}')

    # ── save visualisation ────────────────────────────────────────────────────
    if save_vis_flag:
        vis_path = os.path.splitext(out_path)[0] + '_vis.png'
        src_img_disp = resize(src_img, target_res=IMG_SIZE, resize=True, to_pil=True)
        tgt_img_disp = resize(tgt_img, target_res=IMG_SIZE, resize=True, to_pil=True)
        save_vis(
            src_img_disp, tgt_img_disp,
            src_kps_padded, tgt_kps_padded,
            [m['name'] for m in matches],
            vis_path,
        )

    return result


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='GeoAware-SC semantic keypoint matching.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--src',   required=True, help='Source image path')
    parser.add_argument('--kps',   required=True, help='Source keypoints JSON (from kp_select.py)')
    parser.add_argument('--tgt',   required=True, help='Target image path')
    parser.add_argument('--ckpt',  default=DEFAULT_CKPT, help='AggregationNetwork checkpoint (.PTH)')
    parser.add_argument('--out',   default=None,  help='Output JSON path (default: <tgt_stem>_match.json)')
    parser.add_argument('--no-vis', action='store_true', help='Skip saving visualisation PNG')
    parser.add_argument('--fp16', action='store_true',
                        help='Use FP16 (half-precision) for SD + DINOv2. '
                             'Reduces VRAM from ~10 GB to ~5 GB. Recommended for 30/40-series GPUs.')
    parser.add_argument('--sd-size', type=int, default=NUM_PATCHES * 16, metavar='N',
                        help=f'SD input resolution (default: {NUM_PATCHES * 16}). '
                             'Use 512 to further reduce peak VRAM at slight quality cost.')
    args = parser.parse_args()

    for p in [args.src, args.kps, args.tgt, args.ckpt]:
        if not os.path.isfile(p):
            sys.exit(f'[kp_match] ERROR: file not found: {p}')

    match(
        src_path=args.src,
        kps_path=args.kps,
        tgt_path=args.tgt,
        ckpt_path=args.ckpt,
        out_path=args.out,
        save_vis_flag=not args.no_vis,
        half_precision=args.fp16,
        sd_size=args.sd_size,
    )


if __name__ == '__main__':
    main()

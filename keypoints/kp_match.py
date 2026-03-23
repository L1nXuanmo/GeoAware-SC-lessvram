"""
keypoints/kp_match.py — Topology-aware semantic keypoint matching (GeoAware-SC).

Given a source image with topology keypoints and a target image, finds the
semantically corresponding pixel locations on the target using a two-stage
pipeline:
  Stage 1 — Top-K candidates + topology neighbour re-ranking (scale-invariant)
  Stage 2 — Up-sampled local refinement (60→240, precision ~8px → ~2px)

Usage
-----
    python keypoints/kp_match.py \\
        --src  <source.png> \\
        --kps  <source_topo.npy> \\
        --tgt  <target.png> \\
        [--ckpt results_spair/best_856.PTH] \\
        [--out  result.npy] [--no-vis] \\
        [--fp16] [--sd-size 960] \\
        [--top-k 8] [--alpha 0.7] [--refine 4]

Key options
-----------
  --fp16          FP16 backbone (~5 GB vs ~10 GB VRAM)
  --sd-size N     SD input resolution (960/768/640/512, multiples of 64)
  --top-k K       Stage-1 candidate count (1 = legacy argmax, default 8)
  --alpha A       Appearance vs geometry weight (1.0 = appearance only, default 0.7)
  --refine R      Feature upsample factor for Stage-2 (0 = skip, default 4)

Output  <out>.npy + <out>_vis.png
------
  .npy dict: src_image, src_nodes (N,2), tgt_image, tgt_nodes (N,2),
             node_parts (N,), part_names {id:str}, adj_matrix (N,N)
"""

import argparse
import gc
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

# ── project root on path ──────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.utils_correspondence import resize
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
                         adj_matrix: np.ndarray | None = None,
                         nodes_3d: np.ndarray | None = None,
                         node_parts: np.ndarray | None = None,
                         img_size: int = IMG_SIZE,
                         num_patches: int = NUM_PATCHES,
                         top_k: int = 8,
                         alpha: float = 0.7,
                         sigma: float = 0.3,
                         depth_weight: float = 0.15,
                         refine_factor: int = 4,
                         refine_window: int = 16) -> list[tuple[int, int]]:
    """
    Two-stage keypoint matching with topology-aware refinement.

    Stage 1 — Top-K + topology neighbour consistency + 3D depth constraint:
      For each source keypoint, keep the K highest-cosine candidates in the
      target patch grid.  When *adj_matrix* is provided, re-rank candidates
      using a combined score:
          final(c) = α · sim(c) + (1-α-β) · geo_score(c) + β · depth_score(c)
      where geo_score penalises ratio deviations of edge lengths (scale-invariant),
      and depth_score penalises candidates that break the same-Part height ordering
      from source 3D coordinates (requires nodes_3d).

    Stage 2 — Up-sampled refinement (unchanged).

    Parameters
    ----------
    nodes_3d       : (N, 3) float64 from kp_select (NaN for invalid). None to skip.
    node_parts     : (N,) int32 Part IDs. Needed for same-Part 3D constraint.
    depth_weight   : weight β for 3D depth constraint (default 0.15).
                     geo weight becomes (1 - α - β). Set 0 to disable.

    Returns
    -------
    list of (x, y) in target padded-space (img_size pixels).
    """
    P  = num_patches                     # 60
    PP = P * P                           # 3600
    stride = img_size / P                # 8.0

    # ── similarity matrix (P*P × P*P) ────────────────────────────────────────
    desc1 = feat_src.reshape(1, -1, PP).permute(0, 2, 1)   # (1, 3600, C)
    desc2 = feat_tgt.reshape(1, -1, PP).permute(0, 2, 1)
    sim = torch.matmul(desc1, desc2.permute(0, 2, 1))[0]   # (3600, 3600)

    # source keypoints → patch indices
    kps_arr = np.array(src_kps_padded, dtype=np.float32)    # (N, 2) [x, y]
    src_patch_x = np.clip((P / img_size * kps_arr[:, 0]).astype(np.int32), 0, P - 1)
    src_patch_y = np.clip((P / img_size * kps_arr[:, 1]).astype(np.int32), 0, P - 1)
    patch_idx   = P * src_patch_y + src_patch_x              # (N,)
    N = len(patch_idx)

    sim_indexed = sim[patch_idx]                              # (N, 3600)

    # ── Stage 1a: Top-K candidates ───────────────────────────────────────────
    K = min(top_k, PP)
    topk_vals, topk_ids = torch.topk(sim_indexed, K, dim=-1)  # (N, K)

    # convert top-k patch indices → pixel coords  (patch centre)
    topk_py = topk_ids // P           # (N, K) row in patch grid
    topk_px = topk_ids %  P           # (N, K) col
    topk_x  = (topk_px.float() * stride + stride / 2)  # (N, K) pixel x
    topk_y  = (topk_py.float() * stride + stride / 2)  # (N, K) pixel y

    # ── Stage 1b: Topology-aware re-ranking ──────────────────────────────────
    if adj_matrix is not None and N > 1:
        # first-pass argmax positions (used as neighbour anchors)
        argmax_ids = torch.argmax(sim_indexed, dim=-1)        # (N,)
        anchor_x = (argmax_ids % P).float() * stride + stride / 2   # (N,)
        anchor_y = (argmax_ids // P).float() * stride + stride / 2

        # source keypoint positions as tensor
        src_x = torch.tensor([p[0] for p in src_kps_padded], dtype=torch.float32)
        src_y = torch.tensor([p[1] for p in src_kps_padded], dtype=torch.float32)

        adj = torch.from_numpy((adj_matrix > 0).astype(np.float32))  # (N, N)

        # normalise sim scores to [0, 1] per keypoint for blending
        sim_norm = topk_vals - topk_vals.min(dim=-1, keepdim=True).values
        denom = topk_vals.max(dim=-1, keepdim=True).values - topk_vals.min(dim=-1, keepdim=True).values
        sim_norm = sim_norm / (denom + 1e-8)

        geo_scores = torch.zeros(N, K)
        for i in range(N):
            neighbours = torch.nonzero(adj[i], as_tuple=True)[0]
            if len(neighbours) == 0:
                geo_scores[i] = 1.0           # no constraint → don't penalise
                continue
            for ki in range(K):
                cx, cy = topk_x[i, ki].item(), topk_y[i, ki].item()
                ratios = []
                for j in neighbours:
                    j = j.item()
                    d_src = ((src_x[i] - src_x[j]) ** 2 + (src_y[i] - src_y[j]) ** 2).sqrt().item()
                    d_tgt = ((cx - anchor_x[j].item()) ** 2 + (cy - anchor_y[j].item()) ** 2) ** 0.5
                    if d_src < 1e-3:
                        ratios.append(1.0)
                    else:
                        ratios.append(np.exp(-((d_tgt / d_src - 1.0) ** 2) / (2 * sigma ** 2)))
                geo_scores[i, ki] = sum(ratios) / len(ratios)

        combined = alpha * sim_norm + (1 - alpha) * geo_scores.to(sim_norm.device)

        # ── Stage 1c: 3D depth constraint (same-Part height ordering) ────────
        #  For each pair of same-Part points (i, j) with valid 3D, the
        #  vertical difference (Y_i - Y_j) from source 3D should be preserved
        #  proportionally in image space (v_i - v_j) on the target.
        #  This prevents cup-rim points from jumping to the cup body.
        has_3d = (nodes_3d is not None and node_parts is not None
                  and depth_weight > 0)
        if has_3d:
            beta = min(depth_weight, 1.0 - alpha)  # don't exceed budget
            depth_scores = torch.ones(N, K, device=sim_norm.device)
            sigma_d = 0.3  # tolerance for normalised depth-ordering deviation

            for i in range(N):
                pid_i = int(node_parts[i])
                if np.isnan(nodes_3d[i, 0]):
                    continue
                # find same-Part peers with valid 3D (not necessarily neighbours)
                peers = [j for j in range(N)
                         if j != i and int(node_parts[j]) == pid_i
                         and not np.isnan(nodes_3d[j, 0])]
                if not peers:
                    continue

                # source Y differences (camera frame: Y axis ~ vertical)
                src_y3d_i = nodes_3d[i, 1]
                for ki in range(K):
                    cy_cand = topk_y[i, ki].item()
                    agree = []
                    for j in peers:
                        src_dy = src_y3d_i - nodes_3d[j, 1]   # 3D height diff
                        src_dv = kps_arr[i, 1] - kps_arr[j, 1]  # source image v diff
                        tgt_dv = cy_cand - anchor_y[j].item()    # target image v diff
                        # check if direction of vertical offset is preserved
                        if abs(src_dv) < 1e-3:
                            agree.append(1.0)
                        else:
                            ratio = tgt_dv / src_dv
                            agree.append(np.exp(-((ratio - 1.0) ** 2) / (2 * sigma_d ** 2)))
                    depth_scores[i, ki] = sum(agree) / len(agree)

            # re-weight: alpha * sim + (1-alpha-beta) * geo + beta * depth
            geo_weight = max(0, 1.0 - alpha - beta)
            combined = (alpha * sim_norm
                        + geo_weight * geo_scores.to(sim_norm.device)
                        + beta * depth_scores)

        best_k = torch.argmax(combined, dim=-1)  # (N,)
    else:
        best_k = torch.zeros(N, dtype=torch.long)  # just take top-1

    # gather initial matches (patch-resolution)
    init_x = torch.tensor([topk_x[i, best_k[i]].item() for i in range(N)])
    init_y = torch.tensor([topk_y[i, best_k[i]].item() for i in range(N)])

    # ── Stage 2: Up-sampled local refinement ─────────────────────────────────
    if refine_factor > 0:
        UP = P * refine_factor                                       # 240
        up_stride = img_size / UP                                    # 2.0
        feat_src_up = F.interpolate(feat_src, size=(UP, UP), mode='bilinear', align_corners=False)
        feat_tgt_up = F.interpolate(feat_tgt, size=(UP, UP), mode='bilinear', align_corners=False)
        # normalise after interpolation
        feat_src_up = feat_src_up / (feat_src_up.norm(dim=1, keepdim=True) + 1e-8)
        feat_tgt_up = feat_tgt_up / (feat_tgt_up.norm(dim=1, keepdim=True) + 1e-8)

        C = feat_src_up.shape[1]
        results = []
        W = refine_window   # half-size in upsampled pixels

        for i in range(N):
            # source descriptor at upsampled resolution
            sx_up = int(round(kps_arr[i, 0] / up_stride))
            sy_up = int(round(kps_arr[i, 1] / up_stride))
            sx_up = max(0, min(UP - 1, sx_up))
            sy_up = max(0, min(UP - 1, sy_up))
            src_desc = feat_src_up[0, :, sy_up, sx_up]               # (C,)

            # centre of search in target upsampled grid
            cx_up = int(round(init_x[i].item() / up_stride))
            cy_up = int(round(init_y[i].item() / up_stride))

            # clipped window
            y0 = max(0, cy_up - W)
            y1 = min(UP, cy_up + W + 1)
            x0 = max(0, cx_up - W)
            x1 = min(UP, cx_up + W + 1)

            patch = feat_tgt_up[0, :, y0:y1, x0:x1]                 # (C, h, w)
            h, w = patch.shape[1], patch.shape[2]
            local_sim = torch.einsum('c,chw->hw', src_desc, patch)   # (h, w)
            best_local = local_sim.reshape(-1).argmax().item()
            by, bx = divmod(best_local, w)

            rx = (x0 + bx) * up_stride + up_stride / 2
            ry = (y0 + by) * up_stride + up_stride / 2
            rx = max(0, min(img_size - 1, rx))
            ry = max(0, min(img_size - 1, ry))
            results.append((int(round(rx)), int(round(ry))))

        return results
    else:
        return [(int(round(init_x[i].item())), int(round(init_y[i].item())))
                for i in range(N)]


# ─── visualisation ────────────────────────────────────────────────────────────
# Part-colour palette — identical to kp_select.py so visuals stay consistent.
_PART_COLOURS = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
    '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3',
    '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#ffe119',
]
_EDGE_COLOUR_INTRA = '#ffffff'
_EDGE_COLOUR_INTER = '#ff6600'


def _part_colour(part_id: int) -> str:
    return _PART_COLOURS[part_id % len(_PART_COLOURS)]


def save_vis(src_img, tgt_img, src_kps, tgt_kps,
             node_parts, part_names, adj_matrix, out_path):
    """Side-by-side visualisation with part colours, labels, and topology edges."""
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

    N = len(src_kps)

    # ── draw topology edges ──────────────────────────────────────────────────
    for panel_kps, ax in [(src_kps, axes[0]), (tgt_kps, axes[1])]:
        for a in range(N):
            for b in range(a + 1, N):
                val = adj_matrix[a, b]
                if val == 0:
                    continue
                xa, ya = panel_kps[a]
                xb, yb = panel_kps[b]
                is_inter = (val == 2)
                style = '--' if is_inter else '-'
                colour = _EDGE_COLOUR_INTER if is_inter else _EDGE_COLOUR_INTRA
                ax.plot([xa, xb], [ya, yb], style, color=colour,
                        linewidth=1.5, alpha=0.8, zorder=3)

    # ── draw nodes & labels ──────────────────────────────────────────────────
    for i, ((sx, sy), (tx, ty)) in enumerate(zip(src_kps, tgt_kps)):
        pid = int(node_parts[i])
        c = _part_colour(pid)
        pname = part_names.get(pid, f'part_{pid}')

        for ax, x, y in [(axes[0], sx, sy), (axes[1], tx, ty)]:
            ax.scatter(x, y, s=120, c=c, zorder=5, edgecolors='white', linewidths=1.2)
            ax.annotate(
                f'{i}:{pname}',
                (x, y),
                xytext=(12, 0),
                textcoords='offset points',
                fontsize=7, color='white', fontweight='bold',
                va='center',
                bbox=dict(boxstyle='round,pad=0.15', fc=c, alpha=0.75, ec='none'),
            )

    # ── legend ───────────────────────────────────────────────────────────────
    used = sorted(set(int(p) for p in node_parts))
    patches = [mpatches.Patch(
        color=_part_colour(pid),
        label=f'{part_names.get(pid, f"part_{pid}")} ({int(np.sum(node_parts == pid))} pts)',
    ) for pid in used]
    fig.legend(handles=patches, loc='lower center', ncol=max(1, len(used)),
               fontsize=9, framealpha=0.8)
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
          sd_size: int = NUM_PATCHES * 16,
          top_k: int = 8,
          alpha: float = 0.7,
          refine_factor: int = 4,
          depth_weight: float = 0.15) -> dict:
    """
    Full matching pipeline. Returns the result dict.

    half_precision: use FP16 for SD + DINOv2 weights, reducing VRAM from ~10 GB to ~5 GB.
    sd_size:        SD input resolution (default 960). Use 512 to further reduce peak VRAM.
    top_k:          Stage-1 candidate count for topology-aware re-ranking (1 = legacy argmax).
    alpha:          Blend weight appearance vs geometry (1.0 = appearance only).
    depth_weight:   Blend weight for 3D depth constraint (0 = disable, requires nodes_3d in topo).
    refine_factor:  Feature upsample factor for Stage-2 local refinement (0 = skip).
    """
    set_seed(42)

    # ── load images ──────────────────────────────────────────────────────────
    src_img = Image.open(src_path).convert('RGB')
    tgt_img = Image.open(tgt_path).convert('RGB')
    src_w, src_h = src_img.size
    tgt_w, tgt_h = tgt_img.size

    # ── load topology keypoints (.npy from kp_select.py) ────────────────────
    topo       = np.load(kps_path, allow_pickle=True).item()
    nodes      = topo['nodes']        # (N, 2) int32 — [x, y] pixel coords
    node_parts = topo['node_parts']   # (N,) int32
    part_names = topo['part_names']   # {int: str}

    kps_raw: list[dict] = []
    for i in range(len(nodes)):
        pid = int(node_parts[i])
        pname = part_names.get(pid, f'part_{pid}')
        kps_raw.append({
            'id': i,
            'name': pname,
            'part_id': pid,
            'src_xy': (int(nodes[i][0]), int(nodes[i][1])),
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
    aggre_net.load_pretrained_weights(torch.load(ckpt_path, map_location=device, weights_only=False))
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
    print(f'[kp_match] Running correspondence (top_k={top_k}, alpha={alpha}, refine={refine_factor}x) ...')
    tgt_kps_padded = find_correspondences(
        feat_src, feat_tgt, src_kps_padded,
        adj_matrix=topo['adj_matrix'],
        nodes_3d=topo.get('nodes_3d'),
        node_parts=topo['node_parts'],
        img_size=IMG_SIZE,
        top_k=top_k,
        alpha=alpha,
        depth_weight=depth_weight,
        refine_factor=refine_factor,
    )

    del feat_src, feat_tgt
    gc.collect()
    torch.cuda.empty_cache()

    # ── map target KPs back to original image space ──────────────────────────
    src_nodes_list = []
    tgt_nodes_list = []
    for kp, tgt_pad in zip(kps_raw, tgt_kps_padded):
        tgt_orig = padded_to_orig(tgt_pad, tgt_w, tgt_h, IMG_SIZE)
        src_nodes_list.append(kp['src_xy'])
        tgt_nodes_list.append(tgt_orig)
        print(f'  [{kp["id"]}] {kp["name"]:10s}  src={kp["src_xy"]}  →  tgt={tgt_orig}')

    result = {
        'src_image':   os.path.abspath(src_path),
        'src_nodes':   np.array(src_nodes_list, dtype=np.int32),   # (N, 2)
        'tgt_image':   os.path.abspath(tgt_path),
        'tgt_nodes':   np.array(tgt_nodes_list, dtype=np.int32),   # (N, 2)
        'node_parts':  topo['node_parts'],                         # (N,) int32
        'part_names':  topo['part_names'],                         # {int: str}
        'adj_matrix':  topo['adj_matrix'],                         # (N, N) int32
    }

    # ── save .npy ─────────────────────────────────────────────────────────────
    if out_path is None:
        stem = os.path.splitext(tgt_path)[0]
        out_path = f'{stem}_match.npy'

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.save(out_path, result, allow_pickle=True)
    print(f'[kp_match] Saved .npy → {out_path}')

    # ── save visualisation ────────────────────────────────────────────────────
    if save_vis_flag:
        vis_path = os.path.splitext(out_path)[0] + '_vis.png'
        src_img_disp = resize(src_img, target_res=IMG_SIZE, resize=True, to_pil=True)
        tgt_img_disp = resize(tgt_img, target_res=IMG_SIZE, resize=True, to_pil=True)
        save_vis(
            src_img_disp, tgt_img_disp,
            src_kps_padded, tgt_kps_padded,
            topo['node_parts'], topo['part_names'], topo['adj_matrix'],
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
    parser.add_argument('--kps',   required=True, help='Source topology .npy file (from kp_select.py)')
    parser.add_argument('--tgt',   required=True, help='Target image path')
    parser.add_argument('--ckpt',  default=DEFAULT_CKPT, help='AggregationNetwork checkpoint (.PTH)')
    parser.add_argument('--out',   default=None,  help='Output .npy path (default: <tgt_stem>_match.npy)')
    parser.add_argument('--no-vis', action='store_true', help='Skip saving visualisation PNG')
    parser.add_argument('--fp16', action='store_true',
                        help='Use FP16 (half-precision) for SD + DINOv2. '
                             'Reduces VRAM from ~10 GB to ~5 GB. Recommended for 30/40-series GPUs.')
    parser.add_argument('--sd-size', type=int, default=NUM_PATCHES * 16, metavar='N',
                        help=f'SD input resolution (default: {NUM_PATCHES * 16}). '
                             'Use 512 to further reduce peak VRAM at slight quality cost.')
    parser.add_argument('--top-k', type=int, default=8, metavar='K',
                        help='Number of Top-K candidates per keypoint for topology '
                             're-ranking (default: 8, 1 = legacy argmax).')
    parser.add_argument('--alpha', type=float, default=0.7, metavar='A',
                        help='Blend weight: appearance vs geometry '
                             '(default: 0.7, 1.0 = appearance only).')
    parser.add_argument('--refine', type=int, default=4, metavar='R',
                        help='Feature upsample factor for sub-patch refinement '
                             '(default: 4 → 60→240, 0 = skip).')
    parser.add_argument('--depth-weight', type=float, default=0.15, metavar='B',
                        help='Blend weight for 3D depth constraint '
                             '(default: 0.15, 0 = disable, requires nodes_3d in topo).')
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
        top_k=args.top_k,
        alpha=args.alpha,
        refine_factor=args.refine,
        depth_weight=args.depth_weight,
    )


if __name__ == '__main__':
    main()

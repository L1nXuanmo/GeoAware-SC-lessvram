"""
Template keypoint selector for GeoAware-SC based cross-category grasping pipeline.

Usage:
    python keypoints/select.py <image_path>

Output (saved alongside the input image):
    <image_stem>_kps.json   — keypoints in SPair-compatible format
    <image_stem>_kps.png    — visualization of selected keypoints

JSON format (compatible with project's preprocess_kps_pad / load_spair_data):
    {
        "image_width":  <int>,
        "image_height": <int>,
        "kps": {
            "0": [x, y],   # pixel coords in original image space
            "1": [x, y],
            ...
        },
        "keypoint_names": ["name0", "name1", ...]  # optional names
    }

Controls:
    Left-click   → add keypoint
    Right-click  → remove last keypoint
    'd' key      → remove last keypoint
    'u' key      → undo all (clear)
    Enter / 'q'  → save & exit
    Esc          → exit WITHOUT saving

Keypoint order (fixed):
    Point 0 → Grasp   (grasping contact point)
    Point 1 → Func    (functional reference point)
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('TkAgg')          # interactive backend; fall back to Qt5Agg if needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

# ─── colour palette (cycles if more than 20 points) ────────────────────────
_COLOURS = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
    '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000',
    '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9',
]

def _colour(idx: int) -> str:
    return _COLOURS[idx % len(_COLOURS)]


class KeypointSelector:
    """Interactive keypoint selector backed by Matplotlib."""

    # Fixed semantic names — not configurable via CLI
    FIXED_NAMES = ['Grasp', 'Func']

    def __init__(self, image_path: str):
        self.image_path = os.path.abspath(image_path)
        self.img = Image.open(self.image_path).convert('RGB')
        self.img_w, self.img_h = self.img.size
        self.names = self.FIXED_NAMES

        self.points: list[tuple[float, float]] = []   # (x, y) in image pixels
        self._saved = False

        self._build_ui()

    # ── UI setup ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.canvas.manager.set_window_title(
            f'Select keypoints — {os.path.basename(self.image_path)}'
        )
        self.ax.imshow(np.array(self.img))
        self._set_title()
        self.ax.axis('off')

        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.fig.canvas.mpl_connect('close_event', self._on_close)
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)
        # Store current view limits for zoom
        self._zoom_scale = 1.0

    # ── event handlers ────────────────────────────────────────────────────────

    def _set_title(self):
        next_idx = len(self.points)
        if next_idx < len(self.names):
            next_name = self.names[next_idx]
            prompt = f'Select Grasp first (pt0), then Func (pt1)  --  Next: [{next_idx}] {next_name}'
        else:
            prompt = f'All {len(self.names)} points selected  |  Press Enter/Q to save'
        self.ax.set_title(
            f'{prompt}\nLeft-click: add  |  Right-click/D: undo  |  U: clear  |  Enter/Q: save & quit  |  Esc: cancel  |  Scroll: zoom',
            fontsize=9,
        )

    def _on_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 1:            # left click → add
            if len(self.points) >= len(self.names):
                print(f'[kp_select] All {len(self.names)} points selected (Grasp + Func). Press U to clear and reselect.')
                return
            # snap to integer pixel grid so depth lookup is exact
            x, y = int(round(event.xdata)), int(round(event.ydata))
            self.points.append((x, y))
            self._refresh()
        elif event.button == 3:          # right click → remove last
            if self.points:
                self.points.pop()
                self._refresh()

    def _on_key(self, event):
        if event.key in ('enter', 'q'):
            self._save()
            plt.close(self.fig)
        elif event.key == 'escape':
            print('[select.py] Cancelled — nothing saved.')
            plt.close(self.fig)
        elif event.key in ('d', 'backspace'):
            if self.points:
                self.points.pop()
                self._refresh()
        elif event.key == 'u':
            self.points.clear()
            self._refresh()

    def _on_scroll(self, event):
        """Scroll wheel zoom centred on the cursor position."""
        if event.inaxes != self.ax:
            return
        factor = 0.8 if event.button == 'up' else 1.25
        cx, cy = event.xdata, event.ydata
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        new_xlim = [cx + (x - cx) * factor for x in xlim]
        new_ylim = [cy + (y - cy) * factor for y in ylim]
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.fig.canvas.draw_idle()

    def _on_close(self, event):
        # If the window is closed without explicit save, still save if any points exist
        if not self._saved and self.points:
            ans = input('\n[kp_select] Window closed. Save current points? [Y/n]: ').strip().lower()
            if ans in ('', 'y', 'yes'):
                self._save()

    # ── drawing ──────────────────────────────────────────────────────────────

    def _refresh(self):
        self.ax.cla()
        self.ax.imshow(np.array(self.img))
        self._set_title()
        self.ax.axis('off')

        for idx, (x, y) in enumerate(self.points):
            c = _colour(idx)
            self.ax.scatter(x, y, s=120, c=c, zorder=5, linewidths=1.2, edgecolors='white')
            label = self.names[idx] if idx < len(self.names) else f'kp{idx}'
            self.ax.annotate(
                f' {idx}: {label}',
                (x, y),
                fontsize=8,
                color='white',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc=c, alpha=0.75, ec='none'),
            )

        # legend patch summary
        if self.points:
            patches = [
                mpatches.Patch(
                    color=_colour(i),
                    label=f'{i}: {self.names[i] if i < len(self.names) else f"kp{i}"}  ({x}, {y})',
                )
                for i, (x, y) in enumerate(self.points)
            ]
            self.ax.legend(
                handles=patches,
                loc='upper right',
                fontsize=7,
                framealpha=0.8,
                title=f'{len(self.points)} point(s)',
                title_fontsize=8,
            )

        self.fig.canvas.draw_idle()

    # ── I/O ──────────────────────────────────────────────────────────────────

    def _save(self):
        if not self.points:
            print('[select.py] No points selected — nothing saved.')
            return

        stem = os.path.splitext(self.image_path)[0]
        json_path = f'{stem}_kps.json'
        png_path  = f'{stem}_kps.png'

        # ── build JSON (SPair-compatible interface) ──────────────────────────
        # Coordinates are integer pixel indices — exact for depth map lookup: depth[y, x]
        kps_dict: dict[str, list[int] | None] = {}
        for i, (x, y) in enumerate(self.points):
            kps_dict[str(i)] = [int(x), int(y)]

        payload = {
            'image_width':  self.img_w,
            'image_height': self.img_h,
            'kps': kps_dict,
            'keypoint_names': [
                self.names[i] if i < len(self.names) else f'kp{i}'
                for i in range(len(self.points))
            ],
            # [x, y, visibility=1]  —  x/y are integer pixel coords
            'keypoints_xyv': [[int(x), int(y), 1] for x, y in self.points],
        }

        with open(json_path, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f'[select.py] Saved JSON  → {json_path}')

        # ── save visualisation ───────────────────────────────────────────────
        fig_save, ax_save = plt.subplots(
            figsize=(max(6, self.img_w / 100), max(5, self.img_h / 100)),
            dpi=100,
        )
        ax_save.imshow(np.array(self.img))
        ax_save.axis('off')
        for idx, (x, y) in enumerate(self.points):
            c = _colour(idx)
            ax_save.scatter(x, y, s=120, c=c, zorder=5, linewidths=1.2, edgecolors='white')
            label = self.names[idx] if idx < len(self.names) else f'kp{idx}'
            ax_save.annotate(
                f' {idx}: {label}',
                (x, y),
                fontsize=8,
                color='white',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc=c, alpha=0.75, ec='none'),
            )
        fig_save.tight_layout(pad=0)
        fig_save.savefig(png_path, bbox_inches='tight', dpi=150)
        plt.close(fig_save)
        print(f'[select.py] Saved PNG   → {png_path}')

        self._saved = True

    # ── entry ─────────────────────────────────────────────────────────────────

    def run(self):
        plt.show()


# ─── helper: load saved keypoints back as a torch tensor ────────────────────
def load_keypoints(json_path: str):
    """
    Load saved keypoints and return a torch.Tensor of shape (N, 3) → [x, y, visibility=1],
    compatible with utils.utils_correspondence.preprocess_kps_pad().

    Example
    -------
    >>> kps = load_keypoints('data/images/antelope_kps.json')
    >>> # kps shape: (N, 3)  — x, y in original image pixel space, visibility=1
    """
    import torch
    with open(json_path) as f:
        data = json.load(f)
    xyv = data.get('keypoints_xyv')
    if xyv is None:
        # fall back: reconstruct from 'kps' dict
        kps_dict = data['kps']
        n = len(kps_dict)
        xyv = []
        for i in range(n):
            pt = kps_dict.get(str(i))
            if pt is None:
                xyv.append([0.0, 0.0, 0.0])
            else:
                xyv.append([pt[0], pt[1], 1.0])
    return torch.tensor(xyv, dtype=torch.float32)


# ─── CLI ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Interactive keypoint selector for GeoAware-SC template images.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('image_path', help='Path to the template image (jpg / png)')
    args = parser.parse_args()

    if not os.path.isfile(args.image_path):
        sys.exit(f'[select.py] ERROR: image not found: {args.image_path}')

    print('[kp_select] Point selection order:')
    print('  pt 0 -> Grasp  (grasping contact point)')
    print('  pt 1 -> Func   (functional reference point)')
    print('Select 2 points, then press Enter or Q to save.\n')

    selector = KeypointSelector(image_path=args.image_path)
    selector.run()


if __name__ == '__main__':
    main()

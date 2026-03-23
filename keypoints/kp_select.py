"""
Topology keypoint selector for GeoAware-SC grasping pipeline.

Usage
-----
    # 2D only
    python keypoints/kp_select.py <image_path>

    # With RGBD → 3D back-projection saved into topology
    python keypoints/kp_select.py <image_path> \\
        --depth <depth.png> --intr <cam_intr.npy> \\
        [--depth-scale 1000] [--depth-max 3.0]

Output  <image_stem>_topo.npy + _topo.png
------
    nodes (N,2) int32, node_parts (N,), part_names {id:str},
    adj_matrix (N,N) int32 (0=none, 1=intra, 2=inter)
    + if depth provided: nodes_3d (N,3) float64, camera_intrinsics dict

Controls — SELECT mode: LClick add, RClick undo, 0-9 Part, Tab→EDGE
Controls — EDGE mode:   LClick 2 pts to connect, RClick undo, Tab→SELECT
Enter/Q save | N name-parts & save | Esc cancel | Scroll zoom
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

# ─── colour palette per Part id (cycles after 20) ───────────────────────────
_PART_COLOURS = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
    '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3',
    '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#ffe119',
]

_EDGE_COLOUR_INTRA = '#ffffff'
_EDGE_COLOUR_INTER = '#ff6600'

_SNAP_RADIUS = 15  # pixels — max distance to "click on" a point in EDGE mode


def _part_colour(part_id: int) -> str:
    return _PART_COLOURS[part_id % len(_PART_COLOURS)]


# ─── depth helpers (inlined from kp_3d.py to avoid circular imports) ─────────

def _load_depth(path: str, depth_scale: float) -> np.ndarray:
    """Return depth image as float32 array in **metres**."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        d = np.load(path).astype(np.float32)
    elif ext == '.exr':
        import cv2
        d = cv2.imread(path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
    else:
        d = np.array(Image.open(path)).astype(np.float32)
    return d / depth_scale


def _backproject(u: int, v: int, depth_m: float,
                 fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Pixel (u=col, v=row) + metric depth → 3-D camera frame [X, Y, Z]."""
    X = (u - cx) * depth_m / fx
    Y = (v - cy) * depth_m / fy
    Z = depth_m
    return np.array([X, Y, Z], dtype=np.float64)


class TopologySelector:
    """Interactive topology builder backed by Matplotlib."""

    def __init__(self, image_path: str, *,
                 depth_path: str | None = None,
                 intr_path: str | None = None,
                 fx: float = 0.0, fy: float = 0.0,
                 cx: float = 0.0, cy: float = 0.0,
                 depth_scale: float = 1000.0,
                 depth_max: float = 3.0):
        self.image_path = os.path.abspath(image_path)
        self.img = Image.open(self.image_path).convert('RGB')
        self.img_w, self.img_h = self.img.size

        # ── depth / intrinsics (optional) ────────────────────────────────────
        self.depth_m: np.ndarray | None = None
        self.depth_max = depth_max
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.depth_scale = depth_scale

        if depth_path is not None:
            self.depth_m = _load_depth(depth_path, depth_scale)
            print(f'[topo] Depth loaded: {self.depth_m.shape}  '
                  f'range [{self.depth_m.min():.3f}, {self.depth_m.max():.3f}] m')

        if intr_path is not None:
            K = np.load(intr_path)
            assert K.shape == (3, 3), f'Expected 3×3 intrinsic matrix, got {K.shape}'
            self.fx, self.fy = float(K[0, 0]), float(K[1, 1])
            self.cx, self.cy = float(K[0, 2]), float(K[1, 2])
            print(f'[topo] Intrinsics: fx={self.fx:.1f} fy={self.fy:.1f} '
                  f'cx={self.cx:.1f} cy={self.cy:.1f}')

        # ── state ────────────────────────────────────────────────────────────
        self.mode: str = 'SELECT'              # 'SELECT' or 'EDGE'
        self.current_part: int = 0             # active Part for new points

        self.points: list[tuple[int, int]] = []   # (x, y) per node
        self.node_parts: list[int] = []            # parallel list — part id per node

        self.edges: list[tuple[int, int]] = []     # (node_i, node_j) pairs
        self._edge_first: int | None = None        # first endpoint while building an edge

        # preset Part names: part_0 … part_9 (user can rename used ones on save)
        self.part_names: dict[int, str] = {i: f'part_{i}' for i in range(10)}

        self._saved = False
        self._build_ui()

    # ── UI setup ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.canvas.manager.set_window_title(
            f'Topology selector — {os.path.basename(self.image_path)}'
        )
        self.ax.imshow(np.array(self.img))
        self._set_title()
        self.ax.axis('off')

        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.fig.canvas.mpl_connect('close_event', self._on_close)
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)

    # ── title bar ────────────────────────────────────────────────────────────

    def _set_title(self):
        pname = self.part_names.get(self.current_part, f'part_{self.current_part}')

        if self.mode == 'SELECT':
            line1 = (f'[SELECT]  Part={pname} (id {self.current_part})  |  '
                      f'{len(self.points)} nodes  {len(self.edges)} edges')
            line2 = ('Left: add point  |  Right/D: undo  |  U: clear all  |  '
                      '0-9: switch Part  |  Tab: edge mode  |  N: name & save  |  Enter/Q: quick save')
        else:
            pending = '  (click 2nd point)' if self._edge_first is not None else ''
            line1 = (f'[EDGE]  {len(self.points)} nodes  {len(self.edges)} edges'
                      f'{pending}')
            line2 = ('Left: select point pair  |  Right/D: undo edge  |  '
                      'Tab: select mode  |  N: name & save  |  Enter/Q: quick save')

        self.ax.set_title(f'{line1}\n{line2}', fontsize=9)

    # ── event handlers ────────────────────────────────────────────────────────

    def _on_click(self, event):
        if event.inaxes != self.ax:
            return

        if self.mode == 'SELECT':
            self._on_click_select(event)
        else:
            self._on_click_edge(event)

    def _on_click_select(self, event):
        if event.button == 1:                    # left click → add point
            x, y = int(round(event.xdata)), int(round(event.ydata))
            self.points.append((x, y))
            self.node_parts.append(self.current_part)
            self._refresh()
        elif event.button == 3:                  # right click → remove last point
            self._undo_last_point()

    def _on_click_edge(self, event):
        if event.button == 1:                    # left click → pick point
            hit = self._find_nearest(event.xdata, event.ydata)
            if hit is None:
                return
            if self._edge_first is None:
                self._edge_first = hit
                self._refresh()                  # highlight first point
            else:
                second = hit
                first = self._edge_first
                self._edge_first = None
                if first == second:
                    print('[topo] Self-loop ignored.')
                    self._refresh()
                    return
                # canonical order (smaller first) to ease duplicate check
                edge = (min(first, second), max(first, second))
                if edge in self.edges:
                    print(f'[topo] Edge {edge} already exists.')
                else:
                    self.edges.append(edge)
                self._refresh()
        elif event.button == 3:                  # right click → undo
            self._undo_edge()

    def _on_key(self, event):
        key = event.key

        # ── mode-independent keys ────────────────────────────────────────────
        if key in ('enter', 'q'):
            self._quick_save()
            return
        if key == 'n':
            self._named_save()
            return
        if key == 'escape':
            print('[topo] Cancelled — nothing saved.')
            plt.close(self.fig)
            return
        if key == 'tab':
            self._toggle_mode()
            return

        # ── SELECT mode keys ────────────────────────────────────────────────
        if self.mode == 'SELECT':
            if key in ('d', 'backspace'):
                self._undo_last_point()
            elif key == 'u':
                self.points.clear()
                self.node_parts.clear()
                self.edges.clear()
                self._edge_first = None
                self._refresh()
            elif key in [str(d) for d in range(10)]:
                self.current_part = int(key)
                self._refresh()                  # update title
            return

        # ── EDGE mode keys ───────────────────────────────────────────────────
        if self.mode == 'EDGE':
            if key in ('d', 'backspace'):
                self._undo_edge()
            return

    def _on_scroll(self, event):
        if event.inaxes != self.ax:
            return
        factor = 0.8 if event.button == 'up' else 1.25
        cx, cy = event.xdata, event.ydata
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self.ax.set_xlim([cx + (x - cx) * factor for x in xlim])
        self.ax.set_ylim([cy + (y - cy) * factor for y in ylim])
        self.fig.canvas.draw_idle()

    def _on_close(self, event):
        if not self._saved and self.points:
            # Auto-save with default names (no input() to avoid readline re-entry)
            print('\n[topo] Window closed — auto-saving with default part names.')
            self._do_save()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _toggle_mode(self):
        if self.mode == 'SELECT':
            self.mode = 'EDGE'
            self._edge_first = None
        else:
            self.mode = 'SELECT'
            self._edge_first = None
        self._refresh()

    def _find_nearest(self, mx: float, my: float) -> int | None:
        """Return the index of the point closest to (mx, my), or None if too far."""
        if not self.points:
            return None
        pts = np.array(self.points, dtype=np.float64)
        dists = np.hypot(pts[:, 0] - mx, pts[:, 1] - my)
        idx = int(np.argmin(dists))
        if dists[idx] <= _SNAP_RADIUS:
            return idx
        return None

    def _undo_last_point(self):
        if not self.points:
            return
        removed_idx = len(self.points) - 1
        self.points.pop()
        self.node_parts.pop()
        # remove any edges referencing the removed node
        self.edges = [(a, b) for a, b in self.edges if a != removed_idx and b != removed_idx]
        # cancel pending edge if it references removed node
        if self._edge_first == removed_idx:
            self._edge_first = None
        self._refresh()

    def _undo_edge(self):
        if self._edge_first is not None:
            self._edge_first = None           # just cancel pending
        elif self.edges:
            self.edges.pop()
        self._refresh()

    # ── drawing ──────────────────────────────────────────────────────────────

    def _refresh(self):
        self.ax.cla()
        self.ax.imshow(np.array(self.img))
        self._set_title()
        self.ax.axis('off')

        # draw edges first (below nodes)
        for (a, b) in self.edges:
            xa, ya = self.points[a]
            xb, yb = self.points[b]
            is_inter = self.node_parts[a] != self.node_parts[b]
            style = '--' if is_inter else '-'
            colour = _EDGE_COLOUR_INTER if is_inter else _EDGE_COLOUR_INTRA
            self.ax.plot([xa, xb], [ya, yb], style, color=colour,
                         linewidth=1.5, alpha=0.8, zorder=3)

        # draw nodes
        for idx, (x, y) in enumerate(self.points):
            pid = self.node_parts[idx]
            c = _part_colour(pid)
            marker_size = 140
            edge_col = 'yellow' if (self._edge_first == idx) else 'white'
            edge_w = 2.5 if (self._edge_first == idx) else 1.2
            self.ax.scatter(x, y, s=marker_size, c=c, zorder=5,
                            linewidths=edge_w, edgecolors=edge_col)
            pname = self.part_names.get(pid, f'part_{pid}')
            self.ax.annotate(
                f'{idx}:{pname}',
                (x, y),
                xytext=(12, 0),
                textcoords='offset points',
                fontsize=7,
                color='white',
                fontweight='bold',
                va='center',
                bbox=dict(boxstyle='round,pad=0.15', fc=c, alpha=0.75, ec='none'),
            )

        # legend — one entry per used Part
        used_parts = sorted(set(self.node_parts))
        if used_parts:
            patches = []
            for pid in used_parts:
                count = self.node_parts.count(pid)
                pname = self.part_names.get(pid, f'part_{pid}')
                patches.append(mpatches.Patch(
                    color=_part_colour(pid),
                    label=f'{pname} ({count} pts)',
                ))
            self.ax.legend(
                handles=patches, loc='upper right', fontsize=7,
                framealpha=0.8, title=f'{len(self.points)} nodes  {len(self.edges)} edges',
                title_fontsize=8,
            )

        self.fig.canvas.draw_idle()

    # ── save ─────────────────────────────────────────────────────────────────

    def _quick_save(self):
        """Save immediately with current (default) part names."""
        if not self.points:
            print('[topo] No points selected — nothing saved.')
        else:
            self._do_save()
        self._saved = True          # prevent _on_close from re-saving
        plt.close(self.fig)

    def _named_save(self):
        """Prompt for part names in terminal, then save."""
        if not self.points:
            print('[topo] No points selected — nothing saved.')
            self._saved = True
            plt.close(self.fig)
            return

        # prompt user to rename used Parts
        used_parts = sorted(set(self.node_parts))
        print('\n── Part naming ─────────────────────────────────────')
        print('Press Enter to keep the default name, or type a new name.\n')
        for pid in used_parts:
            count = self.node_parts.count(pid)
            default = self.part_names.get(pid, f'part_{pid}')
            answer = input(f'  Part {pid} ({count} pts) [{default}]: ').strip()
            if answer:
                self.part_names[pid] = answer
        print()

        self._do_save()
        self._saved = True          # prevent _on_close from re-saving
        plt.close(self.fig)

    def _do_save(self):
        if not self.points:
            return

        stem = os.path.splitext(self.image_path)[0]
        npy_path = f'{stem}_topo.npy'
        png_path = f'{stem}_topo.png'

        N = len(self.points)
        nodes = np.array(self.points, dtype=np.int32)                  # (N, 2)
        node_parts = np.array(self.node_parts, dtype=np.int32)         # (N,)

        # build adjacency matrix: 0=no edge, 1=intra-part, 2=inter-part
        adj_matrix = np.zeros((N, N), dtype=np.int32)
        for (a, b) in self.edges:
            val = 1 if node_parts[a] == node_parts[b] else 2
            adj_matrix[a, b] = val
            adj_matrix[b, a] = val

        # only store Parts that are actually used
        used_parts = sorted(set(self.node_parts))
        part_names = {pid: self.part_names.get(pid, f'part_{pid}') for pid in used_parts}

        data = {
            'image_path':   self.image_path,
            'image_width':  self.img_w,
            'image_height': self.img_h,
            'nodes':        nodes,
            'node_parts':   node_parts,
            'part_names':   part_names,
            'adj_matrix':   adj_matrix,
        }

        # ── optional depth back-projection → nodes_3d ────────────────────────
        if self.depth_m is not None and self.fx > 0 and self.fy > 0:
            h_d, w_d = self.depth_m.shape
            nodes_3d = np.full((N, 3), np.nan, dtype=np.float64)
            for i in range(N):
                u, v = int(nodes[i, 0]), int(nodes[i, 1])
                u_c = max(0, min(w_d - 1, u))
                v_c = max(0, min(h_d - 1, v))
                d = float(self.depth_m[v_c, u_c])
                if 0 < d < self.depth_max:
                    nodes_3d[i] = _backproject(u_c, v_c, d, self.fx, self.fy,
                                               self.cx, self.cy)
            data['nodes_3d'] = nodes_3d
            data['camera_intrinsics'] = {
                'fx': self.fx, 'fy': self.fy,
                'cx': self.cx, 'cy': self.cy,
                'depth_scale': self.depth_scale,
            }
            valid = int(np.sum(~np.isnan(nodes_3d[:, 0])))
            print(f'[topo] 3D back-projection: {valid}/{N} valid points')

        np.save(npy_path, data, allow_pickle=True)
        print(f'[topo] Saved topology → {npy_path}')

        # ── save visualisation ───────────────────────────────────────────────
        self._save_png(png_path, data)
        print(f'[topo] Saved PNG      → {png_path}')

        self._saved = True

    def _save_png(self, png_path: str, data: dict):
        fig, ax = plt.subplots(
            figsize=(max(6, self.img_w / 100), max(5, self.img_h / 100)),
            dpi=100,
        )
        ax.imshow(np.array(self.img))
        ax.axis('off')

        nodes = data['nodes']
        nparts = data['node_parts']
        adj = data['adj_matrix']
        part_names = data['part_names']

        # edges from adjacency matrix (upper triangle to avoid duplicates)
        N = len(nodes)
        for a in range(N):
            for b in range(a + 1, N):
                val = adj[a, b]
                if val == 0:
                    continue
                xa, ya = nodes[a]
                xb, yb = nodes[b]
                is_inter = (val == 2)
                style = '--' if is_inter else '-'
                colour = _EDGE_COLOUR_INTER if is_inter else _EDGE_COLOUR_INTRA
                ax.plot([xa, xb], [ya, yb], style, color=colour,
                        linewidth=1.5, alpha=0.8, zorder=3)

        # nodes
        for idx in range(len(nodes)):
            x, y = nodes[idx]
            pid = nparts[idx]
            c = _part_colour(pid)
            ax.scatter(x, y, s=120, c=c, zorder=5, linewidths=1.2, edgecolors='white')
            pname = part_names.get(pid, f'part_{pid}')
            ax.annotate(
                f'{idx}:{pname}',
                (x, y),
                xytext=(12, 0),
                textcoords='offset points',
                fontsize=7, color='white', fontweight='bold',
                va='center',
                bbox=dict(boxstyle='round,pad=0.15', fc=c, alpha=0.75, ec='none'),
            )

        # legend
        n_intra = int(np.sum(adj == 1)) // 2  # symmetric → divide by 2
        n_inter = int(np.sum(adj == 2)) // 2
        n_edges = n_intra + n_inter
        used = sorted(set(int(p) for p in nparts))
        if used:
            patches = [mpatches.Patch(
                color=_part_colour(pid),
                label=f'{part_names.get(pid, f"part_{pid}")} ({int(np.sum(nparts == pid))} pts)',
            ) for pid in used]
            ax.legend(
                handles=patches, loc='upper right', fontsize=7, framealpha=0.8,
                title=f'{N} nodes  {n_edges} edges (intra {n_intra} / inter {n_inter})',
                title_fontsize=7,
            )

        fig.tight_layout(pad=0)
        fig.savefig(png_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

    # ── entry ─────────────────────────────────────────────────────────────────

    def run(self):
        plt.show()


# ─── load helper ─────────────────────────────────────────────────────────────

def load_topology(npy_path: str) -> dict:
    """
    Load a saved topology .npy file and return the dict.

    Keys: image_path, image_width, image_height,
          nodes (N,2), node_parts (N,), part_names {id:str},
          adj_matrix (N,N) — 0=no edge, 1=intra-part, 2=inter-part.
    """
    data = np.load(npy_path, allow_pickle=True).item()
    return data


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Interactive topology keypoint selector.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('image_path', help='Path to the template image (jpg / png)')
    # optional RGBD depth integration
    parser.add_argument('--depth', default=None,
                        help='Depth image (16-bit mm PNG / .npy / .exr). '
                             'If given, 3D coords are saved alongside 2D nodes.')
    parser.add_argument('--intr', default=None,
                        help='Camera intrinsic matrix (.npy 3×3). '
                             'Overrides --fx/--fy/--cx/--cy.')
    parser.add_argument('--fx', type=float, default=0.0, help='Focal length x')
    parser.add_argument('--fy', type=float, default=0.0, help='Focal length y')
    parser.add_argument('--cx', type=float, default=0.0, help='Principal point x')
    parser.add_argument('--cy', type=float, default=0.0, help='Principal point y')
    parser.add_argument('--depth-scale', type=float, default=1000.0,
                        help='raw / scale = metres (default 1000 for mm PNG)')
    parser.add_argument('--depth-max', type=float, default=3.0,
                        help='Discard depth beyond this (metres)')
    args = parser.parse_args()

    if not os.path.isfile(args.image_path):
        sys.exit(f'[topo] ERROR: image not found: {args.image_path}')

    print('[topo] Topology selector')
    print('  SELECT mode: left-click to add points, digit keys 0-9 to switch Part')
    print('  EDGE   mode: Tab to enter, click two points to connect')
    print('  Enter/Q: quick save  |  N: name parts & save  |  Esc: cancel\n')

    selector = TopologySelector(
        image_path=args.image_path,
        depth_path=args.depth,
        intr_path=args.intr,
        fx=args.fx, fy=args.fy,
        cx=args.cx, cy=args.cy,
        depth_scale=args.depth_scale,
        depth_max=args.depth_max,
    )
    selector.run()


if __name__ == '__main__':
    main()

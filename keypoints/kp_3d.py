"""
keypoints/kp_3d.py — Lift 2-D keypoints to 3-D using an RGBD image pair.

Given:
  * a keypoint JSON file (color_kps.json format, produced by kp_select.py)
  * an RGB image
  * a depth image  (16-bit PNG in millimetres, .npy, or 32-bit .exr in metres)

This script:
  1. Reconstructs a dense 3-D point cloud from the RGBD pair.
  2. Back-projects each 2-D keypoint through the depth map → (X, Y, Z) in
     camera frame (metres).
  3. Saves the 3-D coordinates back into the same JSON file under two new
     keys: "kps_3d" and "keypoints_xyzv".
  4. Opens an interactive Open3D viewer showing the coloured point cloud
     with the keypoints rendered as labelled spheres.

Usage
-----
    # Recommended: pass intrinsics file (3×3 .npy matrix)
    python keypoints/kp_3d.py \\
        --kps   path/to/color_kps.json \\
        --rgb   path/to/color.png \\
        --depth path/to/depth.png \\
        --intr  path/to/cam_intr.npy \\
        [--depth-scale 1000.0] \\
        [--depth-max   3.0] \\
        [--no-save] \\
        [--no-vis]

    # Or: specify intrinsics manually (fallback if no --intr)
    python keypoints/kp_3d.py \\
        --kps   path/to/color_kps.json \\
        --rgb   path/to/color.png \\
        --depth path/to/depth.png \\
        --fx 636.3 --fy 636.3 --cx 640 --cy 360

Camera intrinsics (--intr / --fx --fy --cx --cy)
------------------------------------------------
  --intr  path/to/cam_intr.npy   (preferred)
    Loads a 3×3 NumPy intrinsic matrix:
      [[fx,  0, cx],
       [ 0, fy, cy],
       [ 0,  0,  1]]
    This overrides --fx/--fy/--cx/--cy if both are given.

  --fx/--fy/--cx/--cy
    Manual fallback. Defaults are 0 (triggers a warning if no --intr).

Depth encoding (--depth-scale)
-------------------------------
  Raw pixel value divided by depth-scale = depth in metres.
    1000   → 16-bit PNG where each step is 1 mm  (default)
    1      → float32 .npy / .exr already in metres

JSON output (written in-place to the --kps file)
------------------------------------------------
  {
    ...original keys...
    "kps_3d": {
      "0": [X, Y, Z],           # metres, camera frame — null if no valid depth
      "1": [X, Y, Z],
      ...
    },
    "keypoints_xyzv": [
      [X, Y, Z, visibility],    # visibility mirrors the original keypoints_xyv
      ...
    ],
    "camera_intrinsics": {
      "fx": ..., "fy": ..., "cx": ..., "cy": ...,
      "depth_scale": ...
    }
  }

Viewer controls
---------------
  Left-drag    → orbit / rotate
  Middle-drag  → pan
  Scroll       → zoom
  Q / Esc      → close
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

try:
    import open3d as o3d
    _OPEN3D = True
except ImportError:
    _OPEN3D = False


# ──────────────────────────────────────────────────────────────────────────────
# Colour palette for keypoint spheres (cycles if >12 points)
# ──────────────────────────────────────────────────────────────────────────────
_SPHERE_COLOURS = [
    [0.90, 0.12, 0.18],  # 0 – red    (Grasp)
    [0.13, 0.70, 0.30],  # 1 – green  (Func)
    [1.00, 0.75, 0.00],  # 2 – yellow
    [0.15, 0.38, 0.88],  # 3 – blue
    [0.95, 0.42, 0.08],  # 4 – orange
    [0.55, 0.09, 0.72],  # 5 – purple
    [0.20, 0.80, 0.80],  # 6 – cyan
    [0.85, 0.20, 0.75],  # 7 – pink
    [0.60, 0.40, 0.20],  # 8 – brown
    [0.40, 0.80, 0.20],  # 9 – lime
    [0.80, 0.80, 0.80],  # 10 – silver
    [0.60, 0.60, 1.00],  # 11 – lavender
]


def _sphere_colour(idx: int) -> list:
    return _SPHERE_COLOURS[idx % len(_SPHERE_COLOURS)]


# ──────────────────────────────────────────────────────────────────────────────
# Depth / geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_depth(path: str, depth_scale: float) -> np.ndarray:
    """Return depth image as float32 array in **metres**."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        d = np.load(path).astype(np.float32)
    elif ext == ".exr":
        try:
            import cv2
            d = cv2.imread(path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
        except ImportError:
            raise RuntimeError("Reading .exr files requires OpenCV (cv2).")
    else:
        # 16-bit PNG (most common ROS / RealSense format)
        d = np.array(Image.open(path)).astype(np.float32)
    return d / depth_scale


def backproject(u: int, v: int, depth_m: float,
                fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Back-project a pixel (u=col, v=row) + metric depth to 3-D camera frame."""
    X = (u - cx) * depth_m / fx
    Y = (v - cy) * depth_m / fy
    Z = depth_m
    return np.array([X, Y, Z], dtype=np.float64)


def build_pointcloud(rgb: np.ndarray, depth_m: np.ndarray,
                     fx: float, fy: float, cx: float, cy: float,
                     depth_max: float) -> "o3d.geometry.PointCloud":
    """Build a coloured Open3D PointCloud from an RGB + metric-depth pair."""
    h, w = depth_m.shape
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))

    mask = (depth_m > 0) & (depth_m < depth_max)
    Z = depth_m[mask]
    X = (uu[mask] - cx) * Z / fx
    Y = (vv[mask] - cy) * Z / fy

    pts   = np.stack([X, Y, Z], axis=-1)           # (N, 3)  metres
    cols  = rgb.astype(np.float64)[mask] / 255.0   # (N, 3)  [0,1]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    return pcd


def make_sphere(centre: np.ndarray, radius: float,
                colour: list) -> "o3d.geometry.TriangleMesh":
    """Create a small solid sphere centred at *centre* with uniform *colour*."""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=20)
    sphere.translate(centre)
    sphere.paint_uniform_color(colour)
    sphere.compute_vertex_normals()
    return sphere


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lift 2-D keypoints to 3-D from an RGBD image pair.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--kps",   required=True,
                        help="Path to keypoint JSON  (color_kps.json format)")
    parser.add_argument("--rgb",   required=True,
                        help="Path to RGB image  (PNG / JPG)")
    parser.add_argument("--depth", required=True,
                        help="Path to depth image  (16-bit PNG in mm, .npy, or .exr)")
    # camera intrinsics ──────────────────────────────────────────────────────
    parser.add_argument("--intr", type=str, default=None,
                        help="Path to 3×3 intrinsic matrix (.npy)  "
                             "— overrides --fx/fy/cx/cy if provided")
    parser.add_argument("--fx",  type=float, default=0.0,
                        help="Focal length x  (pixels; ignored if --intr given)")
    parser.add_argument("--fy",  type=float, default=0.0,
                        help="Focal length y  (pixels; ignored if --intr given)")
    parser.add_argument("--cx",  type=float, default=0.0,
                        help="Principal point x  (pixels; ignored if --intr given)")
    parser.add_argument("--cy",  type=float, default=0.0,
                        help="Principal point y  (pixels; ignored if --intr given)")
    # depth encoding ─────────────────────────────────────────────────────────
    parser.add_argument("--depth-scale", type=float, default=1000.0,
                        help="Divide raw depth by this to get metres  "
                             "(default 1000 → 16-bit-mm PNG)")
    parser.add_argument("--depth-max",   type=float, default=3.0,
                        help="Clip depth beyond this distance (metres) for visualisation")
    # flags ──────────────────────────────────────────────────────────────────
    parser.add_argument("--no-save", action="store_true",
                        help="Skip writing 3-D coords back to the JSON file")
    parser.add_argument("--no-vis",  action="store_true",
                        help="Skip Open3D visualisation")
    args = parser.parse_args()

    # ── resolve camera intrinsics ─────────────────────────────────────────────
    if args.intr is not None:
        K = np.load(args.intr)
        assert K.shape == (3, 3), f"Expected 3×3 intrinsic matrix, got {K.shape}"
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        print(f"Loaded intrinsics from {args.intr}")
        print(f"  fx={fx:.2f}  fy={fy:.2f}  cx={cx:.2f}  cy={cy:.2f}")
    else:
        fx, fy, cx, cy = args.fx, args.fy, args.cx, args.cy
        if fx == 0 or fy == 0:
            print("[warn] No --intr file and --fx/--fy are 0.  "
                  "Back-projection will be incorrect!\n"
                  "       Pass --intr path/to/cam_intr.npy  or  "
                  "--fx/--fy/--cx/--cy manually.")

    # ── load inputs ──────────────────────────────────────────────────────────
    print(f"Loading keypoints : {args.kps}")
    with open(args.kps, "r") as f:
        kps_data = json.load(f)

    print(f"Loading RGB       : {args.rgb}")
    rgb_img = np.array(Image.open(args.rgb).convert("RGB"))

    print(f"Loading depth     : {args.depth}  (scale={args.depth_scale})")
    depth_m = load_depth(args.depth, args.depth_scale)
    print(f"  depth range: [{depth_m.min():.4f}, {depth_m.max():.4f}] m")

    h_rgb, w_rgb = rgb_img.shape[:2]
    h_d,   w_d   = depth_m.shape
    if h_rgb != h_d or w_rgb != w_d:
        print(f"[warn] RGB ({w_rgb}×{h_rgb}) and depth ({w_d}×{h_d}) "
              "dimensions differ — proceeding with depth dimensions")

    # ── back-project each keypoint to 3-D ────────────────────────────────────
    kps_2d      = kps_data["kps"]                # {"0": [u,v], "1": [u,v], ...}
    names_list  = kps_data.get("keypoint_names") or []
    vis_flags   = {                              # from keypoints_xyv if present
        str(i): v
        for i, (_, _, v) in enumerate(kps_data.get("keypoints_xyv", []))
    }

    kps_3d      : dict[str, list | None] = {}
    kps_xyzv    : list = []

    print("\n── Keypoint back-projection ─────────────────────────────────────")
    for key, (u, v) in kps_2d.items():
        u_i = max(0, min(w_d - 1, int(round(u))))
        v_i = max(0, min(h_d - 1, int(round(v))))
        d   = float(depth_m[v_i, u_i])
        vis = vis_flags.get(key, 1)

        idx  = int(key)
        name = names_list[idx] if idx < len(names_list) else f"kp{key}"

        if d <= 0 or d >= args.depth_max:
            print(f"  [{key}] {name:<12s}  pixel=({u_i},{v_i})  "
                  f"depth={d:.4f} m  → INVALID (no 3-D point)")
            kps_3d[key] = None
            kps_xyzv.append([None, None, None, 0])
            continue

        xyz = backproject(u_i, v_i, d, fx, fy, cx, cy)
        kps_3d[key] = xyz.tolist()
        kps_xyzv.append([float(xyz[0]), float(xyz[1]), float(xyz[2]), int(vis)])
        print(f"  [{key}] {name:<12s}  pixel=({u_i},{v_i})  depth={d:.4f} m"
              f"  → X={xyz[0]:+.4f}  Y={xyz[1]:+.4f}  Z={xyz[2]:+.4f}")

    # ── save back to JSON ─────────────────────────────────────────────────────
    if not args.no_save:
        kps_data["kps_3d"]          = kps_3d
        kps_data["keypoints_xyzv"]  = kps_xyzv
        kps_data["camera_intrinsics"] = {
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "depth_scale": args.depth_scale,
        }
        with open(args.kps, "w") as f:
            json.dump(kps_data, f, indent=2)
        print(f"\n[saved] 3-D keypoints written to  {args.kps}")

    # ── 3-D visualisation ────────────────────────────────────────────────────
    if args.no_vis:
        return

    if not _OPEN3D:
        print("\n[error] open3d is not installed.\n"
              "        Install it with:  pip install open3d\n"
              "        Then re-run this script.")
        sys.exit(1)

    print("\nBuilding point cloud …")
    pcd = build_pointcloud(rgb_img, depth_m, fx, fy, cx, cy, args.depth_max)
    print(f"  raw points : {len(pcd.points):,}")

    # voxel down-sample for smooth interactive performance
    voxel = max(0.003, args.depth_max * 0.002)
    pcd   = pcd.voxel_down_sample(voxel_size=voxel)
    print(f"  after voxel downsample (size={voxel:.3f} m) : {len(pcd.points):,}")

    # determine sphere radius relative to scene depth
    valid_xyz = [v for v in kps_3d.values() if v is not None]
    if valid_xyz:
        mean_z    = np.mean([v[2] for v in valid_xyz])
        sphere_r  = max(0.005, mean_z * 0.012)
    else:
        sphere_r  = 0.01

    # build geometries list
    geoms: list = [pcd]

    print("\n── Keypoint spheres ─────────────────────────────────────────────")
    for key, xyz in kps_3d.items():
        if xyz is None:
            continue
        idx    = int(key)
        colour = _sphere_colour(idx)
        name   = names_list[idx] if idx < len(names_list) else f"kp{key}"
        sphere = make_sphere(np.array(xyz), sphere_r, colour)
        geoms.append(sphere)
        print(f"  sphere [{key}] {name:<12s}  colour={colour}  "
              f"pos=[{xyz[0]:+.4f}, {xyz[1]:+.4f}, {xyz[2]:+.4f}]")

    # coordinate frame at origin
    frame_size = sphere_r * 8
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=frame_size, origin=[0, 0, 0])
    geoms.append(frame)

    print("\n── Open3D viewer ───────────────────────────────────────────────")
    print("  Left-drag    → orbit / rotate")
    print("  Middle-drag  → pan")
    print("  Scroll       → zoom")
    print("  Q / Esc      → close\n")

    o3d.visualization.draw_geometries(
        geoms,
        window_name="3-D Keypoints",
        width=1280,
        height=720,
        point_show_normal=False,
    )


if __name__ == "__main__":
    main()

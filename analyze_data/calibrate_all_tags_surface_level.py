#!/usr/bin/env python3
"""All-tags homography fitted on the table SURFACE instead of the rail tops.

The AprilTags sit on the rails, --tag-height-mm above the playing surface, but the puck
is on the surface. A homography fitted to the tags is exact only on the rail plane: a
surface point is seen along the same camera ray as a rail-plane point closer to the
camera's nadir, so the rail-plane fit pulls the puck toward the nadir, more the farther
it is from it.

For each camera this script:
  1. detects every tag center exactly like calibrate_all_tags.py (same tag positions:
     TCP-measured tags 0-3, the rest extrapolated along each rail);
  2. recovers the camera pose from those rail-plane points with cv2.solvePnP, assuming an
     undistorted pinhole with the principal point at the image center. The focal length is
     --focal-px, or (default) the one that best reprojects the tags (PlayStation Eye, wide
     75 deg setting: f ~ 520 px at 640x480). Check the printed camera height with a tape;
  3. slides each tag along its camera ray down to the surface:
         P_surface = P_rail + (P_rail - C_xy) * h / H
     with C_xy the camera nadir, H its height above the rails and h = --tag-height-mm;
  4. fits the homography (least squares, x3 upscaled px -> robot mm) to those surface
     positions and writes Mimg_<name>_only_all_tags_predicted_surface_level.npy.

    python analyze_data/calibrate_all_tags_surface_level.py --camera-indices 1 2 --names overhead back
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from calibrate_all_tags import (
    ROBOT_TO_WARPED,
    capture_centers,
    fit_mrob,
    spans_both_rails,
    tag_positions,
)
from detect_apriltags import make_passes

_REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_WH = (640, 480)


def camera_pose(px, rob_mm, focal_px):
    """Camera center (robot mm; z = height above the tag plane) from rail-plane points."""
    K = np.array([[focal_px, 0, (IMAGE_WH[0] - 1) / 2], [0, focal_px, (IMAGE_WH[1] - 1) / 2], [0, 0, 1.0]])
    obj = np.column_stack([rob_mm, np.zeros(len(rob_mm))]).astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, np.asarray(px, np.float64), K, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise SystemExit("solvePnP failed")
    R, _ = cv2.Rodrigues(rvec)
    center = (-R.T @ tvec).ravel()
    reproj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
    reproj_px = np.linalg.norm(reproj.reshape(-1, 2) - px, axis=1)
    return center, reproj_px


def to_surface(rob_mm, center, tag_height_mm):
    height = abs(center[2])
    return rob_mm + (rob_mm - center[:2]) * (tag_height_mm / height)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera-indices", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--names", nargs="+", default=["overhead", "back"])
    ap.add_argument("--out-dir", default=str(_REPO_ROOT / "assets" / "real"))
    ap.add_argument("--tag-height-mm", type=float, default=31.75, help="tag (rail top) height above the playing surface")
    ap.add_argument("--focal-px", type=float, default=None,
                    help="pinhole focal length at 640x480; default: best tag reprojection in 300-900 px")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--min-frames", type=int, default=10)
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 1.5, 2.0, 2.5, 3.0])
    ap.add_argument("--max-spread", type=float, default=2.0)
    args = ap.parse_args()
    if len(args.names) != len(args.camera_indices):
        raise SystemExit("--names needs one name per camera index")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    positions = tag_positions()
    passes = make_passes(args.scales, multi=True)
    for index, name in zip(args.camera_indices, args.names):
        centers, n_frames, frame, _ = capture_centers(index, passes, args.max_spread, args.frames, args.min_frames)
        if frame is None:
            raise SystemExit(f"{name} (camera {index}): no frames")
        ids = [t for t, c in centers.items() if t in positions and c["used"]]
        if len(ids) < 4 or not spans_both_rails(ids):
            raise SystemExit(f"{name}: need >= 4 tags covering both rails, have {ids}")
        px = np.array([centers[t]["xy"] for t in ids])
        rail = np.array([positions[t][0] for t in ids], dtype=np.float64)

        focal_px = args.focal_px
        if focal_px is None:
            focal_px = min(np.arange(300.0, 900.0, 5.0), key=lambda f: np.median(camera_pose(px, rail, f)[1]))
        center, reproj_px = camera_pose(px, rail, focal_px)
        surface = to_surface(rail, center, args.tag_height_mm)
        Mrob = fit_mrob(px, surface)
        Mimg = ROBOT_TO_WARPED @ Mrob
        Mimg /= Mimg[2, 2]

        print(f"\n{name} (camera {index}), {n_frames} frames, {len(ids)} tags {ids}")
        print(f"  camera (f = {focal_px:.0f} px): nadir at robot ({center[0]:.0f}, {center[1]:.0f}) mm, "
              f"{abs(center[2]):.0f} mm above the rails; pose reprojection error "
              f"median {np.median(reproj_px):.2f} px, max {reproj_px.max():.2f} px")
        print(f"  {'tag':>5} {'rail x':>8} {'rail y':>8} {'surface x':>10} {'surface y':>10} {'shift':>7}  (mm)")
        for t, r, s in zip(ids, rail, surface):
            print(f"  {t:>5} {r[0]:8.1f} {r[1]:8.1f} {s[0]:10.1f} {s[1]:10.1f} {np.linalg.norm(s - r):7.1f}")
        path = out_dir / f"Mimg_{name}_only_all_tags_predicted_surface_level.npy"
        np.save(path, Mimg)
        print(f"  saved {path}")


if __name__ == "__main__":
    main()

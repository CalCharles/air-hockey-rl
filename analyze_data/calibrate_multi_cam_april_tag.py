#!/usr/bin/env python3
"""Multi-camera homography from the 4 AprilTags and the recorded TCP positions.

For each camera: detect tags 0-3, then compute Mimg / Mrob against the TCP positions
in calibrate_back_cam_april_tag.TCP_MM (same conventions as the single-camera
scripts: frames rotated 180, pixels x3, Mimg -> (x_mm + 2250, -y_mm + 500),
Mrob -> robot mm). Because every Mrob lands in the same robot frame, chaining them
gives a pixel -> pixel homography between cameras (table plane):

    H_<cam>_to_<ref> = inv(Mrob_<ref>) @ Mrob_<cam>      (x3 upscaled pixels)

It also writes a fused top-down view (all cameras warped with their Mimg and
averaged where they overlap).

    python analyze_data/calibrate_multi_cam_april_tag.py
    python analyze_data/calibrate_multi_cam_april_tag.py --camera-indices 1 2 --names overhead back --check

The first camera is the reference. Outputs in --out-dir (default assets/real/):
Mimg_<name>_cam_april_tag.npy, Mrob_<name>_cam_april_tag.npy,
H_<name>_to_<ref>_april_tag.npy, multi_cam_april_tag_fused.png, and a
multi_cam_april_tag.npz with everything.

--check: afterwards, put ONE red puck anywhere on the table. Every camera maps it to
robot (x, y); the spread between cameras is the calibration error away from the
tags. Move it around, especially toward the far end. q / Esc in the window to finish.
Keep the arm clear of the tags during capture.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from calibrate_back_cam_april_tag import (
    OFFSET_CONSTANTS,
    ORIGINAL_SIZE,
    TAG_IDS,
    TCP_MM,
    UPSCALE_CONSTANT,
    VISUAL_DOWNSCALE_CONSTANT,
    _make_detector,
    annotate_tags,
    capture_tag_centers,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
CANVAS_WH = tuple((ORIGINAL_SIZE * UPSCALE_CONSTANT).tolist())
PREVIEW_WH = tuple((ORIGINAL_SIZE * UPSCALE_CONSTANT // VISUAL_DOWNSCALE_CONSTANT).tolist())


def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {index} (is another process using it?)")
    for _ in range(10):  # let exposure settle
        cap.read()
    return cap


def calibrate_camera(index, name, detector, out_dir):
    """Detect the tags in one camera and return its homographies, frame and warp."""
    cap = open_camera(index)
    try:
        points_xy, seen, frame, found = capture_tag_centers(cap, detector)
    finally:
        cap.release()
    if frame is not None:
        cv2.imwrite(str(out_dir / f"{name}_cam_april_tag_capture.png"), annotate_tags(frame, found))
    if points_xy is None:
        raise SystemExit(f"{name} (camera {index}): need tags {list(TAG_IDS)} in at least 10 frames; saw {seen}.")

    warped_px = np.stack([TCP_MM[:, 0], -TCP_MM[:, 1]], axis=1) + OFFSET_CONSTANTS
    pts1 = np.ascontiguousarray(points_xy * UPSCALE_CONSTANT, dtype=np.float32)
    Mrob = cv2.getPerspectiveTransform(pts1, TCP_MM)
    Mimg = cv2.getPerspectiveTransform(pts1, np.ascontiguousarray(warped_px, dtype=np.float32))

    image = cv2.resize(frame, CANVAS_WH, interpolation=cv2.INTER_LINEAR)
    warped = cv2.warpPerspective(image, Mimg, CANVAS_WH)
    coverage = cv2.warpPerspective(np.ones(image.shape[:2], np.uint8), Mimg, CANVAS_WH, flags=cv2.INTER_NEAREST)

    print(f"\n{name} (camera {index}):")
    for tag_id, px, rob in zip(TAG_IDS, points_xy, TCP_MM):
        print(f"  tag {tag_id}: pixel (x, y) = ({px[0]:.2f}, {px[1]:.2f})  ->  robot (mm) = ({rob[0]:.1f}, {rob[1]:.1f})")
    return {
        "name": name,
        "index": index,
        "Mimg": Mimg,
        "Mrob": Mrob,
        "tag_px": points_xy,
        "warped": warped,
        "coverage": coverage,
    }


def fuse(cams):
    """Average the warped cameras where more than one covers a pixel."""
    total = np.zeros((CANVAS_WH[1], CANVAS_WH[0], 3), np.float32)
    weight = np.zeros((CANVAS_WH[1], CANVAS_WH[0]), np.float32)
    for cam in cams:
        w = cam["coverage"].astype(np.float32)
        total += cam["warped"].astype(np.float32) * w[..., None]
        weight += w
    fused = (total / np.maximum(weight, 1.0)[..., None]).astype(np.uint8)
    overlap = float(np.mean(weight >= 2)) / max(float(np.mean(weight >= 1)), 1e-6)
    return fused, overlap


def find_red_puck(frame_bgr, min_area=40.0, max_area=4000.0):
    """Centroid (x, y) of the most circular red blob, or None."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255)) | cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_circ = None, 0.0
    for c in contours:
        area = cv2.contourArea(c)
        perim = cv2.arcLength(c, True)
        if not (min_area <= area <= max_area) or perim <= 0:
            continue
        circ = 4 * np.pi * area / perim**2
        if circ > best_circ:
            m = cv2.moments(c)
            best, best_circ = (m["m10"] / m["m00"], m["m01"] / m["m00"]), circ
    return best


def live_check(cams):
    """Map one red puck through every camera's Mrob and report the spread (mm)."""
    caps = [open_camera(cam["index"]) for cam in cams]
    spreads = []
    print("\nLive check: place ONE red puck on the table. q / Esc in the window to finish.")
    try:
        while True:
            panels, robot_xy = [], []
            for cam, cap in zip(cams, caps):
                ok, frame = cap.read()
                if not ok or frame is None:
                    frame = np.zeros((480, 640, 3), np.uint8)
                frame = cv2.rotate(frame, cv2.ROTATE_180)
                px = find_red_puck(frame)
                label = f"{cam['name']}: no puck"
                if px is not None:
                    pt = np.float32([[px]]) * UPSCALE_CONSTANT
                    xy = cv2.perspectiveTransform(pt, cam["Mrob"])[0, 0]
                    robot_xy.append(xy)
                    cv2.circle(frame, (int(px[0]), int(px[1])), 10, (0, 255, 0), 2)
                    label = f"{cam['name']}: ({xy[0]:.0f}, {xy[1]:.0f}) mm"
                cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                panels.append(frame)
            if len(robot_xy) == len(cams) and len(cams) > 1:
                xy = np.array(robot_xy)
                spread = float(np.max(np.linalg.norm(xy[:, None] - xy[None], axis=-1)))
                spreads.append(spread)
                mean = xy.mean(axis=0)
                cv2.putText(
                    panels[0], f"spread {spread:.0f} mm at ({mean[0]:.0f}, {mean[1]:.0f})",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA,
                )
            cv2.imshow("multi-camera check (q to finish)", np.hstack(panels))
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        for cap in caps:
            cap.release()
        cv2.destroyAllWindows()
    if spreads:
        print(f"Cross-camera spread over {len(spreads)} frames: median {np.median(spreads):.1f} mm, max {np.max(spreads):.1f} mm")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera-indices", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--names", nargs="+", default=["overhead", "back"])
    ap.add_argument("--out-dir", default=str(_REPO_ROOT / "assets" / "real"))
    ap.add_argument("--check", action="store_true", help="live red-puck cross-camera check afterwards")
    args = ap.parse_args()
    if len(args.names) != len(args.camera_indices):
        raise SystemExit("--names needs one name per camera index")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # One camera at a time, so two USB cameras never have to stream together for the capture.
    detector = _make_detector()
    cams = [calibrate_camera(i, n, detector, out_dir) for i, n in zip(args.camera_indices, args.names)]
    ref = cams[0]

    bundle = {"tag_ids": np.array(TAG_IDS), "robot_points_mm": TCP_MM, "offset_constants": OFFSET_CONSTANTS}
    for cam in cams:
        np.save(out_dir / f"Mimg_{cam['name']}_cam_april_tag.npy", cam["Mimg"])
        np.save(out_dir / f"Mrob_{cam['name']}_cam_april_tag.npy", cam["Mrob"])
        bundle[f"Mimg_{cam['name']}"] = cam["Mimg"]
        bundle[f"Mrob_{cam['name']}"] = cam["Mrob"]
        bundle[f"tag_px_{cam['name']}"] = cam["tag_px"]
    for cam in cams[1:]:
        H = np.linalg.inv(ref["Mrob"]) @ cam["Mrob"]
        H /= H[2, 2]
        key = f"H_{cam['name']}_to_{ref['name']}"
        np.save(out_dir / f"{key}_april_tag.npy", H)
        bundle[key] = H
        print(f"\n{key} ({cam['name']} x3 pixel -> {ref['name']} x3 pixel, table plane):")
        print(np.array2string(H, precision=5, suppress_small=True))
    np.savez(out_dir / "multi_cam_april_tag.npz", **bundle)

    fused, overlap = fuse(cams)
    fused_preview = cv2.resize(fused, PREVIEW_WH, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(str(out_dir / "multi_cam_april_tag_fused.png"), fused_preview)
    for cam in cams:
        cv2.imwrite(
            str(out_dir / f"{cam['name']}_cam_april_tag_transformed.png"),
            cv2.resize(cam["warped"], PREVIEW_WH, interpolation=cv2.INTER_LINEAR),
        )
    print(f"\nFused view: cameras overlap on {100 * overlap:.0f}% of the covered top-down area.")
    print(f"Saved homographies and previews to {out_dir}")

    if args.check:
        live_check(cams)


if __name__ == "__main__":
    main()

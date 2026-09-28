#!/usr/bin/env python3
"""Camera homographies from ALL visible AprilTags: TCP-measured tags 0-3 plus tags
extrapolated along each rail.

Tag positions (robot frame, mm):
  * tags 0-3: TCP positions recorded with the paddle centered over each tag (TCP_MM).
  * left rail (robot's view), near -> far: 0, 1, 4, 6, 8, 10, equally spaced, so
    tag k-th along the rail = tag 0 + k * (tag 1 - tag 0).
  * right rail, near -> far: 2, 3, 5, 7, 9, 11, equally spaced with its own step (tag 3 - tag 2).
The step is taken per rail in both x and y, so any small angle of the rail relative to
the robot x axis is carried along. An error in a measured step grows with each step
(tag 10 inherits 5x the error of the 0 -> 1 step).

For each camera: tag centers are found with the multi-scale detector from
detect_apriltags.py (upscales x1..x3, stock + loosened settings, corners mapped back with
(x + 0.5) / s - 0.5, outlier passes dropped) and averaged over frames. Every tag with a
known position that is seen in enough frames goes into cv2.findHomography (least squares
over all points, not 4-point exact). Printed per tag:
  * fit residual: how far the homography puts the tag from its position (mm);
  * leave-one-out error: refit without that tag and predict it. This is the honest
    accuracy estimate, especially for the far tags.

Outputs use the same conventions as the other april-tag scripts (frames rotated 180,
pixels x3, Mimg -> (x_mm + 2250, -y_mm + 500), Mrob -> robot mm), so Mimg works in the
runtime puck pipeline and view_homography.py unchanged. Dual camera: every Mrob maps into
the same robot frame, so H_<cam>_to_<ref> = inv(Mrob_ref) @ Mrob_cam, plus a fused
top-down view.

    python analyze_data/calibrate_all_tags.py
    python analyze_data/calibrate_all_tags.py --camera-indices 1 2 --names overhead back --check

Written to --out-dir (default assets/real/):
  Mimg_<name>_cam_all_tags.npy, Mrob_<name>_cam_all_tags.npy   per camera
  H_<name>_to_<ref>_all_tags.npy                                  camera -> reference camera
  multi_cam_all_tags.npz      everything; use with
      python analyze_data/view_homography.py --camera-index 1 2 --calib assets/real/multi_cam_all_tags.npz
  <name>_cam_all_tags_capture.png / _transformed.png, multi_cam_all_tags_fused.png

--check: afterwards, put ONE red puck on the table; every camera maps it to robot (x, y)
and the window shows the spread between cameras. q / Esc to finish.
Keep the arm clear of the tags during capture.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from calibrate_multi_cam_april_tag import live_check
from detect_apriltags import detect, make_passes, merge

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Actual TCP (x, y) in mm with the paddle centered over tags 0-3.
TCP_MM = {
    0: (-402.5, -452.7),
    1: (-700.5, -452.7),
    2: (-391.6, 522.8),
    3: (-690.3, 528.8),
}
# Equally spaced tags along each rail, nearest the robot first. The first two are measured.
RAILS = {
    "left": (0, 1, 4, 6, 8, 10),
    "right": (2, 3, 5, 7, 9, 11),
}

UPSCALE_CONSTANT = 3
VISUAL_DOWNSCALE_CONSTANT = 2
ORIGINAL_SIZE = np.array([640, 480])
OFFSET_CONSTANTS = np.array([2250.0, 500.0])  # must match image_detection.offset_constants
CANVAS_WH = tuple((ORIGINAL_SIZE * UPSCALE_CONSTANT).tolist())
PREVIEW_WH = tuple((ORIGINAL_SIZE * UPSCALE_CONSTANT // VISUAL_DOWNSCALE_CONSTANT).tolist())
# Robot mm -> warped-image px: (x + 2250, -y + 500). Mimg = ROBOT_TO_WARPED @ Mrob.
ROBOT_TO_WARPED = np.array([[1.0, 0.0, OFFSET_CONSTANTS[0]], [0.0, -1.0, OFFSET_CONSTANTS[1]], [0.0, 0.0, 1.0]])


def tag_positions():
    """{tag_id: (xy_mm, measured)} for every tag on the rails, extrapolated linearly."""
    positions = {}
    for rail in RAILS.values():
        first, second = np.array(TCP_MM[rail[0]]), np.array(TCP_MM[rail[1]])
        step = second - first
        for k, tag_id in enumerate(rail):
            positions[tag_id] = (first + k * step, tag_id in TCP_MM)
    return positions


def capture_centers(index, passes, max_spread, sample_frames, min_frames):
    """Mean center (rotated-frame px) of every accepted tag seen in >= min_frames frames."""
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {index} (is another process using it?)")
    samples, last_frame, last_merged, n_frames = {}, None, {}, 0
    try:
        for _ in range(10):  # let exposure settle
            cap.read()
        print(f"camera {index}: detecting over {sample_frames} frames ({len(passes)} passes per frame)...")
        for _ in range(sample_frames):
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            n_frames += 1
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            merged = merge(detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), passes), max_spread)
            for tag_id, m in merged.items():
                if m["ok"]:
                    samples.setdefault(tag_id, []).append(m["center"])
            last_frame, last_merged = frame, merged
    finally:
        cap.release()

    centers = {}
    for tag_id, pts in sorted(samples.items()):
        pts = np.array(pts)
        centers[tag_id] = {"xy": pts.mean(axis=0), "std": pts.std(axis=0), "frames": len(pts), "used": len(pts) >= min_frames}
    return centers, n_frames, last_frame, last_merged


def fit_mrob(pixels_xy, robot_mm):
    """Least-squares homography (x3 upscaled px -> robot mm) over all points."""
    H, _ = cv2.findHomography(
        np.asarray(pixels_xy, np.float64).reshape(-1, 1, 2) * UPSCALE_CONSTANT,
        np.asarray(robot_mm, np.float64).reshape(-1, 1, 2),
        method=0,
    )
    return H / H[2, 2]


def apply(H, pixels_xy):
    pts = np.asarray(pixels_xy, np.float64).reshape(-1, 1, 2) * UPSCALE_CONSTANT
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def spans_both_rails(ids):
    return all(any(t in ids for t in rail) for rail in RAILS.values())


def calibrate_camera(index, name, positions, passes, args, out_dir):
    centers, n_frames, frame, merged = capture_centers(index, passes, args.max_spread, args.frames, args.min_frames)
    if frame is None:
        raise SystemExit(f"{name} (camera {index}): no frames")

    print(f"\n{name} (camera {index}), {n_frames} frames:")
    ids = []
    for tag_id, c in centers.items():
        if tag_id not in positions:
            print(f"  tag {tag_id:>2}: seen {c['frames']} frames, no known position -> ignored")
            continue
        status = "used" if c["used"] else f"skipped (< {args.min_frames} frames)"
        print(
            f"  tag {tag_id:>2}: pixel ({c['xy'][0]:7.2f}, {c['xy'][1]:7.2f})  std ({c['std'][0]:.2f}, {c['std'][1]:.2f}) px"
            f"  {c['frames']}/{n_frames} frames  {status}"
        )
        if c["used"]:
            ids.append(tag_id)
    missing = sorted(set(positions) - set(centers))
    if missing:
        print(f"  not detected: {missing}")
    if len(ids) < 4 or not spans_both_rails(ids):
        raise SystemExit(f"{name}: need >= 4 tags covering both rails, have {ids}")

    px = np.array([centers[t]["xy"] for t in ids])
    rob = np.array([positions[t][0] for t in ids])
    Mrob = fit_mrob(px, rob)
    Mimg = ROBOT_TO_WARPED @ Mrob
    Mimg /= Mimg[2, 2]

    residual = np.linalg.norm(apply(Mrob, px) - rob, axis=1)
    print(f"  fit on {len(ids)} tags {ids}")
    print(f"  {'tag':>5} {'robot x':>8} {'robot y':>8}  {'source':<12} {'fit resid':>9} {'leave-1-out':>11}  (mm)")
    loo = {}
    for k, tag_id in enumerate(ids):
        rest = [j for j in range(len(ids)) if j != k]
        err = None
        if len(rest) >= 4 and spans_both_rails([ids[j] for j in rest]):
            H = fit_mrob(px[rest], rob[rest])
            err = float(np.linalg.norm(apply(H, px[k:k + 1])[0] - rob[k]))
            loo[tag_id] = err
        source = "TCP" if positions[tag_id][1] else "extrapolated"
        err_s = f"{err:11.1f}" if err is not None else f"{'n/a':>11}"
        print(f"  {tag_id:>5} {rob[k][0]:8.1f} {rob[k][1]:8.1f}  {source:<12} {residual[k]:9.1f} {err_s}")

    image = cv2.resize(frame, CANVAS_WH, interpolation=cv2.INTER_LINEAR)
    warped = cv2.warpPerspective(image, Mimg, CANVAS_WH)
    coverage = cv2.warpPerspective(np.ones(image.shape[:2], np.uint8), Mimg, CANVAS_WH, flags=cv2.INTER_NEAREST)

    capture = frame.copy()
    for tag_id, m in merged.items():
        color = (0, 255, 0) if tag_id in ids else (160, 160, 160)
        cv2.polylines(capture, [np.round(m["corners"]).astype(np.int32).reshape(-1, 1, 2)], True, color, 2)
        cv2.putText(capture, str(tag_id), tuple(np.round(m["center"]).astype(int) + [6, -6]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / f"{name}_cam_all_tags_capture.png"), capture)

    preview = cv2.resize(warped, PREVIEW_WH, interpolation=cv2.INTER_LINEAR)
    for tag_id, (xy, measured) in positions.items():
        u, v = (np.array([xy[0], -xy[1]]) + OFFSET_CONSTANTS) / VISUAL_DOWNSCALE_CONSTANT
        color = (0, 255, 255) if measured else (255, 255, 0)
        cv2.drawMarker(preview, (int(round(u)), int(round(v))), color, cv2.MARKER_CROSS, 16, 2)
        cv2.putText(preview, str(tag_id), (int(u) + 6, int(v) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / f"{name}_cam_all_tags_transformed.png"), preview)

    return {
        "name": name,
        "index": index,
        "ids": ids,
        "tag_px": px,
        "Mrob": Mrob,
        "Mimg": Mimg,
        "residual_mm": residual,
        "loo_mm": loo,
        "warped": warped,
        "coverage": coverage,
        "preview": preview,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera-indices", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--names", nargs="+", default=["overhead", "back"], help="first camera is the reference")
    ap.add_argument("--out-dir", default=str(_REPO_ROOT / "assets" / "real"))
    ap.add_argument("--frames", type=int, default=60, help="frames to average per camera")
    ap.add_argument("--min-frames", type=int, default=10, help="a tag must be accepted in this many frames to be used")
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 1.5, 2.0, 2.5, 3.0])
    ap.add_argument("--max-spread", type=float, default=2.0, help="passes farther than this (px) from the median are dropped")
    ap.add_argument("--show", action="store_true", help="show the transformed / fused views until a key is pressed")
    ap.add_argument("--check", action="store_true", help="live red-puck cross-camera check afterwards")
    args = ap.parse_args()
    if len(args.names) != len(args.camera_indices):
        raise SystemExit("--names needs one name per camera index")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    positions = tag_positions()
    print("Tag positions (robot frame, mm):")
    for rail_name, rail in RAILS.items():
        step = np.array(TCP_MM[rail[1]]) - np.array(TCP_MM[rail[0]])
        print(f"  {rail_name} rail {list(rail)}, step ({step[0]:.1f}, {step[1]:.1f}) mm = {np.linalg.norm(step):.1f} mm")
        for tag_id in rail:
            xy, measured = positions[tag_id]
            print(f"    tag {tag_id:>2}: ({xy[0]:8.1f}, {xy[1]:7.1f})  {'TCP' if measured else 'extrapolated'}"
                  f"  table x {xy[0] / 1000 + 1.2:.3f} m")

    # One camera at a time, so two USB cameras never have to stream together for the capture.
    passes = make_passes(args.scales, multi=True)
    cams = [calibrate_camera(i, n, positions, passes, args, out_dir) for i, n in zip(args.camera_indices, args.names)]
    ref = cams[0]

    bundle = {
        "tag_ids": np.array(sorted(positions)),
        "tag_positions_mm": np.array([positions[t][0] for t in sorted(positions)]),
        "tag_measured": np.array([positions[t][1] for t in sorted(positions)]),
        "offset_constants": OFFSET_CONSTANTS,
    }
    for cam in cams:
        np.save(out_dir / f"Mimg_{cam['name']}_cam_all_tags.npy", cam["Mimg"])
        np.save(out_dir / f"Mrob_{cam['name']}_cam_all_tags.npy", cam["Mrob"])
        bundle[f"Mimg_{cam['name']}"] = cam["Mimg"]
        bundle[f"Mrob_{cam['name']}"] = cam["Mrob"]
        bundle[f"ids_{cam['name']}"] = np.array(cam["ids"])
        bundle[f"tag_px_{cam['name']}"] = cam["tag_px"]
        bundle[f"residual_mm_{cam['name']}"] = cam["residual_mm"]
    for cam in cams[1:]:
        H = np.linalg.inv(ref["Mrob"]) @ cam["Mrob"]
        H /= H[2, 2]
        key = f"H_{cam['name']}_to_{ref['name']}"
        np.save(out_dir / f"{key}_all_tags.npy", H)
        bundle[key] = H
        print(f"\n{key} ({cam['name']} x3 pixel -> {ref['name']} x3 pixel, table plane):")
        print(np.array2string(H, precision=5, suppress_small=True))

    np.savez(out_dir / "multi_cam_all_tags.npz", **bundle)

    total = np.zeros((CANVAS_WH[1], CANVAS_WH[0], 3), np.float32)
    weight = np.zeros((CANVAS_WH[1], CANVAS_WH[0]), np.float32)
    for cam in cams:
        w = cam["coverage"].astype(np.float32)
        total += cam["warped"].astype(np.float32) * w[..., None]
        weight += w
    fused = cv2.resize((total / np.maximum(weight, 1.0)[..., None]).astype(np.uint8), PREVIEW_WH, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(str(out_dir / "multi_cam_all_tags_fused.png"), fused)

    print("\nSummary (mm):")
    for cam in cams:
        loo = list(cam["loo_mm"].values())
        loo_s = f"leave-one-out median {np.median(loo):.1f}, max {np.max(loo):.1f}" if loo else "leave-one-out n/a"
        print(f"  {cam['name']:>9}: {len(cam['ids'])} tags, fit residual max {np.max(cam['residual_mm']):.1f}, {loo_s}")
    print(f"Saved homographies and previews to {out_dir}")
    print(f"View: python analyze_data/view_homography.py --camera-index {' '.join(map(str, args.camera_indices))} "
          f"--calib {out_dir / 'multi_cam_all_tags.npz'} --names {' '.join(args.names)}")

    if args.show:
        for cam in cams:
            cv2.imshow(f"{cam['name']} transformed", cam["preview"])
        cv2.imshow("fused (any key to close)", fused)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if args.check:
        live_check(cams)


if __name__ == "__main__":
    main()

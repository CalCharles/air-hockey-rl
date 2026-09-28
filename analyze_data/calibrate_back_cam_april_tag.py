#!/usr/bin/env python3
"""Homography for the back camera from the 4 AprilTags and already-recorded TCP positions.

No robot needed: the TCP positions over tags 0-3 were recorded with
scripts/real/calibrate_robo_camera_april_tag_tcp.py (overhead camera, 2026-09-25)
and are hardcoded in TCP_MM below. This script only detects the tags in the
back camera and recomputes Mimg / Mrob with the same conventions as that script
(frames rotated 180, pixels x3, Mimg -> (x_mm + 2250, -y_mm + 500), Mrob -> robot mm),
so the Mimg drops into the runtime puck pipeline / view_homography.py unchanged.

    python analyze_data/calibrate_back_cam_april_tag.py
    python analyze_data/calibrate_back_cam_april_tag.py --camera-index 2 --show

Writes Mimg_back_cam_april_tag.npy and Mrob_back_cam_april_tag.npy to --out-dir
(default assets/real/), plus capture / overlay / transformed images next to them.
Keep the arm clear of the tags while it captures.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "real"))

from calibrate_robo_camera_april_tag_tcp import (  # noqa: E402
    OFFSET_CONSTANTS,
    ORIGINAL_SIZE,
    TAG_IDS,
    UPSCALE_CONSTANT,
    VISUAL_DOWNSCALE_CONSTANT,
    _make_detector,
    annotate_tags,
    detect_tag_centers,
)

# Actual TCP (x, y) in mm with the paddle centered over tags 0, 1, 2, 3.
TCP_MM = np.float32([
    [-402.5, -452.7],
    [-700.5, -452.7],
    [-391.6, 522.8],
    [-690.3, 528.8],
])


def capture_tag_centers(cap, detector, sample_frames=60, min_valid_frames=10):
    """Average tag 0-3 centers (x, y) over frames where all 4 are visible. Frames rotated 180."""
    samples = {tag_id: [] for tag_id in TAG_IDS}
    seen = set()
    last_frame = None
    last_found = {}
    for _ in range(sample_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        found = detect_tag_centers(frame, detector)
        seen.update(found)
        last_frame, last_found = frame, found
        if all(tag_id in found for tag_id in TAG_IDS):
            for tag_id in TAG_IDS:
                samples[tag_id].append(found[tag_id][0])

    n_valid = len(samples[TAG_IDS[0]])
    if n_valid < min_valid_frames:
        return None, sorted(seen), last_frame, last_found
    points_xy = np.array([np.mean(samples[t], axis=0) for t in TAG_IDS], dtype=np.float32)
    spread = max(float(np.max(np.std(samples[t], axis=0))) for t in TAG_IDS)
    print(f"Averaged {n_valid}/{sample_frames} frames; worst per-tag std {spread:.2f} px")
    return points_xy, list(TAG_IDS), last_frame, last_found


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera-index", type=int, default=2)
    ap.add_argument("--out-dir", default=str(_REPO_ROOT / "assets" / "real"))
    ap.add_argument("--show", action="store_true", help="show the transformed view until a key is pressed")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera_index} (is another process using it?)")
    try:
        for _ in range(10):  # let exposure settle
            cap.read()
        points_xy, seen, frame, found = capture_tag_centers(cap, _make_detector())
    finally:
        cap.release()

    if frame is not None:
        cv2.imwrite(str(out_dir / "back_cam_april_tag_capture.png"), annotate_tags(frame, found))
    if points_xy is None:
        raise SystemExit(
            f"Need tags {list(TAG_IDS)} in at least 10 frames; saw {seen}. "
            f"See {out_dir / 'back_cam_april_tag_capture.png'}."
        )

    warped_px = np.stack([TCP_MM[:, 0], -TCP_MM[:, 1]], axis=1) + OFFSET_CONSTANTS
    pts1 = np.ascontiguousarray(points_xy * UPSCALE_CONSTANT, dtype=np.float32)
    Mrob = cv2.getPerspectiveTransform(pts1, TCP_MM)
    Mimg = cv2.getPerspectiveTransform(pts1, np.ascontiguousarray(warped_px, dtype=np.float32))

    print("Correspondences:")
    for tag_id, px, rob in zip(TAG_IDS, points_xy, TCP_MM):
        print(f"  tag {tag_id}: pixel (x, y) = ({px[0]:.2f}, {px[1]:.2f})  ->  robot (mm) = ({rob[0]:.1f}, {rob[1]:.1f})")

    np.save(out_dir / "Mimg_back_cam_april_tag.npy", Mimg)
    np.save(out_dir / "Mrob_back_cam_april_tag.npy", Mrob)

    image = cv2.resize(frame, tuple(ORIGINAL_SIZE * UPSCALE_CONSTANT), interpolation=cv2.INTER_LINEAR)
    dst = cv2.warpPerspective(image, Mimg, tuple(ORIGINAL_SIZE * UPSCALE_CONSTANT))
    preview_wh = tuple((ORIGINAL_SIZE * UPSCALE_CONSTANT // VISUAL_DOWNSCALE_CONSTANT).tolist())
    dst_preview = cv2.resize(dst, preview_wh, interpolation=cv2.INTER_LINEAR)
    for u, v in warped_px / VISUAL_DOWNSCALE_CONSTANT:
        cv2.drawMarker(dst_preview, (int(round(u)), int(round(v))), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
    cv2.imwrite(str(out_dir / "back_cam_april_tag_transformed.png"), dst_preview)

    print(f"Saved {out_dir / 'Mimg_back_cam_april_tag.npy'} and {out_dir / 'Mrob_back_cam_april_tag.npy'}")
    print(f"Preview images: back_cam_april_tag_capture.png, back_cam_april_tag_transformed.png in {out_dir}")

    if args.show:
        cv2.imshow("back camera transformed (any key to close)", dst_preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

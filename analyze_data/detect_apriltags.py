#!/usr/bin/env python3
"""Live AprilTag (tag36h11) detection on one camera, with optional multi-scale detection.

    python analyze_data/detect_apriltags.py
    python analyze_data/detect_apriltags.py --camera-index 2 --expected-ids 0 1 2 3 4 5 6 7 8 9 10

Two modes (press m to switch):
  default      one pass of the stock detector on the raw frame
  multi-scale  the frame is upscaled (--scales, cubic) and run through both the stock
               and a loosened detector at every scale. Small / far tags that the stock
               detector rejects become decodable once their cells are a few px wide.

Multi-scale corner coordinates are mapped back to the ORIGINAL frame with
x = (x_up + 0.5) / s - 0.5 (cv2.resize aligns pixel centers, so plain x_up / s would
shift every point by (s - 1) / (2 s) px). Each pass gives a center (diagonal
intersection); passes more than --max-spread px from the median are dropped as
outliers, and the tag's center is the mean of the rest. If most passes disagree
(a misread id, or the same id printed twice) the tag is flagged red and not accepted.

Colors: green = accepted, red = passes disagree, gray = id not in --expected-ids.
The panel lists, per id: center (original px), how many of the last --window frames
it was accepted in, the largest distance of a kept pass from the median, and which
passes were kept this frame (+ = loosened detector settings).

Keys: m = toggle mode, r = reset frame statistics, q / Esc = quit (prints a summary).
Stop any other process holding the camera first.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict, deque

import cv2
import numpy as np

DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)


def make_params(loosened):
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    if loosened:
        p.adaptiveThreshWinSizeMin = 3  # more threshold window sizes
        p.adaptiveThreshWinSizeMax = 53
        p.adaptiveThreshWinSizeStep = 4
        p.minMarkerPerimeterRate = 0.01  # allow smaller outlines
        p.perspectiveRemovePixelPerCell = 8  # sample each cell more finely
        p.perspectiveRemoveIgnoredMarginPerCell = 0.2  # ignore more of each cell's blurry edge
        p.maxErroneousBitsInBorderRate = 0.5  # tolerate a noisier black border
    return p


def make_passes(scales, multi):
    """List of (label, scale, detector). Default mode is the single stock pass at 1x."""
    if not multi:
        return [("1x", 1.0, cv2.aruco.ArucoDetector(DICTIONARY, make_params(False)))]
    passes = []
    for s in scales:
        passes.append((f"{s:g}x", s, cv2.aruco.ArucoDetector(DICTIONARY, make_params(False))))
        passes.append((f"{s:g}x+", s, cv2.aruco.ArucoDetector(DICTIONARY, make_params(True))))
    return passes


def tag_center(corners):
    """Intersection of the diagonals: the true center under perspective (corner mean is not)."""
    p0, p1, p2, p3 = corners.astype(np.float64)
    d1, d2 = p2 - p0, p3 - p1
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-9:
        return corners.mean(axis=0)
    t = ((p1[0] - p0[0]) * d2[1] - (p1[1] - p0[1]) * d2[0]) / denom
    return p0 + t * d1


def detect(gray, passes):
    """Run every pass; return {id: [(label, corners_orig, center_orig), ...]} in original px."""
    found = defaultdict(list)
    for label, s, detector in passes:
        img = gray if s == 1.0 else cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        corners, ids, _ = detector.detectMarkers(img)
        if ids is None:
            continue
        for c, tag_id in zip(corners, ids.ravel()):
            c = (c.reshape(4, 2).astype(np.float64) + 0.5) / s - 0.5
            found[int(tag_id)].append((label, c, tag_center(c)))
    return found


def merge(found, max_spread):
    """Per id: mean center of the passes within max_spread px of the median.

    Single passes scatter ~1 px from frame noise; a misread id or a duplicate tag lands
    many px away. So outlier passes are dropped, and the tag is accepted only if most
    passes agree (otherwise it is flagged and not used).
    """
    merged = {}
    for tag_id, hits in found.items():
        centers = np.array([h[2] for h in hits])
        dist = np.linalg.norm(centers - np.median(centers, axis=0), axis=1)
        inliers = dist <= max_spread
        merged[tag_id] = {
            "center": centers[inliers].mean(axis=0) if inliers.any() else np.median(centers, axis=0),
            "corners": hits[int(np.argmin(dist))][1],
            "labels": [h[0] for h, keep in zip(hits, inliers) if keep],
            "spread": float(np.max(dist[inliers])) if inliers.any() else float(np.max(dist)),
            "ok": int(inliers.sum()) * 2 > len(hits),
        }
    return merged


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera-index", type=int, default=1)
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 1.5, 2.0, 2.5, 3.0])
    ap.add_argument("--expected-ids", type=int, nargs="*", default=None, help="other ids are shown gray and ignored")
    ap.add_argument("--max-spread", type=float, default=2.0, help="passes farther than this (px) from the median are dropped")
    ap.add_argument("--window", type=int, default=30, help="frames for the detection-rate / std statistics")
    ap.add_argument("--start-default", action="store_true", help="start in default (single-pass) mode")
    args = ap.parse_args()
    expected = None if args.expected_ids is None else set(args.expected_ids)

    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera_index} (is another process using it?)")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"camera {args.camera_index}: {w}x{h}   tag36h11   m = toggle mode, r = reset, q / Esc = quit")

    multi = not args.start_default
    passes = make_passes(args.scales, multi)
    history = deque(maxlen=args.window)  # per frame: {id: center} of accepted tags
    window = f"apriltag tag36h11  camera {args.camera_index}"
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                continue
            t0 = time.time()
            merged = merge(detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), passes), args.max_spread)
            dt_ms = 1000.0 * (time.time() - t0)
            history.append({
                i: m["center"] for i, m in merged.items()
                if m["ok"] and (expected is None or i in expected)
            })

            for tag_id, m in merged.items():
                if expected is not None and tag_id not in expected:
                    color = (160, 160, 160)
                else:
                    color = (0, 255, 0) if m["ok"] else (0, 0, 255)
                cv2.polylines(frame, [np.round(m["corners"]).astype(np.int32).reshape(-1, 1, 2)], True, color, 2)
                cx, cy = int(round(m["center"][0])), int(round(m["center"][1]))
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                cv2.putText(frame, str(tag_id), (cx + 6, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

            mode = f"multi-scale ({len(passes)} passes)" if multi else "default (1 pass)"
            lines = [f"{mode}  {dt_ms:.0f} ms/frame"]
            ids_seen = sorted(set(merged) | {i for f in history for i in f})
            for tag_id in ids_seen:
                if expected is not None and tag_id not in expected:
                    continue
                rate = sum(tag_id in f for f in history)
                m = merged.get(tag_id)
                if m is None:
                    lines.append(f"id {tag_id:>2}  --  {rate}/{len(history)}")
                    continue
                passes_str = ",".join(m["labels"]) if multi else ""
                lines.append(
                    f"id {tag_id:>2}  ({m['center'][0]:6.1f},{m['center'][1]:6.1f})  {rate}/{len(history)}"
                    f"  spread {m['spread']:.2f}  {passes_str}"
                )
            # Text goes in a strip below the image so it never covers tags.
            panel = np.zeros((10 + 17 * len(lines), frame.shape[1], 3), np.uint8)
            for k, line in enumerate(lines):
                cv2.putText(panel, line, (8, 18 + 17 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(window, np.vstack([frame, panel]))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                multi = not multi
                passes = make_passes(args.scales, multi)
                history.clear()
            if key == ord("r"):
                history.clear()
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if history:
        print(f"\nLast {len(history)} frames ({'multi-scale' if multi else 'default'}), accepted tags:")
        for tag_id in sorted({i for f in history for i in f}):
            pts = np.array([f[tag_id] for f in history if tag_id in f])
            print(
                f"  id {tag_id:>2}: {len(pts)}/{len(history)} frames  center ({pts[:, 0].mean():.2f}, {pts[:, 1].mean():.2f})"
                f"  std ({pts[:, 0].std():.2f}, {pts[:, 1].std():.2f}) px"
            )


if __name__ == "__main__":
    main()

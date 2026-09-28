#!/usr/bin/env python3
"""Robot-camera homography from 4 AprilTags (tag36h11, ids 0-3) and live TCP readings.

Same homography as ``calibrate_robo_camera.py``, but the correspondences come from
AprilTags instead of red pucks, so there is no puck placement or blob-ordering error:

  1. For each tag id 0, 1, 2, 3 in order: stop external control, move the arm by
     hand so the paddle is centered over the tag, restart external control, press
     space. The actual TCP pose is read over RTDE.
  2. Move the arm clear of the tags. The camera averages the 4 tag centers over
     several frames, using multi-scale detection (below).
  3. Mimg / Mrob are computed with cv2.getPerspectiveTransform, and the TCP
     positions are printed.

Conventions (so the runtime puck pipeline keeps working unchanged):
  * frames are rotated 180 degrees and pixels scaled by upscale_constant = 3,
    exactly like control_parameters.homography_transform;
  * Mimg maps to the warped image, pixel = (x_mm + 2250, -y_mm + 500). The runtime
    (image_detection._pixel_to_robot_xy) inverts this as x = (u - 2250) / 1000,
    y = -(v - 500) / 1000. The old puck calibration got the same -y implicitly from
    its clockwise point ordering; with tag ids we set it explicitly;
  * Mrob maps pixels to true robot (x, y) in mm (no y flip). The runtime never loads Mrob.

Tag detection (same as analyze_data/detect_apriltags.py): each frame is upscaled by
DETECT_SCALES (cubic) and run through the stock and a loosened detector at every scale,
so small / far tags still decode on a 640x480 camera. Corners are mapped back to the
original frame with x = (x_up + 0.5) / s - 0.5 (cv2.resize aligns pixel centers; plain
x_up / s would shift every point by (s - 1) / (2 s) px). Per tag, passes more than
MAX_PASS_SPREAD_PX from the median are dropped and the rest averaged; a tag whose passes
mostly disagree (misread id, same id printed twice) is not used. --single-scale goes
back to one stock pass.

    python scripts/real/calibrate_robo_camera_april_tag_tcp.py --camera-index 1
    python scripts/real/calibrate_robo_camera_april_tag_tcp.py --camera-index 1 -s

``-s / --save-homographies`` writes Mimg.npy and Mrob.npy to the current directory
(copy them to assets/real/). Every run also saves them, the TCP poses, and the
capture images under temp/calibration_collect_april_tcp/<timestamp>/.
"""

from __future__ import annotations

import argparse
import os
import sys
import termios
import time
import tty

import cv2
import numpy as np
from rtde_receive import RTDEReceiveInterface as RTDEReceive

ROBOT_HOST = "172.22.22.2"
TAG_IDS = (0, 1, 2, 3)
TEMP_CALIB_DIR = "temp/calibration_collect_april_tcp"

UPSCALE_CONSTANT = 3
VISUAL_DOWNSCALE_CONSTANT = 2
ORIGINAL_SIZE = np.array([640, 480])
# Must match image_detection.offset_constants / AirHockeyReal.offset_constants.
OFFSET_CONSTANTS = np.float32([2250, 500])
CENTER_OFFSET_M = 1.2  # robot x -> table x

DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
DETECT_SCALES = (1.0, 1.5, 2.0, 2.5, 3.0)
MAX_PASS_SPREAD_PX = 2.0  # single passes scatter ~1 px from noise; mistakes land far away


def read_terminal_command():
    """Space returns ``record`` immediately; other text is submitted with Enter."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        buf = []
        while True:
            ch = sys.stdin.read(1)
            if ch == " " and not buf:
                sys.stdout.write("\n")
                sys.stdout.flush()
                return "record"
            if ch in ("\n", "\r"):
                sys.stdout.write("\n")
                sys.stdout.flush()
                return "".join(buf).strip().lower()
            if ch in ("\x7f", "\b"):
                if buf:
                    buf.pop()
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
                continue
            if ch == "\x03":
                raise KeyboardInterrupt
            if ch == "\x04":
                return "quit"
            buf.append(ch)
            sys.stdout.write(ch)
            sys.stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def read_tcp_pose(host, timeout_s=4.0):
    """Connect, read one actual TCP pose [x, y, z, rx, ry, rz] (m, rad), disconnect.

    A fresh connection per read, so stopping external control between tags
    does not break the script. Returns None on failure.
    """
    rcv = None
    try:
        rcv = RTDEReceive(host)
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if rcv.isConnected():
                pose = rcv.getActualTCPPose()
                if pose is not None and len(pose) >= 6:
                    return [float(v) for v in pose[:6]]
            time.sleep(0.05)
        return None
    except Exception as exc:
        print(f"Could not read the robot ({exc}).")
        return None
    finally:
        if rcv is not None:
            try:
                rcv.disconnect()
            except Exception:
                pass


def collect_tcp_over_tags(host):
    """Record one TCP pose with the paddle centered over each tag id, in order."""
    recorded = []
    print(
        "\nFor each tag: stop external control, move the paddle by hand so it is centered over\n"
        "the tag, restart external control, then press space.\n"
        "Commands: space = record TCP,  redo + Enter = discard last,  q + Enter = quit.\n"
    )
    while len(recorded) < len(TAG_IDS):
        tag_id = TAG_IDS[len(recorded)]
        print(f"[tag {tag_id}] Center the paddle over tag {tag_id}, restart external control, press space.")
        command = read_terminal_command()
        if command in {"q", "quit", "exit"}:
            return None
        if command == "redo":
            if not recorded:
                print("Nothing to redo yet.")
                continue
            recorded.pop()
            print(f"Discarded tag {TAG_IDS[len(recorded)]}. Record it again.")
            continue
        if command != "record":
            print("Unrecognized command. Press space to record, or type redo / q.")
            continue

        pose = read_tcp_pose(host)
        if pose is None:
            print(f"No TCP pose for tag {tag_id}. Is external control running? Press space to retry.")
            continue
        recorded.append(pose)
        print(f"Recorded tag {tag_id}: x={pose[0] * 1000:.1f} mm  y={pose[1] * 1000:.1f} mm  z={pose[2] * 1000:.1f} mm")
    return recorded


def _make_params(loosened):
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


def _make_detector(multi_scale=True):
    """Detection passes as a list of (scale, ArucoDetector). Pass it to detect_tag_centers."""
    if not multi_scale:
        return [(1.0, cv2.aruco.ArucoDetector(DICTIONARY, _make_params(False)))]
    return [
        (s, cv2.aruco.ArucoDetector(DICTIONARY, _make_params(loosened)))
        for s in DETECT_SCALES
        for loosened in (False, True)
    ]


def _tag_center_xy(corners):
    """Intersection of the tag's diagonals: the true center under perspective (corner mean is not)."""
    p0, p1, p2, p3 = corners.astype(np.float64)
    d1, d2 = p2 - p0, p3 - p1
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-9:
        return corners.mean(axis=0)
    t = ((p1[0] - p0[0]) * d2[1] - (p1[1] - p0[1]) * d2[0]) / denom
    return p0 + t * d1


def detect_tag_centers(frame_bgr, detector):
    """Return {tag_id: (center_xy, corners)} in original-frame px for one BGR frame.

    ``detector`` is the pass list from _make_detector. Every pass's corners are mapped
    back to the original frame; per tag, passes farther than MAX_PASS_SPREAD_PX from
    the median center are dropped and the rest averaged. Tags whose passes mostly
    disagree are left out.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    hits = {}
    for s, aruco in detector:
        img = gray if s == 1.0 else cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        corners, ids, _rejected = aruco.detectMarkers(img)
        if ids is None:
            continue
        for tag_corners, tag_id in zip(corners, ids.flatten()):
            pts = (tag_corners.reshape(4, 2).astype(np.float64) + 0.5) / s - 0.5
            hits.setdefault(int(tag_id), []).append((_tag_center_xy(pts), pts))

    found = {}
    for tag_id, tag_hits in hits.items():
        centers = np.array([c for c, _ in tag_hits])
        dist = np.linalg.norm(centers - np.median(centers, axis=0), axis=1)
        inliers = dist <= MAX_PASS_SPREAD_PX
        if int(inliers.sum()) * 2 <= len(tag_hits):
            continue
        found[tag_id] = (centers[inliers].mean(axis=0), tag_hits[int(np.argmin(dist))][1])
    return found


def annotate_tags(frame_bgr, found):
    preview = frame_bgr.copy()
    for tag_id, (center, pts) in found.items():
        cx, cy = int(round(center[0])), int(round(center[1]))
        cv2.polylines(preview, [pts.astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 2)
        cv2.circle(preview, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(preview, f"id {tag_id}", (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return preview


def detect_four_apriltags(cap, detector, sample_frames=60, min_valid_frames=10):
    """Average the centers (x, y) of tags 0-3 over frames where all 4 are visible.

    Frames are rotated 180 degrees, like the runtime pipeline.
    """
    samples = {tag_id: [] for tag_id in TAG_IDS}
    seen = set()
    valid_frames = 0
    last_frame = None
    print(f"Detecting over {sample_frames} frames ({len(detector)} passes per frame)...")
    for _ in range(sample_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        last_frame = frame
        found = detect_tag_centers(frame, detector)
        seen.update(found)
        cv2.imshow("apriltag-detect", annotate_tags(frame, found))
        cv2.waitKey(1)
        if all(tag_id in found for tag_id in TAG_IDS):
            valid_frames += 1
            for tag_id in TAG_IDS:
                samples[tag_id].append(found[tag_id][0])

    if valid_frames < min_valid_frames:
        return None, sorted(seen), last_frame
    points_xy = np.array([np.mean(samples[tag_id], axis=0) for tag_id in TAG_IDS], dtype=np.float32)
    spread = max(float(np.max(np.std(samples[tag_id], axis=0))) for tag_id in TAG_IDS)
    print(f"Averaged {valid_frames} frames; worst per-tag std {spread:.2f} px")
    return points_xy, list(TAG_IDS), last_frame


def pump_gui(duration_s=0.5):
    """Let HighGUI paint windows before blocking on terminal input().

    A single waitKey(1) is not enough to draw a newly created window, which then
    stays black while the script waits in the terminal.
    """
    deadline = time.time() + duration_s
    while time.time() < deadline:
        cv2.waitKey(10)


def print_tcp_table(poses):
    print("\nTCP positions over each tag (actual TCP pose):")
    print(f"  {'tag':>3}  {'x (mm)':>9} {'y (mm)':>9} {'z (mm)':>9}   {'rx':>7} {'ry':>7} {'rz':>7} (rad)   table x (m)")
    for tag_id, pose in zip(TAG_IDS, poses):
        x, y, z, rx, ry, rz = pose
        print(
            f"  {tag_id:>3}  {x * 1000:9.1f} {y * 1000:9.1f} {z * 1000:9.1f}   "
            f"{rx:7.3f} {ry:7.3f} {rz:7.3f}         {x + CENTER_OFFSET_M:.3f}"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera-index", type=int, required=True)
    ap.add_argument("--robot-host", default=ROBOT_HOST)
    ap.add_argument("-s", "--save-homographies", action="store_true")
    ap.add_argument("--single-scale", action="store_true", help="one stock detector pass instead of multi-scale")
    args = ap.parse_args()

    session_dir = os.path.join(TEMP_CALIB_DIR, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(session_dir, exist_ok=True)

    # Open the camera first so a wrong index fails before any robot work.
    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera_index} (is another process using it?)")

    try:
        poses = collect_tcp_over_tags(args.robot_host)
        if poses is None:
            print("Calibration canceled.")
            return
        np.save(os.path.join(session_dir, "tcp_poses.npy"), np.array(poses))
        print_tcp_table(poses)

        detector = _make_detector(multi_scale=not args.single_scale)
        input("\nMove the arm clear of all 4 tags, then press Enter to detect them... ")
        while True:
            points_xy, seen_ids, frame = detect_four_apriltags(cap, detector)
            if points_xy is None:
                retry = input(
                    f"Need tags {list(TAG_IDS)} in at least 10 frames; saw {seen_ids}. "
                    "Adjust and press Enter to retry, or 'q' to quit: "
                ).strip().lower()
                if retry in {"q", "quit", "exit"}:
                    print("Calibration canceled.")
                    return
                continue
            preview = frame.copy()
            for tag_id, (x, y) in zip(TAG_IDS, points_xy):
                cv2.circle(preview, (int(round(x)), int(round(y))), 8, (0, 255, 0), 2)
                cv2.putText(preview, str(tag_id), (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("apriltag-detect-final", preview)
            pump_gui()
            confirm = input("Detected tags 0-3. Type 'done' to accept, or press Enter to recapture: ").strip().lower()
            if confirm in {"done", "d"}:
                break
    finally:
        cap.release()

    robot_mm = np.array([[p[0] * 1000.0, p[1] * 1000.0] for p in poses], dtype=np.float32)
    warped_px = np.stack([robot_mm[:, 0], -robot_mm[:, 1]], axis=1) + OFFSET_CONSTANTS
    pts1 = np.ascontiguousarray(points_xy * UPSCALE_CONSTANT, dtype=np.float32)

    Mrob = cv2.getPerspectiveTransform(pts1, robot_mm)
    Mimg = cv2.getPerspectiveTransform(pts1, np.ascontiguousarray(warped_px, dtype=np.float32))

    print("\nCorrespondences used for calibration:")
    for tag_id, px, rob, wp in zip(TAG_IDS, points_xy, robot_mm, warped_px):
        print(
            f"  tag {tag_id}: pixel (x, y) = ({px[0]:.2f}, {px[1]:.2f})  ->  robot (mm) = ({rob[0]:.1f}, {rob[1]:.1f})"
            f"  ->  warped px = ({wp[0]:.1f}, {wp[1]:.1f})"
        )
    outside = [t for t, (u, v) in zip(TAG_IDS, warped_px) if not (0 <= u < 1920 and 0 <= v < 1440)]
    if outside:
        # Harmless for the fit (tags beside the table can map off-canvas); only the table must fit.
        print(f"Note: tags {outside} map outside the 1920x1440 warped image (fine if the table itself is inside).")

    image = cv2.resize(frame, tuple(ORIGINAL_SIZE * UPSCALE_CONSTANT), interpolation=cv2.INTER_LINEAR)
    dst = cv2.warpPerspective(image, Mimg, tuple(ORIGINAL_SIZE * UPSCALE_CONSTANT))
    preview_wh = tuple((ORIGINAL_SIZE * UPSCALE_CONSTANT // VISUAL_DOWNSCALE_CONSTANT).tolist())
    dst_preview = cv2.resize(dst, preview_wh, interpolation=cv2.INTER_LINEAR)
    for u, v in warped_px / VISUAL_DOWNSCALE_CONSTANT:
        cv2.drawMarker(dst_preview, (int(round(u)), int(round(v))), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
    cv2.imshow("image", cv2.resize(image, preview_wh, interpolation=cv2.INTER_LINEAR))
    cv2.imshow("transformed", dst_preview)
    pump_gui()

    cv2.imwrite(os.path.join(session_dir, "apriltag_capture_raw.png"), frame)
    cv2.imwrite(os.path.join(session_dir, "apriltag_capture_overlay.png"), preview)
    cv2.imwrite(os.path.join(session_dir, "transformed.png"), dst_preview)
    np.save(os.path.join(session_dir, "Mimg.npy"), Mimg)
    np.save(os.path.join(session_dir, "Mrob.npy"), Mrob)
    np.savez(
        os.path.join(session_dir, "calibration.npz"),
        tag_ids=np.array(TAG_IDS),
        tcp_poses=np.array(poses),
        robot_points_mm=robot_mm,
        tag_centers_xy=points_xy,
        offset_constants=OFFSET_CONSTANTS,
        Mimg=Mimg,
        Mrob=Mrob,
    )
    if args.save_homographies:
        np.save("Mimg.npy", Mimg)
        np.save("Mrob.npy", Mrob)
        print("Saved Mimg.npy and Mrob.npy to the current directory (copy to assets/real/).")
    print(f"Saved session to {session_dir}")

    input("Transformed view shown. Press Enter to exit... ")
    cv2.destroyAllWindows()
    print_tcp_table(poses)


if __name__ == "__main__":
    main()

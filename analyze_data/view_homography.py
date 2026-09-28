#!/usr/bin/env python3
"""Raw camera view, the Mimg-warped image the puck detector sees, and a Box2D render of
what the real env reports, for one or more cameras.

    python analyze_data/view_homography.py --camera-index 0
    python analyze_data/view_homography.py --camera-index 1 2 \
        --calib assets/real/multi_cam_april_tag.npz --names overhead back

Multiple cameras: each camera is shown as a column (raw on top, warped below). Each
camera runs the puck detector on its own warped image; the Box2D puck is the mean of
the cameras that detected it (the fallback when none did), and each camera's own
estimate is drawn as a colored ring around it, with their spread in mm.

Puck: same path as AirHockeyReal — rotate 180, upscale, warp with Mimg, run the
puck detector, add the table offsets. Paddle: actual TCP pose over RTDE + the table offsets
(--no-robot skips the robot connection and draws only the puck).
Both are drawn on the raw view (through inv(Mimg)) and on the warped view so you can
see where the homography puts them (puck green/red when occluded, paddle blue).

Homographies: --mimg (one .npy per camera, default assets/real/Mimg.npy), or --calib
(an .npz from calibrate_multi_cam_april_tag.py) with --names picking Mimg_<name>.
q / Esc to quit.
Stop any other process holding the camera first.
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from airhockey.sims.real.image_detection import (  # noqa: E402
    find_red_hockey_puck,
    find_red_hockey_puck_antiglare,
    offset_constants,
    original_size,
    upscale_constant,
    visual_downscale_constant,
)
from airhockey.sims.real.table_calibration import TABLE_CENTER_OFFSET_X, TABLE_CENTER_OFFSET_Y  # noqa: E402
from scripts.visualization.visualize_real_trajectory import RealTrajectoryRenderer  # noqa: E402

PUCK_DETECTORS = {
    "red_puck": find_red_hockey_puck,
    "red_puck_antiglare": find_red_hockey_puck_antiglare,
}
# AirHockeyReal defaults (airhockey/sims/air_hockey_real.py).
ANTIGLARE_KWARGS = {
    "antiglare_bounds_in_raw_image": True,
    "antiglare_min_x_px": 290,
    "antiglare_max_x_px": 451,
    "antiglare_min_y_px": 186,
    "antiglare_max_y_px": 465,
}
PUCK_HISTORY_LEN = 5
CAMERA_COLORS = [(255, 0, 255), (0, 200, 255), (255, 255, 0), (0, 128, 255)]  # BGR, one per camera


def warp_to_detector_frame(frame, Mimg):
    """Mirror of control_parameters.homography_transform (rotate=False), with our own Mimg."""
    image = cv2.rotate(frame, cv2.ROTATE_180)
    image = cv2.resize(
        image,
        (int(640 * upscale_constant), int(480 * upscale_constant)),
        interpolation=cv2.INTER_LINEAR,
    )
    dst = cv2.warpPerspective(image, Mimg, tuple(original_size * upscale_constant))
    return cv2.resize(
        dst,
        (
            int(640 * upscale_constant / visual_downscale_constant),
            int(480 * upscale_constant / visual_downscale_constant),
        ),
        interpolation=cv2.INTER_LINEAR,
    )


def table_xy_to_warped_pixel(table_xy, center_offset):
    """Table-frame (m) -> pixel in the warped image returned by warp_to_detector_frame."""
    robot_x = float(table_xy[0]) - center_offset[0]
    robot_y = float(table_xy[1]) - center_offset[1]
    u = (robot_x * 1000.0 + offset_constants[0]) / visual_downscale_constant
    v = (-robot_y * 1000.0 + offset_constants[1]) / visual_downscale_constant
    return int(round(u)), int(round(v))


def resize_to_height(image, height):
    return cv2.resize(image, (int(round(image.shape[1] * height / image.shape[0])), height))


def table_xy_to_raw_pixel(table_xy, Mimg_inv, center_offset, raw_wh):
    """Inverse of the puck pipeline: table-frame (m) -> pixel in the unrotated raw frame."""
    robot_x = float(table_xy[0]) - center_offset[0]
    robot_y = float(table_xy[1]) - center_offset[1]
    homo = np.array(
        [[[robot_x * 1000.0 + offset_constants[0], -robot_y * 1000.0 + offset_constants[1]]]],
        dtype=np.float64,
    )
    up = cv2.perspectiveTransform(homo, Mimg_inv)[0, 0] / upscale_constant  # rotated 640x480 frame
    w, h = raw_wh
    x = w - 1 - up[0] * w / 640.0
    y = h - 1 - up[1] * h / 480.0
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    return int(round(x)), int(round(y))


def load_homographies(args):
    if args.calib is not None:
        calib = np.load(args.calib)
        missing = [n for n in args.names if f"Mimg_{n}" not in calib.files]
        if missing:
            raise SystemExit(f"{args.calib} has no Mimg for {missing}; available: {calib.files}")
        mimgs = [np.asarray(calib[f"Mimg_{n}"], dtype=np.float64) for n in args.names]
        labels = list(args.names)
    else:
        mimgs = [np.load(path) for path in args.mimg]
        labels = [Path(path).stem for path in args.mimg]
    if len(mimgs) != len(args.camera_index):
        raise SystemExit(f"{len(args.camera_index)} camera indices but {len(mimgs)} homographies")
    return mimgs, labels


def box2d_pixel(renderer, table_xy, frame_h):
    """Table-frame (m) -> pixel in the vertical (rotated CCW) frame from render_frame."""
    x, y = renderer.table_position_to_pixel_coords(float(table_xy[0]), float(table_xy[1]))
    return int(y), int(frame_h - 1 - x)


def render_box2d(renderer, paddle, puck):
    """Box2D panel. Without a paddle (--no-robot), draw only the table and the puck."""
    puck_kwargs = dict(
        puck_x=float(puck[0]) if puck is not None else None,
        puck_y=float(puck[1]) if puck is not None else None,
        puck_occluded=bool(puck[2]) if puck is not None else None,
    )
    if paddle is not None:
        return renderer.render_frame(pos_x=float(paddle[0]), pos_y=float(paddle[1]), **puck_kwargs)
    frame = renderer.table_img.copy()
    if puck is not None:
        renderer.draw_puck(frame, puck_kwargs["puck_x"], puck_kwargs["puck_y"], puck_kwargs["puck_occluded"])
        state = "occluded" if puck_kwargs["puck_occluded"] else "visible"
        cv2.putText(frame, f"Puck: ({puck[0]:.3f}, {puck[1]:.3f})m [{state}]", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 180, 75), 2)
    return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera-index", type=int, nargs="+", required=True)
    ap.add_argument("--mimg", nargs="+", default=[str(_REPO_ROOT / "assets" / "real" / "Mimg.npy")])
    ap.add_argument("--calib", default=None, help="npz from calibrate_multi_cam_april_tag.py (overrides --mimg)")
    ap.add_argument("--names", nargs="+", default=["overhead", "back"], help="camera names in --calib, in --camera-index order")
    ap.add_argument("--robot-host", default="172.22.22.2")
    ap.add_argument("--no-robot", action="store_true", help="don't connect to the robot; no paddle is drawn")
    ap.add_argument("--puck-detector", choices=sorted(PUCK_DETECTORS), default="red_puck_antiglare")
    ap.add_argument("--center-offset", type=float, default=TABLE_CENTER_OFFSET_X,
                    help="table x = robot x + this (m); default from table_calibration.py")
    ap.add_argument("--center-offset-y", type=float, default=TABLE_CENTER_OFFSET_Y,
                    help="table y = robot y + this (m); default from table_calibration.py")
    ap.add_argument("--scale", type=float, default=0.75, help="display scale of the whole window")
    args = ap.parse_args()


    print(args.mimg)

    mimgs, labels = load_homographies(args)


    detector = PUCK_DETECTORS[args.puck_detector]
    center_offset = (args.center_offset, args.center_offset_y)
    detector_kwargs = dict(center_offset_constant=args.center_offset, center_offset_constant_y=args.center_offset_y)
    if args.puck_detector == "red_puck_antiglare":
        detector_kwargs.update(ANTIGLARE_KWARGS)

    rcv = None
    if not args.no_robot:
        from rtde_receive import RTDEReceiveInterface as RTDEReceive

        # RTDEReceive waits indefinitely when the robot is off or unreachable.
        print(f"connecting to robot {args.robot_host} ... (hangs here if the robot is off; use --no-robot)")
        rcv = RTDEReceive(args.robot_host)

    cams = []
    for index, Mimg, label in zip(args.camera_index, mimgs, labels):
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise SystemExit(f"could not open camera {index} (is another process using it?)")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"camera {index} ({label}): {w}x{h}")
        cams.append({
            "index": index,
            "label": label,
            "cap": cap,
            "Mimg": Mimg,
            "Mimg_inv": np.linalg.inv(Mimg),
            "history": deque(maxlen=PUCK_HISTORY_LEN),
            "color": CAMERA_COLORS[len(cams) % len(CAMERA_COLORS)],
        })
    multi = len(cams) > 1
    print(f"robot: {'none (--no-robot)' if rcv is None else args.robot_host}   q / Esc to quit")

    renderer = RealTrajectoryRenderer(orientation="vertical", paddle_input_frame="table", quiet=True)
    fused_history = deque(maxlen=PUCK_HISTORY_LEN)
    window = "homography  cameras " + " ".join(str(c["index"]) for c in cams)
    try:
        while True:
            frames = [cam["cap"].read() for cam in cams]
            if any(not ok or frame is None or frame.size == 0 for ok, frame in frames):
                continue

            paddle = None
            if rcv is not None:
                tcp = np.asarray(rcv.getActualTCPPose(), dtype=float)
                paddle = np.array([tcp[0] + args.center_offset, tcp[1] + args.center_offset_y])

            columns, hits = [], []
            for cam, (_, frame) in zip(cams, frames):
                # The detector may draw on its input, so keep a clean copy for display.
                warped = warp_to_detector_frame(frame, cam["Mimg"])
                warped_view = warped.copy()
                # Detector hit (occluded == 0) is robot-frame x; fallback is already table frame.
                puck = np.array(
                    detector(warped, list(cam["history"]), rotate=False, **detector_kwargs),
                    dtype=float,
                )
                if int(puck[2]) == 0:
                    puck[0] += args.center_offset
                    puck[1] += args.center_offset_y
                    hits.append((cam, puck[:2].copy()))
                have_puck = int(puck[2]) == 0 or len(cam["history"]) > 0
                if have_puck:
                    cam["history"].append(puck)

                raw = frame.copy()
                raw_wh = (raw.shape[1], raw.shape[0])
                if have_puck:
                    color = cam["color"] if multi else (60, 180, 75)
                    if int(puck[2]):
                        color = (30, 30, 220)
                    px = table_xy_to_raw_pixel(puck[:2], cam["Mimg_inv"], center_offset, raw_wh)
                    if px is not None:
                        cv2.circle(raw, px, 10, color, 2)
                    cv2.circle(warped_view, table_xy_to_warped_pixel(puck[:2], center_offset), 16, color, 3)
                if paddle is not None:
                    px = table_xy_to_raw_pixel(paddle, cam["Mimg_inv"], center_offset, raw_wh)
                    if px is not None:
                        cv2.circle(raw, px, 14, (255, 128, 0), 2)
                    cv2.circle(warped_view, table_xy_to_warped_pixel(paddle, center_offset), 26, (255, 128, 0), 3)
                if multi:
                    cv2.putText(raw, f"{cam['label']} (camera {cam['index']})", (10, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, cam["color"], 2, cv2.LINE_AA)
                if multi:
                    columns.append(np.vstack([raw, cv2.resize(warped_view, (raw.shape[1], int(round(raw.shape[1] * 0.75))))]))
                else:
                    columns.extend([raw, resize_to_height(warped_view, raw.shape[0])])

            # Fused puck: mean of the cameras that detected it, else the last fused position (occluded).
            if hits:
                puck = np.array([*np.mean([xy for _, xy in hits], axis=0), 0.0])
            elif fused_history:
                puck = np.array([*fused_history[-1][:2], 1.0])
            else:
                puck = None
            if puck is not None:
                fused_history.append(puck)

            sim = render_box2d(renderer, paddle, puck)
            if multi:
                ring = max(4, int(renderer.puck_radius * renderer.ppm) + 3)
                for cam, xy in hits:
                    cv2.circle(sim, box2d_pixel(renderer, xy, sim.shape[0]), ring, cam["color"], 2)
                if len(hits) > 1:
                    pts = np.array([xy for _, xy in hits])
                    spread_mm = 1000.0 * float(np.max(np.linalg.norm(pts[:, None] - pts[None], axis=-1)))
                    cv2.putText(sim, f"cam spread {spread_mm:.0f} mm", (10, sim.shape[0] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

            height = max(col.shape[0] for col in columns)
            view = np.hstack([resize_to_height(col, height) for col in columns] + [resize_to_height(sim, height)])
            if args.scale != 1.0:
                view = cv2.resize(view, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
            cv2.imshow(window, view)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        for cam in cams:
            cam["cap"].release()
        cv2.destroyAllWindows()
        if rcv is not None:
            rcv.disconnect()


if __name__ == "__main__":
    main()

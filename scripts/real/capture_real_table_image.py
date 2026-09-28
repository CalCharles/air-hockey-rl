#!/usr/bin/env python3
"""Make the sim's table image from the real table, aligned to the puck detector.

Writes ``assets/air_hockey_table_real.png``, a top-down picture of the real playing
surface in the table frame. Every renderer uses it in place of the stock rink drawing
when it exists (see ``airhockey/renderers/table_image.py``), so the circles, lines and
goal slots drawn in the sim sit where the real ones are.

Steps (terminal prompts):
  1. Blank table: move the arm (and anything else) off the table, press Enter. The
     median of --frames frames is rectified straight into the table frame through the
     same homography (--mimg), rotation, upscale and table offsets
     (``table_calibration.py``) that the runtime puck pipeline uses. The picture is
     aligned by construction; nothing is stretched by hand.
  2. Puck checks: put the red puck on the centre dot of a big circle, step out of view,
     press Enter. The puck goes through the runtime detector exactly as the env sees it
     (``homography_transform`` -> ``find_red_hockey_puck_antiglare``). The dot is found
     in the blank picture near that position, and the offset puck - dot is printed.
     Repeat for as many circles as you like (all four is best), then type d.
  3. Alignment: the picture is warped so each dot lands under its reported puck.
     1 check -> translation, 2 -> similarity (shift, rotation, uniform scale),
     >= 3 -> affine (shift, rotation, scale / stretch per axis). The residuals after
     the warp are printed. Offsets larger than --max-correction-cm are treated as a
     failed detection and nothing is warped.

Outputs:
  --out (default assets/air_hockey_table_real.png) and a .json with the geometry,
  checks and alignment next to it; the raw captures, the unaligned picture and the
  sanity images (reported puck drawn on the final picture, and in the real-robot sim
  view) under temp/real_table_image/<timestamp>/.

    python scripts/real/capture_real_table_image.py --camera-index 1
    python scripts/real/capture_real_table_image.py --from-session temp/real_table_image/<timestamp>

--from-session rebuilds everything from the saved raw captures (no camera).
Rerun the script whenever the camera, the homography or table_calibration.py changes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from airhockey.renderers.table_image import REAL_TABLE_IMAGE  # noqa: E402
from airhockey.sims.real import image_detection  # noqa: E402
from airhockey.sims.real.table_calibration import (  # noqa: E402
    TABLE_CENTER_OFFSET_X,
    TABLE_CENTER_OFFSET_Y,
)

# Box2D table (sim and real configs): the rectangle the renderers stretch the image over.
TABLE_LENGTH = 1.9304
TABLE_WIDTH = 0.8636
PUCK_RADIUS = 0.03175

# Runtime camera pipeline (control_parameters.homography_transform + image_detection).
RAW_WH = (640, 480)
UPSCALE = 3
CANVAS_WH = (RAW_WH[0] * UPSCALE, RAW_WH[1] * UPSCALE)
SHOWDST_WH = (CANVAS_WH[0] // 2, CANVAS_WH[1] // 2)
OFFSET_CONSTANTS = np.asarray(image_detection.offset_constants, dtype=float)
DETECTORS = {
    "red_puck_antiglare": image_detection.find_red_hockey_puck_antiglare,
    "red_puck": image_detection.find_red_hockey_puck,
}
# AirHockeyReal defaults for the antiglare detector.
ANTIGLARE_KWARGS = {
    "antiglare_min_x_px": 290,
    "antiglare_max_x_px": 451,
    "antiglare_min_y_px": 186,
    "antiglare_max_y_px": 465,
}


# --------------------------------------------------------------------------- camera

def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {index} (is another process using it?)")
    return cap


def grab_median(cap, n_frames):
    """Median of n_frames frames (after flushing the buffer): removes noise and flicker."""
    for _ in range(10):
        cap.read()
    frames = []
    for _ in range(n_frames):
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)
    if not frames:
        raise SystemExit("camera returned no frames")
    if frames[0].shape[1::-1] != RAW_WH:
        raise SystemExit(f"camera frames are {frames[0].shape[1::-1]}, the homography expects {RAW_WH}")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def ask(prompt):
    try:
        return input(prompt).strip().lower()
    except EOFError:
        return "q"


def show(window, image, enabled):
    if enabled:
        cv2.imshow(window, image)
        cv2.waitKey(300)


# --------------------------------------------------------------------- geometry

def runtime_showdst(raw, mimg):
    """Exactly control_parameters.homography_transform(raw)[0] with the given Mimg."""
    image = cv2.rotate(raw, cv2.ROTATE_180)
    image = cv2.resize(image, CANVAS_WH, interpolation=cv2.INTER_LINEAR)
    dst = cv2.warpPerspective(image, mimg, CANVAS_WH)
    return cv2.resize(dst, SHOWDST_WH, interpolation=cv2.INTER_LINEAR)


def detect_puck_table_xy(raw, mimg, detector):
    """Puck in the table frame as the env reports it, or None when occluded."""
    kwargs = ANTIGLARE_KWARGS if detector == "red_puck_antiglare" else {}
    x, y, occluded = DETECTORS[detector](runtime_showdst(raw, mimg), rotate=False, **kwargs)
    if int(occluded):
        return None
    return np.array([x + TABLE_CENTER_OFFSET_X, y + TABLE_CENTER_OFFSET_Y])


def table_to_raw_px(table_xy, mimg):
    """Table-frame metres (N, 2) -> raw camera pixels (N, 2), inverting the runtime chain.

    table -> robot (table_calibration) -> canvas px (u = x_mm + 2250, v = -y_mm + 500)
    -> inverse Mimg -> x3 upscaled index -> rotated-frame px (cv2.resize pixel-centre
    convention) -> raw px (undo the 180 degree rotation).
    """
    robot = table_xy - np.array([TABLE_CENTER_OFFSET_X, TABLE_CENTER_OFFSET_Y])
    canvas = np.column_stack([robot[:, 0] * 1000.0 + OFFSET_CONSTANTS[0], -robot[:, 1] * 1000.0 + OFFSET_CONSTANTS[1]])
    up = cv2.perspectiveTransform(canvas.reshape(-1, 1, 2), np.linalg.inv(mimg)).reshape(-1, 2)
    rot = (up + 0.5) / UPSCALE - 0.5
    return np.column_stack([RAW_WH[0] - 1 - rot[:, 0], RAW_WH[1] - 1 - rot[:, 1]])


class TableGrid:
    """Stored-PNG pixel <-> table metres (row 0 = far wall x = -L/2, column 0 = y = -W/2)."""

    def __init__(self, ppm):
        self.rows = int(round(TABLE_LENGTH * ppm))
        self.cols = int(round(TABLE_WIDTH * ppm))
        self.ppm_x = self.rows / TABLE_LENGTH
        self.ppm_y = self.cols / TABLE_WIDTH

    def pixel_to_table(self, col, row):
        return np.column_stack([-TABLE_LENGTH / 2 + (np.asarray(row) + 0.5) / self.ppm_x,
                                -TABLE_WIDTH / 2 + (np.asarray(col) + 0.5) / self.ppm_y])

    def table_to_pixel(self, xy):
        xy = np.atleast_2d(xy)
        return np.column_stack([(xy[:, 1] + TABLE_WIDTH / 2) * self.ppm_y - 0.5,
                                (xy[:, 0] + TABLE_LENGTH / 2) * self.ppm_x - 0.5])

    def render(self, raw, mimg, align=None):
        """Rectify a raw frame into the table image. ``align`` (2x3, table metres) maps
        unaligned table positions to where they should be drawn."""
        rr, cc = np.mgrid[0:self.rows, 0:self.cols]
        xy = self.pixel_to_table(cc.ravel(), rr.ravel())
        if align is not None:
            inv = cv2.invertAffineTransform(np.asarray(align, dtype=np.float64))
            xy = xy @ inv[:, :2].T + inv[:, 2]
        src = table_to_raw_px(xy, mimg).astype(np.float32)
        map_x = src[:, 0].reshape(self.rows, self.cols)
        map_y = src[:, 1].reshape(self.rows, self.cols)
        return cv2.remap(raw, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def canvas_coverage(mimg):
    """Parts of the table the runtime detector cannot see because they fall off the
    1920x1440 canvas (or outside the detector's crop)."""
    grid = TableGrid(200)
    rr, cc = np.mgrid[0:grid.rows, 0:grid.cols]
    xy = grid.pixel_to_table(cc.ravel(), rr.ravel())
    robot = xy - np.array([TABLE_CENTER_OFFSET_X, TABLE_CENTER_OFFSET_Y])
    u = robot[:, 0] * 1000.0 + OFFSET_CONSTANTS[0]
    v = -robot[:, 1] * 1000.0 + OFFSET_CONSTANTS[1]
    # _preprocess_puck_image (rotate=False): detector px = canvas / 4; rows >= 249 and cols >= 470 blanked.
    visible = (u >= 0) & (u < 470 * 4) & (v >= 9 * 4) & (v < 249 * 4)
    raw = table_to_raw_px(xy, mimg)
    in_camera = (raw[:, 0] >= 0) & (raw[:, 0] <= RAW_WH[0] - 1) & (raw[:, 1] >= 0) & (raw[:, 1] <= RAW_WH[1] - 1)
    hidden = xy[~(visible & in_camera)]
    return float(1.0 - (visible & in_camera).mean()), hidden


# ----------------------------------------------------------------------- landmarks

def find_dot(table_img, grid, near_xy, search_m=0.06):
    """Centre (table metres) of the black circle-centre dot nearest near_xy, or None."""
    c, r = grid.table_to_pixel(near_xy)[0]
    half = int(round(search_m * grid.ppm_x))
    c0, r0 = max(0, int(c) - half), max(0, int(r) - half)
    c1, r1 = min(grid.cols, int(c) + half + 1), min(grid.rows, int(r) + half + 1)
    gray = cv2.cvtColor(table_img[r0:r1, c0:c1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    if gray.size == 0:
        return None
    darkness = np.clip(np.median(gray) - gray, 0, None)
    mask = (darkness > 45).astype(np.uint8)
    k = max(3, int(round(0.006 * grid.ppm_x)) | 1)  # removes the thin printed lines, keeps the dot
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    area_lo = np.pi * (0.003 * grid.ppm_x) ** 2
    area_hi = np.pi * (0.02 * grid.ppm_x) ** 2
    best = None
    for i in range(1, n):
        if not area_lo <= stats[i, cv2.CC_STAT_AREA] <= area_hi:
            continue
        ys, xs = np.nonzero(labels == i)
        w = darkness[ys, xs]
        px = np.array([c0 + (xs * w).sum() / w.sum(), r0 + (ys * w).sum() / w.sum()])
        dist = np.hypot(px[0] - c, px[1] - r)
        if best is None or dist < best[0]:
            best = (dist, px)
    if best is None:
        return None
    return grid.pixel_to_table(best[1][0], best[1][1])[0]


def fit_alignment(dots, pucks):
    """2x3 affine (table metres) taking dot positions onto reported puck positions."""
    d, p = np.asarray(dots, float), np.asarray(pucks, float)
    if len(d) == 1:
        return np.hstack([np.eye(2), (p - d).T]), "translation"
    if len(d) == 2:
        # least-squares similarity p = a * d + b over complex numbers
        dc, pc = d[:, 0] + 1j * d[:, 1], p[:, 0] + 1j * p[:, 1]
        dm, pm = dc.mean(), pc.mean()
        a = np.vdot(dc - dm, pc - pm) / np.vdot(dc - dm, dc - dm)
        b = pm - a * dm
        return np.array([[a.real, -a.imag, b.real], [a.imag, a.real, b.imag]]), "similarity"
    design = np.column_stack([d, np.ones(len(d))])
    m, *_ = np.linalg.lstsq(design, p, rcond=None)
    return m.T, "affine"


def describe_alignment(m):
    lin = m[:, :2]
    u, s, vt = np.linalg.svd(lin)
    rot = u @ vt
    return {
        "shift_mm": [round(float(v) * 1000, 1) for v in m[:, 2]],
        "scale_x": round(float(np.linalg.norm(lin[:, 0])), 4),
        "scale_y": round(float(np.linalg.norm(lin[:, 1])), 4),
        "rotation_deg": round(float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0]))), 3),
    }


# ------------------------------------------------------------------ sanity images

def draw_check(table_img, grid, pucks, dots):
    img = table_img.copy()
    rad = int(round(PUCK_RADIUS * grid.ppm_x))
    for i, p in enumerate(pucks):
        c, r = np.round(grid.table_to_pixel(p)[0]).astype(int)
        cv2.circle(img, (c, r), rad, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.drawMarker(img, (c, r), (0, 0, 255), cv2.MARKER_CROSS, 10, 1)
        cv2.putText(img, str(i), (c + rad + 2, r), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        if dots[i] is not None:
            dc, dr = np.round(grid.table_to_pixel(dots[i])[0]).astype(int)
            cv2.drawMarker(img, (dc, dr), (255, 128, 0), cv2.MARKER_TILTED_CROSS, 8, 1)
    return img


def zoom_strip(img, grid, pucks, half_m=0.08, scale=4):
    tiles = []
    half = int(round(half_m * grid.ppm_x))
    padded = cv2.copyMakeBorder(img, half, half, half, half, cv2.BORDER_CONSTANT)
    for p in pucks:
        c, r = np.round(grid.table_to_pixel(p)[0]).astype(int)
        tile = padded[r:r + 2 * half + 1, c:c + 2 * half + 1]
        tiles.append(cv2.resize(tile, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST))
    return np.hstack(tiles) if tiles else None


def sim_view_check(table_png, pucks):
    """Draw the reported pucks with the real-robot sim view renderer on the new image."""
    from scripts.visualization.visualize_real_trajectory import RealTrajectoryRenderer

    renderer = RealTrajectoryRenderer(
        robot_x_offset=TABLE_CENTER_OFFSET_X, robot_y_offset=TABLE_CENTER_OFFSET_Y,
        paddle_input_frame="table", quiet=True,
    )
    table = cv2.imread(str(table_png))
    table = cv2.rotate(table, cv2.ROTATE_90_CLOCKWISE)
    frame = cv2.resize(table, (renderer.render_length, renderer.render_width))
    rad = int(round(PUCK_RADIUS * renderer.ppm))
    for p in pucks:
        px = tuple(int(v) for v in renderer.table_position_to_pixel_coords(p[0], p[1]))
        cv2.circle(frame, px, rad, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.drawMarker(frame, px, (0, 0, 255), cv2.MARKER_CROSS, 8, 1)
    return frame


# --------------------------------------------------------------------------- main

def collect(args, session):
    """Capture blank + puck frames from the camera; returns (blank, [puck frames])."""
    cap = open_camera(args.camera_index)
    gui = not args.no_gui
    grid = TableGrid(args.ppm)
    try:
        while True:
            print("\n=== Step 1: blank table ===")
            print("Move the robot arm (and anything else) off the table and out of the camera view.")
            if ask("Press Enter to take the blank-table picture (q to quit): ") == "q":
                raise SystemExit("aborted")
            blank = grab_median(cap, args.frames)
            show("blank table (rectified)", grid.render(blank, args.mimg_matrix), gui)
            if ask("Keep this picture? [Y/n]: ") != "n":
                break
        cv2.imwrite(str(session / "raw_blank.png"), blank)

        pucks = []
        while True:
            print(f"\n=== Step 2: puck check {len(pucks) + 1} ===")
            print("Place the red puck on the CENTRE DOT of one of the big circles and step out of view.")
            reply = ask("Press Enter to capture, d when done" + (" (at least one needed)" if not pucks else "") + ": ")
            if reply == "d":
                if pucks:
                    break
                print("Need at least one puck check.")
                continue
            if reply == "q":
                raise SystemExit("aborted")
            frame = grab_median(cap, args.frames)
            if detect_puck_table_xy(frame, args.mimg_matrix, args.detector) is None:
                print("The detector does not see the puck (occluded). Adjust and try again.")
                continue
            path = session / f"raw_puck_{len(pucks)}.png"
            cv2.imwrite(str(path), frame)
            pucks.append(frame)
            print(f"Saved {path.name}. Move the puck to another circle for a better fit, or type d.")
        return blank, pucks
    finally:
        cap.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--camera-index", type=int, help="the camera the rollout config uses (camera_index)")
    src.add_argument("--from-session", type=Path, help="rebuild from a saved session directory (no camera)")
    ap.add_argument("--mimg", type=Path, default=REPO_ROOT / "assets" / "real" / "Mimg.npy",
                    help="homography the runtime loads (control_parameters.py: assets/real/Mimg.npy)")
    ap.add_argument("--detector", choices=sorted(DETECTORS), default="red_puck_antiglare")
    ap.add_argument("--ppm", type=float, default=500.0, help="output pixels per metre")
    ap.add_argument("--frames", type=int, default=30, help="frames per capture (median)")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "assets" / REAL_TABLE_IMAGE)
    ap.add_argument("--max-correction-cm", type=float, default=4.0)
    ap.add_argument("--no-align", action="store_true", help="save the rectified picture without the dot->puck warp")
    ap.add_argument("--no-gui", action="store_true")
    args = ap.parse_args()

    if not args.mimg.exists():
        candidates = sorted(str(p.relative_to(REPO_ROOT)) for p in (REPO_ROOT / "assets" / "real").glob("**/Mimg*.npy"))
        raise SystemExit(f"{args.mimg} not found. Pass the homography the runtime uses with --mimg, e.g.\n  "
                         + "\n  ".join(candidates))
    args.mimg_matrix = np.load(args.mimg).astype(np.float64)
    grid = TableGrid(args.ppm)

    if args.from_session:
        session = args.from_session
        blank = cv2.imread(str(session / "raw_blank.png"))
        puck_frames = [cv2.imread(str(p)) for p in sorted(session.glob("raw_puck_*.png"))]
        if blank is None or not puck_frames:
            raise SystemExit(f"{session} needs raw_blank.png and raw_puck_*.png")
    else:
        session = REPO_ROOT / "temp" / "real_table_image" / datetime.now().strftime("%Y%m%d_%H%M%S")
        session.mkdir(parents=True, exist_ok=True)
        print(f"Session directory: {session}")
        blank, puck_frames = collect(args, session)

    hidden_frac, _ = canvas_coverage(args.mimg_matrix)
    if hidden_frac > 0.001:
        print(f"\nNote: {hidden_frac * 100:.1f} % of the table is outside what the runtime detector can see "
              "(off the camera or off the 1920x1440 homography canvas); the picture is still drawn there.")

    unaligned = grid.render(blank, args.mimg_matrix)
    cv2.imwrite(str(session / "table_unaligned.png"), unaligned)

    print("\n=== Puck vs circle-centre dot (unaligned picture) ===")
    pucks, dots = [], []
    for i, frame in enumerate(puck_frames):
        p = detect_puck_table_xy(frame, args.mimg_matrix, args.detector)
        if p is None:
            print(f"  check {i}: puck not detected, skipped")
            continue
        d = find_dot(unaligned, grid, p)
        pucks.append(p)
        dots.append(d)
        if d is None:
            print(f"  check {i}: puck at table ({p[0]:+.4f}, {p[1]:+.4f}) m, no dot found within 6 cm")
        else:
            e = (p - d) * 1000
            print(f"  check {i}: puck ({p[0]:+.4f}, {p[1]:+.4f})  dot ({d[0]:+.4f}, {d[1]:+.4f})  "
                  f"puck - dot = ({e[0]:+.1f}, {e[1]:+.1f}) mm, |{np.hypot(*e):.1f}| mm")

    pairs = [(d, p) for d, p in zip(dots, pucks) if d is not None]
    align, kind, summary = None, None, None
    too_big = [np.hypot(*(p - d)) for d, p in pairs if np.hypot(*(p - d)) > args.max_correction_cm / 100]
    if args.no_align:
        print("\n--no-align: saving the rectified picture as is.")
    elif not pairs:
        print("\nNo dot found next to any puck: saving the rectified picture without alignment.")
    elif too_big:
        print(f"\nA puck is {max(too_big) * 100:.1f} cm from its dot (> --max-correction-cm "
              f"{args.max_correction_cm}): probably not on a centre dot, or a detection problem. "
              "Saving the rectified picture without alignment.")
    else:
        align, kind = fit_alignment([d for d, _ in pairs], [p for _, p in pairs])
        summary = describe_alignment(align)
        print(f"\nAlignment ({kind}, {len(pairs)} check(s)): {summary}")

    final = grid.render(blank, args.mimg_matrix, align) if align is not None else unaligned

    print("\n=== Check on the saved picture ===")
    final_dots = [find_dot(final, grid, p) for p in pucks]
    residuals = []
    for i, (p, d) in enumerate(zip(pucks, final_dots)):
        if d is None:
            print(f"  check {i}: dot not found")
            continue
        e = (p - d) * 1000
        residuals.append(float(np.hypot(*e)))
        print(f"  check {i}: reported puck - dot = ({e[0]:+.1f}, {e[1]:+.1f}) mm, |{residuals[-1]:.1f}| mm")
    if kind == "translation":
        print("  (one check fixes the shift only; the residual is 0 by construction. "
              "Add checks on the other circles to test the rest of the table.)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), final)
    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "session": str(session),
        "mimg": str(args.mimg),
        "mimg_matrix": args.mimg_matrix.tolist(),
        "table_center_offset": [TABLE_CENTER_OFFSET_X, TABLE_CENTER_OFFSET_Y],
        "table_length_m": TABLE_LENGTH,
        "table_width_m": TABLE_WIDTH,
        "image_rows_cols": [grid.rows, grid.cols],
        "layout": "row 0 = far wall (table x = -L/2), last row = robot wall; column 0 = table y = -W/2",
        "detector": args.detector,
        "checks": [{"puck_table_xy": p.tolist(), "dot_unaligned_xy": None if d is None else d.tolist()}
                   for p, d in zip(pucks, dots)],
        "alignment": None if align is None else {"kind": kind, "matrix": align.tolist(), **summary},
        "residual_mm": residuals,
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    check = draw_check(final, grid, pucks, final_dots)
    cv2.imwrite(str(session / "check_table.png"), check)
    strip = zoom_strip(check, grid, pucks)
    if strip is not None:
        cv2.imwrite(str(session / "check_zoom.png"), strip)
    sim_view = sim_view_check(args.out, pucks)
    cv2.imwrite(str(session / "check_sim_view.png"), sim_view)
    print(f"\nSaved {args.out} (+ .json)")
    print(f"Sanity images in {session}: check_table.png, check_zoom.png (red = reported puck, "
          "blue x = dot), check_sim_view.png (sim-view renderer)")
    if not args.no_gui and strip is not None:
        cv2.imshow("reported puck (red) on the new table image", strip)
        cv2.imshow("sim view", sim_view)
        print("Press any key in an image window to finish.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

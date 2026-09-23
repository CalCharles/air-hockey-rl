#!/usr/bin/env python3
"""
Side by side: raw camera (left) | Box2D scene of the observed positions (right).

The right panel is a Box2D env (`--sim-config`) whose puck and paddle bodies are placed
at the positions the policy observes: puck = camera -> `Mimg` homography -> robot meters
-> + center_offset_constant, paddle = UR5 TCP pose + x_offset. No physics step; the
scene is drawn with `AirHockeyRenderer`. Both panels are vertical, robot end at the bottom.

Compare each panel against its own table: when the real puck touches a rail / the
paddle / crosses the painted centre line, the Box2D puck should do the same. The Box2D
panel prints the puck's gap to the nearest wall and to the paddle surface (red when < 1 cm).

Two modes:

  recorded (default)  `.hdf5` trials -> GIFs in analyze_data/results/gif/raw_vs_box2d/<session>/.
                      Uses the recorded `observations` (what the policy received);
                      `train_img[t]` is paired with `observations[t + 1]`.
  --live              camera + optional robot, shown in a window (q / Esc to quit).
                      Runs the live stack's own chain per frame: `homography_transform`
                      (assets/real/Mimg.npy) -> `find_red_hockey_puck_antiglare` with the
                      real env's default bounds -> + center_offset. The paddle is read with
                      the read-only RTDE receive interface (never commands the robot);
                      `--no-robot` skips it. Stop any other process holding the camera first.

Usage:
    python analyze_data/raw_vs_box2d.py data/robot_data_collection_puck_collision_change_angle_20260910_1818 --every 6
    python analyze_data/raw_vs_box2d.py --live                        # camera 0 + robot 172.22.22.2
    python analyze_data/raw_vs_box2d.py --live --no-robot --center-offset 1.302
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from airhockey import AirHockeyEnv  # noqa: E402
from airhockey.renderers.render import AirHockeyRenderer  # noqa: E402

PANEL_H = 720
OFF_TABLE = (10.0, 10.0)
WHITE, RED, BLACK = (255, 255, 255), (0, 0, 255), (0, 0, 0)

# Real env defaults (airhockey/sims/air_hockey_real.py): puck_detector red_puck_antiglare + bounds.
ANTIGLARE_KWARGS = dict(antiglare_bounds_in_raw_image=True, antiglare_min_x_px=290, antiglare_max_x_px=451,
                        antiglare_min_y_px=186, antiglare_max_y_px=465)


class SideBySide:
    def __init__(self, sim_config: Path):
        cfg = yaml.safe_load(sim_config.read_text())["air_hockey"]
        sp = cfg["simulator_params"]
        for key in ("puck_noise", "enable_random_occlusions", "enable_observation_delay",
                    "enable_action_delay", "enable_puck_delay_interpolation"):
            sp[key] = False
        self.env = AirHockeyEnv(copy.deepcopy(cfg))
        self.env.reset(seed=0)
        self.sim = self.env.simulator
        self.renderer = AirHockeyRenderer(self.env, orientation="horizontal", show_target_position=False)

    def render_box2d(self, paddle_xy, puck_xy):
        for body, xy in ((self.sim.paddles["paddle_ego"], paddle_xy), (next(iter(self.sim.pucks.values())), puck_xy)):
            bx, by = self.sim.base_coord_to_box2d(xy if xy is not None else OFF_TABLE)
            body.position = (float(bx), float(by))
            body.linearVelocity = (0.0, 0.0)
        self.env.current_state = self.sim.get_current_state()  # AirHockeyRenderer draws from this snapshot
        # Horizontal render has the robot end on the left; 90 deg counter-clockwise puts it at the bottom.
        return cv2.rotate(self.renderer.get_frame(), cv2.ROTATE_90_COUNTERCLOCKWISE)

    def compose(self, raw_bgr, paddle_xy, puck_xy, title, subtitle=""):
        """raw_bgr: camera frame as the stack saves it (rotated 180 deg, robot end on the right)."""
        half_l, half_w = self.sim.length / 2, self.sim.width / 2
        rp, rk = self.sim.puck_radius, self.sim.paddle_radius
        lines = [(title, WHITE)]
        if puck_xy is not None:
            wall_gap = min(half_l - abs(puck_xy[0]), half_w - abs(puck_xy[1])) - rp
            lines.append((f"puck obs ({puck_xy[0]:+.3f}, {puck_xy[1]:+.3f})", WHITE))
            lines.append((f"wall gap {wall_gap * 100:+.1f} cm", RED if wall_gap < 0.01 else WHITE))
            if paddle_xy is not None:
                paddle_gap = np.linalg.norm(np.asarray(puck_xy) - np.asarray(paddle_xy)) - rp - rk
                lines.append((f"puck-paddle gap {paddle_gap * 100:+.1f} cm", RED if paddle_gap < 0.01 else WHITE))
        else:
            lines.append(("puck not detected", WHITE))
        if paddle_xy is None:
            lines.append(("paddle unavailable", WHITE))
        right = fit_height(self.render_box2d(paddle_xy, puck_xy))
        put_lines(right, lines)
        left = fit_height(cv2.rotate(raw_bgr, cv2.ROTATE_90_CLOCKWISE))
        put_lines(left, [("camera", WHITE), (subtitle, WHITE)])
        return np.hstack([left, right])


def put_lines(img, lines, origin=(10, 22)):
    for i, (text, color) in enumerate(lines):
        y = origin[1] + 22 * i
        cv2.putText(img, text, (origin[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, BLACK, 3, cv2.LINE_AA)
        cv2.putText(img, text, (origin[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def fit_height(img, h=PANEL_H):
    return cv2.resize(img, (int(round(img.shape[1] * h / img.shape[0])), h), interpolation=cv2.INTER_LINEAR)


def run_recorded(args, view: SideBySide):
    files = sorted(args.data_dir.glob("*.hdf5"))[:: args.every]
    if args.limit is not None:
        files = files[: args.limit]
    name = args.data_dir.resolve().name
    if args.center_offset is not None:
        name += f"_offset{args.center_offset:g}"
    out_root = args.out_dir / name
    print(f"{len(files)} trials -> {out_root}")
    for path in files:
        with h5py.File(path, "r") as h:
            imgs = h["train_img"][:]
            obs = h["observations"][:].copy()
            recorded_offset = float(h.attrs.get("center_offset_constant", 1.2))
        if args.center_offset is not None:
            obs[:, [12, 27]] += args.center_offset - recorded_offset
        frames = []
        for t, raw in enumerate(imgs):
            o = obs[t + 1]
            paddle = o[12:14] if o[14] == 0 else None
            puck = o[27:29] if o[29] == 0 else None
            frame = view.compose(raw, paddle, puck, f"Box2D  step {t}/{len(imgs) - 1}", path.stem)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        out_path = out_root / f"{path.stem}.gif"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        images = [Image.fromarray(f).quantize(colors=256, method=Image.Quantize.FASTOCTREE,
                                              dither=Image.Dither.NONE) for f in frames]
        images[0].save(out_path, save_all=True, append_images=images[1:], loop=0, duration=int(1000 / args.fps))
        print(f"  {path.name}: {len(frames)} frames")


def run_live(args, view: SideBySide):
    from airhockey.sims.real.control_parameters import homography_transform
    from airhockey.sims.real.image_detection import find_red_hockey_puck_antiglare

    center_offset = 1.2 if args.center_offset is None else args.center_offset
    x_offset = center_offset  # the real env's paddle offset; both default to 1.2

    rcv = None
    if not args.no_robot:
        try:
            from rtde_receive import RTDEReceiveInterface
            rcv = RTDEReceiveInterface(args.robot_host)
            print(f"reading TCP pose from {args.robot_host} (receive interface only)")
        except Exception as exc:
            print(f"robot unavailable ({exc}); continuing without the paddle")

    cap = cv2.VideoCapture(int(args.camera_index), cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera_index} (is another process using it?)")
    print("q / Esc to quit")

    puck_history = [(-2 + center_offset, 0.0, 1)] * 5
    last = time.time()
    try:
        while True:
            ok, image = cap.read()
            if not ok or image is None or image.size == 0:
                time.sleep(0.01)
                continue
            showdst, raw = homography_transform(image, get_save=True)
            puck = np.array(find_red_hockey_puck_antiglare(showdst, puck_history, rotate=False,
                                                           center_offset_constant=center_offset,
                                                           **ANTIGLARE_KWARGS), dtype=float)
            visible = int(puck[2]) == 0
            if visible:
                puck[0] += center_offset
            puck_history = puck_history[1:] + [tuple(puck)]

            paddle = None
            if rcv is not None:
                try:
                    pose = rcv.getActualTCPPose()
                    paddle = np.array([pose[0] + x_offset, pose[1]])
                except Exception:
                    paddle = None

            now = time.time()
            fps = 1.0 / max(now - last, 1e-6)
            last = now
            frame = view.compose(raw, paddle, puck[:2] if visible else None, f"Box2D  live  {fps:4.1f} fps",
                                 f"center_offset {center_offset:g}")
            cv2.imshow("raw vs Box2D", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if rcv is not None:
            rcv.disconnect()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir", type=Path, nargs="?", help="session folder of .hdf5 trials (recorded mode)")
    ap.add_argument("--live", action="store_true", help="live camera (+ robot) instead of recorded trials")
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--robot-host", default="172.22.22.2")
    ap.add_argument("--no-robot", action="store_true", help="live mode without reading the paddle pose")
    ap.add_argument("--sim-config", type=Path, default=REPO_ROOT / "configs/new_juggle/sysid_v2_hist2.yaml")
    ap.add_argument("--center-offset", type=float, default=None,
                    help="center_offset_constant (and x_offset) to express observations with; "
                         "recorded mode default = the value stored in the trial, live default = 1.2")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "analyze_data/results/gif/raw_vs_box2d")
    ap.add_argument("--limit", type=int, default=None, help="recorded mode: render only the first N trials")
    ap.add_argument("--every", type=int, default=1, help="recorded mode: take every k-th trial")
    ap.add_argument("--fps", type=float, default=10.0, help="recorded mode: GIF fps (real steps are ~40 ms)")
    args = ap.parse_args()

    view = SideBySide(args.sim_config)
    if args.live:
        run_live(args, view)
    elif args.data_dir is None:
        ap.error("give a session folder, or --live")
    else:
        run_recorded(args, view)


if __name__ == "__main__":
    main()

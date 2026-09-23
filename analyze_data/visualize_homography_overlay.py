#!/usr/bin/env python3
"""
Overlay the Box2D sim, fed with the observed real positions, on homography-warped real camera frames.

For each `.hdf5` trial (robot data-collection schema: `train_img`, `train_vals`,
`observations`), every saved raw camera frame is warped exactly like the live
stack does (`airhockey/sims/real/control_parameters.py:homography_transform`:
upscale x3 -> `warpPerspective(Mimg)` -> downscale /2, 1 px = 2 mm). Then:

  Box2D scene        the paddle and puck bodies of a Box2D env (`--sim-config`) are
                     placed at the observed positions (no physics step), the scene is
                     drawn with `AirHockeyRenderer` and blended over the real frame.
                     The render is scaled to 2 mm / px and placed so that its halfway
                     line (obs x = 0) lies on the real table's painted centre line
                     (detected per trial, or `--real-centre-u`); obs y = 0 stays at
                     robot y = 0.
  orange / yellow    outline of the Box2D puck / paddle at the positions Box2D reports
                     after placement (read back from the sim state), drawn 4 px wider
                     so they stay visible when they coincide with green / blue
  green / blue       the same observed puck / paddle positions mapped the way the real
                     stack maps them (robot frame -> warped pixels via `offset_constants`),
                     i.e. where the observation says the object is on the real table

With the halfway lines aligned, the gap between green / blue and orange / yellow is
how far the sim's view of the scene is from the real one along the table. The puck
(camera) and paddle (kinematics) are shown only when their valid flag is 0.

`train_img[t]` is paired with `observations[t + 1]` (the observation written
after step t; it equals `train_vals[t]` for the paddle).

Output: `analyze_data/results/gif/<data folder name>/<trial>.gif`.

Usage:
    python analyze_data/visualize_homography_overlay.py \
        data/robot_data_collection_puck_collision_change_angle_20260910_1818 --limit 5
"""

from __future__ import annotations

import argparse
import copy
import sys
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
from airhockey.sims.real.overlay_utils import robot_to_display_pixel  # noqa: E402

UPSCALE = 3
ORIGINAL_SIZE = (640, 480)
VISUAL_DOWNSCALE = 2
OFFSET_CONSTANTS = (2250.0, 500.0)
PX_PER_M = 1000.0 / VISUAL_DOWNSCALE
OFF_TABLE = (10.0, 10.0)

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
ORANGE = (0, 140, 255)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def warp_frame(raw_bgr: np.ndarray, mimg: np.ndarray) -> np.ndarray:
    w, h = ORIGINAL_SIZE[0] * UPSCALE, ORIGINAL_SIZE[1] * UPSCALE
    up = cv2.resize(raw_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    dst = cv2.warpPerspective(up, mimg, (w, h))
    return cv2.resize(dst, (w // VISUAL_DOWNSCALE, h // VISUAL_DOWNSCALE), interpolation=cv2.INTER_AREA)


def real_px(x_obs: float, y_obs: float, center_offset: float) -> np.ndarray:
    """Observation -> warped pixel the way the real stack maps it (robot frame + offset_constants)."""
    return np.asarray(robot_to_display_pixel(x_obs - center_offset, y_obs, offset_constants=OFFSET_CONSTANTS,
                                             visual_downscale_constant=VISUAL_DOWNSCALE))


def detect_centre_line_u(warped_frames) -> float:
    """Column of the painted (blue) halfway line in the warped frames, median over frames."""
    us = []
    for f in warped_frames:
        roi = f[60:440, 300:700].astype(np.float32)
        blueness = (roi[..., 0] - roi[..., 2]).mean(axis=0)
        us.append(300 + int(np.argmax(blueness)))
    return float(np.median(us))


class Box2DScene:
    """Box2D env whose puck / paddle are teleported to observed positions and rendered into the warped frame."""

    def __init__(self, sim_config: Path, centre_u: float, frame_hw):
        cfg = yaml.safe_load(sim_config.read_text())["air_hockey"]
        sp = cfg["simulator_params"]
        for key in ("puck_noise", "enable_random_occlusions", "enable_observation_delay",
                    "enable_action_delay", "enable_puck_delay_interpolation"):
            sp[key] = False
        sp["render_size"] = int(round(float(sp["width"]) * PX_PER_M))
        self.env = AirHockeyEnv(copy.deepcopy(cfg))
        self.env.reset(seed=0)
        self.sim = self.env.simulator
        self.renderer = AirHockeyRenderer(self.env, orientation="horizontal", show_target_position=False)
        self.frame_hw = frame_hw
        self.centre_u = centre_u
        self.v0 = OFFSET_CONSTANTS[1] / VISUAL_DOWNSCALE

        world = [(0.0, 0.0), (0.5, 0.0), (0.0, 0.3)]
        src = np.float32([self.renderer.world_xy_to_output_pixel(x, y) for x, y in world])
        dst = np.float32([self.sim_px(x, y) for x, y in world])
        self.affine = cv2.getAffineTransform(src, dst)
        blank = np.full(self.renderer.get_frame().shape[:2], 255, np.uint8)
        self.mask = cv2.warpAffine(blank, self.affine, (frame_hw[1], frame_hw[0])) > 0

    def sim_px(self, x_obs: float, y_obs: float) -> np.ndarray:
        """Box2D (obs-frame) position -> warped pixel with the halfway lines aligned."""
        return np.array([self.centre_u + x_obs * PX_PER_M, self.v0 - y_obs * PX_PER_M])

    def place(self, paddle_xy, puck_xy):
        for body, xy in ((self.sim.paddles["paddle_ego"], paddle_xy), (next(iter(self.sim.pucks.values())), puck_xy)):
            bx, by = self.sim.base_coord_to_box2d(xy if xy is not None else OFF_TABLE)
            body.position = (float(bx), float(by))
            body.linearVelocity = (0.0, 0.0)
            body.angularVelocity = 0.0
        state = self.sim.get_current_state()
        self.env.current_state = state  # AirHockeyRenderer draws from this snapshot, not from the bodies
        return np.asarray(state["paddles"]["paddle_ego"]["position"]), np.asarray(state["pucks"][0]["position"])

    def blend(self, frame, alpha: float):
        render = self.renderer.get_frame()
        warped = cv2.warpAffine(render, self.affine, (self.frame_hw[1], self.frame_hw[0]), flags=cv2.INTER_LINEAR)
        m = self.mask
        frame[m] = np.round(frame[m] * (1 - alpha) + warped[m] * alpha).astype(np.uint8)
        return frame


def draw_marker(frame, center, radius_m, color, thickness=2, pad_px=0):
    c = tuple(int(round(v)) for v in center)
    cv2.circle(frame, c, max(1, int(round(radius_m * PX_PER_M)) + pad_px), color, thickness)
    cv2.drawMarker(frame, c, color, cv2.MARKER_CROSS, 10, 1)


def draw_legend(frame, lines):
    items = [("paddle obs, real mapping (kinematics)", BLUE), ("puck obs, real mapping (camera)", GREEN),
             ("Box2D paddle at the same obs", YELLOW), ("Box2D puck at the same obs", ORANGE)]
    h = 20 * (len(items) + len(lines)) + 12
    y = frame.shape[0] - h
    cv2.rectangle(frame, (5, y - 6), (400, frame.shape[0] - 5), BLACK, -1)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (10, y + 12 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)
    y += 20 * len(lines)
    for i, (label, color) in enumerate(items):
        yy = y + 14 + 20 * i
        cv2.line(frame, (12, yy - 4), (40, yy - 4), color, 3)
        cv2.putText(frame, label, (48, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)


def render_trial(path: Path, out_path: Path, mimg, args) -> dict:
    with h5py.File(path, "r") as h:
        imgs = h["train_img"][:]
        obs = h["observations"][:]
        center_offset = float(h.attrs.get("center_offset_constant", 1.2))
    if args.center_offset is not None:
        # obs x = robot x + center_offset for both the kinematic paddle and the camera puck.
        obs = obs.copy()
        obs[:, 12] += args.center_offset - center_offset
        obs[:, 27] += args.center_offset - center_offset
        center_offset = args.center_offset
    warped = [warp_frame(im, mimg) for im in imgs]
    centre_u = args.real_centre_u if args.real_centre_u is not None else detect_centre_line_u(warped)
    scene = Box2DScene(args.sim_config, centre_u, warped[0].shape[:2])
    rp, rk = scene.sim.puck_radius, scene.sim.paddle_radius
    shift_cm = (scene.sim_px(0, 0) - real_px(0, 0, center_offset)) * 1000.0 / PX_PER_M / 10.0

    frames = []
    for t, frame in enumerate(warped):
        o = obs[t + 1]
        paddle = o[12:14] if o[14] == 0 else None
        puck = o[27:29] if o[29] == 0 else None
        sim_paddle, sim_puck = scene.place(paddle, puck)
        scene.blend(frame, args.alpha)
        if paddle is not None:
            draw_marker(frame, real_px(*paddle, center_offset), rk, BLUE)
            draw_marker(frame, scene.sim_px(*sim_paddle), rk, YELLOW, pad_px=4)
        if puck is not None:
            draw_marker(frame, real_px(*puck, center_offset), rp, GREEN)
            draw_marker(frame, scene.sim_px(*sim_puck), rp, ORANGE, pad_px=4)
        draw_legend(frame, [f"{path.stem}  step {t}/{len(warped) - 1}",
                            f"Box2D - real mapping: dx {shift_cm[0]:+.1f} cm, dy {shift_cm[1]:+.1f} cm "
                            f"(centre line u={centre_u:.0f})"])
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if args.width and args.width != rgb.shape[1]:
            rgb = cv2.resize(rgb, (args.width, int(round(rgb.shape[0] * args.width / rgb.shape[1]))),
                             interpolation=cv2.INTER_AREA)
        frames.append(rgb)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_gif(frames, out_path, args.fps)
    return {"frames": len(frames), "centre_u": centre_u, "shift_cm": shift_cm}


def save_gif(frames_rgb, out_path: Path, fps: float):
    # Per-frame adaptive palettes wash out the thin overlay lines; use one shared palette
    # with the overlay colors reserved exactly.
    reserved = [BLUE, GREEN, ORANGE, YELLOW, WHITE, BLACK]
    reserved_rgb = [c[::-1] for c in reserved]
    sample = np.concatenate(frames_rgb[:: max(1, len(frames_rgb) // 8)], axis=0)
    base = Image.fromarray(sample).quantize(colors=256 - len(reserved_rgb), method=Image.Quantize.FASTOCTREE)
    pal = base.getpalette()[: 3 * (256 - len(reserved_rgb))]
    for c in reserved_rgb:
        pal.extend(c)
    pal_img = Image.new("P", (1, 1))
    pal_img.putpalette(pal)
    images = [Image.fromarray(f).quantize(palette=pal_img, dither=Image.Dither.NONE) for f in frames_rgb]
    images[0].save(out_path, save_all=True, append_images=images[1:], loop=0,
                   duration=int(1000 / fps), optimize=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("--sim-config", type=Path, default=REPO_ROOT / "configs/new_juggle/sysid_v2_hist2.yaml")
    ap.add_argument("--mimg", type=Path, default=REPO_ROOT / "assets/real/Mimg.npy")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "analyze_data/results/gif")
    ap.add_argument("--limit", type=int, default=None, help="render only the first N trials")
    ap.add_argument("--every", type=int, default=1, help="take every k-th trial")
    ap.add_argument("--alpha", type=float, default=0.35, help="opacity of the Box2D render over the real frame")
    ap.add_argument("--real-centre-u", type=float, default=None,
                    help="column of the real halfway line in the warped 960 px frame (default: detect per trial)")
    ap.add_argument("--center-offset", type=float, default=None,
                    help="re-express the observations with this center_offset_constant instead of the recorded one "
                         "(output goes to <session>_offset<value>/)")
    ap.add_argument("--fps", type=float, default=10.0, help="real steps are ~40 ms, so 10 fps is ~0.4x speed")
    ap.add_argument("--width", type=int, default=960, help="output GIF width in px (0 = native 960)")
    args = ap.parse_args()

    mimg = np.load(args.mimg)
    files = sorted(args.data_dir.glob("*.hdf5"))[:: args.every]
    if args.limit is not None:
        files = files[: args.limit]
    out_root = args.out_dir / args.data_dir.resolve().name
    if args.center_offset is not None:
        out_root = out_root.with_name(f"{out_root.name}_offset{args.center_offset:g}")
    print(f"{len(files)} trials -> {out_root}  (Mimg {args.mimg}, sim config {args.sim_config})")
    for f in files:
        r = render_trial(f, out_root / f"{f.stem}.gif", mimg, args)
        print(f"  {f.name}: {r['frames']} frames, centre line u={r['centre_u']:.0f}, "
              f"Box2D - real mapping dx {r['shift_cm'][0]:+.1f} cm dy {r['shift_cm'][1]:+.1f} cm")


if __name__ == "__main__":
    main()

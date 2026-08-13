#!/usr/bin/env python3
"""Render before/after MP4s of the staggered multi-puck spawn.

The paddle is held idle, so the clip shows only the reset distribution: the
left panel is the legacy independent per-puck sampling (`multipuck_stagger:
false`), the right panel is the staggered juggle cycle, both on the same seed.

Each puck carries a colored ring, a motion trail (dot spacing is the speed), a
velocity arrow (where it will be 0.4 s from now), and a HUD row with its live
x-velocity and the time it crossed the paddle-reachable boundary.

See notes/docs/environments/multi-puck-spawning.md.

Usage:
    python scripts/visualization/render_multipuck_spawn.py \
        --pucks 2 3 4 --seed 0 -o gifs/multipuck_stagger
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio
import numpy as np
import yaml

from airhockey import AirHockeyEnv
from airhockey.renderers import AirHockeyRenderer

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = _REPO_ROOT / "configs" / "new_juggle" / "sysid_best_params_hist2.yaml"
DEFAULT_OUT = _REPO_ROOT / "gifs" / "multipuck_stagger"

TRAIL_LEN = 12
# RGB, one per puck index.
PUCK_COLORS = [
    (230, 97, 0),    # orange
    (26, 105, 205),  # blue
    (204, 41, 158),  # magenta
    (0, 140, 128),   # teal
    (120, 94, 240),  # violet
    (140, 90, 30),   # brown
]


def make_env(config_path, num_pucks, stagger):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["air_hockey"]
    cfg["num_pucks"] = num_pucks
    cfg["obs_type"] = "multipuck_history"
    cfg["task"] = "puck_juggle_upper_half_reward"
    # Let the clip run its full length: no early termination, no observation
    # noise to blur the overlays.
    cfg["max_timesteps"] = 10_000
    cfg["terminate_on_puck_hit_bottom"] = False
    cfg["terminate_on_puck_pass_paddle"] = False
    cfg["multipuck_stagger"] = stagger
    cfg["simulator_params"]["puck_noise"] = False
    cfg["simulator_params"]["enable_random_occlusions"] = False
    return AirHockeyEnv(cfg)


def true_puck_states(env):
    """Spawned/current (position, velocity) per puck in base coords, from the bodies."""
    return [
        ((-b.position[1], b.position[0]), (-b.linearVelocity[1], b.linearVelocity[0]))
        for b in env.simulator.pucks.values()
    ]


def draw_overlays(frame, renderer, env, trails, arrivals, t_now, label):
    """Rings / trails / velocity arrows on the table, plus a HUD header."""
    reach_x = env._multipuck_reach_x()
    states = true_puck_states(env)

    # Paddle-reachable boundary: what the arrival times are measured to.
    left = renderer.world_xy_to_output_pixel(reach_x, env.table_y_left)
    right = renderer.world_xy_to_output_pixel(reach_x, env.table_y_right)
    overlay = frame.copy()
    cv2.line(overlay, (int(left[0]), int(left[1])), (int(right[0]), int(right[1])),
             (40, 40, 40), 2)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    ring_radius = int(env.puck_radius * renderer.ppm) + 4
    for i, ((x, y), (vx, vy)) in enumerate(states):
        color = PUCK_COLORS[i % len(PUCK_COLORS)]
        px, py = renderer.world_xy_to_output_pixel(x, y)

        trail = trails[i]
        for k, (tx, ty) in enumerate(reversed(trail)):
            fade = 1.0 - (k + 1) / (TRAIL_LEN + 1)
            faded = tuple(int(c * fade + 255 * (1 - fade)) for c in color)
            cv2.circle(frame, (int(tx), int(ty)), max(2, int(4 * fade)), faded, -1, cv2.LINE_AA)
        trail.append((px, py))
        if len(trail) > TRAIL_LEN:
            trail.pop(0)

        cv2.circle(frame, (int(px), int(py)), ring_radius, color, 2, cv2.LINE_AA)
        ahead_x, ahead_y = renderer.world_xy_to_output_pixel(x + 0.4 * vx, y + 0.4 * vy)
        if abs(ahead_x - px) + abs(ahead_y - py) > 4:
            cv2.arrowedLine(frame, (int(px), int(py)), (int(ahead_x), int(ahead_y)),
                            color, 2, cv2.LINE_AA, tipLength=0.25)
        cv2.putText(frame, f"P{i}", (int(px) + ring_radius + 2, int(py) + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    header = np.full((52 + 18 * len(states), frame.shape[1], 3), 255, np.uint8)
    cv2.putText(header, label, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(header, f"t = {t_now:5.2f}s", (8, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(header, "grey line = paddle-reachable boundary",
                (frame.shape[1] - 235, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 90, 90),
                1, cv2.LINE_AA)
    for i, (_, (vx, _)) in enumerate(states):
        color = PUCK_COLORS[i % len(PUCK_COLORS)]
        row = 52 + 18 * i
        cv2.rectangle(header, (8, row - 9), (20, row + 3), color, -1)
        note = (
            f"reached boundary @ {arrivals[i]:.2f}s"
            if arrivals[i] is not None
            else "falling" if vx > 0 else "rising"
        )
        cv2.putText(header, f"P{i}  vx {vx:+.2f} m/s   {note}", (28, row),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1, cv2.LINE_AA)
    return np.concatenate([header, frame], axis=0)


def rollout(config_path, num_pucks, stagger, seed, seconds, fps, label):
    env = make_env(config_path, num_pucks, stagger)
    renderer = AirHockeyRenderer(env, orientation="vertical")
    env.reset(seed=seed)
    reach_x = env._multipuck_reach_x()
    trails = [[] for _ in range(num_pucks)]
    arrivals = [None] * num_pucks
    frames = []
    for step in range(int(seconds * fps)):
        t_now = step / fps
        for i, ((x, _), (vx, _)) in enumerate(true_puck_states(env)):
            if arrivals[i] is None and x >= reach_x and vx > 0:
                arrivals[i] = t_now
        frame = cv2.cvtColor(renderer.get_frame(), cv2.COLOR_BGR2RGB)
        frames.append(draw_overlays(frame, renderer, env, trails, arrivals, t_now, label))
        env.step(np.array([0.0, 0.0], dtype=np.float32))
    return frames


def pad_to_even(frame):
    """ffmpeg needs even dimensions."""
    height, width = frame.shape[:2]
    return cv2.copyMakeBorder(frame, 0, height % 2, 0, width % 2,
                              cv2.BORDER_CONSTANT, value=(255, 255, 255))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pucks", type=int, nargs="+", default=[2, 3, 4],
                        help="puck counts to render, one clip each (default: 2 3 4)")
    parser.add_argument("--seed", type=int, default=0, help="reset seed (default: 0)")
    parser.add_argument("--seconds", type=float, default=9.0,
                        help="clip length; one juggle cycle is ~3 s (default: 9)")
    parser.add_argument("--fps", type=int, default=20,
                        help="playback fps; 20 matches the sim rate, i.e. real time")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help=f"sim config to render (default: {DEFAULT_CONFIG.name})")
    parser.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT,
                        help=f"output directory (default: {DEFAULT_OUT})")
    return parser.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for num_pucks in args.pucks:
        panels = [
            rollout(args.config, num_pucks, stagger, args.seed, args.seconds, args.fps,
                    label=f"{num_pucks} pucks  {name}")
            for stagger, name in ((False, "BEFORE: independent spawns"),
                                  (True, "AFTER: staggered spawns"))
        ]
        divider = np.full((panels[0][0].shape[0], 6, 3), 60, np.uint8)
        pair = [
            pad_to_even(np.concatenate([before, divider, after], axis=1))
            for before, after in zip(*panels)
        ]
        path = args.out / f"{num_pucks}puck_before_vs_after_seed{args.seed}.mp4"
        imageio.mimsave(path, pair, fps=args.fps, quality=9, macro_block_size=1)
        print(f"{path}  {pair[0].shape[1]}x{pair[0].shape[0]}  {len(pair) / args.fps:.1f}s")


if __name__ == "__main__":
    main()

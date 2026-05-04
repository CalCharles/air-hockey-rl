"""Visualize the effect of the Box2D collision randomization options.

Renders paired GIFs (same seed / same heuristic policy) with and without the
per-collision-source randomization flags enabled so the visual effect can
be judged by eye:

  paddle-puck:
    - enable_paddle_puck_strength_randomization (default range [0.5, 1.0])
    - enable_paddle_puck_direction_randomization (default cone = 10 deg per side)
  puck-wall:
    - enable_wall_direction_randomization (default cone = 10 deg per side)
    (no wall-strength knob by design)

Defaults match the canonical Box2D config used elsewhere in the repo.
"""

import argparse
import copy
import os
from pathlib import Path

import cv2
import imageio
import numpy as np
import yaml

from airhockey import AirHockeyEnv
from airhockey.renderers import AirHockeyRenderer


DEFAULT_CONFIG = (
    "scripts/smooth_policy/amp_history/configs/new_juggle/"
    "pid_noise_constant_upper_half_custom_sim_params.yaml"
)


def _annotate(frame_rgb, label, reward, cum_reward):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    color = (0, 0, 0)
    thickness = 1
    cv2.putText(frame_rgb, label, (5, 15), font, font_scale, color, thickness)
    cv2.putText(
        frame_rgb,
        f"r {reward:+.2f}",
        (frame_rgb.shape[1] - 70, 15),
        font,
        font_scale,
        color,
        thickness,
    )
    cv2.putText(
        frame_rgb,
        f"R {cum_reward:+.2f}",
        (frame_rgb.shape[1] - 70, 30),
        font,
        font_scale,
        color,
        thickness,
    )
    return frame_rgb


def _resize(frame_rgb, width=160):
    aspect_ratio = frame_rgb.shape[1] / frame_rgb.shape[0]
    return cv2.resize(frame_rgb, (width, int(width / aspect_ratio)))


def _heuristic_action(obs):
    """Tracks the puck's x so gravity pulls it into the paddle repeatedly."""
    paddle_x = float(obs[12])
    puck_x = float(obs[27])
    ax = np.clip((puck_x - paddle_x) * 6.0, -1.0, 1.0)
    return np.array([ax, 0.0], dtype=np.float32)


def run_and_save(env, renderer, label, gif_path, seed, max_steps):
    obs, _ = env.reset(seed=seed)
    frames = []
    reward = 0.0
    cum_reward = 0.0
    for _ in range(max_steps):
        bgr = renderer.get_frame()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = _resize(rgb, width=160)
        rgb = _annotate(rgb, label, reward, cum_reward)
        frames.append(rgb)
        action = _heuristic_action(obs)
        obs, reward, term, trunc, _ = env.step(action)
        cum_reward += float(reward)
        if term or trunc:
            break
    imageio.mimsave(
        gif_path,
        frames,
        format="GIF",
        loop=0,
        duration=int(1000 / 20),
    )
    stats = env.simulator.get_episode_collision_stats()
    paddle_counts = {k: v["count"] for k, v in stats["paddle"].items()}
    wall_counts = {k: v["count"] for k, v in stats["wall"].items()}
    return paddle_counts, wall_counts, len(frames)


def build_params(config_path, enable_rand, strength_range, cone_deg):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    params = copy.deepcopy(cfg["air_hockey"])
    # Force puck to spawn near the paddle so the heuristic reliably juggles,
    # and let trajectories run uninterrupted long enough to see many collisions.
    params["puck_spawn_near_paddle_prob"] = 1.0
    params["terminate_on_puck_hit_bottom"] = False
    params["terminate_on_puck_pass_paddle"] = False
    params["terminate_on_enemy_goal"] = False
    sim = params.setdefault("simulator_params", {})
    # Paddle-puck: both strength + direction.
    sim["enable_paddle_puck_strength_randomization"] = bool(enable_rand)
    sim["paddle_puck_strength_range"] = list(strength_range)
    sim["enable_paddle_puck_direction_randomization"] = bool(enable_rand)
    sim["paddle_puck_direction_cone_deg"] = float(cone_deg)
    # Puck-wall: direction only (no strength knob for walls).
    sim["enable_wall_direction_randomization"] = bool(enable_rand)
    sim["wall_direction_cone_deg"] = float(cone_deg)
    return params


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env-config-path", type=str, default=DEFAULT_CONFIG)
    ap.add_argument("--output-dir", type=str, default="runs/collision_randomization_viz")
    ap.add_argument("--n-trajectories", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--strength-min", type=float, default=0.5)
    ap.add_argument("--strength-max", type=float, default=1.0)
    ap.add_argument("--direction-cone-deg", type=float, default=10.0)
    args = ap.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = {
        "off": False,
        "on": True,
    }

    summary_rows = []
    for variant_label, enable_rand in variants.items():
        params = build_params(
            args.env_config_path,
            enable_rand=enable_rand,
            strength_range=(args.strength_min, args.strength_max),
            cone_deg=args.direction_cone_deg,
        )
        env = AirHockeyEnv(params)
        renderer = AirHockeyRenderer(
            env, orientation="vertical", show_target_position=True, show_acceleration_arrow=False
        )
        for traj_idx in range(args.n_trajectories):
            seed = args.base_seed + traj_idx
            label = (
                f"rand={variant_label} seed={seed}"
                if variant_label == "off"
                else f"rand=on[{args.strength_min:.2f},{args.strength_max:.2f}] cone={args.direction_cone_deg:.0f}deg seed={seed}"
            )
            gif_path = out_dir / f"traj_{traj_idx}_rand_{variant_label}.gif"
            paddle_counts, wall_counts, n_frames = run_and_save(
                env, renderer, label, str(gif_path), seed=seed, max_steps=args.max_steps
            )
            summary_rows.append(
                (variant_label, seed, n_frames, paddle_counts, wall_counts, str(gif_path))
            )
            print(f"Saved {gif_path} ({n_frames} frames)  paddle={paddle_counts}  wall={wall_counts}")

    print("\nSummary:")
    for row in summary_rows:
        variant, seed, n_frames, paddle_counts, wall_counts, path = row
        p_total = sum(paddle_counts.values())
        w_total = sum(wall_counts.values())
        print(
            f"  [rand={variant}] seed={seed}  frames={n_frames}  paddle_hits={p_total}  "
            f"wall_hits={w_total}  file={path}"
        )


if __name__ == "__main__":
    main()

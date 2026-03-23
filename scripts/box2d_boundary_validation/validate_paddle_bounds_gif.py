#!/usr/bin/env python3
"""
Validate Box2D paddle boundary alignment with a looped perimeter GIF.

The script:
- Loads simulator defaults from YAML (`air_hockey.simulator_params`).
- Builds a Box2D AirHockey env with caller-provided `paddle_bounds`.
- Applies corner-aware edge limits (`top_abs`, `max_bias_*`) through
  `paddle_edge_bounds` and validates against `get_clip_limits`.
- Drives the paddle around the clipped perimeter for configurable loops.
- Saves one GIF plus a JSON summary, and exits non-zero on validation failure.

Example (raw-robot x limits):
    python scripts/box2d_boundary_validation/validate_paddle_bounds_gif.py \
      --x-min -0.80 --x-max -0.37 --y-min -0.33 --y-max 0.33 \
      --top-abs 0.35 --max-bias-m -0.30 --max-bias-p -0.30 \
      --limits-frame raw_robot \
      --loops 3 --steps-per-edge 48 --fps 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio
import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from airhockey import AirHockeyEnv
from airhockey.renderers import AirHockeyRenderer
from airhockey.sims.real.coordinate_transform import get_clip_limits


EPS = 1e-6


@dataclass
class BoundaryConfig:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    top_abs: float
    bot_abs: float
    max_bias_m: float
    max_bias_p: float

    @property
    def bounds(self) -> list[float]:
        return [self.x_min, self.x_max, self.y_min, self.y_max]

    @property
    def edge_lims(self) -> list[float]:
        # Coordinate transform expects (top_abs, bot_abs, max_bias_m, max_bias_p).
        return [self.top_abs, self.bot_abs, self.max_bias_m, self.max_bias_p]


def _convert_raw_to_centered_limits(cfg: BoundaryConfig, center_offset_constant: float) -> BoundaryConfig:
    """Convert raw-robot x-style limits into centered-x limits used by env state."""
    shift = float(center_offset_constant)
    return BoundaryConfig(
        x_min=cfg.x_min + shift,
        x_max=cfg.x_max + shift,
        y_min=cfg.y_min,
        y_max=cfg.y_max,
        top_abs=cfg.top_abs,
        bot_abs=cfg.bot_abs,
        max_bias_m=cfg.max_bias_m + shift,
        max_bias_p=cfg.max_bias_p + shift,
    )


def _load_simulator_params_from_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config at {config_path}: expected dictionary.")
    air_hockey_cfg = config.get("air_hockey", {})
    if not isinstance(air_hockey_cfg, dict):
        raise ValueError(f"Invalid config at {config_path}: expected 'air_hockey' dictionary.")
    sim_params = air_hockey_cfg.get("simulator_params", {})
    if not isinstance(sim_params, dict):
        raise ValueError(f"Invalid config at {config_path}: expected 'air_hockey.simulator_params' dictionary.")
    return dict(sim_params)


def _build_env_from_yaml(
    *,
    config_path: Path,
    seed: int,
    max_timesteps: int,
    boundary_cfg: BoundaryConfig,
) -> object:
    sim_params = _load_simulator_params_from_config(config_path)
    sim_params["seed"] = int(seed)
    sim_params["puck_noise"] = False
    sim_params["enable_random_occlusions"] = False
    sim_params["paddle_bounds"] = boundary_cfg.bounds
    sim_params["paddle_edge_bounds"] = boundary_cfg.edge_lims

    env_cfg = {
        "task": "paddle_free_movement",
        "simulator": "box2d",
        "simulator_params": sim_params,
        "paddle_bounds": boundary_cfg.bounds,
        "paddle_edge_bounds": boundary_cfg.edge_lims,
        "seed": int(seed),
        "n_training_steps": 1,
        "obs_type": "paddle",
        "num_pucks": 0,
        "num_blocks": 0,
        "num_obstacles": 0,
        "num_targets": 0,
        "num_paddles": 1,
        "max_timesteps": int(max_timesteps),
        "goal_max_x_velocity": 1,
        "goal_max_y_velocity": 5,
        "goal_min_y_velocity": 1,
        "terminate_on_enemy_goal": False,
        "terminate_on_out_of_bounds": False,
        "terminate_on_puck_stop": False,
        "terminate_on_puck_hit_bottom": False,
        "terminate_on_puck_hit_paddle": False,
        "terminate_on_puck_pass_paddle": False,
        "truncate_rew": -1.0,
        "wall_bumping_rew": 0.0,
        "direction_change_rew": 0.0,
        "horizontal_vel_rew": 0.0,
        "diagonal_motion_rew": 0.0,
        "stand_still_rew": 0.0,
        "return_goal_obs": False,
        "use_reward_shaping": False,
        "use_smooth_penalty": False,
        "base_reward_scaling": 1.0,
        "jerk_penalty_coeff": 0.0,
        "velocity_penalty_coeff": 0.0,
    }
    return AirHockeyEnv(env_cfg)


def _set_paddle_state(env: object, base_pos: tuple[float, float], base_vel: tuple[float, float] = (0.0, 0.0)) -> None:
    body = env.simulator.paddles["paddle_ego"]
    body.position = env.simulator.base_coord_to_box2d(base_pos)
    body.linearVelocity = env.simulator.base_coord_to_box2d(base_vel)
    env.current_state = env.simulator.get_current_state()


def _capture_frame(renderer: AirHockeyRenderer) -> np.ndarray:
    frame = renderer.get_frame()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _resize_for_gif(frame: np.ndarray, target_width: int = 160) -> np.ndarray:
    aspect_ratio = frame.shape[1] / max(frame.shape[0], 1)
    return cv2.resize(frame, (int(target_width), max(1, int(target_width / max(aspect_ratio, EPS)))))


def _draw_overlay(
    frame: np.ndarray,
    *,
    step_idx: int,
    loop_idx: int,
    edge_label: str,
    cur: tuple[float, float],
    tgt: tuple[float, float],
    cfg: BoundaryConfig,
    latest_x_limits: tuple[float, float],
) -> np.ndarray:
    lines = [
        f"step={step_idx} loop={loop_idx} edge={edge_label}",
        f"cur=({cur[0]:+.3f},{cur[1]:+.3f}) tgt=({tgt[0]:+.3f},{tgt[1]:+.3f})",
        f"x:[{cfg.x_min:+.3f},{cfg.x_max:+.3f}] y:[{cfg.y_min:+.3f},{cfg.y_max:+.3f}]",
        f"x_clip_now:[{latest_x_limits[0]:+.3f},{latest_x_limits[1]:+.3f}]",
        f"top_abs={cfg.top_abs:.3f} max_bias_m={cfg.max_bias_m:+.3f} max_bias_p={cfg.max_bias_p:+.3f}",
    ]
    y = 16
    for line in lines:
        cv2.putText(frame, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (0, 0, 0), 1, cv2.LINE_AA)
        y += 15
    return frame


def _base_to_frame_pixel(renderer: AirHockeyRenderer, base_pos: tuple[float, float]) -> tuple[int, int]:
    """
    Map base-frame coordinates to pixel coordinates in the already-rendered frame.
    Supports both horizontal and vertical renderer orientations.
    """
    base_x, base_y = float(base_pos[0]), float(base_pos[1])
    x_h = (-base_x + float(renderer.length) / 2.0) * float(renderer.ppm)
    y_h = (base_y + float(renderer.width) / 2.0) * float(renderer.ppm)

    if renderer.orientation == "vertical":
        # Renderer rotates frame 90deg counterclockwise at end of get_frame().
        # Rotation is applied on the original frame with width=render_length.
        x_pix = y_h
        y_pix = float(renderer.render_length) - 1.0 - x_h
    else:
        x_pix = x_h
        y_pix = y_h

    return int(round(x_pix)), int(round(y_pix))


def _draw_effective_bounds_overlay(
    frame: np.ndarray,
    renderer: AirHockeyRenderer,
    cfg: BoundaryConfig,
    *,
    color: tuple[int, int, int] = (255, 0, 255),
    thickness: int = 2,
    right_samples: int = 120,
) -> np.ndarray:
    """
    Draw the actual clipped boundary polygon:
    left + top + bottom straight segments and right clipped edge with corner cuts.
    """
    ys = np.linspace(cfg.y_min, cfg.y_max, max(8, int(right_samples)), dtype=float)
    right_points: list[tuple[float, float]] = []
    for y in ys:
        _, x_max_clip, _, _ = get_clip_limits(0.0, float(y), cfg.bounds, cfg.edge_lims)
        right_points.append((float(x_max_clip), float(y)))

    left_bottom = (float(cfg.x_min), float(cfg.y_min))
    left_top = (float(cfg.x_min), float(cfg.y_max))
    right_bottom = right_points[0]
    right_top = right_points[-1]

    boundary_points_base: list[tuple[float, float]] = [left_bottom, right_bottom]
    boundary_points_base.extend(right_points[1:])
    boundary_points_base.extend([left_top, left_bottom])

    poly_pts = np.array(
        [_base_to_frame_pixel(renderer, p) for p in boundary_points_base],
        dtype=np.int32,
    )
    cv2.polylines(frame, [poly_pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    return frame


def _xmax_terms(y: float, cfg: BoundaryConfig) -> tuple[float, float, float]:
    return (
        cfg.x_max,
        cfg.max_bias_m - cfg.top_abs * y,
        cfg.max_bias_p + cfg.top_abs * y,
    )


def _sample_right_boundary(cfg: BoundaryConfig, n: int) -> tuple[np.ndarray, np.ndarray]:
    ys = np.linspace(cfg.y_min, cfg.y_max, max(2, int(n)), dtype=float)
    xs = np.empty_like(ys)
    for i, y in enumerate(ys):
        x_min_clip, x_max_clip, _, _ = get_clip_limits(0.0, float(y), cfg.bounds, cfg.edge_lims)
        _ = x_min_clip
        xs[i] = x_max_clip
    return xs, ys


def _build_perimeter_waypoints(cfg: BoundaryConfig, steps_per_edge: int) -> list[tuple[np.ndarray, str]]:
    right_x, right_y = _sample_right_boundary(cfg, steps_per_edge + 1)
    top_right = np.array([right_x[-1], cfg.y_max], dtype=float)
    bottom_right = np.array([right_x[0], cfg.y_min], dtype=float)
    left_top = np.array([cfg.x_min, cfg.y_max], dtype=float)
    left_bottom = np.array([cfg.x_min, cfg.y_min], dtype=float)

    segments: list[tuple[np.ndarray, np.ndarray, str]] = [
        (left_bottom, bottom_right, "bottom"),
        (bottom_right, top_right, "right_clipped"),
        (top_right, left_top, "top"),
        (left_top, left_bottom, "left"),
    ]

    waypoints: list[tuple[np.ndarray, str]] = []
    for start, end, label in segments:
        for t in np.linspace(0.0, 1.0, max(2, int(steps_per_edge)), endpoint=False):
            target = (1.0 - t) * start + t * end
            waypoints.append((target.astype(float), label))
    waypoints.append((left_bottom, "loop_close"))
    return waypoints


def _classify_right_edge_contacts(
    *,
    x: float,
    y: float,
    cfg: BoundaryConfig,
    x_max_at_y: float,
    near_tol: float,
    active_tol: float,
    touched: set[str],
) -> None:
    if abs(x - x_max_at_y) > near_tol:
        return
    touched.add("right_clipped")
    cap, m_line, p_line = _xmax_terms(y, cfg)
    current_min = min(cap, m_line, p_line)
    if abs(cap - current_min) <= active_tol:
        touched.add("right_vertical")
    if abs(m_line - current_min) <= active_tol:
        touched.add("corner_minus")
    if abs(p_line - current_min) <= active_tol:
        touched.add("corner_plus")


def _run_validation_rollout(
    env: object,
    renderer: AirHockeyRenderer,
    cfg: BoundaryConfig,
    *,
    loops: int,
    steps_per_edge: int,
    control_substeps: int,
    action_scale: float,
    position_tol: float,
) -> tuple[list[np.ndarray], dict]:
    center = np.array([(cfg.x_min + cfg.x_max) * 0.5, (cfg.y_min + cfg.y_max) * 0.5], dtype=float)
    _set_paddle_state(env, (float(center[0]), float(center[1])))
    env.reset(seed=env.simulator.rng.randint(0, int(1e8)))
    _set_paddle_state(env, (float(center[0]), float(center[1])))

    waypoints = _build_perimeter_waypoints(cfg, steps_per_edge)
    norm_scale = np.array(
        [
            max((cfg.x_max - cfg.x_min) * 0.5, 1e-4),
            max((cfg.y_max - cfg.y_min) * 0.5, 1e-4),
        ],
        dtype=np.float32,
    )

    frames: list[np.ndarray] = []
    touched: set[str] = set()
    violations: list[dict[str, float]] = []
    stats = {"max_x_over": 0.0, "max_x_under": 0.0, "max_y_over": 0.0, "max_y_under": 0.0}
    pos_stats = {
        "x_min_observed": float("inf"),
        "x_max_observed": float("-inf"),
        "y_min_observed": float("inf"),
        "y_max_observed": float("-inf"),
        "min_dist_left": float("inf"),
        "min_dist_bottom": float("inf"),
        "min_dist_top": float("inf"),
        "min_dist_right_clipped": float("inf"),
    }

    step_counter = 0
    for loop_idx in range(max(1, int(loops))):
        for target, edge_label in waypoints:
            for _ in range(max(1, int(control_substeps))):
                cur = np.array(env.current_state["paddles"]["paddle_ego"]["position"], dtype=float)
                err = target - cur
                desired = cur + np.clip(err, -norm_scale * float(action_scale), norm_scale * float(action_scale))
                x_min_clip_des, x_max_clip_des, y_min_clip_des, y_max_clip_des = get_clip_limits(
                    float(desired[0]), float(desired[1]), cfg.bounds, cfg.edge_lims
                )
                clipped = np.array(
                    [
                        np.clip(float(desired[0]), float(x_min_clip_des), float(x_max_clip_des)),
                        np.clip(float(desired[1]), float(y_min_clip_des), float(y_max_clip_des)),
                    ],
                    dtype=float,
                )
                _set_paddle_state(env, (float(clipped[0]), float(clipped[1])), base_vel=(0.0, 0.0))

                cur = np.array(env.current_state["paddles"]["paddle_ego"]["position"], dtype=float)
                x_min_clip, x_max_clip, y_min_clip, y_max_clip = get_clip_limits(cur[0], cur[1], cfg.bounds, cfg.edge_lims)
                x_under = max(0.0, x_min_clip - cur[0])
                x_over = max(0.0, cur[0] - x_max_clip)
                y_under = max(0.0, y_min_clip - cur[1])
                y_over = max(0.0, cur[1] - y_max_clip)
                stats["max_x_under"] = max(stats["max_x_under"], float(x_under))
                stats["max_x_over"] = max(stats["max_x_over"], float(x_over))
                stats["max_y_under"] = max(stats["max_y_under"], float(y_under))
                stats["max_y_over"] = max(stats["max_y_over"], float(y_over))
                pos_stats["x_min_observed"] = min(pos_stats["x_min_observed"], float(cur[0]))
                pos_stats["x_max_observed"] = max(pos_stats["x_max_observed"], float(cur[0]))
                pos_stats["y_min_observed"] = min(pos_stats["y_min_observed"], float(cur[1]))
                pos_stats["y_max_observed"] = max(pos_stats["y_max_observed"], float(cur[1]))
                pos_stats["min_dist_left"] = min(pos_stats["min_dist_left"], abs(float(cur[0]) - cfg.x_min))
                pos_stats["min_dist_bottom"] = min(pos_stats["min_dist_bottom"], abs(float(cur[1]) - cfg.y_min))
                pos_stats["min_dist_top"] = min(pos_stats["min_dist_top"], abs(float(cur[1]) - cfg.y_max))
                pos_stats["min_dist_right_clipped"] = min(pos_stats["min_dist_right_clipped"], abs(float(cur[0]) - float(x_max_clip)))

                if x_under > position_tol or x_over > position_tol or y_under > position_tol or y_over > position_tol:
                    violations.append(
                        {
                            "step": float(step_counter),
                            "x": float(cur[0]),
                            "y": float(cur[1]),
                            "x_min_clip": float(x_min_clip),
                            "x_max_clip": float(x_max_clip),
                            "y_min_clip": float(y_min_clip),
                            "y_max_clip": float(y_max_clip),
                            "x_under": float(x_under),
                            "x_over": float(x_over),
                            "y_under": float(y_under),
                            "y_over": float(y_over),
                        }
                    )

                if abs(cur[0] - cfg.x_min) <= position_tol * 1.5:
                    touched.add("left")
                if abs(cur[1] - cfg.y_min) <= position_tol * 1.5:
                    touched.add("bottom")
                if abs(cur[1] - cfg.y_max) <= position_tol * 1.5:
                    touched.add("top")
                _classify_right_edge_contacts(
                    x=float(cur[0]),
                    y=float(cur[1]),
                    cfg=cfg,
                    x_max_at_y=float(x_max_clip),
                    near_tol=position_tol * 2.0,
                    active_tol=position_tol * 2.0,
                    touched=touched,
                )

                frame = _capture_frame(renderer)
                frame = _draw_effective_bounds_overlay(frame, renderer, cfg, color=(255, 0, 255), thickness=2)
                frame = _draw_overlay(
                    frame,
                    step_idx=step_counter,
                    loop_idx=loop_idx,
                    edge_label=edge_label,
                    cur=(float(cur[0]), float(cur[1])),
                    tgt=(float(target[0]), float(target[1])),
                    cfg=cfg,
                    latest_x_limits=(float(x_min_clip), float(x_max_clip)),
                )
                frames.append(_resize_for_gif(frame, target_width=160))
                step_counter += 1

    required = {"left", "bottom", "top", "right_clipped"}
    _, sample_ys = _sample_right_boundary(cfg, 41)
    right_modes: set[str] = set()
    for y in sample_ys:
        cap, m_line, p_line = _xmax_terms(float(y), cfg)
        min_term = min(cap, m_line, p_line)
        if abs(cap - min_term) <= 1e-8:
            right_modes.add("right_vertical")
        if abs(m_line - min_term) <= 1e-8:
            right_modes.add("corner_minus")
        if abs(p_line - min_term) <= 1e-8:
            right_modes.add("corner_plus")
    required.update(right_modes)
    missing = sorted(required - touched)

    summary = {
        "required_boundary_segments": sorted(required),
        "touched_boundary_segments": sorted(touched),
        "missing_boundary_segments": missing,
        "violations_count": len(violations),
        "violations_head": violations[:20],
        "max_violation": stats,
        "position_stats": pos_stats,
        "validation_passed": len(missing) == 0 and len(violations) == 0,
    }
    return frames, summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Box2D paddle clipping limits with corner-aware perimeter GIF."
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="scripts/smooth_policy/amp_history/configs/new_juggle/pid_noise_constant_upper_half_custom_sim_params.yaml",
        help="YAML source for air_hockey.simulator_params.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-timesteps", type=int, default=5000)

    parser.add_argument("--x-min", type=float, required=True)
    parser.add_argument("--x-max", type=float, required=True)
    parser.add_argument("--y-min", type=float, required=True)
    parser.add_argument("--y-max", type=float, required=True)
    parser.add_argument(
        "--limits-frame",
        type=str,
        choices=["raw_robot", "centered"],
        default="raw_robot",
        help=(
            "Interpretation for x-related limits. "
            "'raw_robot' expects real-style raw x limits (for example -0.8..-0.37) and "
            "converts to centered-x internally using center_offset_constant. "
            "'centered' expects already centered x limits."
        ),
    )
    parser.add_argument("--top-abs", type=float, default=0.0)
    parser.add_argument("--bot-abs", type=float, default=0.0)
    parser.add_argument("--max-bias-m", type=float, default=100.0)
    parser.add_argument("--max-bias-p", type=float, default=100.0)

    parser.add_argument("--loops", type=int, default=3)
    parser.add_argument("--steps-per-edge", type=int, default=48)
    parser.add_argument("--control-substeps", type=int, default=3)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--position-tol", type=float, default=0.015)

    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="runs/paddle_boundary_validation")
    parser.add_argument("--name", type=str, default="paddle_boundary_validation")
    parser.add_argument(
        "--renderer-orientation",
        type=str,
        choices=["horizontal", "vertical"],
        default="vertical",
        help="Renderer orientation. Use vertical for less axis confusion in boundary checks.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_cfg = BoundaryConfig(
        x_min=float(args.x_min),
        x_max=float(args.x_max),
        y_min=float(args.y_min),
        y_max=float(args.y_max),
        top_abs=float(args.top_abs),
        bot_abs=float(args.bot_abs),
        max_bias_m=float(args.max_bias_m),
        max_bias_p=float(args.max_bias_p),
    )
    if input_cfg.x_min >= input_cfg.x_max:
        raise ValueError("Expected x_min < x_max.")
    if input_cfg.y_min >= input_cfg.y_max:
        raise ValueError("Expected y_min < y_max.")

    config_path = Path(args.config_path).expanduser().resolve()
    sim_params_preview = _load_simulator_params_from_config(config_path)
    center_offset_constant = float(sim_params_preview.get("center_offset_constant", 1.2))
    cfg = (
        _convert_raw_to_centered_limits(input_cfg, center_offset_constant)
        if args.limits_frame == "raw_robot"
        else input_cfg
    )

    env = _build_env_from_yaml(
        config_path=config_path,
        seed=int(args.seed),
        max_timesteps=int(args.max_timesteps),
        boundary_cfg=cfg,
    )
    renderer = AirHockeyRenderer(
        env,
        orientation=args.renderer_orientation,
        show_target_position=False,
        show_acceleration_arrow=False,
    )

    frames, summary = _run_validation_rollout(
        env=env,
        renderer=renderer,
        cfg=cfg,
        loops=int(args.loops),
        steps_per_edge=int(args.steps_per_edge),
        control_substeps=int(args.control_substeps),
        action_scale=float(args.action_scale),
        position_tol=float(args.position_tol),
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = output_dir / f"{args.name}.gif"
    summary_path = output_dir / f"{args.name}.json"
    imageio.mimsave(
        gif_path,
        frames,
        format="GIF",
        loop=0,
        duration=int(1000 / max(1, int(args.fps))),
    )

    result = {
        "gif_path": str(gif_path),
        "summary_path": str(summary_path),
        "seed": int(args.seed),
        "config_path": str(config_path),
        "limits_frame": str(args.limits_frame),
        "center_offset_constant": center_offset_constant,
        "input_bounds": input_cfg.bounds,
        "input_edge_lims": input_cfg.edge_lims,
        "bounds": cfg.bounds,
        "edge_lims": cfg.edge_lims,
        "loops": int(args.loops),
        "steps_per_edge": int(args.steps_per_edge),
        "control_substeps": int(args.control_substeps),
        "fps": int(args.fps),
        "renderer_orientation": str(args.renderer_orientation),
        **summary,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print("-" * 80)
    print(f"GIF: {gif_path}")
    print(f"Summary: {summary_path}")
    print(f"limits_frame={result['limits_frame']} center_offset={result['center_offset_constant']:.3f}")
    print(f"input_bounds={result['input_bounds']} effective_bounds={result['bounds']}")
    print(f"Touched: {', '.join(result['touched_boundary_segments'])}")
    print(f"Required: {', '.join(result['required_boundary_segments'])}")
    print(f"Missing: {', '.join(result['missing_boundary_segments']) if result['missing_boundary_segments'] else '(none)'}")
    print(f"Violations: {result['violations_count']}")
    print(f"Max violation metrics: {result['max_violation']}")
    print(f"Validation passed: {result['validation_passed']}")

    if not result["validation_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

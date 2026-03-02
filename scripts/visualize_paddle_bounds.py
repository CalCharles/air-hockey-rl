#!/usr/bin/env python3
"""
Visualize free paddle movement limits in Box2D with configurable bounds.

This script drives the paddle toward each cardinal direction (up/down/left/right)
until motion settles, then exports:
  - combined sweep GIF
  - optional per-direction GIFs
  - per-direction final images
  - optional stitched summary image
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np

from airhockey import AirHockeyEnv
from airhockey.renderers import AirHockeyRenderer


def fps_to_duration_ms(fps: int) -> int:
    return int(1000 * (1.0 / float(max(fps, 1))))


def build_env_config(
    seed: int,
    max_timesteps: int,
    paddle_bounds: List[float],
    paddle_edge_bounds: List[float],
) -> Dict:
    """Build a minimal config dict accepted by AirHockeyEnv."""
    simulator_params = {
        "absorb_target": False,
        "block_density": 500,
        "block_width": 0.0254,
        "force_scaling": 600,
        "gravity": -0.5,
        "length": 1.9304,
        "max_force_timestep": 100,
        "max_paddle_vel": 2.0,
        "paddle_damping": 3.0,
        "paddle_density": 2500,
        "paddle_radius": 0.0508,
        "puck_damping": 0.5,
        "puck_density": 250,
        "puck_radius": 0.03175,
        "render_size": 360,
        "wall_bounce_scale": 0.02,
        "width": 0.8636,
        "action_lag": 0.0,
        "paddle_bounds": paddle_bounds,
        "paddle_edge_bounds": paddle_edge_bounds,
    }
    return {
        "task": "paddle_free_movement",
        "simulator": "box2d",
        "simulator_params": simulator_params,
         "paddle_bounds": paddle_bounds,
         "paddle_edge_bounds": paddle_edge_bounds,
        "seed": seed,
        "n_training_steps": 1,
        "obs_type": "paddle",
        "num_pucks": 0,
        "num_blocks": 0,
        "num_obstacles": 0,
        "num_targets": 0,
        "num_paddles": 1,
        "max_timesteps": max_timesteps,
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


def set_paddle_state(env: AirHockeyEnv, base_pos: Tuple[float, float], base_vel=(0.0, 0.0)) -> None:
    """Force paddle to a known base-frame state for deterministic sweeps."""
    body = env.simulator.paddles["paddle_ego"]
    box2d_pos = env.simulator.base_coord_to_box2d(base_pos)
    box2d_vel = env.simulator.base_coord_to_box2d(base_vel)
    body.position = box2d_pos
    body.linearVelocity = box2d_vel
    env.current_state = env.simulator.get_current_state()


def base_to_pixel(renderer: AirHockeyRenderer, base_pos: Tuple[float, float]) -> Tuple[int, int]:
    """Convert base-frame position to pixel coordinates for horizontal orientation."""
    render_coords = renderer.convert_to_render_coords_sys(base_pos)
    center = np.array(render_coords, dtype=float) + np.array((renderer.width / 2.0, renderer.length / 2.0))
    center = np.array((center[1], center[0])) * renderer.ppm
    return int(center[0]), int(center[1])


def draw_bounds_overlay(frame: np.ndarray, renderer: AirHockeyRenderer, bounds: List[float], color=(255, 0, 255)) -> None:
    x_min, x_max, y_min, y_max = bounds
    top_left = base_to_pixel(renderer, (x_min, y_min))
    top_right = base_to_pixel(renderer, (x_min, y_max))
    bottom_left = base_to_pixel(renderer, (x_max, y_min))
    bottom_right = base_to_pixel(renderer, (x_max, y_max))
    min_x = min(top_left[0], top_right[0], bottom_left[0], bottom_right[0])
    max_x = max(top_left[0], top_right[0], bottom_left[0], bottom_right[0])
    min_y = min(top_left[1], top_right[1], bottom_left[1], bottom_right[1])
    max_y = max(top_left[1], top_right[1], bottom_left[1], bottom_right[1])
    cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), color, 2)


def annotate_frame(
    frame_bgr: np.ndarray,
    renderer: AirHockeyRenderer,
    bounds: List[float],
    direction_name: str,
    step_idx: int,
    paddle_pos: Tuple[float, float],
) -> np.ndarray:
    frame = frame_bgr.copy()
    draw_bounds_overlay(frame, renderer, bounds)
    text_color = (255, 255, 255)
    shadow_color = (0, 0, 0)
    lines = [
        f"Direction: {direction_name}",
        f"Step: {step_idx}",
        f"Paddle(x,y): ({paddle_pos[0]:.3f}, {paddle_pos[1]:.3f})",
        f"Bounds: x[{bounds[0]:.3f}, {bounds[1]:.3f}] y[{bounds[2]:.3f}, {bounds[3]:.3f}]",
    ]
    y0 = 24
    for i, line in enumerate(lines):
        y = y0 + i * 24
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, shadow_color, 3, cv2.LINE_AA)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv2.LINE_AA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def run_direction_sweep(
    env: AirHockeyEnv,
    renderer: AirHockeyRenderer,
    bounds: List[float],
    direction_name: str,
    direction_target: np.ndarray,
    action_norm_scales: np.ndarray,
    action_scale: float,
    max_steps: int,
    settle_tol: float,
    settle_window: int,
    start_pos: Tuple[float, float],
) -> Tuple[List[np.ndarray], Tuple[float, float], int]:
    # Reset and place the paddle in a known center state before each sweep.
    env.reset(seed=env.simulator.rng.randint(0, int(1e8)))
    set_paddle_state(env, start_pos, base_vel=(0.0, 0.0))

    frames: List[np.ndarray] = []
    stagnant_count = 0
    prev_pos = np.array(start_pos, dtype=float)
    final_pos = prev_pos.copy()
    steps_run = 0

    for step_idx in range(max_steps):
        current_pos = np.array(env.current_state["paddles"]["paddle_ego"]["position"], dtype=float)
        # Closed-loop directional drive toward the requested bound target.
        # This ensures custom bounds are actually tested (not only wall collisions).
        error = direction_target - current_pos
        scaled = error / np.maximum(action_norm_scales, 1e-8)
        action = np.clip(scaled, -1.0, 1.0).astype(np.float32) * float(action_scale)
        _, _, done, truncated, _ = env.step(action)
        state = env.current_state
        paddle_pos = np.array(state["paddles"]["paddle_ego"]["position"], dtype=float)
        final_pos = paddle_pos.copy()

        frame_bgr = renderer.get_frame()
        frame_rgb = annotate_frame(
            frame_bgr=frame_bgr,
            renderer=renderer,
            bounds=bounds,
            direction_name=direction_name,
            step_idx=step_idx,
            paddle_pos=(float(paddle_pos[0]), float(paddle_pos[1])),
        )
        frames.append(frame_rgb)
        steps_run = step_idx + 1

        displacement = float(np.linalg.norm(paddle_pos - prev_pos))
        if displacement < settle_tol:
            stagnant_count += 1
        else:
            stagnant_count = 0
        prev_pos = paddle_pos

        if stagnant_count >= settle_window or done or truncated:
            break

    return frames, (float(final_pos[0]), float(final_pos[1])), steps_run


def save_stitched_summary(target_path: Path, final_frames: Dict[str, np.ndarray]) -> None:
    """Create a 2x2 summary image of final directional boundary positions."""
    order = ["up", "right", "left", "down"]
    tiles = [final_frames[d] for d in order if d in final_frames]
    if len(tiles) != 4:
        return
    h, w = tiles[0].shape[:2]
    canvas = np.zeros((2 * h, 2 * w, 3), dtype=np.uint8)
    canvas[0:h, 0:w] = tiles[0]
    canvas[0:h, w:2 * w] = tiles[1]
    canvas[h:2 * h, 0:w] = tiles[2]
    canvas[h:2 * h, w:2 * w] = tiles[3]
    imageio.imwrite(target_path, canvas)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Box2D paddle movement bounds.")
    parser.add_argument(
        "--paddle-bounds",
        type=float,
        nargs=4,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
        default=[0.0, 0.9142, -0.5332, 0.5332],
        help="Paddle bounds in base coordinates: x_min x_max y_min y_max.",
    )
    parser.add_argument(
        "--paddle-edge-bounds",
        type=float,
        nargs=4,
        metavar=("TOP_ABS", "BOT_ABS", "MAX_BIAS_P", "MAX_BIAS_M"),
        default=[0.0, 0.0, 100.0, 100.0],
        help="Optional edge limits passed through environment config.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-timesteps", type=int, default=400, help="Env max timesteps.")
    parser.add_argument("--max-steps-per-direction", type=int, default=120, help="Maximum rollout steps per direction.")
    parser.add_argument("--action-scale", type=float, default=1.0, help="Action magnitude toward each direction.")
    parser.add_argument("--settle-tol", type=float, default=1e-4, help="Position delta threshold for settle detection.")
    parser.add_argument("--settle-window", type=int, default=8, help="Consecutive settled frames to stop a sweep.")
    parser.add_argument("--fps", type=int, default=30, help="Output GIF FPS.")
    parser.add_argument("--output-dir", type=str, default="runs/paddle_bounds_viz", help="Output directory.")
    parser.add_argument("--prefix", type=str, default="paddle_bounds", help="Filename prefix.")
    parser.add_argument("--save-combined-gif", action="store_true", help="Save one GIF with all directions.")
    parser.add_argument("--save-per-direction-gifs", action="store_true", help="Save a GIF per direction.")
    parser.add_argument("--save-final-images", action="store_true", help="Save final frame image per direction.")
    parser.add_argument("--save-summary-image", action="store_true", help="Save 2x2 stitched summary image.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Default to saving all output artifacts unless user specifies otherwise.
    if not (args.save_combined_gif or args.save_per_direction_gifs or args.save_final_images or args.save_summary_image):
        args.save_combined_gif = True
        args.save_per_direction_gifs = True
        args.save_final_images = True
        args.save_summary_image = True

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_env_config(
        seed=args.seed,
        max_timesteps=args.max_timesteps,
        paddle_bounds=[float(v) for v in args.paddle_bounds],
        paddle_edge_bounds=[float(v) for v in args.paddle_edge_bounds],
    )
    env = AirHockeyEnv(cfg)
    renderer = AirHockeyRenderer(env, orientation="horizontal", show_target_position=False, show_acceleration_arrow=False)

    x_min, x_max, y_min, y_max = [float(v) for v in args.paddle_bounds]
    start_pos = ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)

    directions: DirectionSpec = {
        "up": np.array([x_min, start_pos[1]], dtype=np.float32),
        "down": np.array([x_max, start_pos[1]], dtype=np.float32),
        "left": np.array([start_pos[0], y_min], dtype=np.float32),
        "right": np.array([start_pos[0], y_max], dtype=np.float32),
    }
    action_norm_scales = np.array(
        [
            max((x_max - x_min) * 0.5, 1e-3),
            max((y_max - y_min) * 0.5, 1e-3),
        ],
        dtype=np.float32,
    )

    all_frames: List[np.ndarray] = []
    final_frames: Dict[str, np.ndarray] = {}
    summary_rows = []
    duration_ms = fps_to_duration_ms(args.fps)

    for direction_name, direction_target in directions.items():
        frames, final_pos, steps_run = run_direction_sweep(
            env=env,
            renderer=renderer,
            bounds=[x_min, x_max, y_min, y_max],
            direction_name=direction_name,
            direction_target=direction_target,
            action_norm_scales=action_norm_scales,
            action_scale=args.action_scale,
            max_steps=args.max_steps_per_direction,
            settle_tol=args.settle_tol,
            settle_window=args.settle_window,
            start_pos=start_pos,
        )
        if not frames:
            continue

        final_frames[direction_name] = frames[-1]
        all_frames.extend(frames)
        summary_rows.append((direction_name, final_pos, steps_run))

        if args.save_per_direction_gifs:
            per_dir_gif = out_dir / f"{args.prefix}_{direction_name}.gif"
            imageio.mimsave(per_dir_gif, frames, format="GIF", loop=0, duration=duration_ms)

        if args.save_final_images:
            final_image = out_dir / f"{args.prefix}_{direction_name}_final.png"
            imageio.imwrite(final_image, frames[-1])

    if args.save_combined_gif and all_frames:
        combined_gif = out_dir / f"{args.prefix}_combined.gif"
        imageio.mimsave(combined_gif, all_frames, format="GIF", loop=0, duration=duration_ms)

    if args.save_summary_image:
        stitched = out_dir / f"{args.prefix}_summary.png"
        save_stitched_summary(stitched, final_frames)

    print(f"Saved outputs to: {out_dir}")
    for direction_name, final_pos, steps_run in summary_rows:
        print(f"{direction_name:>5s}: final_pos=({final_pos[0]: .4f}, {final_pos[1]: .4f}), steps={steps_run}")


if __name__ == "__main__":
    main()

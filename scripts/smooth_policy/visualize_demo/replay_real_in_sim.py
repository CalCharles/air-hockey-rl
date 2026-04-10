#!/usr/bin/env python3
"""
Replay a real-world robot trajectory in the Box2D simulator and render both
side-by-side as a single GIF.

This is the inner primitive for sim-to-real system identification: load a recorded
HDF5 episode, reconstruct normalized actions from (desired_pose, pose), reset the
sim to the real episode's initial state, step the sim through the recorded actions,
and produce a side-by-side GIF using the canonical Box2D rendering style on both
panels.

Usage:
    python scripts/smooth_policy/visualize_demo/replay_real_in_sim.py
    python scripts/smooth_policy/visualize_demo/replay_real_in_sim.py --episode path/to/trajectory.hdf5
    python scripts/smooth_policy/visualize_demo/replay_real_in_sim.py --enable-noise
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
import yaml

# Ensure repo root is importable when running as a script.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Local imports from the existing visualization utilities.
try:
    from visualize_real_trajectory import RealTrajectoryRenderer
    from visualize_real_trajectory_split import load_split_trajectory_data
except ModuleNotFoundError:
    from scripts.smooth_policy.visualize_demo.visualize_real_trajectory import (
        RealTrajectoryRenderer,
    )
    from scripts.smooth_policy.visualize_demo.visualize_real_trajectory_split import (
        load_split_trajectory_data,
    )

from airhockey import AirHockeyEnv
from airhockey.renderers.render import AirHockeyRenderer
from airhockey.sims.real.velocity_estimator import fit_velocity_from_positions


# Canonical field indices in the 35-field train_vals layout built by
# load_split_trajectory_data (see SPLIT_DATASETS in visualize_real_trajectory_split.py).
_POSE_XY = slice(5, 7)         # actual paddle position (table frame, m)
_SPEED_XY = slice(11, 13)      # paddle velocity (table frame, m/s)
_DESIRED_XY = slice(26, 28)    # policy target position (table frame, m)
_PUCK_XY = slice(32, 34)       # puck position (table frame, m)
_CUR_TIME = 0                  # unix timestamp (s)


DEFAULT_EPISODE = (
    "real_runs/online_run/episode_hdf5/100-200/trajectory_data451.hdf5"
)
DEFAULT_CONFIG = (
    "scripts/smooth_policy/amp_history/configs/new_juggle/"
    "pid_noise_constant_upper_half_custom_sim_params.yaml"
)

# Keys to disable under air_hockey.simulator_params when --enable-noise is off.
_NOISE_SIM_KEYS = (
    "puck_noise",
    "enable_random_occlusions",
    "enable_observation_delay",
    "enable_action_delay",
    "enable_action_force_attenuation",
)
# Keys to disable under air_hockey when running a pure replay.
_TERMINATION_KEYS = (
    "terminate_on_enemy_goal",
    "terminate_on_puck_hit_bottom",
    "terminate_on_puck_pass_paddle",
    "terminate_on_puck_stop",
    "terminate_on_out_of_bounds",
)


def load_real_episode(path: str) -> dict:
    """Load a real-world HDF5 episode into the arrays we need for replay."""
    train_vals = load_split_trajectory_data(path)
    if train_vals.shape[0] < 2:
        raise ValueError(
            f"Episode too short ({train_vals.shape[0]} steps); need >= 2 for replay."
        )
    pose_xy = np.asarray(train_vals[:, _POSE_XY], dtype=np.float64)
    speed_xy = np.asarray(train_vals[:, _SPEED_XY], dtype=np.float64)
    desired_xy = np.asarray(train_vals[:, _DESIRED_XY], dtype=np.float64)
    puck_xy = np.asarray(train_vals[:, _PUCK_XY], dtype=np.float64)
    cur_time = np.asarray(train_vals[:, _CUR_TIME], dtype=np.float64)
    return {
        "pose_xy": pose_xy,
        "speed_xy": speed_xy,
        "desired_xy": desired_xy,
        "puck_xy": puck_xy,
        "cur_time": cur_time,
        "num_steps": int(train_vals.shape[0]),
    }


def reconstruct_actions(
    pose_xy: np.ndarray, desired_xy: np.ndarray, move_lims: np.ndarray
) -> np.ndarray:
    """Invert desired = pose + action * move_lims to get normalized actions in [-1, 1].

    Matches the pattern used in
    scripts/smooth_policy/amp_history/amp_training/td3/extras/async_td3_real_reset_policy.py:318-320.
    """
    move_lims = np.asarray(move_lims, dtype=np.float64).reshape(-1)[:2]
    move_lims[np.abs(move_lims) < 1e-6] = 1.0
    actions = (desired_xy - pose_xy) / move_lims[None, :]
    return np.clip(actions, -1.0, 1.0)


def load_sim_config(config_path: str, enable_noise: bool) -> dict:
    """Load the sim YAML and optionally disable noise/delay/termination for a clean replay."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    ah = cfg["air_hockey"]
    if not enable_noise:
        sim_params = ah.setdefault("simulator_params", {})
        for key in _NOISE_SIM_KEYS:
            if key in sim_params:
                sim_params[key] = False
        for key in _TERMINATION_KEYS:
            if key in ah:
                ah[key] = False
    return ah


def build_sim_env(sim_cfg: dict):
    """Instantiate AirHockeyEnv + AirHockeyRenderer from a prepared sim-config dict.

    `show_target_position=False` so that we can overlay our own target marker at the
    same physical position and with the same visual style on both panels (consistency
    with the real-side rendering).
    """
    env = AirHockeyEnv(copy.deepcopy(sim_cfg))
    renderer = AirHockeyRenderer(
        env, show_target_position=False, show_acceleration_arrow=False
    )
    return env, renderer


def estimate_puck_velocity_fit(
    episode: dict,
    start_frame: int,
    half_window: int,
    gravity: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Estimate puck velocity at `start_frame` via the gravity-linear LSQ fit.

    Slices a window `[start_frame - half_window, start_frame + half_window]` (clamped
    to episode bounds) and fits the kinematic model `pos(t) = pos0 + v0*t - 0.5*g*t^2`
    using `airhockey.sims.real.velocity_estimator.fit_velocity_from_positions`. We then
    take `v_at_times[k]` where `k` is the offset of `start_frame` inside the window —
    i.e. the velocity exactly at the replay's first frame.

    Default gravity=(0,0) suits flat real-table data; pass a non-zero vector to model
    a constant in-plane deceleration if needed.
    """
    puck_xy = episode["puck_xy"]
    cur_time = episode["cur_time"]
    n_total = puck_xy.shape[0]
    lo = max(0, start_frame - half_window)
    hi = min(n_total, start_frame + half_window + 1)
    if hi - lo < 2:
        raise ValueError(
            f"Velocity-fit window [{lo}, {hi}) is too small (need >= 2 samples)."
        )
    result = fit_velocity_from_positions(
        positions=puck_xy[lo:hi],
        times=cur_time[lo:hi],
        valid_mask=None,
        gravity=gravity,
    )
    if result is None:
        raise RuntimeError(
            f"fit_velocity_from_positions returned None for window [{lo}, {hi})."
        )
    k = start_frame - lo
    v = np.asarray(result["v_at_times"][k], dtype=np.float64)
    print(
        f"Puck velocity (fit, window [{lo}, {hi}), k={k}): "
        f"v=({v[0]:.4f}, {v[1]:.4f}) m/s, snr={result['snr']:.2f}, n_valid={result['n_valid']}"
    )
    return v


def initial_state_vector(
    episode: dict,
    start_frame: int = 0,
    puck_vel_fit: bool = False,
    puck_vel_half_window: int = 5,
    puck_vel_gravity: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Build the 8-vector [paddle_pos, paddle_vel, puck_pos, puck_vel] for reset_from_state.

    Order matches the definition that AirHockeyPuckJuggleUpperHalfRewardEnv actually
    inherits: airhockey/airhockey_base.py:906 (AirHockeyBaseEnv.create_world_objects_from_state).
    Note there's ANOTHER definition in airhockey_simple_tasks.py:537 with the OPPOSITE
    ordering — don't be misled; only the base-env one is active for this task.

    Puck velocity is either:
      - the simple two-point finite difference at `start_frame` (default), or
      - the gravity-linear LSQ fit over a ±half_window neighborhood (if `puck_vel_fit`).
    """
    pose_xy = episode["pose_xy"]
    speed_xy = episode["speed_xy"]
    puck_xy = episode["puck_xy"]
    cur_time = episode["cur_time"]
    n_total = pose_xy.shape[0]

    if not 0 <= start_frame < n_total:
        raise IndexError(
            f"start_frame={start_frame} out of range for episode with {n_total} steps."
        )

    paddle_pos = pose_xy[start_frame]
    paddle_vel = speed_xy[start_frame]
    puck_pos = puck_xy[start_frame]

    if puck_vel_fit:
        puck_vel = estimate_puck_velocity_fit(
            episode,
            start_frame=start_frame,
            half_window=puck_vel_half_window,
            gravity=puck_vel_gravity,
        )
    else:
        # Simple two-point finite difference centered on start_frame's local
        # neighborhood. Prefer the (start, start+1) pair; fall back to (start-1, start)
        # if start_frame is the last frame.
        if start_frame + 1 < n_total:
            i0, i1 = start_frame, start_frame + 1
        else:
            i0, i1 = start_frame - 1, start_frame
        dt0 = float(cur_time[i1] - cur_time[i0])
        if not np.isfinite(dt0) or dt0 <= 1e-6:
            dt0 = 0.05  # fall back to ~20 Hz nominal
        puck_vel = (puck_xy[i1] - puck_xy[i0]) / dt0

    return np.concatenate([paddle_pos, paddle_vel, puck_pos, puck_vel]).astype(np.float64)


def _postprocess_frame(frame_bgr: np.ndarray, width: int = 160) -> np.ndarray:
    """Canonical td3_training.py postprocess: BGR->RGB + resize to `width` pixels wide."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    aspect_ratio = rgb.shape[1] / rgb.shape[0]
    new_h = max(1, int(round(width / aspect_ratio)))
    return cv2.resize(rgb, (width, new_h))


def _target_pixel_coords(
    tx: float,
    ty: float,
    width_m: float,
    length_m: float,
    ppm: float,
    render_length: int,
) -> tuple[float, float]:
    """Map a table-frame (x, y) to post-rotation pixel coords for the "vertical" panel.

    Mirrors AirHockeyRenderer.world_xy_to_output_pixel for orientation='vertical':
    - convert_to_render_coords_sys: (x, y) -> (y, -x)
    - + (width/2, length/2), swap axes, scale by ppm
    - apply CCW rotation by mapping (x_px, y_px) -> (y_px, render_length - 1 - x_px)

    Both renderers share identical (width, length, ppm, render_length) in our setup, so
    one helper produces the same pixel coords for both panels.
    """
    # convert_to_render_coords_sys
    render_x = float(ty)
    render_y = -float(tx)
    center_u = render_x + width_m / 2.0
    center_v = render_y + length_m / 2.0
    # swap and scale
    pre_rot_x = center_v * float(ppm)
    pre_rot_y = center_u * float(ppm)
    # CCW rotation (orientation='vertical')
    post_x = pre_rot_y
    post_y = float(render_length) - 1.0 - pre_rot_x
    return post_x, post_y


def _draw_sim_ghost_overlay(
    real_frame_bgr: np.ndarray,
    sim_paddle_xy: np.ndarray,
    sim_puck_xy: np.ndarray,
    width_m: float,
    length_m: float,
    ppm: float,
    render_length: int,
    paddle_radius_m: float,
    puck_radius_m: float,
    alpha: float = 0.35,
) -> None:
    """Draw light semi-transparent ghost circles at the sim's paddle+puck positions.

    Renders on top of the post-rotation real-side BGR frame (in place). Used to
    visualize how far the sim has drifted from the real trajectory at each step.
    Uses alpha blending for the "light" look the user asked for, with a thin
    outline so the ghosts stay legible when they land on busy background.
    """
    px_paddle = _target_pixel_coords(
        float(sim_paddle_xy[0]), float(sim_paddle_xy[1]),
        width_m, length_m, ppm, render_length,
    )
    px_puck = _target_pixel_coords(
        float(sim_puck_xy[0]), float(sim_puck_xy[1]),
        width_m, length_m, ppm, render_length,
    )
    paddle_r = max(1, int(round(paddle_radius_m * ppm)))
    puck_r = max(1, int(round(puck_radius_m * ppm)))
    paddle_center = (int(round(px_paddle[0])), int(round(px_paddle[1])))
    puck_center = (int(round(px_puck[0])), int(round(px_puck[1])))

    # Semi-transparent filled circles via addWeighted on a scratch overlay.
    overlay = real_frame_bgr.copy()
    # Paddle ghost: neutral gray (distinguishable from red/blue table markings).
    cv2.circle(overlay, paddle_center, paddle_r, (180, 180, 180), thickness=-1)
    # Puck ghost: same neutral gray for visual coherence.
    cv2.circle(overlay, puck_center, puck_r, (180, 180, 180), thickness=-1)
    cv2.addWeighted(overlay, alpha, real_frame_bgr, 1.0 - alpha, 0, dst=real_frame_bgr)

    # Thin dark outline for legibility on any background.
    cv2.circle(real_frame_bgr, paddle_center, paddle_r, (40, 40, 40), thickness=1, lineType=cv2.LINE_AA)
    cv2.circle(real_frame_bgr, puck_center, puck_r, (40, 40, 40), thickness=1, lineType=cv2.LINE_AA)


def _draw_consistent_target(
    frame_bgr: np.ndarray,
    target_xy_table: np.ndarray,
    width_m: float,
    length_m: float,
    ppm: float,
    render_length: int,
    marker_size: int = 15,
    thickness: int = 3,
    color_bgr: tuple[int, int, int] = (0, 165, 255),  # orange (BGR)
) -> None:
    """Draw an orange cross+circle target marker on a post-rotation BGR frame.

    Visual style matches AirHockeyRenderer.draw_target_marker exactly.
    Clips the target into table bounds before drawing.
    """
    tx = float(np.clip(target_xy_table[0], -length_m / 2.0, length_m / 2.0))
    ty = float(np.clip(target_xy_table[1], -width_m / 2.0, width_m / 2.0))
    u, v = _target_pixel_coords(tx, ty, width_m, length_m, ppm, render_length)
    center = (int(round(u)), int(round(v)))

    # Outer circle (black outline), inner circle (colored).
    cv2.circle(frame_bgr, center, marker_size, (0, 0, 0), thickness + 2)
    cv2.circle(frame_bgr, center, marker_size, color_bgr, thickness)

    # Cross: black outline first, then colored.
    horiz_a = (center[0] - marker_size, center[1])
    horiz_b = (center[0] + marker_size, center[1])
    vert_a = (center[0], center[1] - marker_size)
    vert_b = (center[0], center[1] + marker_size)
    cv2.line(frame_bgr, horiz_a, horiz_b, (0, 0, 0), thickness + 2)
    cv2.line(frame_bgr, vert_a, vert_b, (0, 0, 0), thickness + 2)
    cv2.line(frame_bgr, horiz_a, horiz_b, color_bgr, thickness)
    cv2.line(frame_bgr, vert_a, vert_b, color_bgr, thickness)


def _put_label(frame: np.ndarray, text: str, pos=(5, 20)) -> None:
    cv2.putText(
        frame,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )


def _side_by_side(real_rgb: np.ndarray, sim_rgb: np.ndarray) -> np.ndarray:
    """Horizontal concat with a 3px light-gray separator, matching viz_collision_tiers.py."""
    if sim_rgb.shape[0] != real_rgb.shape[0]:
        # Defensive: match height if for some reason they differ.
        new_w = int(round(sim_rgb.shape[1] * (real_rgb.shape[0] / sim_rgb.shape[0])))
        sim_rgb = cv2.resize(sim_rgb, (max(1, new_w), real_rgb.shape[0]))
    sep = np.full((real_rgb.shape[0], 3, 3), 200, dtype=np.uint8)
    return np.concatenate([real_rgb, sep, sim_rgb], axis=1)


def render_real_frame(
    real_renderer: RealTrajectoryRenderer,
    episode: dict,
    step_idx: int,
) -> np.ndarray:
    """Render a single real-side frame via RealTrajectoryRenderer.

    The target marker is intentionally NOT drawn here — the replay loop draws
    a consistent orange cross overlay on both sim and real panels afterward.
    """
    pose = episode["pose_xy"][step_idx]
    vel = episode["speed_xy"][step_idx]
    puck = episode["puck_xy"][step_idx]
    return real_renderer.render_frame(
        pos_x=float(pose[0]),
        pos_y=float(pose[1]),
        vel_x=float(vel[0]),
        vel_y=float(vel[1]),
        puck_x=float(puck[0]),
        puck_y=float(puck[1]),
        target_x=None,
        target_y=None,
    )


def replay_episode(
    episode_path: str,
    config_path: str,
    output_path: str,
    enable_noise: bool,
    max_steps: int | None,
    fps: int,
    frame_width: int,
    start_frame: int = 0,
    puck_vel_fit: bool = False,
    puck_vel_half_window: int = 5,
) -> dict:
    # --- load real data --------------------------------------------------------
    episode = load_real_episode(episode_path)
    total = episode["num_steps"]
    if not 0 <= start_frame < total:
        raise IndexError(
            f"--start-frame={start_frame} out of range for episode with {total} steps."
        )
    end = total
    if max_steps is not None:
        end = min(total, start_frame + int(max_steps))
    n_replay = end - start_frame
    print(f"Loaded real episode '{episode_path}' with {total} steps")
    print(
        f"Replaying frames [{start_frame}, {end}) → {n_replay} frames "
        f"(start_frame={start_frame})."
    )

    # --- build sim env ---------------------------------------------------------
    sim_cfg = load_sim_config(config_path, enable_noise=enable_noise)
    if not enable_noise:
        print(
            "Sim noise/delay/occlusion/termination disabled "
            "(use --enable-noise to match the training config verbatim)."
        )
    env, sim_renderer = build_sim_env(sim_cfg)

    # --- reconstruct normalized actions ---------------------------------------
    move_lims = getattr(env.simulator, "move_lims", (0.26, 0.12))
    actions = reconstruct_actions(
        episode["pose_xy"], episode["desired_xy"], np.asarray(move_lims)
    )
    print(f"Reconstructed actions from (desired_pose - pose) / move_lims={tuple(move_lims)}")
    print(
        f"  action range per-axis: "
        f"x[{actions[:, 0].min():.3f}, {actions[:, 0].max():.3f}], "
        f"y[{actions[:, 1].min():.3f}, {actions[:, 1].max():.3f}]"
    )

    # --- initialize sim to the real episode's start frame --------------------
    state0 = initial_state_vector(
        episode,
        start_frame=start_frame,
        puck_vel_fit=puck_vel_fit,
        puck_vel_half_window=puck_vel_half_window,
    )
    env.reset_from_state(state0)
    print(
        f"Initialized sim state @ frame {start_frame}: "
        f"paddle_pos={state0[0:2]}, paddle_vel={state0[2:4]}, "
        f"puck_pos={state0[4:6]}, puck_vel={state0[6:8]} "
        f"(puck_vel_fit={'on' if puck_vel_fit else 'off'})"
    )

    # --- real-side renderer (matches sim render_size for pixel alignment) ----
    sim_params = sim_cfg.get("simulator_params", {})
    real_renderer = RealTrajectoryRenderer(
        table_length=float(sim_params.get("length", 1.9304)),
        table_width=float(sim_params.get("width", 0.8636)),
        paddle_radius=float(sim_params.get("paddle_radius", 0.0508)),
        puck_radius=float(sim_params.get("puck_radius", 0.03175)),
        render_size=int(sim_params.get("render_size", 360)),
        orientation="vertical",
        paddle_input_frame="table",  # pose/puck are already in table frame
        quiet=True,
    )

    # --- replay loop -----------------------------------------------------------
    # Indexing model (verified against async_td3_real.py:1604→1675 + airhockey_base.py:784):
    #   row i in HDF5 stores the POST-step paddle state `pose[i] = T_{i+1}` and the action
    #   `actions[i] = a_{i+1}` that was applied in the step producing `pose[i]`.
    #   desired_pose[i] = pose[i] + actions[i] * move_lims (synthetic, from post-step pose).
    #
    # After `reset_from_state(pose[0], ...)` the sim sits at T_1. To reach T_2 we apply
    # actions[1] (not actions[0], which is the action that ALREADY produced T_1 in real).
    # So at iteration i we render the state that should match pose[i], then step with
    # actions[i+1] to prepare the state for iteration i+1. Renders n frames total.
    move_lims_arr = np.asarray(move_lims, dtype=np.float64).reshape(-1)[:2]
    width_m = float(sim_params.get("width", 0.8636))
    length_m = float(sim_params.get("length", 1.9304))
    sim_ppm = float(env.ppm)
    sim_render_length = int(env.render_length)

    frames = []
    sim_terminated_early = False
    terminate_reason = ""
    paddle_radius_m = float(sim_params.get("paddle_radius", 0.0508))
    puck_radius_m = float(sim_params.get("puck_radius", 0.03175))
    paddle_errors: list[float] = []
    puck_errors: list[float] = []
    for offset in range(n_replay):
        i = start_frame + offset
        # At this point the sim is at its current step-i state. Render both panels.
        sim_paddle_pos = np.asarray(
            env.current_state["paddles"]["paddle_ego"]["position"][:2], dtype=np.float64
        )
        sim_puck_pos = np.asarray(
            env.current_state["pucks"][0]["position"][:2], dtype=np.float64
        )

        real_paddle_pos = episode["pose_xy"][i]
        real_puck_pos = episode["puck_xy"][i]
        paddle_errors.append(float(np.linalg.norm(sim_paddle_pos - real_paddle_pos)))
        puck_errors.append(float(np.linalg.norm(sim_puck_pos - real_puck_pos)))
        action_i = actions[i]  # last action the real policy applied at frame i
        sim_target = sim_paddle_pos + action_i * move_lims_arr
        real_target = episode["desired_xy"][i]  # == pose[i] + action_i * move_lims

        sim_bgr = sim_renderer.get_frame()
        real_bgr = render_real_frame(real_renderer, episode, i)

        # Ghost overlay on the real frame showing where the sim currently thinks
        # the paddle and puck are — drift from the real trajectory is immediately
        # visible as an offset between the real rendering and the gray ghosts.
        _draw_sim_ghost_overlay(
            real_bgr,
            sim_paddle_pos,
            sim_puck_pos,
            width_m,
            length_m,
            sim_ppm,
            sim_render_length,
            paddle_radius_m,
            puck_radius_m,
        )

        # Draw the consistent target marker on the pre-resize frames so the marker
        # scales naturally with the downstream resize.
        _draw_consistent_target(
            sim_bgr, sim_target, width_m, length_m, sim_ppm, sim_render_length
        )
        _draw_consistent_target(
            real_bgr, real_target, width_m, length_m, sim_ppm, sim_render_length
        )

        sim_rgb = _postprocess_frame(sim_bgr, width=frame_width)
        real_rgb = _postprocess_frame(real_bgr, width=frame_width)

        _put_label(real_rgb, "REAL")
        _put_label(sim_rgb, "SIM")
        step_label = f"step {i}"
        _put_label(real_rgb, step_label, pos=(5, real_rgb.shape[0] - 8))
        _put_label(sim_rgb, step_label, pos=(5, sim_rgb.shape[0] - 8))

        frames.append(_side_by_side(real_rgb, sim_rgb))

        # Advance sim to match state at iteration i+1 (if any).
        if offset < n_replay - 1:
            _, _, terminated, truncated, _ = env.step(actions[i + 1])
            if terminated or truncated:
                sim_terminated_early = True
                terminate_reason = "terminated" if terminated else "truncated"
                if enable_noise:
                    print(
                        f"Sim {terminate_reason} early at step {i + 1}; stopping replay."
                    )
                    break

    # --- write output ----------------------------------------------------------
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(1, int(round(1000.0 / max(1, fps))))
    imageio.mimsave(
        str(out_path),
        frames,
        format="GIF",
        loop=0,
        duration=duration_ms,
    )
    print(f"\nWrote {len(frames)} frames to: {out_path}")
    if sim_terminated_early and not enable_noise:
        print(
            f"Note: sim reported {terminate_reason} during replay "
            "but replay continued because termination flags were disabled."
        )

    # --- position error metrics ------------------------------------------------
    paddle_err = np.asarray(paddle_errors)
    puck_err = np.asarray(puck_errors)
    cum_paddle = float(np.sum(paddle_err))
    cum_puck = float(np.sum(puck_err))
    metrics = {
        "episode": str(episode_path),
        "num_frames": len(paddle_errors),
        "paddle": {
            "cumulative_error_m": round(cum_paddle, 6),
            "mean_error_m": round(float(np.mean(paddle_err)), 6),
            "max_error_m": round(float(np.max(paddle_err)), 6),
            "final_error_m": round(paddle_errors[-1], 6),
            "per_step_errors_m": [round(e, 6) for e in paddle_errors],
        },
        "puck": {
            "cumulative_error_m": round(cum_puck, 6),
            "mean_error_m": round(float(np.mean(puck_err)), 6),
            "max_error_m": round(float(np.max(puck_err)), 6),
            "final_error_m": round(puck_errors[-1], 6),
            "per_step_errors_m": [round(e, 6) for e in puck_errors],
        },
    }
    metrics_path = out_path.with_suffix(".json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nPosition error summary ({len(paddle_errors)} frames):")
    print(f"  Paddle — cumulative: {cum_paddle:.4f} m, "
          f"mean: {np.mean(paddle_err):.4f} m, "
          f"max: {np.max(paddle_err):.4f} m, "
          f"final: {paddle_errors[-1]:.4f} m")
    print(f"  Puck   — cumulative: {cum_puck:.4f} m, "
          f"mean: {np.mean(puck_err):.4f} m, "
          f"max: {np.max(puck_err):.4f} m, "
          f"final: {puck_errors[-1]:.4f} m")
    print(f"  Metrics saved to: {metrics_path}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a real-world HDF5 episode in the Box2D sim and render a "
            "side-by-side GIF for sim-to-real system identification."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--episode",
        type=str,
        default=str(_REPO_ROOT / DEFAULT_EPISODE),
        help="Path to real-world HDF5 episode (split schema).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(_REPO_ROOT / DEFAULT_CONFIG),
        help="Path to Box2D sim YAML config.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output GIF path. Defaults to ./sim_vs_real_<episode-stem>.gif",
    )
    parser.add_argument(
        "--enable-noise",
        action="store_true",
        help="Use the config's noise/delay/occlusion/termination settings verbatim.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap on replay length (in steps).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="GIF playback frame rate.",
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        default=160,
        help="Width (px) each panel is resized to before side-by-side concat.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help=(
            "Frame index in the real episode at which to begin the comparison. "
            "Sim is reset to this frame's paddle/puck state and the replay starts here."
        ),
    )
    parser.add_argument(
        "--puck-vel-fit",
        action="store_true",
        help=(
            "Estimate initial puck velocity by fitting a gravity-linear model to a "
            "small window around --start-frame, instead of a two-point finite "
            "difference. Uses airhockey.sims.real.velocity_estimator."
        ),
    )
    parser.add_argument(
        "--puck-vel-half-window",
        type=int,
        default=5,
        help=(
            "Half-window (in frames) for --puck-vel-fit. Window is "
            "[start_frame - h, start_frame + h] inclusive (so 11 frames for the default)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episode_path = args.episode
    if not os.path.isabs(episode_path):
        episode_path = str((_REPO_ROOT / episode_path).resolve())
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = str((_REPO_ROOT / config_path).resolve())

    output_path = args.output
    if output_path is None:
        stem = Path(episode_path).stem
        output_path = f"./sim_vs_real_{stem}.gif"

    replay_episode(
        episode_path=episode_path,
        config_path=config_path,
        output_path=output_path,
        enable_noise=args.enable_noise,
        max_steps=args.max_steps,
        fps=args.fps,
        frame_width=args.frame_width,
        start_frame=args.start_frame,
        puck_vel_fit=args.puck_vel_fit,
        puck_vel_half_window=args.puck_vel_half_window,
    )


if __name__ == "__main__":
    main()

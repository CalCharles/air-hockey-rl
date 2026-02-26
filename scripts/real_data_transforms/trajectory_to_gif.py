"""Render real trajectories and sim rollouts as GIFs for visual debugging.

Subcommands:
    render       – Render a real trajectory from HDF5 data
    rollout      – Roll out actions in simulation and save a GIF
    sync         – Side-by-side real vs. sim GIF
    inflection   – Trajectory with peak/contact markers highlighted
    parabolic    – Trajectory with fitted curves and velocity arrows
    puck_segments– Short peak→contact segments as side-by-side real vs. sim

Usage:
    python trajectory_to_gif.py render --traj-idx 100
    python trajectory_to_gif.py parabolic --traj-idx 100
    python trajectory_to_gif.py puck_segments --cfg config.yaml --traj-indices 100,200

Pipeline call stack (visualization / debugging):
  trajectory_to_gif.py  (standalone CLI)  ◄── YOU ARE HERE
    ├─ data_loading.py          load_trajectory()
    ├─ real_to_sim_observations.py  real_trajectory_to_sim_format()
    ├─ rendering_utils.py       pos_to_pixel(), overlay_sprite(), save_gif()
    ├─ puck_inflection.py       analyze_trajectory(), load_peak_start_intervals()
    ├─ puck_velocity_fit.py     segment_free_flights(), fit_segment(), velocity_from_fit()
    └─ AirHockeyEnv             rollout_trajectory() for sim comparison
"""

import os
import sys
import copy
import argparse

import cv2
import numpy as np
import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from data_loading import (
    load_trajectory,
    BOX2D_TABLE_WIDTH,
    BOX2D_PADDLE_RADIUS,
    BOX2D_PUCK_RADIUS,
)
from real_to_sim_observations import real_trajectory_to_sim_format
from rendering_utils import (
    RENDER_SIZE,
    load_assets,
    overlay_sprite,
    pos_to_pixel,
    annotate_loss,
    save_gif,
    produce_synchronized_gifs,
)
from puck_inflection import analyze_trajectory, analyze_and_log, load_peak_start_intervals
from puck_velocity_fit import (
    segment_free_flights,
    fit_segment,
    velocity_from_fit,
    find_contact_events,
    find_puck_x_peaks,
    _split_at_peaks,
)
from airhockey import AirHockeyEnv
from airhockey.renderers import AirHockeyRenderer
from scripts.domain_adaptation.encode_env_params import assign_values


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_and_convert_trajectory(data_dir, traj_idx, start_step=0, end_step=None,
                                dt=0.05, velocity_mode="finite_diff"):
    """Load a real trajectory and convert to sim format, optionally slicing.

    When start_step > 0 or end_step is set, the trajectory is sliced accordingly.
    Velocities at the start point are estimated from the full trajectory before
    slicing, so the initial state has valid velocity estimates.

    Returns:
        converted: dict with 'observations' (N, 8), 'actions' (N, 2), 'hits_array'
    """
    filename = f"trajectory_data{traj_idx}.hdf5"
    traj = load_trajectory(data_dir, filename, load_images=False)
    converted = real_trajectory_to_sim_format(traj, dt=dt, velocity_mode=velocity_mode)

    stop = None if end_step is None else end_step + 1
    if start_step > 0 or stop is not None:
        converted["observations"] = converted["observations"][start_step:stop]
        converted["actions"] = converted["actions"][start_step:stop]
        converted["hits_array"] = converted["hits_array"][start_step:stop]

    return converted


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def rollout_trajectory(action_sequence, base_config, param_vector=None,
                       param_names=None, state_vector=None, gif_path=None,
                       fps=20, actual_paddle_positions=None,
                       show_velocity=False):
    """Roll out actions in an AirHockeyEnv and collect results.

    Args:
        action_sequence: (T, action_dim) actions to replay.
        base_config: Air-hockey config dict.
        param_vector: Optional parameter values to inject via assign_values.
        param_names: Parameter names matching param_vector.
        state_vector: Optional 8D initial state for _from_state.
        gif_path: If set, save a GIF of the rollout.
        fps: GIF playback speed.
        actual_paddle_positions: (T, 4) ground-truth paddle pos+vel for loss overlay.
        show_velocity: If True, draw puck vx/vy arrows on each frame.

    Returns:
        dict with actions, rewards, new_config, episode_return, episode_length,
        trajectory_loss.
    """
    if param_vector is not None and param_names is not None:
        new_config = assign_values(param_vector, param_names, base_config)
    else:
        new_config = copy.deepcopy(base_config)
    env = AirHockeyEnv(new_config)

    if state_vector is not None:
        obs, _ = env.reset_from_state(state_vector)
    else:
        obs, _ = env.reset()

    save = gif_path is not None
    renderer = AirHockeyRenderer(env, show_target_position=False) if save else None
    # Always render horizontal so we can draw target markers and velocity
    # arrows in the same coordinate system, then rotate to vertical manually.
    if renderer is not None:
        renderer.orientation = 'horizontal'
        sim_ppm = renderer.ppm
    frames = []
    rewards = []
    paddle_losses = []
    episode_return = 0.0
    cumulative_loss = 0.0
    n_steps = 0
    font = cv2.FONT_HERSHEY_SIMPLEX

    def _sim_pos_to_px(pos_base):
        """Base coords -> pixel on pre-rotation (horizontal) renderer frame."""
        rx, ry = pos_base[1], -pos_base[0]
        center = np.array([rx + renderer.width / 2, ry + renderer.length / 2])
        return np.array([center[1], center[0]]) * sim_ppm

    for t, action in enumerate(action_sequence):
        step_loss = None
        if actual_paddle_positions is not None and t < len(actual_paddle_positions):
            sim_paddle = np.array(obs[:4])
            step_loss = float(np.linalg.norm(sim_paddle - actual_paddle_positions[t]))
            paddle_losses.append(step_loss)
            cumulative_loss += step_loss

        # Step first so the PID controller sets the correct last_target_position
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        episode_return += reward
        n_steps += 1

        if save:
            frame = renderer.get_frame()
            if step_loss is not None:
                annotate_loss(frame, step_loss, cumulative_loss)

            # Draw puck velocity arrows from the simulator state
            if show_velocity:
                state_info = env.current_state
                if 'pucks' in state_info and len(state_info['pucks']) > 0:
                    puck_pos_base = np.array(state_info['pucks'][0]['position'])
                    puck_vel_base = np.array(state_info['pucks'][0]['velocity'])
                    vx, vy = float(puck_vel_base[0]), float(puck_vel_base[1])

                    center = _sim_pos_to_px(puck_pos_base).astype(int)
                    arrow_scale = 0.1

                    vx_end = _sim_pos_to_px(puck_pos_base + np.array([vx * arrow_scale, 0])).astype(int)
                    if np.linalg.norm(vx_end - center) > 1:
                        cv2.arrowedLine(frame, tuple(center), tuple(vx_end),
                                        (0, 0, 255), 2, tipLength=0.3)

                    vy_end = _sim_pos_to_px(puck_pos_base + np.array([0, vy * arrow_scale])).astype(int)
                    if np.linalg.norm(vy_end - center) > 1:
                        cv2.arrowedLine(frame, tuple(center), tuple(vy_end),
                                        (255, 0, 0), 2, tipLength=0.3)

                    cv2.putText(frame, f"vx={vx:+.3f}", (center[0] + 15, center[1] + 20),
                                font, 0.35, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.putText(frame, f"vy={vy:+.3f}", (center[0] + 15, center[1] + 35),
                                font, 0.35, (255, 0, 0), 1, cv2.LINE_AA)

            # Draw target position marker (blue cross with circle, same as real traj)
            target_base = env.simulator.last_target_position
            if target_base is not None:
                target_px = pos_to_pixel(target_base, sim_ppm).astype(int)
                target_color = (255, 100, 0)  # blue in BGR
                marker_size = 12
                cv2.circle(frame, tuple(target_px), marker_size, (0, 0, 0), 4)
                cv2.circle(frame, tuple(target_px), marker_size, target_color, 2)
                cv2.line(frame, (target_px[0] - marker_size, target_px[1]),
                         (target_px[0] + marker_size, target_px[1]), target_color, 2)
                cv2.line(frame, (target_px[0], target_px[1] - marker_size),
                         (target_px[0], target_px[1] + marker_size), target_color, 2)

            # Rotate to vertical orientation (same as render_trajectory_gif)
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if save:
        save_gif(frames, gif_path, fps)

    return {
        "actions": np.array(action_sequence),
        "rewards": np.array(rewards),
        "new_config": new_config,
        "episode_return": episode_return,
        "episode_length": n_steps,
        "trajectory_loss": float(np.mean(paddle_losses)) if paddle_losses else None,
    }


def render_trajectory_gif(data_dir, traj_idx, output_dir, name=None, fps=20,
                          start_step=0, end_step=None, show_velocity=False,
                          target_positions=None):
    """Render a real trajectory (from HDF5) as a GIF with sprite overlays.

    Args:
        target_positions: Optional (T, 2) array of target positions in base
            coordinates. When provided, a blue cross marker is drawn at each
            target to show where the paddle was commanded to go.
    """
    filename = f"trajectory_data{traj_idx}.hdf5"
    traj = load_trajectory(data_dir, filename, load_images=False)

    stop = None if end_step is None else end_step + 1
    paddle_positions = traj["paddle"][start_step:stop, :2]
    puck_data = traj["puck"][start_step:stop]

    # Compute target positions from actions if not provided
    if target_positions is None and "action" in traj:
        actions = traj["action"][start_step:stop, :2]
        target_positions = paddle_positions + actions

    # Pre-compute parabolic-fit puck velocities (full trajectory, then slice)
    if show_velocity:
        from puck_velocity_fit import fit_trajectory_velocities
        all_vel = fit_trajectory_velocities(traj)  # (N, 2)
        puck_vel = all_vel[start_step:stop]

    ppm = RENDER_SIZE / BOX2D_TABLE_WIDTH
    table_img, paddle_sprite, puck_sprite = load_assets(ppm)
    min_contact_dist = BOX2D_PADDLE_RADIUS + BOX2D_PUCK_RADIUS
    font = cv2.FONT_HERSHEY_SIMPLEX

    frames = []
    overlap_events = []

    for t in tqdm.tqdm(range(len(paddle_positions)), desc=f"Rendering traj {traj_idx}"):
        frame = table_img.copy()
        paddle_pos = paddle_positions[t]
        puck_visible = puck_data[t, 2] < 0.5
        puck_pos = puck_data[t, :2] if puck_visible else None

        # Overlap detection
        overlap, dist = False, None
        if puck_pos is not None:
            dist = np.linalg.norm(puck_pos - paddle_pos)
            if dist < min_contact_dist:
                overlap = True
                overlap_events.append((start_step + t, dist, min_contact_dist - dist))

        # Draw sprites
        if puck_pos is not None:
            puck_px = pos_to_pixel(puck_pos, ppm)
            overlay_sprite(frame, puck_sprite, puck_px)

        paddle_px = pos_to_pixel(paddle_pos, ppm)
        overlay_sprite(frame, paddle_sprite, paddle_px)

        # Draw target position marker (blue cross with circle)
        if target_positions is not None and t < len(target_positions):
            target_px = pos_to_pixel(target_positions[t], ppm).astype(int)
            target_color = (255, 100, 0)  # blue in BGR
            marker_size = 12
            cv2.circle(frame, tuple(target_px), marker_size, (0, 0, 0), 4)
            cv2.circle(frame, tuple(target_px), marker_size, target_color, 2)
            cv2.line(frame, (target_px[0] - marker_size, target_px[1]),
                     (target_px[0] + marker_size, target_px[1]), target_color, 2)
            cv2.line(frame, (target_px[0], target_px[1] - marker_size),
                     (target_px[0], target_px[1] + marker_size), target_color, 2)

        if overlap:
            mid = ((paddle_px + puck_px) / 2).astype(int)
            cv2.circle(frame, tuple(mid), int(min_contact_dist * ppm), (0, 0, 255), 2)

        # Velocity arrows (vx=red, vy=blue)
        if show_velocity and puck_pos is not None:
            vx, vy = puck_vel[t]
            center = puck_px.astype(int)
            arrow_scale = 0.1  # metres of displacement per m/s
            # vx arrow (red)
            vx_end = pos_to_pixel(puck_pos + np.array([vx * arrow_scale, 0]), ppm).astype(int)
            if np.linalg.norm(vx_end - center) > 1:
                cv2.arrowedLine(frame, tuple(center), tuple(vx_end),
                                (0, 0, 255), 2, tipLength=0.3)
            # vy arrow (blue)
            vy_end = pos_to_pixel(puck_pos + np.array([0, vy * arrow_scale]), ppm).astype(int)
            if np.linalg.norm(vy_end - center) > 1:
                cv2.arrowedLine(frame, tuple(center), tuple(vy_end),
                                (255, 0, 0), 2, tipLength=0.3)
            cv2.putText(frame, f"vx={vx:+.3f}", (center[0] + 15, center[1] + 20),
                        font, 0.35, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"vy={vy:+.3f}", (center[0] + 15, center[1] + 35),
                        font, 0.35, (255, 0, 0), 1, cv2.LINE_AA)

        # Labels
        cv2.putText(frame, f"P({paddle_pos[0]:.3f},{paddle_pos[1]:.3f})",
                    (int(paddle_px[0]) + 15, int(paddle_px[1]) - 10),
                    font, 0.35, (0, 0, 200), 1, cv2.LINE_AA)
        if puck_pos is not None:
            cv2.putText(frame, f"K({puck_pos[0]:.3f},{puck_pos[1]:.3f})",
                        (int(puck_px[0]) + 15, int(puck_px[1]) - 10),
                        font, 0.35, (0, 140, 0), 1, cv2.LINE_AA)
            if dist is not None:
                color = (0, 0, 255) if overlap else (0, 140, 0)
                cv2.putText(frame, f"d={dist:.4f}",
                            (int(puck_px[0]) + 15, int(puck_px[1]) + 10),
                            font, 0.3, color, 1, cv2.LINE_AA)

        cv2.putText(frame, f"t={start_step + t}", (10, 25), font, 0.5, (0, 0, 0), 2)
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    os.makedirs(output_dir, exist_ok=True)
    if name is None:
        name = f"trajectory_{traj_idx}.gif"
    gif_path = os.path.join(output_dir, name)
    save_gif(frames, gif_path, fps)


def render_inflection_gif(data_dir, traj_idx, output_dir, log_path=None,
                          name=None, fps=20, start_step=0, end_step=None):
    """Render a real trajectory GIF with puck y-inflection peaks highlighted.

    Peaks (up-up-down) are drawn as yellow circles on the puck. Contact events
    (paddle/wall) are drawn as red circles. Peaks-before-contact are connected
    with a dashed line.

    Also logs the analysis results to a JSON file.
    """
    filename = f"trajectory_data{traj_idx}.hdf5"
    traj = load_trajectory(data_dir, filename, load_images=False)

    # Run inflection analysis
    analysis = analyze_trajectory(traj)
    peaks = {p["timestep"] for p in analysis["peaks"]}
    contact_timesteps = {c["timestep"] for c in analysis["contacts"]}
    # Map each individual peak timestep to its contact info for annotation.
    # Each entry in peaks_before_contacts now has a list of peak_timesteps.
    peak_before_map = {}
    for entry in analysis["peaks_before_contacts"]:
        for pt in entry["peak_timesteps"]:
            peak_before_map[pt] = entry

    # Build set of "falling" timesteps (between each peak and the next valley)
    falling_timesteps = set()
    sorted_peaks = sorted(p["timestep"] for p in analysis["peaks"])
    sorted_valleys = sorted(analysis["valleys"])
    n_total = len(traj["puck"])
    for pk in sorted_peaks:
        # Find the next valley after this peak
        next_valleys = [v for v in sorted_valleys if v > pk]
        end = next_valleys[0] if next_valleys else n_total - 1
        for t in range(pk, end + 1):
            falling_timesteps.add(t)

    stop = None if end_step is None else end_step + 1
    paddle_positions = traj["paddle"][start_step:stop, :2]
    puck_data = traj["puck"][start_step:stop]

    ppm = RENDER_SIZE / BOX2D_TABLE_WIDTH
    table_img, paddle_sprite, puck_sprite = load_assets(ppm)
    min_contact_dist = BOX2D_PADDLE_RADIUS + BOX2D_PUCK_RADIUS
    font = cv2.FONT_HERSHEY_SIMPLEX

    frames = []

    for t in tqdm.tqdm(range(len(paddle_positions)),
                       desc=f"Rendering inflection traj {traj_idx}"):
        frame = table_img.copy()
        abs_t = start_step + t
        paddle_pos = paddle_positions[t]
        puck_visible = puck_data[t, 2] < 0.5
        puck_pos = puck_data[t, :2] if puck_visible else None

        # Draw sprites
        if puck_pos is not None:
            puck_px = pos_to_pixel(puck_pos, ppm)
            overlay_sprite(frame, puck_sprite, puck_px)
        paddle_px = pos_to_pixel(paddle_pos, ppm)
        overlay_sprite(frame, paddle_sprite, paddle_px)

        # Highlight peak (yellow circle + "PEAK" label)
        if abs_t in peaks and puck_pos is not None:
            center = tuple(puck_px.astype(int))
            radius = int(BOX2D_PUCK_RADIUS * ppm) + 8
            cv2.circle(frame, center, radius, (0, 255, 255), 3)
            cv2.putText(frame, "PEAK", (center[0] + 12, center[1] - 15),
                        font, 0.45, (57, 255, 20), 2, cv2.LINE_AA)
            # If this peak is linked to a contact, show that info
            if abs_t in peak_before_map:
                info = peak_before_map[abs_t]
                label = f"-> {info['contact_type']} t={info['contact_timestep']}"
                cv2.putText(frame, label, (center[0] + 12, center[1] + 5),
                            font, 0.3, (0, 180, 180), 1, cv2.LINE_AA)

        # Highlight contact events (red circle)
        if abs_t in contact_timesteps and puck_pos is not None:
            center = tuple(puck_px.astype(int))
            radius = int(BOX2D_PUCK_RADIUS * ppm) + 8
            cv2.circle(frame, center, radius, (0, 0, 255), 3)
            cv2.putText(frame, "CONTACT", (center[0] + 12, center[1] - 15),
                        font, 0.45, (0, 0, 255), 2, cv2.LINE_AA)

        # Purple arrow on puck during falling segments (peak -> valley)
        if abs_t in falling_timesteps and puck_pos is not None:
            # Compute velocity direction from adjacent visible frames
            next_t = t + 1
            if next_t < len(puck_data) and puck_data[next_t, 2] < 0.5:
                next_puck = puck_data[next_t, :2]
                next_px = pos_to_pixel(next_puck, ppm)
                direction = next_px - puck_px
                length = np.linalg.norm(direction)
                if length > 1e-3:
                    arrow_scale = 30.0
                    direction = direction / length * arrow_scale
                    pt1 = tuple(puck_px.astype(int))
                    pt2 = tuple((puck_px + direction).astype(int))
                    cv2.arrowedLine(frame, pt1, pt2, (128, 0, 128), 2,
                                   tipLength=0.35)

        # Coordinate labels
        cv2.putText(frame, f"P({paddle_pos[0]:.3f},{paddle_pos[1]:.3f})",
                    (int(paddle_px[0]) + 15, int(paddle_px[1]) - 10),
                    font, 0.35, (0, 0, 200), 1, cv2.LINE_AA)
        if puck_pos is not None:
            cv2.putText(frame, f"K({puck_pos[0]:.3f},{puck_pos[1]:.3f})",
                        (int(puck_px[0]) + 15, int(puck_px[1]) - 10),
                        font, 0.35, (0, 140, 0), 1, cv2.LINE_AA)

        cv2.putText(frame, f"t={abs_t}", (10, 25), font, 0.5, (0, 0, 0), 2)
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Save GIF
    os.makedirs(output_dir, exist_ok=True)
    if name is None:
        name = f"inflection_{traj_idx}.gif"
    gif_path = os.path.join(output_dir, name)
    save_gif(frames, gif_path, fps)
    print(f"Saved GIF to {gif_path}")

    # Log analysis results
    if log_path is None:
        log_dir = os.path.join(os.path.dirname(__file__), "logs")
        log_path = os.path.join(log_dir, f"inflection_{traj_idx}.json")
    analyze_and_log(data_dir, traj_idx, log_path=log_path)


def render_parabolic_fit_gif(data_dir, traj_idx, output_dir, name=None,
                             fps=20, start_step=0, end_step=None,
                             x_degree=2, y_degree=2):
    """Render a trajectory GIF showing parabolic fits and velocity estimates.

    Uses the same peak-split fitting as fit_trajectory_velocities():
    each free-flight segment is split at x-peaks into monotonic sub-segments,
    and a separate parabola is fit to each. vx is forced to 0 at peaks.

    For each sub-segment, draws:
    - The fitted parabolic curve (colored polyline, different color per sub-segment)
    - Velocity arrows at the current puck position: vx (red), vy (blue)
    - Velocity text labels
    - Peak markers (yellow circle, "PEAK vx=0")
    - Contact markers (red circle)

    Args:
        end_step: Last timestep to render (inclusive). None = end of trajectory.
    """
    filename = f"trajectory_data{traj_idx}.hdf5"
    traj = load_trajectory(data_dir, filename, load_images=False)

    puck_pos = traj["puck"][:, :2]
    puck_occluded = traj["puck"][:, 2] > 0.5
    N = len(traj["puck"])

    if end_step is None:
        end_step = N - 1
    end_step = min(end_step, N - 1)

    paddle_positions = traj["paddle"][start_step:end_step + 1, :2]
    puck_data = traj["puck"][start_step:end_step + 1]

    # --- Pre-compute fits using peak-split logic (matches fit_trajectory_velocities) ---
    contacts = find_contact_events(traj)
    peaks = find_puck_x_peaks(puck_pos, puck_occluded)
    segments = segment_free_flights(contacts, N, gap_tolerance=2, min_length=3)

    # Split each segment at peaks and fit sub-segments
    # Only keep fits that overlap with the visible [start_step, end_step] window
    sub_segment_fits = []  # list of (sub_start, sub_end, fit)
    for seg_start, seg_end in segments:
        sub_segs = _split_at_peaks(seg_start, seg_end, peaks, min_length=3)
        for sub_start, sub_end in sub_segs:
            if sub_end < start_step or sub_start > end_step:
                continue
            fit = fit_segment(puck_pos, puck_occluded, sub_start, sub_end,
                              dt=0.05, x_degree=x_degree, y_degree=y_degree)
            sub_segment_fits.append((sub_start, sub_end, fit))

    # Build lookup: timestep -> (sub_start, sub_end, fit)
    timestep_to_fit = {}
    for sub_start, sub_end, fit in sub_segment_fits:
        if fit is None:
            continue
        for t in range(sub_start, sub_end + 1):
            timestep_to_fit[t] = (sub_start, sub_end, fit)

    peak_set = {p for p in peaks if start_step <= p <= end_step}
    contact_set = {e["timestep"] for e in contacts if start_step <= e["timestep"] <= end_step}

    # Assign distinct colors to sub-segments for visual separation
    SUB_SEG_COLORS = [
        (180, 120, 0),   # teal
        (0, 180, 120),   # green
        (120, 0, 180),   # purple
        (0, 120, 220),   # orange
        (180, 0, 120),   # pink
    ]

    ppm = RENDER_SIZE / BOX2D_TABLE_WIDTH
    table_img, paddle_sprite, puck_sprite = load_assets(ppm)
    font = cv2.FONT_HERSHEY_SIMPLEX

    frames = []

    for t in tqdm.tqdm(range(len(paddle_positions)),
                       desc=f"Rendering parabolic fit traj {traj_idx} "
                            f"[{start_step}:{end_step}]"):
        frame = table_img.copy()
        abs_t = start_step + t
        paddle_pos_t = paddle_positions[t]
        puck_visible = puck_data[t, 2] < 0.5
        puck_pos_t = puck_data[t, :2] if puck_visible else None

        # --- Draw fitted trajectory curves progressively up to current time ---
        for seg_idx, (sub_start, sub_end, fit) in enumerate(sub_segment_fits):
            if fit is None or sub_start > abs_t:
                continue

            draw_end = min(sub_end, abs_t)
            curve_points = []
            dt_phys = 0.05
            for s in range(sub_start, draw_end + 1):
                if puck_occluded[s]:
                    continue
                t_rel = (s - sub_start) * dt_phys
                fx = np.polyval(fit["x_coeffs"], t_rel)
                fy = np.polyval(fit["y_coeffs"], t_rel)
                px = pos_to_pixel(np.array([fx, fy]), ppm)
                curve_points.append(px.astype(int))

            if len(curve_points) > 1:
                color = SUB_SEG_COLORS[seg_idx % len(SUB_SEG_COLORS)]
                pts = np.array(curve_points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, color, 3, cv2.LINE_AA)

        # --- Draw sprites ---
        if puck_pos_t is not None:
            puck_px = pos_to_pixel(puck_pos_t, ppm)
            overlay_sprite(frame, puck_sprite, puck_px)
        paddle_px = pos_to_pixel(paddle_pos_t, ppm)
        overlay_sprite(frame, paddle_sprite, paddle_px)

        # --- Peak marker ---
        if abs_t in peak_set and puck_pos_t is not None:
            center = tuple(puck_px.astype(int))
            radius = int(BOX2D_PUCK_RADIUS * ppm) + 8
            cv2.circle(frame, center, radius, (0, 255, 255), 3)
            cv2.putText(frame, "PEAK vx=0", (center[0] + 12, center[1] - 15),
                        font, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

        # --- Contact marker ---
        if abs_t in contact_set and puck_pos_t is not None:
            center = tuple(puck_px.astype(int))
            radius = int(BOX2D_PUCK_RADIUS * ppm) + 8
            cv2.circle(frame, center, radius, (0, 0, 255), 3)
            cv2.putText(frame, "CONTACT", (center[0] + 12, center[1] - 15),
                        font, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

        # --- Velocity arrows from parabolic fit ---
        if puck_pos_t is not None and abs_t in timestep_to_fit:
            sub_start, sub_end, fit = timestep_to_fit[abs_t]
            t_rel = (abs_t - sub_start) * 0.05
            vx = float(velocity_from_fit(fit["x_coeffs"], t_rel))
            vy = float(velocity_from_fit(fit["y_coeffs"], t_rel))

            # Enforce vx=0 at peaks
            if abs_t in peak_set:
                vx = 0.0

            center = puck_px.astype(int)

            # vx arrow (red) — displace in base x direction then convert to pixel
            vx_endpoint_base = puck_pos_t + np.array([vx * 0.1, 0])
            vx_endpoint_px = pos_to_pixel(vx_endpoint_base, ppm).astype(int)
            if np.linalg.norm(vx_endpoint_px - center) > 1:
                cv2.arrowedLine(frame, tuple(center), tuple(vx_endpoint_px),
                                (0, 0, 255), 2, tipLength=0.3)

            # vy arrow (blue) — displace in base y direction then convert to pixel
            vy_endpoint_base = puck_pos_t + np.array([0, vy * 0.1])
            vy_endpoint_px = pos_to_pixel(vy_endpoint_base, ppm).astype(int)
            if np.linalg.norm(vy_endpoint_px - center) > 1:
                cv2.arrowedLine(frame, tuple(center), tuple(vy_endpoint_px),
                                (255, 0, 0), 2, tipLength=0.3)

            # Velocity text
            cv2.putText(frame, f"vx={vx:+.3f}", (center[0] + 15, center[1] + 20),
                        font, 0.35, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"vy={vy:+.3f}", (center[0] + 15, center[1] + 35),
                        font, 0.35, (255, 0, 0), 1, cv2.LINE_AA)

        # --- Coordinate labels ---
        cv2.putText(frame, f"P({paddle_pos_t[0]:.3f},{paddle_pos_t[1]:.3f})",
                    (int(paddle_px[0]) + 15, int(paddle_px[1]) - 10),
                    font, 0.35, (0, 0, 200), 1, cv2.LINE_AA)
        if puck_pos_t is not None:
            cv2.putText(frame, f"K({puck_pos_t[0]:.3f},{puck_pos_t[1]:.3f})",
                        (int(puck_px[0]) + 15, int(puck_px[1]) - 10),
                        font, 0.35, (0, 140, 0), 1, cv2.LINE_AA)

        cv2.putText(frame, f"t={abs_t}", (10, 25), font, 0.5, (0, 0, 0), 2)

        # Legend
        cv2.putText(frame, "vx", (10, frame.shape[0] - 30),
                    font, 0.35, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "vy", (10, frame.shape[0] - 15),
                    font, 0.35, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, "fit curves (color = sub-segment)",
                    (50, frame.shape[0] - 15),
                    font, 0.3, (100, 100, 100), 1, cv2.LINE_AA)

        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    os.makedirs(output_dir, exist_ok=True)
    if name is None:
        name = f"parabolic_{traj_idx}_t{start_step}-{end_step}.gif"
    gif_path = os.path.join(output_dir, name)
    save_gif(frames, gif_path, fps)
    print(f"Saved parabolic fit GIF to {gif_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR = "/data2/air_hockey/air_hockey_state_data/datastor1/calebc/public/data/mouse/state_data_all_new/"
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "gifs")


def _add_common_args(parser):
    """Add arguments shared across all subcommands."""
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--traj-idx", type=int, default=100)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--start-step", type=int, default=0,
                        help="Timestep to start from (0 = beginning). "
                             "Velocities are estimated from the full trajectory.")
    parser.add_argument("--end-step", type=int, default=None,
                        help="Last timestep to render (default: end of trajectory)")
    parser.add_argument("--show-velocity", action="store_true",
                        help="Draw puck vx/vy arrows on each frame")


def _build_parser():
    parser = argparse.ArgumentParser(description="Air hockey trajectory GIF tools")
    sub = parser.add_subparsers(dest="command")

    rp = sub.add_parser("render", help="Render a real trajectory from HDF5 data")
    _add_common_args(rp)
    rp.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    rp.add_argument("--name", type=str, default=None)

    ro = sub.add_parser("rollout", help="Roll out actions in simulation")
    _add_common_args(ro)
    ro.add_argument("--cfg", type=str, required=True)
    ro.add_argument("--gif-path", type=str, default=None)

    sp = sub.add_parser("sync", help="Side-by-side real vs. sim GIF")
    _add_common_args(sp)
    sp.add_argument("--cfg", type=str, required=True)
    sp.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)

    ip = sub.add_parser("inflection",
                         help="Render trajectory with puck y-inflection peaks highlighted")
    _add_common_args(ip)
    ip.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    ip.add_argument("--name", type=str, default=None)
    ip.add_argument("--log-path", type=str, default=None,
                    help="Path for JSON inflection log (default: logs/inflection_<idx>.json)")

    pb = sub.add_parser("parabolic",
                         help="Render trajectory with parabolic fit curves and velocity arrows")
    _add_common_args(pb)
    pb.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    pb.add_argument("--name", type=str, default=None)
    pb.add_argument("--x-degree", type=int, default=2,
                    help="Polynomial degree for x(t) fit (default: 2)")
    pb.add_argument("--y-degree", type=int, default=2,
                    help="Polynomial degree for y(t) fit (default: 2)")

    ps = sub.add_parser("puck_segments",
                         help="Render short puck trajectories from peak starts "
                              "as side-by-side real vs. sim GIFs")
    ps.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    ps.add_argument("--cfg", type=str, required=True,
                    help="Path to sim config YAML")
    ps.add_argument("--log-dir", type=str,
                    default=os.path.join(os.path.dirname(__file__), "logs"),
                    help="Directory containing inflection_<idx>.json logs")
    ps.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    ps.add_argument("--traj-indices", type=str, default=None,
                    help="Comma-separated trajectory indices (default: all logs in log-dir)")
    ps.add_argument("--max-segments", type=int, default=None,
                    help="Max number of segments to render (default: all)")
    ps.add_argument("--fps", type=int, default=20)
    ps.add_argument("--show-velocity", action="store_true",
                    help="Draw puck vx/vy arrows on each frame")

    return parser


def cmd_render(args):
    render_trajectory_gif(
        data_dir=args.data_dir,
        traj_idx=args.traj_idx,
        output_dir=args.output_dir,
        name=args.name,
        fps=args.fps,
        start_step=args.start_step,
        end_step=args.end_step,
        show_velocity=args.show_velocity,
    )


def cmd_rollout(args):
    import yaml
    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)

    converted = load_and_convert_trajectory(
        args.data_dir, args.traj_idx, start_step=args.start_step,
        end_step=args.end_step, velocity_mode="parabolic")

    results = rollout_trajectory(
        action_sequence=converted["actions"],
        base_config=cfg["air_hockey"],
        state_vector=converted["observations"][0],
        gif_path=args.gif_path,
        fps=args.fps,
        actual_paddle_positions=converted["observations"][:, :4],
    )
    print(f"Episode length: {results['episode_length']}, "
          f"return: {results['episode_return']:.2f}")
    if results["trajectory_loss"] is not None:
        print(f"Mean paddle trajectory loss: {results['trajectory_loss']:.4f}")


def cmd_inflection(args):
    output_dir = os.path.join(os.path.dirname(args.output_dir), "inflection_gifs")
    render_inflection_gif(
        data_dir=args.data_dir,
        traj_idx=args.traj_idx,
        output_dir=output_dir,
        log_path=args.log_path,
        name=args.name,
        fps=args.fps,
        start_step=args.start_step,
        end_step=args.end_step,
    )


def cmd_parabolic(args):
    output_dir = os.path.join(os.path.dirname(args.output_dir), "parabolic_gifs")
    render_parabolic_fit_gif(
        data_dir=args.data_dir,
        traj_idx=args.traj_idx,
        output_dir=output_dir,
        name=args.name,
        fps=args.fps,
        start_step=args.start_step,
        end_step=args.end_step,
        x_degree=args.x_degree,
        y_degree=args.y_degree,
    )


def cmd_puck_segments(args):
    import yaml
    import glob as globmod
    import tempfile
    import imageio

    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)

    # Determine which trajectory indices to use
    if args.traj_indices is not None:
        traj_indices = [int(x.strip()) for x in args.traj_indices.split(",")]
    else:
        # Discover all available log files
        pattern = os.path.join(args.log_dir, "inflection_*.json")
        traj_indices = sorted(
            int(os.path.basename(p).replace("inflection_", "").replace(".json", ""))
            for p in globmod.glob(pattern)
        )

    if not traj_indices:
        print("No trajectory indices found. Run inflection analysis first.")
        return

    # Load peak-start intervals: {traj_idx: [(peak_start, contact_end), ...]}
    intervals_by_traj = load_peak_start_intervals(args.log_dir, traj_indices)
    if not intervals_by_traj:
        print("No peak-start intervals found.")
        return

    output_dir_fd = os.path.join(args.output_dir, "short_puck_trajectories")
    output_dir_para = os.path.join(args.output_dir, "parabolic_short_puck_trajectories")
    os.makedirs(output_dir_fd, exist_ok=True)
    os.makedirs(output_dir_para, exist_ok=True)

    vel_modes = [
        ("finite_diff", "fd", output_dir_fd),
        ("parabolic", "parabolic", output_dir_para),
    ]

    segment_count = 0
    min_traj_length = 30
    for traj_idx in sorted(intervals_by_traj.keys()):
        intervals = intervals_by_traj[traj_idx]
        for iv_idx, (peak_start, contact_end) in enumerate(intervals):
            if args.max_segments is not None and segment_count >= args.max_segments:
                print(f"\nReached --max-segments={args.max_segments}, stopping.")
                return
            if contact_end - peak_start < min_traj_length:
                continue

            seg_name = f"traj{traj_idx}_seg{iv_idx}_t{peak_start}-{contact_end}"

            # Check if all sync GIFs already exist across both directories
            all_exist = all(
                os.path.exists(os.path.join(out_dir, seg_name, "sync.gif"))
                for _, _, out_dir in vel_modes
            )
            if all_exist:
                print(f"Skipping {seg_name} (already exists)")
                segment_count += 1
                continue

            print(f"\n{'='*60}")
            print(f"Segment: traj={traj_idx}, interval=[{peak_start}, {contact_end}]")
            print(f"{'='*60}")

            seg_len = contact_end - peak_start

            # Render real trajectory segment once (reused for both modes)
            # Use fd directory to hold the shared real GIF
            real_seg_dir = os.path.join(output_dir_fd, seg_name)
            os.makedirs(real_seg_dir, exist_ok=True)
            real_gif_name = "real.gif"
            render_trajectory_gif(
                data_dir=args.data_dir,
                traj_idx=traj_idx,
                output_dir=real_seg_dir,
                name=real_gif_name,
                fps=args.fps,
                start_step=peak_start,
                end_step=contact_end,
                show_velocity=args.show_velocity,
            )

            real_gif_path = os.path.join(real_seg_dir, real_gif_name)
            trimmed_real_path = real_gif_path

            # Rollout + sync for each velocity mode in its own directory
            for vel_mode, tag, out_dir in vel_modes:
                seg_dir = os.path.join(out_dir, seg_name)
                os.makedirs(seg_dir, exist_ok=True)
                print(f"  -- velocity mode: {vel_mode}")

                converted = load_and_convert_trajectory(
                    args.data_dir, traj_idx, start_step=peak_start,
                    end_step=contact_end, velocity_mode=vel_mode)

                obs_seg = converted["observations"]
                act_seg = converted["actions"]

                rollout_gif_path = os.path.join(seg_dir, "rollout.gif")
                results = rollout_trajectory(
                    action_sequence=act_seg,
                    base_config=cfg["air_hockey"],
                    state_vector=obs_seg[0],
                    gif_path=rollout_gif_path,
                    fps=args.fps,
                    actual_paddle_positions=obs_seg[:, :4],
                    show_velocity=args.show_velocity,
                )

                if results["trajectory_loss"] is not None:
                    print(f"    Paddle loss ({tag}): {results['trajectory_loss']:.4f}")

                # Side-by-side sync
                sync_path = os.path.join(seg_dir, "sync.gif")
                produce_synchronized_gifs(
                    trimmed_real_path, rollout_gif_path, sync_path,
                    title_left="Real",
                    title_right=f"Sim ({tag})",
                    fps=args.fps,
                )

            segment_count += 1

    print(f"\nDone. Rendered {segment_count} segments to:")
    print(f"  finite_diff: {output_dir_fd}")
    print(f"  parabolic:   {output_dir_para}")


def cmd_sync(args):
    import yaml

    # 1. Render real trajectory
    traj_gif_name = f"trajectory_{args.traj_idx}.gif"
    print(f"--- 1/3: Rendering real trajectory {args.traj_idx} ---")
    render_trajectory_gif(
        data_dir=args.data_dir,
        traj_idx=args.traj_idx,
        output_dir=args.output_dir,
        name=traj_gif_name,
        fps=args.fps,
        start_step=args.start_step,
        end_step=args.end_step,
        show_velocity=args.show_velocity,
    )
    traj_gif_path = os.path.join(args.output_dir, traj_gif_name)

    # 2. Sim rollout
    print(f"\n--- 2/3: Sim rollout ---")
    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)

    converted = load_and_convert_trajectory(
        args.data_dir, args.traj_idx, start_step=args.start_step,
        end_step=args.end_step, velocity_mode="finite_diff")

    rollout_gif_path = os.path.join(args.output_dir, "rollout.gif")
    results = rollout_trajectory(
        action_sequence=converted["actions"],
        base_config=cfg["air_hockey"],
        state_vector=converted["observations"][0],
        gif_path=rollout_gif_path,
        fps=args.fps,
        actual_paddle_positions=converted["observations"][:, :4],
        show_velocity=args.show_velocity,
    )
    print(f"Episode length: {results['episode_length']}, "
          f"return: {results['episode_return']:.2f}")
    if results["trajectory_loss"] is not None:
        print(f"Mean paddle trajectory loss: {results['trajectory_loss']:.4f}")

    # 3. Side-by-side
    print(f"\n--- 3/3: Synchronized GIF ---")
    sync_gif_path = os.path.join(args.output_dir, f"sync_{args.traj_idx}.gif")
    produce_synchronized_gifs(traj_gif_path, rollout_gif_path, sync_gif_path,
                              fps=args.fps)


if __name__ == "__main__":
    parser = _build_parser()
    parser.add_argument("--data-dir", type = str, default = "/data2/air_hockey/air_hockey_state_data/datastor1/calebc/public/data/mouse/state_data_all_new/")
    args = parser.parse_args()

    commands = {"render": cmd_render, "rollout": cmd_rollout, "sync": cmd_sync,
                "inflection": cmd_inflection, "parabolic": cmd_parabolic,
                "puck_segments": cmd_puck_segments}
    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()

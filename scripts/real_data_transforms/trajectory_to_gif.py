"""Render real trajectories and sim rollouts as GIFs.

Subcommands:
    render   – Render a real trajectory from HDF5 data
    rollout  – Roll out actions in simulation and save a GIF
    sync     – Side-by-side real vs. sim GIF

All subcommands accept --start-step to begin mid-trajectory.

Usage:
    python scripts/real_data_transforms/trajectory_to_gif.py render --traj-idx 100
    python scripts/real_data_transforms/trajectory_to_gif.py rollout --cfg config.yaml --traj-idx 100 --start-step 50
    python scripts/real_data_transforms/trajectory_to_gif.py sync --cfg config.yaml --traj-idx 100
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
from puck_inflection import analyze_trajectory, analyze_and_log
from airhockey import AirHockeyEnv
from airhockey.renderers import AirHockeyRenderer
from scripts.domain_adaptation.encode_env_params import assign_values


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_and_convert_trajectory(data_dir, traj_idx, start_step=0, dt=0.05):
    """Load a real trajectory and convert to sim format, optionally starting mid-way.

    When start_step > 0, the trajectory is sliced from that timestep onward.
    Velocities at the start point are estimated from the full trajectory before
    slicing, so the initial state has valid velocity estimates.

    Returns:
        converted: dict with 'observations' (N, 8), 'actions' (N, 2), 'hits_array'
    """
    filename = f"trajectory_data{traj_idx}.hdf5"
    traj = load_trajectory(data_dir, filename, load_images=False)
    converted = real_trajectory_to_sim_format(traj, dt=dt)

    if start_step > 0:
        converted["observations"] = converted["observations"][start_step:]
        converted["actions"] = converted["actions"][start_step:]
        converted["hits_array"] = converted["hits_array"][start_step:]

    return converted


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def rollout_trajectory(action_sequence, base_config, param_vector=None,
                       param_names=None, state_vector=None, gif_path=None,
                       fps=20, actual_paddle_positions=None):
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
    renderer = AirHockeyRenderer(env) if save else None
    frames = []
    rewards = []
    paddle_losses = []
    episode_return = 0.0
    cumulative_loss = 0.0
    n_steps = 0

    for t, action in enumerate(action_sequence):
        step_loss = None
        if actual_paddle_positions is not None and t < len(actual_paddle_positions):
            sim_paddle = np.array(obs[:4])
            step_loss = float(np.linalg.norm(sim_paddle - actual_paddle_positions[t]))
            paddle_losses.append(step_loss)
            cumulative_loss += step_loss

        if save:
            frame = renderer.get_frame()
            if step_loss is not None:
                annotate_loss(frame, step_loss, cumulative_loss)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        episode_return += reward
        n_steps += 1

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
                          start_step=0):
    """Render a real trajectory (from HDF5) as a GIF with sprite overlays."""
    filename = f"trajectory_data{traj_idx}.hdf5"
    traj = load_trajectory(data_dir, filename, load_images=False)

    paddle_positions = traj["paddle"][start_step:, :2]
    puck_data = traj["puck"][start_step:]

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

        if overlap:
            mid = ((paddle_px + puck_px) / 2).astype(int)
            cv2.circle(frame, tuple(mid), int(min_contact_dist * ppm), (0, 0, 255), 2)

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

    # Overlap report
    # if overlap_events:
    #     print(f"\nOverlap at {len(overlap_events)} timesteps "
    #           f"(min_contact_dist={min_contact_dist:.5f}):")
    #     for t, d, pen in overlap_events:
    #         print(f"  t={t:4d}  dist={d:.5f}  penetration={pen:.5f}")
    # else:
    #     print("\nNo puck-paddle overlap detected.")

    os.makedirs(output_dir, exist_ok=True)
    if name is None:
        name = f"trajectory_{traj_idx}.gif"
    gif_path = os.path.join(output_dir, name)
    save_gif(frames, gif_path, fps)


def render_inflection_gif(data_dir, traj_idx, output_dir, log_path=None,
                          name=None, fps=20, start_step=0):
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
    peaks = set(analysis["peaks"])
    contact_timesteps = {c["timestep"] for c in analysis["contacts"]}
    # Map each individual peak timestep to its contact info for annotation.
    # Each entry in peaks_before_contacts now has a list of peak_timesteps.
    peak_before_map = {}
    for entry in analysis["peaks_before_contacts"]:
        for pt in entry["peak_timesteps"]:
            peak_before_map[pt] = entry

    # Build set of "falling" timesteps (between each peak and the next valley)
    falling_timesteps = set()
    sorted_peaks = sorted(analysis["peaks"])
    sorted_valleys = sorted(analysis["valleys"])
    n_total = len(traj["puck"])
    for pk in sorted_peaks:
        # Find the next valley after this peak
        next_valleys = [v for v in sorted_valleys if v > pk]
        end = next_valleys[0] if next_valleys else n_total - 1
        for t in range(pk, end + 1):
            falling_timesteps.add(t)

    paddle_positions = traj["paddle"][start_step:, :2]
    puck_data = traj["puck"][start_step:]

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

    return parser


def cmd_render(args):
    render_trajectory_gif(
        data_dir=args.data_dir,
        traj_idx=args.traj_idx,
        output_dir=args.output_dir,
        name=args.name,
        fps=args.fps,
        start_step=args.start_step,
    )


def cmd_rollout(args):
    import yaml
    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)

    converted = load_and_convert_trajectory(
        args.data_dir, args.traj_idx, start_step=args.start_step)

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
    )


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
    )
    traj_gif_path = os.path.join(args.output_dir, traj_gif_name)

    # 2. Sim rollout
    print(f"\n--- 2/3: Sim rollout ---")
    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)

    converted = load_and_convert_trajectory(
        args.data_dir, args.traj_idx, start_step=args.start_step)

    rollout_gif_path = os.path.join(args.output_dir, "rollout.gif")
    results = rollout_trajectory(
        action_sequence=converted["actions"],
        base_config=cfg["air_hockey"],
        state_vector=converted["observations"][0],
        gif_path=rollout_gif_path,
        fps=args.fps,
        actual_paddle_positions=converted["observations"][:, :4],
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
                "inflection": cmd_inflection}
    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()

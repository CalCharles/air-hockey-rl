"""Generate ~20 parabolic short-trajectory GIFs for approach intervals.

Selects approach intervals (peak → paddle contact, no wall hits) from
inflection logs and renders:
  1. Parabolic-fit GIF (real data with fitted curves + velocity arrows)
  2. Sim rollout GIF (using estimate.yaml config)
  3. Side-by-side sync GIF (real vs sim)

Usage:
    # Batch mode: select ~20 diverse approach intervals
    python generate_approach_gifs.py \
        --cfg ../domain_adaptation/realworld_paddle_config/estimate.yaml \
        --num-gifs 20

    # Single trajectory with explicit start/end steps
    python generate_approach_gifs.py \
        --cfg ../domain_adaptation/realworld_paddle_config/estimate.yaml \
        --traj-idx 100 --start-step 254 --end-step 269
"""

import os
import sys
import argparse
import glob as globmod
import json

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from puck_inflection import load_approach_intervals
from trajectory_to_gif import (
    render_parabolic_fit_gif,
    render_trajectory_gif,
    rollout_trajectory,
    load_and_convert_trajectory,
)
from rendering_utils import produce_synchronized_gifs

DEFAULT_DATA_DIR = "/data2/air_hockey/air_hockey_state_data/datastor1/calebc/public/data/mouse/state_data_all_new/"
DEFAULT_LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")


def select_intervals(log_dir, traj_indices, num_gifs=20, min_duration=15,
                     max_duration=50, seed=42):
    """Select diverse approach intervals spread across durations."""
    intervals_by_traj, _ = load_approach_intervals(
        log_dir, traj_indices, require_fitted_peak=True)

    # Flatten all intervals with their durations
    all_intervals = []
    for traj_idx, ivs in intervals_by_traj.items():
        for start, end in ivs:
            dur = end - start
            if min_duration <= dur <= max_duration:
                all_intervals.append((traj_idx, start, end, dur))

    if not all_intervals:
        print("No intervals found in the specified duration range.")
        return []

    # Sort by duration for stratified sampling
    all_intervals.sort(key=lambda x: x[3])
    print(f"Found {len(all_intervals)} intervals in duration range "
          f"[{min_duration}, {max_duration}]")

    if len(all_intervals) <= num_gifs:
        selected = all_intervals
    else:
        # Stratified sampling: evenly spaced across sorted durations
        indices = np.linspace(0, len(all_intervals) - 1, num_gifs, dtype=int)
        selected = [all_intervals[i] for i in indices]

    print(f"Selected {len(selected)} intervals:")
    for traj_idx, start, end, dur in selected:
        print(f"  traj={traj_idx:4d}  [{start:4d}, {end:4d}]  dur={dur}")

    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Generate parabolic approach interval GIFs")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--log-dir", type=str, default=DEFAULT_LOG_DIR)
    parser.add_argument("--cfg", type=str, required=True,
                        help="Path to sim config YAML (e.g. estimate.yaml)")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "gifs", "approach_intervals"))
    parser.add_argument("--num-gifs", type=int, default=20)
    parser.add_argument("--min-duration", type=int, default=15)
    parser.add_argument("--max-duration", type=int, default=50)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    # Single-trajectory mode
    parser.add_argument("--traj-idx", type=int, default=None,
                        help="Single trajectory index to render")
    parser.add_argument("--start-step", type=int, default=None,
                        help="Start timestep (peak). Required with --traj-idx.")
    parser.add_argument("--end-step", type=int, default=None,
                        help="End timestep (contact). Required with --traj-idx.")
    args = parser.parse_args()

    import yaml
    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)

    # Single-trajectory mode
    if args.traj_idx is not None:
        if args.start_step is None or args.end_step is None:
            parser.error("--start-step and --end-step are required with --traj-idx")
        dur = args.end_step - args.start_step
        selected = [(args.traj_idx, args.start_step, args.end_step, dur)]
    else:
        # Batch mode: discover all trajectory indices from logs
        pattern = os.path.join(args.log_dir, "inflection_*.json")
        traj_indices = sorted(
            int(os.path.basename(p).replace("inflection_", "").replace(".json", ""))
            for p in globmod.glob(pattern)
        )

        selected = select_intervals(
            args.log_dir, traj_indices, num_gifs=args.num_gifs,
            min_duration=args.min_duration, max_duration=args.max_duration,
            seed=args.seed)

    if not selected:
        return

    os.makedirs(args.output_dir, exist_ok=True)

    for i, (traj_idx, start, end, dur) in enumerate(selected):
        seg_name = f"traj{traj_idx}_t{start}-{end}"
        seg_dir = os.path.join(args.output_dir, seg_name)
        os.makedirs(seg_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(selected)}] traj={traj_idx}  "
              f"[{start}, {end}]  dur={dur}")
        print(f"{'='*60}")

        # 1. Parabolic fit GIF (real data with fitted curves)
        parabolic_path = os.path.join(seg_dir, "parabolic.gif")
        if not os.path.exists(parabolic_path):
            print("  Rendering parabolic fit...")
            render_parabolic_fit_gif(
                data_dir=args.data_dir,
                traj_idx=traj_idx,
                output_dir=seg_dir,
                name="parabolic.gif",
                fps=args.fps,
                start_step=start,
                end_step=end,
            )
        else:
            print("  Parabolic fit GIF already exists, skipping.")

        # 2. Real trajectory GIF
        real_path = os.path.join(seg_dir, "real.gif")
        if not os.path.exists(real_path):
            print("  Rendering real trajectory...")
            render_trajectory_gif(
                data_dir=args.data_dir,
                traj_idx=traj_idx,
                output_dir=seg_dir,
                name="real.gif",
                fps=args.fps,
                start_step=start,
                end_step=end,
                show_velocity=True,
            )
        else:
            print("  Real GIF already exists, skipping.")

        # 3. Sim rollout GIF (using estimate.yaml)
        rollout_path = os.path.join(seg_dir, "sim_rollout.gif")
        if not os.path.exists(rollout_path):
            print("  Running sim rollout...")
            converted = load_and_convert_trajectory(
                args.data_dir, traj_idx,
                start_step=start, end_step=end,
                velocity_mode="parabolic")

            results = rollout_trajectory(
                action_sequence=converted["actions"],
                base_config=cfg["air_hockey"],
                state_vector=converted["observations"][0],
                gif_path=rollout_path,
                fps=args.fps,
                actual_paddle_positions=converted["observations"][:, :4],
                show_velocity=True,
            )

            loss = results["trajectory_loss"]
            if loss is not None:
                print(f"    Paddle loss: {loss:.4f}")
        else:
            print("  Sim rollout GIF already exists, skipping.")

        # 4. Side-by-side sync GIF
        sync_path = os.path.join(seg_dir, "sync.gif")
        if not os.path.exists(sync_path):
            print("  Producing sync GIF...")
            produce_synchronized_gifs(
                real_path, rollout_path, sync_path,
                title_left="Real",
                title_right="Sim (estimate.yaml)",
                fps=args.fps,
            )
        else:
            print("  Sync GIF already exists, skipping.")

    print(f"\nDone! {len(selected)} approach interval GIF sets saved to:")
    print(f"  {args.output_dir}")


if __name__ == "__main__":
    main()

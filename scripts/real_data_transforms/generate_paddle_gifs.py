"""Generate ~20 longer-horizon paddle trajectory GIFs.

Selects contiguous segments from real trajectory HDF5 files and renders:
  1. Real trajectory GIF (paddle + puck + target marker)
  2. Sim rollout GIF (using estimate.yaml config)
  3. Side-by-side sync GIF (real vs sim)

Unlike generate_approach_gifs.py (which uses short puck approach intervals),
this script picks longer segments (default 100-200 steps) sampled uniformly
across the available trajectory files.

Usage:
    python generate_paddle_gifs.py \
        --cfg ../domain_adaptation/realworld_paddle_config/estimate.yaml \
        --num-gifs 20 --segment-length 150 --fps 20
"""

import os
import sys
import argparse
import glob as globmod

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from data_loading import load_trajectory
from trajectory_to_gif import (
    render_trajectory_gif,
    rollout_trajectory,
    load_and_convert_trajectory,
)
from rendering_utils import produce_synchronized_gifs

DEFAULT_DATA_DIR = (
    "/data2/air_hockey/air_hockey_state_data/datastor1/calebc/public/data/"
    "mouse/state_data_all_new/"
)


# ---------------------------------------------------------------------------
# Segment selection
# ---------------------------------------------------------------------------

def discover_trajectory_indices(data_dir):
    """Return sorted list of available trajectory indices from HDF5 files."""
    files = sorted(
        globmod.glob(os.path.join(data_dir, "trajectory_data*.hdf5")),
        key=lambda p: int(
            os.path.basename(p)[len("trajectory_data"):-len(".hdf5")]
        ),
    )
    return [
        int(os.path.basename(p)[len("trajectory_data"):-len(".hdf5")])
        for p in files
    ]


def select_segments(data_dir, traj_indices, num_gifs=20, segment_length=150,
                    seed=42):
    """Pick diverse (traj_idx, start, end) segments spread across trajectories.

    Strategy:
      1. For each trajectory, compute how many non-overlapping segments of
         `segment_length` fit in it.
      2. Flatten all candidate (traj_idx, start, end) tuples.
      3. Stratified-sample `num_gifs` of them evenly spaced across the list.
    """
    rng = np.random.default_rng(seed)

    all_candidates = []
    for tidx in traj_indices:
        filename = f"trajectory_data{tidx}.hdf5"
        try:
            traj = load_trajectory(data_dir, filename, load_images=False)
        except Exception as e:
            print(f"  Warning: could not load traj {tidx}: {e}")
            continue

        n = len(traj["paddle"])
        if n < segment_length + 1:
            continue

        # Generate all possible non-overlapping starts
        starts = list(range(0, n - segment_length, segment_length // 2))
        for start in starts:
            end = start + segment_length - 1
            all_candidates.append((tidx, start, end))

    if not all_candidates:
        print("No valid segments found.")
        return []

    print(f"Found {len(all_candidates)} candidate segments "
          f"(segment_length={segment_length})")

    if len(all_candidates) <= num_gifs:
        selected = all_candidates
    else:
        indices = np.linspace(0, len(all_candidates) - 1, num_gifs, dtype=int)
        selected = [all_candidates[i] for i in indices]

    print(f"Selected {len(selected)} segments:")
    for tidx, start, end in selected:
        print(f"  traj={tidx:4d}  [{start:4d}, {end:4d}]  len={end-start+1}")

    return selected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate longer-horizon paddle trajectory GIFs")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cfg", type=str, required=True,
                        help="Path to sim config YAML (e.g. estimate.yaml)")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "gifs", "paddle-trajectories-1"))
    parser.add_argument("--num-gifs", type=int, default=20)
    parser.add_argument("--segment-length", type=int, default=150,
                        help="Number of timesteps per segment")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    # Single-segment mode
    parser.add_argument("--traj-idx", type=int, default=None,
                        help="Render a single trajectory index")
    parser.add_argument("--start-step", type=int, default=None)
    parser.add_argument("--end-step", type=int, default=None)
    args = parser.parse_args()

    import yaml
    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)

    if args.traj_idx is not None:
        start = args.start_step or 0
        traj = load_trajectory(
            args.data_dir, f"trajectory_data{args.traj_idx}.hdf5",
            load_images=False)
        end = args.end_step if args.end_step is not None else len(traj["paddle"]) - 1
        selected = [(args.traj_idx, start, end)]
    else:
        traj_indices = discover_trajectory_indices(args.data_dir)
        if not traj_indices:
            print(f"No trajectory HDF5 files found in {args.data_dir}")
            return
        print(f"Discovered {len(traj_indices)} trajectory files.")

        selected = select_segments(
            args.data_dir, traj_indices,
            num_gifs=args.num_gifs,
            segment_length=args.segment_length,
            seed=args.seed,
        )

    if not selected:
        return

    os.makedirs(args.output_dir, exist_ok=True)

    for i, (traj_idx, start, end) in enumerate(selected):
        seg_name = f"traj{traj_idx}_t{start}-{end}"
        seg_dir = os.path.join(args.output_dir, seg_name)
        os.makedirs(seg_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(selected)}] traj={traj_idx}  "
              f"[{start}, {end}]  len={end-start+1}")
        print(f"{'='*60}")

        # 1. Real trajectory GIF
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
                show_velocity=False,
            )
        else:
            print("  Real GIF already exists, skipping.")

        # 2. Sim rollout GIF
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
                show_velocity=False,
            )

            loss = results["trajectory_loss"]
            if loss is not None:
                print(f"    Paddle loss: {loss:.4f}")
        else:
            print("  Sim rollout GIF already exists, skipping.")

        # 3. Side-by-side sync GIF
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

    print(f"\nDone! {len(selected)} paddle trajectory GIF sets saved to:")
    print(f"  {args.output_dir}")


if __name__ == "__main__":
    main()

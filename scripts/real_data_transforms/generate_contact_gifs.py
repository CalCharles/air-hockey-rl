"""Generate ~20 GIFs centred on paddle contact events.

Each segment starts a few steps *before* a paddle contact and ends
`--post-steps` (default 25) steps *after* it, so you can clearly see
the puck behaviour immediately before and after impact.

Renders three GIFs per contact:
  1. real.gif         real trajectory (paddle + puck + target marker)
  2. sim_rollout.gif  sim replay using estimate.yaml
  3. sync.gif         side-by-side real vs. sim

Contact events are loaded from the pre-computed inflection JSON logs.
Only paddle contacts (not wall hits) are used.

Usage:
    python generate_contact_gifs.py \
        --cfg ../domain_adaptation/realworld_paddle_config/estimate.yaml \
        --num-gifs 20 --pre-steps 5 --post-steps 25 --fps 20
"""

import os
import sys
import argparse
import json
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
DEFAULT_LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")


# ---------------------------------------------------------------------------
# Contact selection
# ---------------------------------------------------------------------------

def load_paddle_contacts(log_dir, traj_indices):
    """Return list of (traj_idx, contact_timestep) for all paddle contacts."""
    all_contacts = []
    missing = 0

    for traj_idx in traj_indices:
        log_path = os.path.join(log_dir, f"inflection_{traj_idx}.json")
        if not os.path.exists(log_path):
            missing += 1
            continue
        with open(log_path) as f:
            data = json.load(f)

        for c in data.get("contacts", []):
            if c.get("type") == "paddle":
                all_contacts.append((traj_idx, c["timestep"]))

    print(f"Found {len(all_contacts)} paddle contacts "
          f"({missing} logs missing)")
    return all_contacts


def select_contacts(all_contacts, num_gifs=20, seed=42):
    """Stratified-sample `num_gifs` contacts evenly spread across the list."""
    if not all_contacts:
        return []

    if len(all_contacts) <= num_gifs:
        selected = all_contacts
    else:
        # Sort by (traj_idx, timestep) so we sample across the full dataset
        sorted_contacts = sorted(all_contacts)
        indices = np.linspace(0, len(sorted_contacts) - 1, num_gifs, dtype=int)
        selected = [sorted_contacts[i] for i in indices]

    print(f"Selected {len(selected)} contacts:")
    for traj_idx, t in selected:
        print(f"  traj={traj_idx:4d}  contact_t={t:5d}")
    return selected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate paddle-contact GIFs (pre→contact→post)")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--log-dir", type=str, default=DEFAULT_LOG_DIR)
    parser.add_argument("--cfg", type=str, required=True,
                        help="Path to sim config YAML (e.g. estimate.yaml)")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "gifs", "contact-comparisons-2"))
    parser.add_argument("--num-gifs", type=int, default=20)
    parser.add_argument("--pre-steps", type=int, default=5,
                        help="Steps before contact to start segment")
    parser.add_argument("--post-steps", type=int, default=25,
                        help="Steps after contact to end segment")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import yaml
    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)

    # Discover all available log indices
    log_files = sorted(globmod.glob(os.path.join(args.log_dir, "inflection_*.json")))
    traj_indices = [
        int(os.path.basename(p).replace("inflection_", "").replace(".json", ""))
        for p in log_files
    ]
    print(f"Found {len(traj_indices)} inflection logs.")

    all_contacts = load_paddle_contacts(args.log_dir, traj_indices)
    selected = select_contacts(all_contacts, num_gifs=args.num_gifs, seed=args.seed)

    if not selected:
        print("No contacts found.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    for i, (traj_idx, contact_t) in enumerate(selected):
        # Load traj to know its length
        try:
            traj = load_trajectory(
                args.data_dir, f"trajectory_data{traj_idx}.hdf5",
                load_images=False)
        except Exception as e:
            print(f"  Could not load traj {traj_idx}: {e}, skipping.")
            continue

        n = len(traj["paddle"])
        start = max(0, contact_t - args.pre_steps)
        end = min(n - 1, contact_t + args.post_steps)

        seg_name = f"traj{traj_idx}_contact{contact_t}_t{start}-{end}"
        seg_dir = os.path.join(args.output_dir, seg_name)
        os.makedirs(seg_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(selected)}] traj={traj_idx}  "
              f"contact_t={contact_t}  [{start}, {end}]  len={end-start+1}")
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
                show_velocity=True,
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
                show_velocity=True,
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

    print(f"\nDone! {len(selected)} contact GIF sets saved to:")
    print(f"  {args.output_dir}")


if __name__ == "__main__":
    main()

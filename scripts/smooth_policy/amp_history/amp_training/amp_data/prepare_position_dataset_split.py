#!/usr/bin/env python3
"""
Prepare AMP dataset with 5 consecutive position states from split-schema real robot trajectories.

This script mirrors prepare_position_dataset.py behavior but reads trajectories stored as
separate HDF5 datasets:
  cur_time, tidx, i, estop, safety, pose, speed, force, acc, desired_pose, puck

Those fields are assembled into canonical train_vals layout:
  [cur_time, tidx, i, estop, safety, pose(6), speed(6), force(6), acc(3), desired_pose(6), puck(3)]
with width 35, then processed using the same filtering/extraction logic as the train_vals pipeline.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from scripts.smooth_policy.amp_history.amp_training.amp_data.prepare_position_dataset import (
    extract_position_sequences,
    filter_trajectory,
    find_trajectory_files,
    load_demo_list,
    print_dataset_statistics,
    save_dataset,
    validate_dataset,
)


SPLIT_DATASET_SPECS = (
    ("cur_time", 1),
    ("tidx", 1),
    ("i", 1),
    ("estop", 1),
    ("safety", 1),
    ("pose", 6),
    ("speed", 6),
    ("force", 6),
    ("acc", 3),
    ("desired_pose", 6),
    ("puck", 3),
)


def _read_2d_dataset(h5_file, key, width):
    """Read one split-schema dataset as shape (T, width)."""
    if key not in h5_file:
        raise KeyError(f"Missing required dataset '{key}'")
    arr = np.asarray(h5_file[key][:])
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"Dataset '{key}' must be 1D/2D, got shape {arr.shape}")
    if arr.shape[1] != width:
        raise ValueError(
            f"Dataset '{key}' expected width {width}, got shape {arr.shape}"
        )
    return arr


def load_train_vals_from_split_hdf5(file_path):
    """
    Load one split-schema trajectory and assemble canonical train_vals (T, 35).
    """
    with h5py.File(file_path, "r") as f:
        missing = [name for name, _ in SPLIT_DATASET_SPECS if name not in f]
        if missing:
            raise ValueError(
                f"File {Path(file_path).name} is not split-schema compatible. "
                f"Missing datasets: {missing}"
            )

        parts = []
        n_rows = None
        for name, width in SPLIT_DATASET_SPECS:
            part = _read_2d_dataset(f, name, width)
            if n_rows is None:
                n_rows = part.shape[0]
            elif part.shape[0] != n_rows:
                raise ValueError(
                    f"Inconsistent timestep count in {Path(file_path).name}: "
                    f"dataset '{name}' has {part.shape[0]} rows, expected {n_rows}"
                )
            parts.append(part)

    return np.concatenate(parts, axis=1)


def process_all_trajectories(
    data_dir,
    min_length=50,
    max_trajectories=None,
    safety_check=True,
    demo_list=None,
    sequence_length=5,
    include_actions=False,
    include_puck=False,
):
    """
    Process all split-schema trajectory files and collect position sequences.

    Returns shape/stat formats identical to prepare_position_dataset.py.
    """
    trajectory_files = find_trajectory_files(data_dir, demo_list)

    if max_trajectories is not None:
        trajectory_files = trajectory_files[:max_trajectories]

    print(f"Found {len(trajectory_files)} trajectory files to process")
    if demo_list is not None:
        print(f"  (filtered by demo list: {len(demo_list)} trajectories specified)")

    all_position_sequences = []
    all_action_sequences = [] if include_actions else None
    all_puck_sequences = [] if include_puck else None
    stats = {
        "total_files": len(trajectory_files),
        "valid_trajectories": 0,
        "skipped_trajectories": 0,
        "total_sequences": 0,
        "total_timesteps": 0,
        "error_files": [],
        "include_actions": include_actions,
        "include_puck": include_puck,
    }

    for file_path in tqdm(trajectory_files, desc="Processing trajectories"):
        try:
            train_vals = load_train_vals_from_split_hdf5(file_path)

            if not filter_trajectory(
                train_vals,
                min_length,
                safety_check,
                include_actions=include_actions,
                include_puck=include_puck,
            ):
                stats["skipped_trajectories"] += 1
                continue

            result = extract_position_sequences(
                train_vals,
                sequence_length=sequence_length,
                include_actions=include_actions,
                include_puck=include_puck,
            )

            if include_actions and include_puck:
                position_sequences, action_sequences, puck_sequences = result
                all_position_sequences.append(position_sequences)
                all_action_sequences.append(action_sequences)
                all_puck_sequences.append(puck_sequences)
            elif include_actions:
                position_sequences, action_sequences = result
                all_position_sequences.append(position_sequences)
                all_action_sequences.append(action_sequences)
            elif include_puck:
                position_sequences, puck_sequences = result
                all_position_sequences.append(position_sequences)
                all_puck_sequences.append(puck_sequences)
            else:
                all_position_sequences.append(result)

            stats["valid_trajectories"] += 1
            stats["total_sequences"] += (
                position_sequences.shape[0]
                if (include_actions or include_puck)
                else result.shape[0]
            )
            stats["total_timesteps"] += train_vals.shape[0]

        except Exception as exc:
            print(f"\nError processing {file_path.name}: {exc}")
            stats["skipped_trajectories"] += 1
            stats["error_files"].append((file_path.name, str(exc)))
            continue

    if not all_position_sequences:
        raise ValueError("No valid trajectories found!")

    position_dataset = np.concatenate(all_position_sequences, axis=0)

    if include_actions and include_puck:
        action_dataset = np.concatenate(all_action_sequences, axis=0)
        puck_dataset = np.concatenate(all_puck_sequences, axis=0)
        return position_dataset, action_dataset, puck_dataset, stats
    if include_actions:
        action_dataset = np.concatenate(all_action_sequences, axis=0)
        return position_dataset, action_dataset, stats
    if include_puck:
        puck_dataset = np.concatenate(all_puck_sequences, axis=0)
        return position_dataset, puck_dataset, stats

    return position_dataset, stats


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Prepare AMP dataset with 5 consecutive position states from split-schema "
            "real robot trajectories"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/nfs/data/airhockey",
        help="Directory containing split-schema trajectory HDF5 files",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=(
            "scripts/smooth_policy/amp_history/amp_training/amp_data/"
            "amp_position_dataset_split.pt"
        ),
        help="Output path for the PyTorch dataset file",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=50,
        help="Minimum trajectory length to include",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="Maximum number of trajectories to process (for testing)",
    )
    parser.add_argument(
        "--no-safety-check",
        action="store_true",
        help="Disable safety flag filtering",
    )
    parser.add_argument(
        "--demo-list",
        type=str,
        default=None,
        help="Path to text file containing list of trajectory names to process (one per line)",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=5,
        help="Number of consecutive states per sequence",
    )
    parser.add_argument(
        "--include-actions",
        action="store_true",
        help="Include action sequences (delta target positions) in the dataset",
    )
    parser.add_argument(
        "--include-puck",
        action="store_true",
        help="Include aligned 5-step puck windows [t..t+4] for each paddle sequence",
    )

    return parser.parse_args()


def main():
    """Main function to prepare AMP position dataset from split-schema trajectories."""
    args = parse_args()
    if args.include_puck and args.sequence_length != 5:
        raise ValueError("--include-puck currently requires --sequence-length 5.")

    print("=" * 80)
    print("AMP POSITION DATASET PREPARATION (SPLIT SCHEMA)")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Output path: {args.output_path}")
    print(f"  Min trajectory length: {args.min_length}")
    print(f"  Max trajectories: {args.max_trajectories or 'unlimited'}")
    print(f"  Safety check: {not args.no_safety_check}")
    print(f"  Sequence length: {args.sequence_length}")
    print(f"  Include actions: {args.include_actions}")
    print(f"  Include puck windows: {args.include_puck}")

    demo_list = None
    if args.demo_list:
        print(f"  Demo list file: {args.demo_list}")
        demo_list = load_demo_list(args.demo_list)
        print(f"  Loaded {len(demo_list)} trajectory IDs from demo list")

    result = process_all_trajectories(
        args.data_dir,
        min_length=args.min_length,
        max_trajectories=args.max_trajectories,
        safety_check=not args.no_safety_check,
        demo_list=demo_list,
        sequence_length=args.sequence_length,
        include_actions=args.include_actions,
        include_puck=args.include_puck,
    )

    if args.include_actions and args.include_puck:
        position_dataset, action_dataset, puck_dataset, stats = result
    elif args.include_actions:
        position_dataset, action_dataset, stats = result
        puck_dataset = None
    elif args.include_puck:
        position_dataset, puck_dataset, stats = result
        action_dataset = None
    else:
        position_dataset, stats = result
        action_dataset = None
        puck_dataset = None

    print_dataset_statistics(position_dataset, stats, action_dataset, puck_dataset)

    if not validate_dataset(
        position_dataset, args.sequence_length, action_dataset, puck_dataset
    ):
        print("\nWARNING: Validation failed! Proceeding with save anyway...")

    output_path = Path(args.output_path)
    save_dataset(position_dataset, output_path, stats, action_dataset, puck_dataset)

    print("\n" + "=" * 80)
    print("✓ Dataset preparation complete!")
    print("=" * 80)

    print("\nTo load the dataset in Python:")
    print("  import torch")
    print(f"  data = torch.load('{args.output_path}')")
    print("  position_sequences = data['position_sequences']  # Shape: (N, 5, 2)")
    print("  state1 = position_sequences[:, 0, :]  # (N, 2)")
    print("  state2 = position_sequences[:, 1, :]  # (N, 2)")
    print("  state3 = position_sequences[:, 2, :]  # (N, 2)")
    print("  state4 = position_sequences[:, 3, :]  # (N, 2)")
    print("  state5 = position_sequences[:, 4, :]  # (N, 2)")

    if args.include_actions:
        print("\n  # With --include-actions flag:")
        print("  action_sequences = data['action_sequences']  # Shape: (N, 4, 2)")
        print("  # transition actions aligned with state transitions in each 5-state window")
        print("  action12 = action_sequences[:, 0, :]  # (N, 2) - action s1->s2")
        print("  action45 = action_sequences[:, 3, :]  # (N, 2) - action s4->s5")
    if args.include_puck:
        print("\n  # With --include-puck flag:")
        print("  puck_sequences = data['puck_sequences']  # Shape: (N, 5, 2)")
        print("  # aligned with paddle windows [t..t+4]; index 2 is two timesteps before s5")


if __name__ == "__main__":
    main()

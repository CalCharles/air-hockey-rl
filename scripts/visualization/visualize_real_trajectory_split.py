#!/usr/bin/env python3
"""
Visualize split-schema real robot trajectory data as GIFs.

This script is for HDF5 files that store state in separate datasets, e.g.:
  cur_time, tidx, i, estop, safety, pose, speed, force, acc, desired_pose, puck

It converts those datasets into the canonical train_vals layout and reuses the
renderer/pipeline from visualize_real_trajectory.py.
"""

from pathlib import Path
import argparse
import sys

import h5py
import numpy as np

try:
    # Works when running this file directly from its own folder.
    from visualize_real_trajectory import (
        RealTrajectoryRenderer,
        create_trajectory_gif,
        extract_paddle_data,
        print_trajectory_statistics,
    )
except ModuleNotFoundError:
    # Works when running from repo root.
    from scripts.visualization.visualize_real_trajectory import (
        RealTrajectoryRenderer,
        create_trajectory_gif,
        extract_paddle_data,
        print_trajectory_statistics,
    )


SPLIT_DATASETS = (
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

OPTIONAL_SPLIT_DATASETS = (
    ("timing", 9),
    ("paddle_actual", 6),
    ("paddle_cmd", 12),
    ("puck_meta", 2),
    ("stop_flags", 3),
    ("reset_stage_id", 1),
    # Policy-only fields (present on policy-collection HDF5s, absent on
    # reset-FSM HDF5s). Together with `desired_pose` + `puck` they make a
    # policy episode's HDF5 self-sufficient for offline policy replay /
    # comparison without needing the runtime replay buffer.
    ("policy_action", 2),   # Raw normalized [-1, 1] action executed this step
    ("task_reward", 1),     # Per-step env task reward
    ("done", 1),            # Same no-bootstrap done flag stored in replay buffer
)
OPTIONAL_ALLOWED_WIDTHS = {
    "timing": (8, 9),
}


def _read_2d_dataset(h5_file, key, width):
    """Read dataset as 2D array of shape (N, width)."""
    if key not in h5_file:
        raise KeyError(f"Missing dataset '{key}'")

    arr = np.asarray(h5_file[key][:], dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"Dataset '{key}' must be 1D/2D, got shape {arr.shape}")
    if arr.shape[1] != width:
        raise ValueError(
            f"Dataset '{key}' expected width {width}, got shape {arr.shape}"
        )
    return arr


def _read_2d_dataset_with_allowed_widths(h5_file, key, allowed_widths):
    """Read dataset as 2D array with width in allowed_widths."""
    if key not in h5_file:
        raise KeyError(f"Missing dataset '{key}'")

    arr = np.asarray(h5_file[key][:], dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"Dataset '{key}' must be 1D/2D, got shape {arr.shape}")
    if arr.shape[1] not in tuple(int(w) for w in allowed_widths):
        raise ValueError(
            f"Dataset '{key}' expected widths {tuple(allowed_widths)}, got shape {arr.shape}"
        )
    return arr


def load_split_trajectory_data(filepath):
    """
    Load split-schema trajectory and concatenate into canonical train_vals order.

    Returns:
        numpy.ndarray: Array with shape (N, 35) in canonical train_vals layout.
    """
    print(f"Loading split-schema trajectory data from: {filepath}")
    with h5py.File(filepath, "r") as f:
        keys = set(f.keys())
        missing = [name for name, _ in SPLIT_DATASETS if name not in keys]
        if missing:
            raise KeyError(
                f"File does not match split schema; missing datasets: {missing}"
            )

        parts = []
        n_rows = None
        for name, width in SPLIT_DATASETS:
            part = _read_2d_dataset(f, name, width)
            if n_rows is None:
                n_rows = part.shape[0]
            elif part.shape[0] != n_rows:
                raise ValueError(
                    f"Inconsistent timestep count for '{name}': "
                    f"{part.shape[0]} vs expected {n_rows}"
                )
            parts.append(part)

    train_vals = np.concatenate(parts, axis=1)
    print(f"  Shape (assembled train_vals): {train_vals.shape}")
    print(f"  Timesteps: {train_vals.shape[0]}")
    print(f"  Features per timestep: {train_vals.shape[1]}")
    return train_vals


def load_split_optional_data(filepath):
    """Load optional split-schema datasets when present.

    Returns:
        dict[str, np.ndarray]: Mapping from optional dataset name to array.
    """
    optional_data = {}
    with h5py.File(filepath, "r") as f:
        n_rows = None
        for name, width in OPTIONAL_SPLIT_DATASETS:
            if name not in f:
                continue
            if name in OPTIONAL_ALLOWED_WIDTHS:
                data = _read_2d_dataset_with_allowed_widths(
                    f, name, OPTIONAL_ALLOWED_WIDTHS[name]
                )
            else:
                data = _read_2d_dataset(f, name, width)
            if n_rows is None and "cur_time" in f:
                n_rows = _read_2d_dataset(f, "cur_time", 1).shape[0]
            if n_rows is not None and data.shape[0] != n_rows:
                raise ValueError(
                    f"Inconsistent timestep count for optional '{name}': "
                    f"{data.shape[0]} vs expected {n_rows}"
                )
            optional_data[name] = data
    return optional_data


def visualize_single_file(data_path, output_dir, args):
    """Generate one visualization GIF for a single split-schema HDF5 file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "trajectory_visualization.gif"

    print("=" * 80)
    print("REAL ROBOT TRAJECTORY VISUALIZATION (SPLIT SCHEMA)")
    print("=" * 80)
    print(f"\nInput file: {data_path}")
    print(f"Output directory: {output_dir}")
    print(f"Output file: {output_path}")

    train_vals = load_split_trajectory_data(data_path)
    paddle_data = extract_paddle_data(train_vals, require_puck=args.require_puck)
    print_trajectory_statistics(paddle_data)

    renderer = RealTrajectoryRenderer(
        table_length=args.table_length,
        table_width=args.table_width,
        paddle_radius=args.paddle_radius,
        puck_radius=args.puck_radius,
        render_size=args.render_size,
        robot_x_offset=args.robot_x_offset,
        orientation="vertical",
    )

    create_trajectory_gif(
        paddle_data,
        renderer,
        output_path,
        max_frames=args.max_frames,
        subsample=args.subsample,
        fps=args.fps,
    )

    print("\n" + "=" * 80)
    print("✓ Visualization complete!")
    print("=" * 80)
    print(f"\nTo view the GIF, open: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize split-schema real robot trajectories (pose/speed/puck datasets) "
            "as GIFs matching Box2D style."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to one .hdf5 file or a directory containing trajectory_data*.hdf5",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory root",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to render per trajectory",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=1,
        help="Subsample factor (1=all frames, 2=every other frame, etc.)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Frames per second for GIF playback",
    )
    parser.add_argument(
        "--table-length",
        type=float,
        default=1.9304,
        help="Table length in meters",
    )
    parser.add_argument(
        "--table-width",
        type=float,
        default=0.8636,
        help="Table width in meters",
    )
    parser.add_argument(
        "--robot-x-offset",
        type=float,
        default=1.2,
        help="Robot base X offset from table center in meters",
    )
    parser.add_argument(
        "--paddle-radius",
        type=float,
        default=0.0508,
        help="Paddle radius in meters",
    )
    parser.add_argument(
        "--puck-radius",
        type=float,
        default=0.03175,
        help="Puck radius in meters",
    )
    parser.add_argument(
        "--render-size",
        type=int,
        default=360,
        help="Render width (pixels) used for visualization",
    )
    parser.add_argument(
        "--require-puck",
        action="store_true",
        help="Fail if puck data is unavailable after schema conversion",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input_path)

    if not input_path.exists():
        print(f"Error: Input path not found: {input_path}")
        sys.exit(1)

    if input_path.is_file():
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = (
                Path(__file__).parent / f"{input_path.stem}_visualization_split_schema"
            )
        visualize_single_file(input_path, output_dir, args)
        return

    trajectory_files = sorted(input_path.glob("trajectory_data*.hdf5"))
    if not trajectory_files:
        trajectory_files = sorted(input_path.glob("*.hdf5"))

    if not trajectory_files:
        print(f"Error: No .hdf5 trajectory files found in directory: {input_path}")
        sys.exit(1)

    if args.output_dir:
        batch_output_root = Path(args.output_dir)
    else:
        batch_output_root = (
            Path(__file__).parent / f"{input_path.name}_visualization_split_schema"
        )
    batch_output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("REAL ROBOT TRAJECTORY VISUALIZATION (SPLIT SCHEMA, BATCH MODE)")
    print("=" * 80)
    print(f"Input directory: {input_path}")
    print(f"Found {len(trajectory_files)} trajectory files")
    print(f"Output root: {batch_output_root}")

    success = 0
    failures = 0
    for idx, data_path in enumerate(trajectory_files, start=1):
        print(f"\n[{idx}/{len(trajectory_files)}] Processing {data_path.name}")
        output_dir = batch_output_root / data_path.stem
        try:
            visualize_single_file(data_path, output_dir, args)
            success += 1
        except Exception as exc:
            failures += 1
            print(f"  ✗ Failed to process {data_path}: {exc}")

    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Successful: {success}")
    print(f"Failed: {failures}")
    print(f"Output root: {batch_output_root}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render a teleop HDF5 trajectory in fixed-length frame segments as GIFs.

Each segment is saved as a sliced HDF5 + a trajectory_visualization.gif under
the output directory, using the same rendering pipeline as the split-schema
visualizer.

Usage:
    python scripts/visualization/render_teleop_segments.py \
        sysid/teleop/trajectory_data1.hdf5 \
        -o sysid/teleop/trajectory_data1_segments \
        --interval 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import h5py
import imageio
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
_REPO_ROOT_STR = str(_REPO_ROOT)
while _REPO_ROOT_STR in sys.path:
    sys.path.remove(_REPO_ROOT_STR)
sys.path.insert(0, _REPO_ROOT_STR)
# ROS puts a `scripts` package on PYTHONPATH that shadows ours; evict it.
if "scripts" in sys.modules:
    del sys.modules["scripts"]

from scripts.visualization.visualize_real_trajectory import (
    RealTrajectoryRenderer,
    extract_paddle_data,
)
from scripts.visualization.visualize_real_trajectory_split import (
    SPLIT_DATASETS,
    load_split_trajectory_data,
)

OUTPUT_WIDTH = 320
FPS = 20


def slice_hdf5(src_path: Path, start: int, end: int, dst_path: Path) -> None:
    """Copy split-schema datasets from src[start:end] into dst."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        for name, _ in SPLIT_DATASETS:
            arr = np.asarray(src[name][:], dtype=np.float64)
            if arr.ndim == 1:
                arr = arr[:, None]
            dst.create_dataset(name, data=arr[start:end])


def render_segment_gif(
    train_vals: np.ndarray,
    renderer: RealTrajectoryRenderer,
    gif_path: Path,
    frame_offset: int,
) -> None:
    """Render a GIF with original frame numbers overlaid."""
    paddle_data = extract_paddle_data(train_vals, require_puck=False)

    pos_x = paddle_data["pos_x"]
    pos_y = paddle_data["pos_y"]
    vel_x = paddle_data["vel_x"]
    vel_y = paddle_data["vel_y"]
    timestamps = paddle_data["timestamps"]
    has_puck = paddle_data.get("has_puck", False)
    puck_x = paddle_data.get("puck_x")
    puck_y = paddle_data.get("puck_y")
    puck_occluded = paddle_data.get("puck_occluded")
    has_target = paddle_data.get("has_target", False)
    target_x = paddle_data.get("target_x")
    target_y = paddle_data.get("target_y")

    relative_time = timestamps - timestamps[0]
    frames = []

    for i in range(len(pos_x)):
        frame = renderer.render_frame(
            pos_x[i],
            pos_y[i],
            vel_x=vel_x[i],
            vel_y=vel_y[i],
            puck_x=(puck_x[i] if has_puck else None),
            puck_y=(puck_y[i] if has_puck else None),
            puck_occluded=(
                puck_occluded[i]
                if (has_puck and puck_occluded is not None)
                else None
            ),
            target_x=(target_x[i] if has_target else None),
            target_y=(target_y[i] if has_target else None),
            timestep=frame_offset + i,
            total_time=relative_time[i],
        )
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        aspect = rgb.shape[1] / rgb.shape[0]
        h = int(OUTPUT_WIDTH / aspect)
        rgb = cv2.resize(rgb, (OUTPUT_WIDTH, h))
        frames.append(rgb)

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(1, int(round(1000.0 / FPS)))
    imageio.mimsave(str(gif_path), frames, format="GIF", loop=0, duration=duration_ms)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a teleop HDF5 into fixed-length segments and render GIFs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=str, help="Path to split-schema HDF5 file.")
    parser.add_argument(
        "-o", "--output-dir", type=str, default=None,
        help="Output directory. Defaults to <input_stem>_segments next to the input.",
    )
    parser.add_argument(
        "--interval", type=int, default=100,
        help="Number of frames per segment.",
    )
    args = parser.parse_args()

    src = Path(args.input).expanduser().resolve()
    if not src.exists():
        print(f"Error: {src} not found")
        sys.exit(1)

    if args.output_dir:
        out_root = Path(args.output_dir).expanduser().resolve()
    else:
        out_root = src.parent / f"{src.stem}_segments"

    with h5py.File(src, "r") as f:
        n_total = f["cur_time"].shape[0]
    print(f"Input: {src}  ({n_total} frames)")
    print(f"Output: {out_root}")
    print(f"Interval: {args.interval} frames\n")

    renderer = RealTrajectoryRenderer(
        table_length=1.9304,
        table_width=0.8636,
        paddle_radius=0.0508,
        puck_radius=0.03175,
        render_size=360,
        robot_x_offset=1.2,
        orientation="vertical",
        paddle_input_frame="table",
        quiet=True,
    )

    start = 0
    while start < n_total:
        end = min(start + args.interval, n_total)
        seg_dir = out_root / f"frames_{start}_{end}"
        hdf5_out = seg_dir / f"segment_{start}_{end}.hdf5"
        gif_out = seg_dir / "trajectory_visualization.gif"

        print(f"  frames {start:>5d}–{end:>5d}  ({end - start} frames) ...", end=" ", flush=True)

        slice_hdf5(src, start, end, hdf5_out)
        train_vals = load_split_trajectory_data(hdf5_out)
        render_segment_gif(train_vals, renderer, gif_out, frame_offset=start)

        print(f"done -> {gif_out}")
        start = end

    print(f"\nAll segments saved under: {out_root}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Split a teleop HDF5 trajectory into named system-ID categories and render GIFs.

Each category is saved as:
  <output_root>/<category_name>/<category_name>.hdf5   (full frame range)

If the category spans > 200 frames, GIFs are chunked into 100-frame sub-segments:
  <output_root>/<category_name>/frames_<start>_<end>/
      segment_<start>_<end>.hdf5
      trajectory_visualization.gif

Otherwise a single GIF is placed alongside the HDF5:
  <output_root>/<category_name>/trajectory_visualization.gif

Usage:
    PYTHONPATH=/home/pearl/air-hockey-rl \
    python scripts/smooth_policy/visualize_demo/split_teleop_categories.py \
        sysid/teleop/trajectory_data1.hdf5 \
        -o sysid/teleop/system_id
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
if "scripts" in sys.modules:
    del sys.modules["scripts"]

from scripts.smooth_policy.visualize_demo.visualize_real_trajectory import (
    RealTrajectoryRenderer,
    extract_paddle_data,
)
from scripts.smooth_policy.visualize_demo.visualize_real_trajectory_split import (
    SPLIT_DATASETS,
    load_split_trajectory_data,
)

CATEGORIES = [
    ("side_to_side_fast",    120,  300),
    ("side_to_side_slow",    310,  650),
    ("side_to_side_dynamic", 675,  1150),
    ("up_and_down_slow",     1200, 1525),
    ("up_and_down_fast",     1550, 1725),
    ("up_and_down_dynamic",  1780, 2050),
    ("circle_fast",          2100, 2500),
    ("circle_slow",          2575, 2850),
    ("random",               3150, 3350),
]

CHUNK_THRESHOLD = 200
CHUNK_SIZE = 100
OUTPUT_WIDTH = 320
FPS = 20


def slice_hdf5(src_path: Path, start: int, end: int, dst_path: Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        for name, _ in SPLIT_DATASETS:
            arr = np.asarray(src[name][:], dtype=np.float64)
            if arr.ndim == 1:
                arr = arr[:, None]
            dst.create_dataset(name, data=arr[start:end])


def render_gif(
    hdf5_path: Path,
    renderer: RealTrajectoryRenderer,
    gif_path: Path,
    frame_offset: int,
) -> None:
    train_vals = load_split_trajectory_data(hdf5_path)
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
        description="Split teleop trajectory into named system-ID categories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=str, help="Path to split-schema HDF5 file.")
    parser.add_argument(
        "-o", "--output-dir", type=str, default=None,
        help="Output root directory. Defaults to <input_stem>_system_id next to input.",
    )
    args = parser.parse_args()

    src = Path(args.input).expanduser().resolve()
    if not src.exists():
        print(f"Error: {src} not found")
        sys.exit(1)

    out_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else src.parent / f"{src.stem}_system_id"
    )

    with h5py.File(src, "r") as f:
        n_total = f["cur_time"].shape[0]
    print(f"Input: {src}  ({n_total} frames)")
    print(f"Output: {out_root}\n")

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

    for name, frame_start, frame_end in CATEGORIES:
        if frame_end > n_total:
            print(f"SKIP {name}: frames {frame_start}-{frame_end} exceeds file length ({n_total})")
            continue

        n_frames = frame_end - frame_start
        cat_dir = out_root / name
        cat_dir.mkdir(parents=True, exist_ok=True)

        hdf5_full = cat_dir / f"{name}.hdf5"
        print(f"[{name}] frames {frame_start}-{frame_end} ({n_frames} frames)")
        slice_hdf5(src, frame_start, frame_end, hdf5_full)
        print(f"  hdf5 -> {hdf5_full}")

        if n_frames > CHUNK_THRESHOLD:
            chunk_start = frame_start
            while chunk_start < frame_end:
                chunk_end = min(chunk_start + CHUNK_SIZE, frame_end)
                chunk_dir = cat_dir / f"frames_{chunk_start}_{chunk_end}"
                chunk_hdf5 = chunk_dir / f"segment_{chunk_start}_{chunk_end}.hdf5"
                chunk_gif = chunk_dir / "trajectory_visualization.gif"

                slice_hdf5(src, chunk_start, chunk_end, chunk_hdf5)
                render_gif(renderer=renderer, hdf5_path=chunk_hdf5, gif_path=chunk_gif, frame_offset=chunk_start)
                print(f"  gif  -> {chunk_gif}")
                chunk_start = chunk_end
        else:
            gif_path = cat_dir / "trajectory_visualization.gif"
            render_gif(renderer=renderer, hdf5_path=hdf5_full, gif_path=gif_path, frame_offset=frame_start)
            print(f"  gif  -> {gif_path}")

    print(f"\nAll categories saved under: {out_root}")


if __name__ == "__main__":
    main()

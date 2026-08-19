#!/usr/bin/env python3
"""Split trajectory_data3.hdf5 into named categories, render trajectory GIFs
and side-by-side sim-vs-real replay GIFs for each 100-frame chunk.

For each category:
  <output_root>/<category_name>/<category_name>.hdf5   (full frame range)
  <output_root>/<category_name>/frames_<start>_<end>/
      segment_<start>_<end>.hdf5
      trajectory_visualization.gif
      sim_vs_real.gif            (side-by-side replay, sim reset per chunk)
      sim_vs_real.json           (position error metrics)

Usage:
    PYTHONPATH=/home/pearl/air-hockey-rl \
    python scripts/visualization/split_teleop_categories_data3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import h5py
import imageio
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT_STR = str(_REPO_ROOT)
while _REPO_ROOT_STR in sys.path:
    sys.path.remove(_REPO_ROOT_STR)
sys.path.insert(0, _REPO_ROOT_STR)
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
from scripts.visualization.replay_real_in_sim import replay_episode

INPUT_PATH = _REPO_ROOT / "sysid" / "teleop_hist4" / "trajectory_data0.hdf5"
OUTPUT_ROOT = _REPO_ROOT / "sysid" / "teleop_hist4" / "system_id"
SIM_CONFIG = str(_REPO_ROOT / "configs/new_juggle/sysid_best_params_hist4.yaml")

CATEGORIES = [
    ("side_to_side_fast",    101,  600),
    ("side_to_side_slow",    625,  1000),
    ("side_to_side_dynamic", 1001,  1100),
    ("up_and_down_fast",     1125,  1650),
    ("up_and_down_slow",     1655, 1900),
    ("up_and_down_dynamic",  2100, 2300),
    ("diagonal_fast",        2350, 3000),
    ("diagonal_dynamic",     3025, 3700),
    ("circle_fast",          3725, 4000),
    ("circle_slow",          4025, 4200),
    ("random",               4225, 4800),
]

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
    if not INPUT_PATH.exists():
        print(f"Error: {INPUT_PATH} not found")
        sys.exit(1)

    with h5py.File(INPUT_PATH, "r") as f:
        n_total = f["cur_time"].shape[0]
    print(f"Input: {INPUT_PATH}  ({n_total} frames)")
    print(f"Output: {OUTPUT_ROOT}\n")

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
            print(
                f"SKIP {name}: frames {frame_start}-{frame_end} "
                f"exceeds file length ({n_total})"
            )
            continue

        n_frames = frame_end - frame_start
        cat_dir = OUTPUT_ROOT / name
        cat_dir.mkdir(parents=True, exist_ok=True)

        hdf5_full = cat_dir / f"{name}.hdf5"
        print(f"\n{'='*60}")
        print(f"[{name}] frames {frame_start}-{frame_end} ({n_frames} frames)")
        print(f"{'='*60}")
        slice_hdf5(INPUT_PATH, frame_start, frame_end, hdf5_full)
        print(f"  hdf5 -> {hdf5_full}")

        chunk_start = frame_start
        while chunk_start < frame_end:
            chunk_end = min(chunk_start + CHUNK_SIZE, frame_end)
            chunk_dir = cat_dir / f"frames_{chunk_start}_{chunk_end}"
            chunk_hdf5 = chunk_dir / f"segment_{chunk_start}_{chunk_end}.hdf5"
            chunk_gif = chunk_dir / "trajectory_visualization.gif"
            replay_gif = chunk_dir / "sim_vs_real.gif"

            print(f"\n  --- chunk {chunk_start}-{chunk_end} "
                  f"({chunk_end - chunk_start} frames) ---")

            slice_hdf5(INPUT_PATH, chunk_start, chunk_end, chunk_hdf5)
            render_gif(
                renderer=renderer,
                hdf5_path=chunk_hdf5,
                gif_path=chunk_gif,
                frame_offset=chunk_start,
            )
            print(f"  traj gif  -> {chunk_gif}")

            # Side-by-side replay: use the full trajectory so that step labels
            # show the original (absolute) frame numbers. The sim resets fresh
            # at each chunk boundary because each replay_episode call creates a
            # new env and calls reset_from_state at start_frame.
            try:
                replay_episode(
                    episode_path=str(INPUT_PATH),
                    config_path=SIM_CONFIG,
                    output_path=str(replay_gif),
                    enable_noise=False,
                    max_steps=chunk_end - chunk_start,
                    fps=FPS,
                    frame_width=160,
                    start_frame=chunk_start,
                )
                print(f"  replay gif -> {replay_gif}")
            except Exception as exc:
                print(f"  replay FAILED: {exc}")

            chunk_start = chunk_end

    print(f"\n{'='*60}")
    print(f"All categories saved under: {OUTPUT_ROOT}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

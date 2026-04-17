#!/usr/bin/env python3
"""Split puck-wall and puck-paddle collision segments into per-segment HDF5 + GIFs.

Same convention as ``split_system_id_segments.py``:
    <output_root>/trajectory_data<id>/<obj>_<start>_<end>/{<seg>.hdf5, trajectory_visualization.gif}

A 10-timestep buffer is applied at both ends (clamped to file bounds), and the
original frame numbers are preserved in the GIF overlay.

Outputs:
    sysid/wall_collision/         -- puck-wall collision segments
    sysid/paddle_puck_collision/  -- puck-paddle collision segments
"""

from __future__ import annotations

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

from scripts.smooth_policy.visualize_demo.visualize_real_trajectory import (
    RealTrajectoryRenderer,
    extract_paddle_data,
)
from scripts.smooth_policy.visualize_demo.visualize_real_trajectory_split import (
    SPLIT_DATASETS,
    load_split_trajectory_data,
)

# ── HDF5 source paths ──────────────────────────────────────────────────────────

HDF5_MAP = {
    # 50-100 bucket
    457: "real_runs/online_run/episode_hdf5/50-100/trajectory_data457.hdf5",
    467: "real_runs/online_run/episode_hdf5/50-100/trajectory_data467.hdf5",
    481: "real_runs/online_run/episode_hdf5/50-100/trajectory_data481.hdf5",
    486: "real_runs/online_run/episode_hdf5/50-100/trajectory_data486.hdf5",
    # 100-200 bucket
    461: "real_runs/online_run/episode_hdf5/100-200/trajectory_data461.hdf5",
    478: "real_runs/online_run/episode_hdf5/100-200/trajectory_data478.hdf5",
    # >200 bucket
    446: "real_runs/online_run/episode_hdf5/>200/trajectory_data446.hdf5",
    452: "real_runs/online_run/episode_hdf5/>200/trajectory_data452.hdf5",
    463: "real_runs/online_run/episode_hdf5/>200/trajectory_data463.hdf5",
    476: "real_runs/online_run/episode_hdf5/>200/trajectory_data476.hdf5",
    # teleop wall-collision recording
    0: "sysid/wall_collision_teleop/trajectory_data0.hdf5",
}

# ── Segment definitions: (traj_id, start_frame, end_frame) ────────────────────

# Puck-wall collisions. Segment dirname: ``wall_<start>_<end>``.
WALL_SEGMENTS = [
    (478, 0, 30),
    (461, 65, 100),
    (486, 65, 90),
    (481, 20, 55),
    (467, 5, 50),
    (457, 10, 45),
    # teleop wall collisions (trajectory_data0)
    (0, 200, 300),
    (0, 440, 490),
    (0, 655, 700),
    (0, 850, 880),
    (0, 1030, 1080),
    (0, 1190, 1280),
]

# Puck-paddle collisions. Segment dirname: ``paddle_<start>_<end>``.
PADDLE_SEGMENTS = [
    (476, 45, 90),
    (476, 130, 180),
    (463, 35, 85),
    (463, 100, 130),
    (463, 175, 205),
    (452, 40, 80),
    (452, 120, 165),
    (452, 180, 220),
    (452, 270, 310),
    (452, 300, 345),
    (452, 330, 370),
    (452, 460, 490),
    (446, 45, 80),
    (446, 200, 260),
]

BUFFER = 10
WALL_OUTPUT_ROOT = _REPO_ROOT / "sysid" / "wall_collision"
PADDLE_OUTPUT_ROOT = _REPO_ROOT / "sysid" / "paddle_puck_collision"
OUTPUT_WIDTH = 320
FPS = 20


def slice_split_hdf5(src_path: Path, start: int, end: int, dst_path: Path) -> None:
    """Copy the split-schema datasets from src_path[start:end] into dst_path."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        for name, _ in SPLIT_DATASETS:
            arr = np.asarray(src[name][:], dtype=np.float64)
            if arr.ndim == 1:
                arr = arr[:, None]
            dst.create_dataset(name, data=arr[start:end])
        optional_names = {n for n, _ in SPLIT_DATASETS}
        for key in src.keys():
            if key not in optional_names:
                arr = np.asarray(src[key][:])
                if arr.ndim == 0:
                    dst.create_dataset(key, data=arr)
                elif arr.shape[0] >= end:
                    dst.create_dataset(key, data=arr[start:end])


def render_segment_gif(
    train_vals: np.ndarray,
    renderer: RealTrajectoryRenderer,
    gif_path: Path,
    frame_offset: int,
) -> None:
    """Render a GIF from train_vals, labelling frames with their original numbers."""
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


def process_segments(
    segments: list[tuple[int, int, int]],
    output_root: Path,
    seg_label: str,
    renderer: RealTrajectoryRenderer,
    cache: dict[int, int],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    for traj_id, frame_start, frame_end in segments:
        src_rel = HDF5_MAP[traj_id]
        src_path = _REPO_ROOT / src_rel

        if not src_path.exists():
            print(f"SKIP (not found): {src_path}")
            continue

        if traj_id not in cache:
            with h5py.File(src_path, "r") as f:
                cache[traj_id] = f["cur_time"].shape[0]
        n_total = cache[traj_id]

        buf_start = max(0, frame_start - BUFFER)
        buf_end = min(n_total, frame_end + BUFFER)

        seg_name = f"{seg_label}_{frame_start}_{frame_end}"
        seg_dir = output_root / f"trajectory_data{traj_id}" / seg_name

        hdf5_out = seg_dir / f"{seg_name}.hdf5"
        gif_out = seg_dir / "trajectory_visualization.gif"

        print(f"[{traj_id}] {seg_label} frames {frame_start}-{frame_end}  "
              f"(buffered {buf_start}-{buf_end}, n={buf_end - buf_start})")

        slice_split_hdf5(src_path, buf_start, buf_end, hdf5_out)

        train_vals = load_split_trajectory_data(hdf5_out)
        render_segment_gif(train_vals, renderer, gif_out, frame_offset=buf_start)
        print(f"  -> {gif_out}")


def main() -> None:
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

    cache: dict[int, int] = {}  # traj_id -> total timesteps

    print(f"\n=== Puck-wall collisions -> {WALL_OUTPUT_ROOT} ===")
    process_segments(WALL_SEGMENTS, WALL_OUTPUT_ROOT, "wall", renderer, cache)

    print(f"\n=== Puck-paddle collisions -> {PADDLE_OUTPUT_ROOT} ===")
    process_segments(PADDLE_SEGMENTS, PADDLE_OUTPUT_ROOT, "paddle", renderer, cache)

    print("\nAll collision segments written.")


if __name__ == "__main__":
    main()

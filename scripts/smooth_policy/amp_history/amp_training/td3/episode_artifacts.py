"""Episode artifact utilities for async TD3 real-world collection."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import cv2
import h5py
import numpy as np

from scripts.smooth_policy.visualize_demo.visualize_real_trajectory import (
    RealTrajectoryRenderer,
    create_trajectory_gif,
    extract_paddle_data,
)
from scripts.smooth_policy.visualize_demo.visualize_real_trajectory_split import (
    OPTIONAL_SPLIT_DATASETS,
    SPLIT_DATASETS,
    load_split_trajectory_data,
)

OPTIONAL_ALLOWED_WIDTHS = {
    "timing": (8, 9),
}


@dataclass
class CleanResult:
    kept: bool
    reason: str
    timesteps: int
    path: Path


def save_split_episode_hdf5(
    output_dir: str | Path,
    episode_id: int,
    episode_rows: List[Dict[str, np.ndarray]],
    episode_images: List[np.ndarray] | np.ndarray | None = None,
) -> Path:
    """Save one completed episode in split-schema HDF5 format."""
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"trajectory_data{int(episode_id)}.hdf5"

    if not episode_rows:
        raise ValueError("Cannot save episode artifact: episode_rows is empty.")

    stacked: Dict[str, np.ndarray] = {}
    for dataset_name, width in SPLIT_DATASETS:
        values = [np.asarray(row[dataset_name], dtype=np.float64).reshape(1, width) for row in episode_rows]
        stacked[dataset_name] = np.concatenate(values, axis=0)
    optional_stacked: Dict[str, np.ndarray] = {}
    for dataset_name, width in OPTIONAL_SPLIT_DATASETS:
        if all(dataset_name in row for row in episode_rows):
            values = [np.asarray(row[dataset_name], dtype=np.float64).reshape(1, width) for row in episode_rows]
            optional_stacked[dataset_name] = np.concatenate(values, axis=0)

    image_array: np.ndarray | None = None
    if episode_images is not None:
        image_array = np.asarray(episode_images, dtype=np.uint8)
        if image_array.ndim != 4:
            raise ValueError(
                "episode_images must have shape (T, H, W, C) when provided."
            )
        if image_array.shape[0] != stacked["cur_time"].shape[0]:
            raise ValueError(
                "episode_images length must match number of episode timesteps."
            )

    with h5py.File(output_path, "w") as h5_file:
        for dataset_name, _ in SPLIT_DATASETS:
            data = stacked[dataset_name]
            h5_file.create_dataset(
                dataset_name,
                shape=data.shape,
                compression="gzip",
                compression_opts=9,
                data=data,
            )
        for dataset_name, _ in OPTIONAL_SPLIT_DATASETS:
            if dataset_name not in optional_stacked:
                continue
            data = optional_stacked[dataset_name]
            h5_file.create_dataset(
                dataset_name,
                shape=data.shape,
                compression="gzip",
                compression_opts=9,
                data=data,
            )
        if image_array is not None:
            h5_file.create_dataset(
                "train_img",
                shape=image_array.shape,
                compression="gzip",
                compression_opts=9,
                data=image_array,
            )
        h5_file.attrs["schema_version"] = (
            "split_v2_optional" if len(optional_stacked) > 0 else "split_v1"
        )
    return output_path


def clean_episode_hdf5(path: str | Path, min_timesteps: int = 30) -> CleanResult:
    """Validate one split-schema HDF5 and remove it if short/invalid."""
    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        return CleanResult(kept=False, reason="missing", timesteps=0, path=file_path)

    try:
        with h5py.File(file_path, "r") as h5_file:
            lengths = []
            for dataset_name, width in SPLIT_DATASETS:
                if dataset_name not in h5_file:
                    file_path.unlink(missing_ok=True)
                    return CleanResult(kept=False, reason=f"missing_{dataset_name}", timesteps=0, path=file_path)
                data = np.asarray(h5_file[dataset_name][:], dtype=np.float64)
                if data.ndim == 1:
                    data = data[:, None]
                if data.ndim != 2 or data.shape[1] != width:
                    file_path.unlink(missing_ok=True)
                    return CleanResult(
                        kept=False,
                        reason=f"bad_shape_{dataset_name}",
                        timesteps=int(data.shape[0]) if data.ndim >= 1 else 0,
                        path=file_path,
                    )
                if not np.all(np.isfinite(data)):
                    file_path.unlink(missing_ok=True)
                    return CleanResult(
                        kept=False,
                        reason=f"non_finite_{dataset_name}",
                        timesteps=int(data.shape[0]),
                        path=file_path,
                    )
                lengths.append(int(data.shape[0]))
            for dataset_name, width in OPTIONAL_SPLIT_DATASETS:
                if dataset_name not in h5_file:
                    continue
                data = np.asarray(h5_file[dataset_name][:], dtype=np.float64)
                if data.ndim == 1:
                    data = data[:, None]
                expected_widths = OPTIONAL_ALLOWED_WIDTHS.get(dataset_name, (width,))
                if data.ndim != 2 or data.shape[1] not in expected_widths:
                    file_path.unlink(missing_ok=True)
                    return CleanResult(
                        kept=False,
                        reason=f"bad_shape_{dataset_name}",
                        timesteps=int(data.shape[0]) if data.ndim >= 1 else 0,
                        path=file_path,
                    )
                if not np.all(np.isfinite(data)):
                    file_path.unlink(missing_ok=True)
                    return CleanResult(
                        kept=False,
                        reason=f"non_finite_{dataset_name}",
                        timesteps=int(data.shape[0]),
                        path=file_path,
                    )
                lengths.append(int(data.shape[0]))

            if "train_img" in h5_file:
                image_data = np.asarray(h5_file["train_img"][:])
                if image_data.ndim != 4:
                    file_path.unlink(missing_ok=True)
                    return CleanResult(
                        kept=False,
                        reason="bad_shape_train_img",
                        timesteps=int(image_data.shape[0]) if image_data.ndim >= 1 else 0,
                        path=file_path,
                    )
                if image_data.shape[3] not in (1, 3, 4):
                    file_path.unlink(missing_ok=True)
                    return CleanResult(
                        kept=False,
                        reason="bad_channels_train_img",
                        timesteps=int(image_data.shape[0]),
                        path=file_path,
                    )
                lengths.append(int(image_data.shape[0]))
    except Exception:
        file_path.unlink(missing_ok=True)
        return CleanResult(kept=False, reason="read_error", timesteps=0, path=file_path)

    timesteps = lengths[0] if lengths else 0
    if timesteps < int(min_timesteps):
        file_path.unlink(missing_ok=True)
        return CleanResult(kept=False, reason="short_episode", timesteps=timesteps, path=file_path)
    if any(length != timesteps for length in lengths):
        file_path.unlink(missing_ok=True)
        return CleanResult(kept=False, reason="inconsistent_lengths", timesteps=timesteps, path=file_path)
    return CleanResult(kept=True, reason="kept", timesteps=timesteps, path=file_path)


def generate_episode_gif(
    episode_hdf5_path: str | Path,
    gif_root: str | Path,
    fps: int = 20,
    max_frames: int | None = None,
    subsample: int = 1,
    require_puck: bool = False,
) -> Path:
    """Generate one Box2D-style GIF from split-schema episode HDF5."""
    if fps <= 0:
        raise ValueError("fps must be > 0")
    if subsample <= 0:
        raise ValueError("subsample must be > 0")

    hdf5_path = Path(episode_hdf5_path).expanduser().resolve()
    gif_root_path = Path(gif_root).expanduser().resolve()
    output_dir = gif_root_path / hdf5_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "trajectory_visualization.gif"

    train_vals = load_split_trajectory_data(hdf5_path)
    paddle_data = extract_paddle_data(train_vals, require_puck=require_puck)

    renderer = RealTrajectoryRenderer(
        table_length=1.9304,
        table_width=0.8636,
        paddle_radius=0.0508,
        puck_radius=0.03175,
        render_size=360,
        robot_x_offset=1.2,
        orientation="vertical",
        paddle_input_frame="table",
    )
    create_trajectory_gif(
        paddle_data,
        renderer,
        output_path,
        max_frames=max_frames,
        subsample=subsample,
        fps=fps,
    )
    return output_path


def generate_episode_camera_video(
    episode_hdf5_path: str | Path,
    video_root: str | Path,
    fps: int = 20,
    max_frames: int | None = None,
    subsample: int = 1,
    codec: str = "mp4v",
) -> Path:
    """Generate one camera-ground-truth video from train_img in episode HDF5."""
    if fps <= 0:
        raise ValueError("fps must be > 0")
    if subsample <= 0:
        raise ValueError("subsample must be > 0")

    hdf5_path = Path(episode_hdf5_path).expanduser().resolve()
    video_root_path = Path(video_root).expanduser().resolve()
    output_dir = video_root_path / hdf5_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ground_truth_camera.mp4"

    with h5py.File(hdf5_path, "r") as h5_file:
        if "train_img" not in h5_file:
            raise ValueError(f"Missing train_img dataset in {hdf5_path}")
        frames = np.asarray(h5_file["train_img"][:], dtype=np.uint8)

    if frames.ndim != 4:
        raise ValueError(f"train_img must be rank-4 (T,H,W,C), got {frames.shape}")
    if frames.shape[0] == 0:
        raise ValueError("train_img contains zero frames")
    if frames.shape[3] not in (1, 3, 4):
        raise ValueError(f"Unsupported train_img channel count: {frames.shape[3]}")

    total_frames = int(frames.shape[0])
    if max_frames is not None:
        frames = frames[: int(max_frames)]
    frames = frames[:: int(subsample)]
    if frames.shape[0] == 0:
        raise ValueError("No frames available after max_frames/subsample filtering")

    n_render = int(frames.shape[0])
    height, width = int(frames.shape[1]), int(frames.shape[2])
    duration_s = n_render / max(fps, 1)

    print(f"\nGenerating camera video:")
    print(f"  Total camera frames: {total_frames}")
    print(f"  Frames to render: {n_render}")
    print(f"  Subsample factor: {subsample}")
    print(f"  Frame size: {width}x{height}")
    print(f"  Duration: {duration_s:.2f} seconds")
    print(f"  Playback FPS: {fps}")
    print(f"  Codec: {codec}")

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, int(fps), (width, height))

    if not writer.isOpened():
        fallback_path = output_dir / "ground_truth_camera.avi"
        fallback_fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(str(fallback_path), fallback_fourcc, int(fps), (width, height))
        if not writer.isOpened():
            raise RuntimeError("Failed to initialize video writer for ground-truth video.")
        output_path = fallback_path
        print(f"  Fallback codec: XVID (.avi)")

    try:
        for idx, frame in enumerate(frames):
            if idx % 50 == 0 and idx > 0:
                print(f"  Writing frame {idx}/{n_render}...")
            if frame.shape[2] == 1:
                frame_to_write = np.repeat(frame, 3, axis=2)
            elif frame.shape[2] == 4:
                frame_to_write = frame[:, :, :3]
            else:
                frame_to_write = frame
            writer.write(frame_to_write)
    finally:
        writer.release()

    print(f"\nSaving camera video...")
    print(f"  Output path: {output_path}")
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  File size: {file_size_mb:.2f} MB")

    return output_path

"""Episode artifact utilities for async TD3 real-world collection."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import cv2
import h5py
import imageio
import numpy as np

from scripts.visualization.visualize_real_trajectory import (
    RealTrajectoryRenderer,
    create_trajectory_gif,
    extract_paddle_data,
)
from scripts.visualization.visualize_real_trajectory_split import (
    OPTIONAL_SPLIT_DATASETS,
    SPLIT_DATASETS,
    load_split_optional_data,
    load_split_trajectory_data,
)
from airhockey.sims.real.control_parameters import (
    homography_showdst_from_saved_frame,
    offset_constants as DEFAULT_OFFSET_CONSTANTS,
    visual_downscale_constant as DEFAULT_VISUAL_DOWNSCALE,
)
from airhockey.sims.real.overlay_utils import (
    draw_homography_episode_markers,
    enlarged_goal_marker_radius_m,
    observation_to_robot_xy,
)

OPTIONAL_ALLOWED_WIDTHS = {
    "timing": (8, 9),
}

_SYSID_TABLE_WIDTH = 0.8636


def _sysid_sim_goal_bounds():
    w = _SYSID_TABLE_WIDTH
    return type("_Sim", (), {"min_goal_radius": w / 16, "max_goal_radius": w / 4})()


def _prepare_goal_data_for_display(goal_data: np.ndarray | None) -> np.ndarray | None:
    """Ensure GIF rendering uses the enlarged live-overlay goal radius."""
    if goal_data is None or len(goal_data) == 0:
        return goal_data
    out = np.asarray(goal_data, dtype=np.float64).copy()
    display_r = enlarged_goal_marker_radius_m(_sysid_sim_goal_bounds())
    if display_r is not None and display_r > 0:
        out[:, 2] = np.maximum(out[:, 2], display_r)
    return out


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
    panel_height: int = 240,
) -> Path:
    """Generate a side-by-side GIF: Box2D projection of the HDF5 trajectory
    on the left, real-world camera frame (``train_img``) on the right.

    Falls back to Box2D-only rendering via ``create_trajectory_gif`` when
    ``train_img`` is unavailable in the HDF5.
    """
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
    # Optional datasets (incl. `goal` for goal-conditioned tasks). Returns
    # an empty dict when none are present, so non-goal episodes (e.g. juggle,
    # puck_velocity) keep their existing GIF output bit-identical.
    optional_data = load_split_optional_data(hdf5_path)
    goal_data = _prepare_goal_data_for_display(optional_data.get("goal"))

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

    with h5py.File(hdf5_path, "r") as h5_file:
        camera_frames = (
            np.asarray(h5_file["train_img"][:], dtype=np.uint8)
            if "train_img" in h5_file
            else None
        )

    if camera_frames is None:
        create_trajectory_gif(
            paddle_data,
            renderer,
            output_path,
            max_frames=max_frames,
            subsample=subsample,
            fps=fps,
            goal_data=goal_data,
        )
        return output_path

    _create_joint_trajectory_gif(
        paddle_data=paddle_data,
        camera_frames=camera_frames,
        renderer=renderer,
        output_path=output_path,
        fps=fps,
        max_frames=max_frames,
        subsample=subsample,
        panel_height=panel_height,
        goal_data=goal_data,
    )
    return output_path


def _resize_to_height(frame: np.ndarray, height: int) -> np.ndarray:
    """Resize an image preserving aspect ratio to a target height (pixels)."""
    if frame.shape[0] == height:
        return frame
    new_w = max(1, int(round(frame.shape[1] * (height / frame.shape[0]))))
    return cv2.resize(frame, (new_w, height))


def _side_by_side(left_rgb: np.ndarray, right_rgb: np.ndarray) -> np.ndarray:
    """Horizontal concat with a 3px light-gray separator (mirrors
    ``replay_real_in_sim._side_by_side``; inlined to avoid pulling Box2D)."""
    if right_rgb.shape[0] != left_rgb.shape[0]:
        new_w = int(round(right_rgb.shape[1] * (left_rgb.shape[0] / right_rgb.shape[0])))
        right_rgb = cv2.resize(right_rgb, (max(1, new_w), left_rgb.shape[0]))
    sep = np.full((left_rgb.shape[0], 3, 3), 200, dtype=np.uint8)
    return np.concatenate([left_rgb, sep, right_rgb], axis=1)


def _create_joint_trajectory_gif(
    paddle_data: Dict[str, np.ndarray],
    camera_frames: np.ndarray,
    renderer: RealTrajectoryRenderer,
    output_path: Path,
    fps: int,
    max_frames: int | None,
    subsample: int,
    panel_height: int,
    goal_data: np.ndarray | None = None,
) -> None:
    """Render a side-by-side GIF using the existing Box2D renderer and the
    raw ``train_img`` camera frames. Stitching uses ``_side_by_side`` from
    ``replay_real_in_sim.py``.

    ``goal_data`` (when provided) is an ``(N, 3)`` array of
    ``[goal_x_table, goal_y_table, goal_radius]`` per step, drawn on the
    left (Box2D) panel as a green goal region.
    """
    if camera_frames.ndim != 4:
        raise ValueError(
            f"camera_frames must be rank-4 (T,H,W,C), got {camera_frames.shape}"
        )

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
    has_goal = goal_data is not None and len(goal_data) > 0

    n_traj = len(pos_x)
    n_cam = int(camera_frames.shape[0])
    n_frames = min(n_traj, n_cam)
    if n_frames == 0:
        raise ValueError("No frames available to render (trajectory or train_img empty).")
    if max_frames is not None:
        n_frames = min(n_frames, int(max_frames))

    indices = np.arange(0, n_frames, int(subsample))
    relative_time = timestamps - timestamps[0]

    frames: List[np.ndarray] = []
    for i in indices:
        sim_bgr = renderer.render_frame(
            pos_x[i],
            pos_y[i],
            vel_x=vel_x[i],
            vel_y=vel_y[i],
            puck_x=(puck_x[i] if has_puck else None),
            puck_y=(puck_y[i] if has_puck else None),
            puck_occluded=(
                puck_occluded[i] if (has_puck and puck_occluded is not None) else None
            ),
            target_x=(target_x[i] if has_target else None),
            target_y=(target_y[i] if has_target else None),
            goal_x=(float(goal_data[i, 0]) if has_goal else None),
            goal_y=(float(goal_data[i, 1]) if has_goal else None),
            goal_radius=(float(goal_data[i, 2]) if has_goal else None),
            timestep=int(i),
            total_time=float(relative_time[i]),
        )
        sim_rgb = cv2.cvtColor(sim_bgr, cv2.COLOR_BGR2RGB)

        cam = camera_frames[i]
        if cam.shape[2] == 1:
            cam = np.repeat(cam, 3, axis=2)
        elif cam.shape[2] == 4:
            cam = cam[:, :, :3]

        sim_rgb = _resize_to_height(sim_rgb, panel_height)
        cam = _resize_to_height(cam, panel_height)

        frames.append(_side_by_side(sim_rgb, cam))

    duration = int(1000 / fps)
    imageio.mimsave(output_path, frames, format="GIF", loop=0, duration=duration)


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


def generate_episode_homography_gif(
    episode_hdf5_path: str | Path,
    gif_root: str | Path,
    fps: int = 20,
    max_frames: int | None = None,
    subsample: int = 1,
    center_offset_constant: float = 1.2,
    require_goal: bool = True,
) -> Path:
    """Generate a GIF of the homography-warped camera feed with live-style overlays.

    Reconstructs ``showdst`` from each ``train_img`` row, then draws target /
    puck / paddle / goal markers using the same pixel projection as the live
    robot stack. Intended for GCRL / goal-conditioned eval review.

    Output: ``<gif_root>/trajectory_dataN/homography_visualization.gif``
    (``gif_root`` is the run-level ``episode_homography_gifs/`` dir — not
    length-bucketed like ``episode_gifs/``).
    """
    if fps <= 0:
        raise ValueError("fps must be > 0")
    if subsample <= 0:
        raise ValueError("subsample must be > 0")

    hdf5_path = Path(episode_hdf5_path).expanduser().resolve()
    gif_root_path = Path(gif_root).expanduser().resolve()
    output_dir = gif_root_path / hdf5_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "homography_visualization.gif"

    with h5py.File(hdf5_path, "r") as h5_file:
        if "train_img" not in h5_file:
            raise ValueError(f"Missing train_img dataset in {hdf5_path}")
        camera_frames = np.asarray(h5_file["train_img"][:], dtype=np.uint8)
        if camera_frames.ndim != 4 or camera_frames.shape[0] == 0:
            raise ValueError(f"train_img must be non-empty rank-4, got {camera_frames.shape}")

        pose = np.asarray(h5_file["pose"][:], dtype=np.float64)
        puck = np.asarray(h5_file["puck"][:], dtype=np.float64)
        desired_pose = np.asarray(h5_file["desired_pose"][:], dtype=np.float64)
        goal = (
            np.asarray(h5_file["goal"][:], dtype=np.float64)
            if "goal" in h5_file
            else None
        )

    if require_goal and goal is None:
        raise ValueError(
            f"Homography GIF requires optional 'goal' dataset in {hdf5_path} "
            "(record a new eval with goal-conditioned task after goal logging is enabled)."
        )

    n_frames = int(camera_frames.shape[0])
    for name, arr in (("pose", pose), ("puck", puck), ("desired_pose", desired_pose)):
        if int(arr.shape[0]) != n_frames:
            raise ValueError(
                f"Inconsistent lengths: train_img has {n_frames} frames but "
                f"{name} has {arr.shape[0]}"
            )
    if goal is not None and int(goal.shape[0]) != n_frames:
        raise ValueError(
            f"Inconsistent lengths: train_img has {n_frames} frames but goal has {goal.shape[0]}"
        )

    if max_frames is not None:
        n_frames = min(n_frames, int(max_frames))

    indices = np.arange(0, n_frames, int(subsample))
    gif_frames: List[np.ndarray] = []

    for i in indices:
        showdst = homography_showdst_from_saved_frame(camera_frames[i])
        goal_xy = goal[i, :2] if goal is not None else None
        goal_radius = float(goal[i, 2]) if goal is not None and goal.shape[1] >= 3 else None
        if goal_radius is not None and (not np.isfinite(goal_radius) or goal_radius <= 0):
            goal_radius = None

        paddle_robot_x, paddle_robot_y = observation_to_robot_xy(
            pose[i, 0], pose[i, 1], center_offset_constant
        )
        target_robot_x, target_robot_y = observation_to_robot_xy(
            desired_pose[i, 0], desired_pose[i, 1], center_offset_constant
        )
        overlay_bgr = draw_homography_episode_markers(
            np.asarray(showdst, dtype=np.uint8).copy(),
            target_xy_robot=(target_robot_x, target_robot_y),
            puck_state_table=puck[i, :3],
            paddle_xy_robot=(paddle_robot_x, paddle_robot_y),
            goal_xy_table=goal_xy,
            goal_radius_m=goal_radius,
            center_offset_constant=center_offset_constant,
            offset_constants=DEFAULT_OFFSET_CONSTANTS,
            visual_downscale_constant=DEFAULT_VISUAL_DOWNSCALE,
        )
        gif_frames.append(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB))

    duration = int(1000 / fps)
    imageio.mimsave(output_path, gif_frames, format="GIF", loop=0, duration=duration)
    return output_path

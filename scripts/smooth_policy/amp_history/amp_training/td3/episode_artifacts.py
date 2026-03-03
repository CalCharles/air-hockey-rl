"""Episode artifact utilities for async TD3 real-world collection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np

from scripts.smooth_policy.visualize_demo.visualize_real_trajectory import (
    RealTrajectoryRenderer,
    create_trajectory_gif,
    extract_paddle_data,
)
from scripts.smooth_policy.visualize_demo.visualize_real_trajectory_split import (
    SPLIT_DATASETS,
    load_split_trajectory_data,
)


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

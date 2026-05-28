#!/usr/bin/env python3
"""Render all eval goal positions on a single homography overhead PNG.

Uses the same 4x5 row-major grid as ``async_td3_real_eval`` and the same
green goal-ring overlay as ``generate_episode_homography_gif``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np
import yaml

from airhockey.sims.real.control_parameters import (
    homography_showdst_from_saved_frame,
    offset_constants as DEFAULT_OFFSET_CONSTANTS,
    visual_downscale_constant as DEFAULT_VISUAL_DOWNSCALE,
)
from airhockey.sims.real.overlay_utils import (
    draw_goal_marker,
    enlarged_goal_marker_radius_m,
    observation_to_robot_xy,
    robot_to_display_pixel_int,
)
from scripts.td3.helper.eval_goal_grid import (
    GOAL_GRID_COLS,
    GOAL_GRID_ROWS,
    build_eval_goal_grid,
)


def _table_dims_from_config(config_path: Path) -> tuple[float, float]:
    with open(config_path, "r") as handle:
        cfg = yaml.load(handle, Loader=yaml.FullLoader)
    sim = dict(cfg["air_hockey"]["simulator_params"])
    length = float(sim["length"])
    width = float(sim["width"])
    return length / 2.0, width / 2.0


def _goal_display_radius_m(table_width: float) -> float:
    sim_bounds = type(
        "_Sim",
        (),
        {"min_goal_radius": table_width / 16.0, "max_goal_radius": table_width / 4.0},
    )()
    radius = enlarged_goal_marker_radius_m(sim_bounds)
    if radius is None or radius <= 0:
        raise ValueError("Could not derive a positive goal display radius.")
    return float(radius)


def _load_base_homography_frame(hdf5_path: Path, frame_index: int = 0) -> np.ndarray:
    with h5py.File(hdf5_path, "r") as h5_file:
        if "train_img" not in h5_file:
            raise ValueError(f"Missing train_img dataset in {hdf5_path}")
        camera_frames = np.asarray(h5_file["train_img"], dtype=np.uint8)
    if camera_frames.ndim != 4 or camera_frames.shape[0] == 0:
        raise ValueError(f"train_img must be non-empty rank-4, got {camera_frames.shape}")
    idx = int(frame_index) % int(camera_frames.shape[0])
    return homography_showdst_from_saved_frame(camera_frames[idx])


def render_eval_goal_grid_homography_png(
    *,
    hdf5_path: Path,
    output_path: Path,
    table_x_bot: float,
    table_y_right: float,
    center_offset_constant: float = 1.2,
    goal_radius_m: float | None = None,
    frame_index: int = 0,
    label_goals: bool = True,
) -> Path:
    """Draw all eval goals onto one homography-warped frame and save PNG."""
    if goal_radius_m is None:
        goal_radius_m = _goal_display_radius_m(table_y_right * 2.0)

    goals = build_eval_goal_grid(
        table_x_bot=table_x_bot,
        table_y_right=table_y_right,
    )
    frame = _load_base_homography_frame(hdf5_path, frame_index=frame_index)
    overlay = np.asarray(frame, dtype=np.uint8).copy()

    for idx, (goal_x, goal_y) in enumerate(goals, start=1):
        goal_robot_x, goal_robot_y = observation_to_robot_xy(
            goal_x,
            goal_y,
            center_offset_constant,
        )
        draw_goal_marker(
            overlay,
            (goal_robot_x, goal_robot_y),
            goal_radius_m=goal_radius_m,
            offset_constants=DEFAULT_OFFSET_CONSTANTS,
            visual_downscale_constant=DEFAULT_VISUAL_DOWNSCALE,
        )
        if label_goals:
            center = robot_to_display_pixel_int(
                goal_robot_x,
                goal_robot_y,
                offset_constants=DEFAULT_OFFSET_CONSTANTS,
                visual_downscale_constant=DEFAULT_VISUAL_DOWNSCALE,
            )
            label = str(idx)
            cv2.putText(
                overlay,
                label,
                (center[0] - 8, center[1] + 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                label,
                (center[0] - 8, center[1] + 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), overlay):
        raise RuntimeError(f"Failed to write PNG to {output_path}")

    print(
        f"[eval_goal_grid_png] wrote {output_path} "
        f"({GOAL_GRID_ROWS}x{GOAL_GRID_COLS}={len(goals)} goals, "
        f"base_frame={hdf5_path.name}[{frame_index}])"
    )
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render the deterministic eval goal grid on a homography overhead PNG."
        )
    )
    parser.add_argument(
        "--hdf5",
        type=Path,
        required=True,
        help="Episode HDF5 with train_img (any frame from a real eval run).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Default: <hdf5_dir>/eval_goal_grid_homography.png",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/gcrl/gcrl_s4.yaml"),
        help="Env config YAML for table length/width (default: gcrl_s4).",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Which train_img row to use as the background frame.",
    )
    parser.add_argument(
        "--center-offset-constant",
        type=float,
        default=1.2,
        help="Table->robot x offset used by the real env overlays.",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Omit 1..20 index labels on each goal ring.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    hdf5_path = args.hdf5.expanduser().resolve()
    if not hdf5_path.exists():
        raise SystemExit(f"--hdf5 does not exist: {hdf5_path}")

    table_x_bot, table_y_right = _table_dims_from_config(args.config.expanduser().resolve())
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else hdf5_path.parent / "eval_goal_grid_homography.png"
    )

    render_eval_goal_grid_homography_png(
        hdf5_path=hdf5_path,
        output_path=output_path,
        table_x_bot=table_x_bot,
        table_y_right=table_y_right,
        center_offset_constant=float(args.center_offset_constant),
        frame_index=int(args.frame_index),
        label_goals=not bool(args.no_labels),
    )


if __name__ == "__main__":
    main()

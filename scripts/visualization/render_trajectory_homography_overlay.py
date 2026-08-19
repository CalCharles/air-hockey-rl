#!/usr/bin/env python3
"""Render a trajectory overlay PNG from weighted-sum of homography frames.

Each timestep's ``train_img`` row is warped with the same homography pipeline
as ``generate_episode_homography_gif``, then combined into one image:

    overlay = sum_i w_i * frame_i / sum_i w_i

Default weights are uniform; ``linear`` emphasizes later timesteps (motion
trail toward the end of the episode).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np

from airhockey.sims.real.control_parameters import homography_showdst_from_saved_frame


def _frame_weights(n_frames: int, mode: str, *, exp_alpha: float) -> np.ndarray:
    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    idx = np.arange(n_frames, dtype=np.float64)
    if mode == "uniform":
        weights = np.ones(n_frames, dtype=np.float64)
    elif mode == "linear":
        weights = idx + 1.0
    elif mode == "exp":
        denom = max(n_frames - 1, 1)
        weights = np.exp(float(exp_alpha) * idx / denom)
    else:
        raise ValueError(f"Unknown weight mode {mode!r}; choose uniform, linear, or exp.")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError(f"Degenerate frame weights for mode={mode!r}")
    return weights / total


def _load_train_img_frames(hdf5_path: Path) -> np.ndarray:
    with h5py.File(hdf5_path, "r") as h5_file:
        if "train_img" not in h5_file:
            raise ValueError(f"Missing train_img dataset in {hdf5_path}")
        camera_frames = np.asarray(h5_file["train_img"][:], dtype=np.uint8)
    if camera_frames.ndim != 4 or camera_frames.shape[0] == 0:
        raise ValueError(f"train_img must be non-empty rank-4, got {camera_frames.shape}")
    return camera_frames


def _select_frame_indices(
    n_frames: int,
    *,
    subsample: int,
    max_frames: int | None,
) -> np.ndarray:
    if subsample <= 0:
        raise ValueError("subsample must be > 0")
    stop = n_frames if max_frames is None else min(n_frames, int(max_frames))
    return np.arange(0, int(stop), int(subsample), dtype=int)


def weighted_sum_homography_frames(
    camera_frames: np.ndarray,
    *,
    frame_indices: np.ndarray,
    weight_mode: str = "uniform",
    exp_alpha: float = 3.0,
    emphasize_motion: bool = True,
    background_blend: float = 0.45,
    trail_gain: float = 1.8,
) -> np.ndarray:
    """Weighted average of homography-warped BGR frames."""
    if len(frame_indices) == 0:
        raise ValueError("No frame indices selected for overlay.")

    warped_frames = np.stack(
        [homography_showdst_from_saved_frame(camera_frames[int(i)]) for i in frame_indices],
        axis=0,
    ).astype(np.float64)
    weights = _frame_weights(len(frame_indices), weight_mode, exp_alpha=exp_alpha)
    weights_bc = weights.reshape(-1, 1, 1, 1)

    if emphasize_motion:
        background = warped_frames[0]
        residuals = np.abs(warped_frames - background)
        motion = (residuals * weights_bc).sum(axis=0)
        motion_luma = motion.mean(axis=2)
        peak = float(motion_luma.max())
        if peak > 0:
            motion_luma = motion_luma / peak
        trail = np.zeros_like(background)
        trail[..., 1] = motion_luma * 255.0
        overlay = background * float(background_blend) + trail * float(trail_gain)
        return np.clip(overlay, 0.0, 255.0).astype(np.uint8)

    overlay = (warped_frames * weights_bc).sum(axis=0)
    return np.clip(overlay, 0.0, 255.0).astype(np.uint8)


def render_trajectory_homography_overlay_png(
    *,
    hdf5_path: Path,
    output_path: Path,
    weight_mode: str = "uniform",
    exp_alpha: float = 3.0,
    subsample: int = 1,
    max_frames: int | None = None,
    emphasize_motion: bool = True,
    background_blend: float = 0.45,
    trail_gain: float = 1.8,
) -> Path:
    """Build and save a weighted-sum homography overlay PNG for one episode."""
    hdf5_path = Path(hdf5_path).expanduser().resolve()
    camera_frames = _load_train_img_frames(hdf5_path)
    frame_indices = _select_frame_indices(
        int(camera_frames.shape[0]),
        subsample=int(subsample),
        max_frames=max_frames,
    )
    overlay = weighted_sum_homography_frames(
        camera_frames,
        frame_indices=frame_indices,
        weight_mode=weight_mode,
        exp_alpha=exp_alpha,
        emphasize_motion=emphasize_motion,
        background_blend=background_blend,
        trail_gain=trail_gain,
    )

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), overlay):
        raise RuntimeError(f"Failed to write PNG to {output_path}")

    print(
        f"[trajectory_overlay] wrote {output_path} "
        f"(frames={len(frame_indices)}/{camera_frames.shape[0]}, "
        f"weight_mode={weight_mode}, subsample={subsample}, "
        f"emphasize_motion={emphasize_motion})"
    )
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a homography trajectory overlay PNG by weighted-averaging "
            "episode train_img frames."
        )
    )
    parser.add_argument(
        "--hdf5",
        type=Path,
        required=True,
        help="Episode HDF5 with train_img.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Default: <hdf5_dir>/trajectory_homography_overlay.png",
    )
    parser.add_argument(
        "--weight-mode",
        choices=("uniform", "linear", "exp"),
        default="uniform",
        help="Per-frame weights before normalization (default: uniform).",
    )
    parser.add_argument(
        "--exp-alpha",
        type=float,
        default=3.0,
        help="Exponent scale when --weight-mode exp.",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=1,
        help="Use every Nth frame from the episode.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Cap the number of leading frames considered.",
    )
    parser.add_argument(
        "--raw-average",
        action="store_true",
        help="Literal weighted average of frames (no motion emphasis).",
    )
    parser.add_argument(
        "--background-blend",
        type=float,
        default=0.45,
        help="Background strength when motion emphasis is enabled.",
    )
    parser.add_argument(
        "--trail-gain",
        type=float,
        default=1.8,
        help="Motion-trail strength when motion emphasis is enabled.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    hdf5_path = args.hdf5.expanduser().resolve()
    if not hdf5_path.exists():
        raise SystemExit(f"--hdf5 does not exist: {hdf5_path}")

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else hdf5_path.parent / "trajectory_homography_overlay.png"
    )

    render_trajectory_homography_overlay_png(
        hdf5_path=hdf5_path,
        output_path=output_path,
        weight_mode=str(args.weight_mode),
        exp_alpha=float(args.exp_alpha),
        subsample=int(args.subsample),
        max_frames=args.max_frames,
        emphasize_motion=not bool(args.raw_average),
        background_blend=float(args.background_blend),
        trail_gain=float(args.trail_gain),
    )


if __name__ == "__main__":
    main()

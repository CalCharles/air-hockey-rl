#!/usr/bin/env python3
"""Trim reset HDF5 files to data after final first-upward completion.

This utility targets reset artifacts stored in split-schema HDF5 files.
It keeps only the segment that starts right after the final "first upward"
transition (after the final bottom-wrap sweep), then optionally renders a GIF.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np

# Ensure local repo modules shadow external "scripts" packages (e.g., ROS).
REPO_ROOT = Path(__file__).resolve().parents[6]
_REPO_ROOT_STR = str(REPO_ROOT)
while _REPO_ROOT_STR in sys.path:
    sys.path.remove(_REPO_ROOT_STR)
sys.path.insert(0, _REPO_ROOT_STR)

from scripts.smooth_policy.amp_history.amp_training.td3.helper.episode_artifacts import (
    generate_episode_gif,
)
from scripts.smooth_policy.visualize_demo.visualize_real_trajectory_split import (
    OPTIONAL_SPLIT_DATASETS,
    SPLIT_DATASETS,
)


def _read_2d_dataset(h5_file: h5py.File, key: str, width: int) -> np.ndarray:
    if key not in h5_file:
        raise KeyError(f"Missing dataset '{key}'")
    arr = np.asarray(h5_file[key][:], dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"Dataset '{key}' must be 1D/2D, got shape={arr.shape}")
    if arr.shape[1] != int(width):
        raise ValueError(f"Dataset '{key}' expected width={width}, got shape={arr.shape}")
    return arr


def _list_hdf5_files(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".hdf5":
            raise ValueError(f"Input file must be .hdf5: {input_path}")
        return [input_path.resolve()]
    pattern = "**/*.hdf5" if recursive else "*.hdf5"
    return sorted(path.resolve() for path in input_path.glob(pattern) if path.is_file())


def _load_split_hdf5(path: Path) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray | None, dict]:
    required: Dict[str, np.ndarray] = {}
    optional: Dict[str, np.ndarray] = {}
    train_img: np.ndarray | None = None
    attrs: dict = {}

    with h5py.File(path, "r") as h5_file:
        for name, width in SPLIT_DATASETS:
            required[name] = _read_2d_dataset(h5_file, name, width)

        n_rows = required["cur_time"].shape[0]
        for name, width in OPTIONAL_SPLIT_DATASETS:
            if name not in h5_file:
                continue
            arr = np.asarray(h5_file[name][:], dtype=np.float64)
            if arr.ndim == 1:
                arr = arr[:, None]
            if arr.ndim != 2 or arr.shape[1] != int(width):
                raise ValueError(f"Optional dataset '{name}' has invalid shape {arr.shape}")
            if arr.shape[0] != n_rows:
                raise ValueError(
                    f"Optional dataset '{name}' length mismatch: {arr.shape[0]} vs {n_rows}"
                )
            optional[name] = arr

        if "train_img" in h5_file:
            train_img = np.asarray(h5_file["train_img"][:], dtype=np.uint8)
            if train_img.ndim != 4:
                raise ValueError(f"train_img must be rank-4, got {train_img.shape}")
            if train_img.shape[0] != n_rows:
                raise ValueError(
                    f"train_img length mismatch: {train_img.shape[0]} vs {n_rows}"
                )

        attrs = dict(h5_file.attrs.items())

    return required, optional, train_img, attrs


def _detect_trim_start_from_stage_id(
    optional: Dict[str, np.ndarray],
) -> tuple[int | None, str]:
    stage_arr = optional.get("reset_stage_id")
    if stage_arr is None:
        return None, "reset_stage_id_missing"
    stage = np.rint(np.asarray(stage_arr, dtype=np.float64).reshape(-1)).astype(np.int32)
    if stage.size <= 1:
        return None, "reset_stage_id_too_short"

    # Preferred signal: final transition from stage<=0 to stage>=1.
    transitions = np.where((stage[:-1] <= 0) & (stage[1:] >= 1))[0] + 1
    if transitions.size > 0:
        return int(transitions[-1]), "stage_transition_0_to_1"

    # Fallback if stage starts already in 1 (or all >=1): final stage>=1 run start.
    stage_one_mask = stage >= 1
    if not np.any(stage_one_mask):
        return None, "reset_stage_id_no_stage1"
    starts = np.where(stage_one_mask & np.concatenate(([True], ~stage_one_mask[:-1])))[0]
    if starts.size <= 0:
        return None, "reset_stage_id_no_stage1_run_start"
    return int(starts[-1]), "stage1_run_start"


def _detect_trim_start_heuristic(required: Dict[str, np.ndarray]) -> tuple[int | None, str]:
    pose = np.asarray(required["pose"], dtype=np.float64)
    desired_pose = np.asarray(required["desired_pose"], dtype=np.float64)
    if pose.shape[0] <= 8:
        return None, "trajectory_too_short"

    paddle_x = pose[:, 0]
    delta_x = desired_pose[:, 0] - pose[:, 0]
    delta_y = desired_pose[:, 1] - pose[:, 1]

    abs_dx = np.abs(delta_x)
    abs_dy = np.abs(delta_y)
    upward_dx_threshold = max(0.010, float(np.quantile(abs_dx, 0.75)) * 0.8)
    sweep_dy_threshold = max(0.005, float(np.quantile(abs_dy, 0.70)) * 0.7)
    sweep_dx_cap = max(0.010, float(np.quantile(abs_dx, 0.55)))
    bottom_x_threshold = float(np.quantile(paddle_x, 0.65))

    upward_mask = delta_x <= (-upward_dx_threshold)
    sweep_mask = (
        (np.abs(delta_y) >= sweep_dy_threshold)
        & (np.abs(delta_x) <= sweep_dx_cap)
        & (paddle_x >= bottom_x_threshold)
    )

    def _find_candidates(min_sweep: int, max_post_sweep: int) -> list[int]:
        candidates: list[int] = []
        n_steps = pose.shape[0]
        for idx in range(2, n_steps - 1):
            pre_upward = upward_mask[max(0, idx - 6):idx]
            pre_sweep = sweep_mask[max(0, idx - 40):idx]
            post_sweep = sweep_mask[idx:min(n_steps, idx + 20)]
            if (
                int(np.count_nonzero(pre_upward)) >= 2
                and int(np.count_nonzero(pre_sweep)) >= min_sweep
                and int(np.count_nonzero(post_sweep)) <= max_post_sweep
                and (not bool(upward_mask[idx]))
            ):
                candidates.append(idx)
        return candidates

    strict_candidates = _find_candidates(min_sweep=6, max_post_sweep=2)
    if strict_candidates:
        return int(strict_candidates[-1]), "heuristic_strict"

    relaxed_candidates = _find_candidates(min_sweep=4, max_post_sweep=4)
    if relaxed_candidates:
        return int(relaxed_candidates[-1]), "heuristic_relaxed"

    return None, "heuristic_no_candidate"


def _slice_episode(
    required: Dict[str, np.ndarray],
    optional: Dict[str, np.ndarray],
    train_img: np.ndarray | None,
    start_idx: int,
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray | None]:
    required_out = {name: np.asarray(arr[start_idx:], dtype=np.float64) for name, arr in required.items()}
    optional_out = {name: np.asarray(arr[start_idx:], dtype=np.float64) for name, arr in optional.items()}
    train_img_out = train_img[start_idx:] if train_img is not None else None
    return required_out, optional_out, train_img_out


def _write_split_hdf5(
    output_path: Path,
    required: Dict[str, np.ndarray],
    optional: Dict[str, np.ndarray],
    train_img: np.ndarray | None,
    attrs: dict,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5_file:
        for name, _ in SPLIT_DATASETS:
            data = required[name]
            h5_file.create_dataset(
                name,
                shape=data.shape,
                compression="gzip",
                compression_opts=9,
                data=data,
            )
        for name, _ in OPTIONAL_SPLIT_DATASETS:
            if name not in optional:
                continue
            data = optional[name]
            h5_file.create_dataset(
                name,
                shape=data.shape,
                compression="gzip",
                compression_opts=9,
                data=data,
            )
        if train_img is not None:
            h5_file.create_dataset(
                "train_img",
                shape=train_img.shape,
                compression="gzip",
                compression_opts=9,
                data=train_img,
            )
        for key, value in attrs.items():
            h5_file.attrs[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Trim reset split-schema HDF5 files to data following the final "
            "first-upward-motion completion."
        )
    )
    parser.add_argument("input_path", type=str, help="Input .hdf5 file or directory")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where trimmed .hdf5 files will be written",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively scan input directory")
    parser.add_argument("--dry-run", action="store_true", help="Only print decisions without writing output")
    parser.add_argument(
        "--force-heuristic",
        action="store_true",
        help="Ignore reset_stage_id and always use heuristic detection",
    )
    parser.add_argument(
        "--keep-original-on-ambiguous",
        action="store_true",
        help="Copy original file when trim boundary is ambiguous (default: skip file)",
    )
    parser.add_argument(
        "--min-keep-steps",
        type=int,
        default=1,
        help="Minimum required timesteps after trim; otherwise treat as ambiguous",
    )
    parser.add_argument(
        "--render-gif",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-render GIF for each trimmed output file",
    )
    parser.add_argument("--gif-fps", type=int, default=20)
    parser.add_argument("--gif-subsample", type=int, default=1)
    parser.add_argument("--gif-max-frames", type=int, default=0, help="0 disables frame cap")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gif_root = output_dir / "gifs"

    files = _list_hdf5_files(input_path, recursive=bool(args.recursive))
    if not files:
        raise FileNotFoundError(f"No .hdf5 files found under {input_path}")

    stats = {
        "files_total": 0,
        "files_written": 0,
        "files_skipped_ambiguous": 0,
        "files_copied_original": 0,
        "files_failed": 0,
        "gifs_written": 0,
        "gifs_failed": 0,
    }

    for file_path in files:
        stats["files_total"] += 1
        rel_name = file_path.name
        out_hdf5 = output_dir / rel_name

        try:
            required, optional, train_img, attrs = _load_split_hdf5(file_path)
            n_steps = int(required["cur_time"].shape[0])

            trim_idx = None
            detection_reason = "unknown"
            if not bool(args.force_heuristic):
                trim_idx, detection_reason = _detect_trim_start_from_stage_id(optional)

            if trim_idx is None:
                trim_idx, heuristic_reason = _detect_trim_start_heuristic(required)
                detection_reason = f"{detection_reason}+{heuristic_reason}"

            if trim_idx is None:
                print(f"[trim][ambiguous] file={rel_name} reason={detection_reason} action=skip")
                stats["files_skipped_ambiguous"] += 1
                if bool(args.keep_original_on_ambiguous):
                    if not bool(args.dry_run):
                        shutil.copy2(file_path, out_hdf5)
                    stats["files_copied_original"] += 1
                    print(f"[trim][copied] file={rel_name} -> {out_hdf5}")
                continue

            trim_idx = int(max(0, min(trim_idx, n_steps)))
            kept_steps = int(n_steps - trim_idx)
            if kept_steps < int(args.min_keep_steps):
                print(
                    f"[trim][ambiguous] file={rel_name} trim_idx={trim_idx} "
                    f"kept_steps={kept_steps} < min_keep_steps={int(args.min_keep_steps)} action=skip"
                )
                stats["files_skipped_ambiguous"] += 1
                continue

            required_out, optional_out, train_img_out = _slice_episode(
                required, optional, train_img, trim_idx
            )
            print(
                f"[trim] file={rel_name} detection={detection_reason} "
                f"trim_idx={trim_idx} orig_steps={n_steps} kept_steps={kept_steps}"
            )

            if not bool(args.dry_run):
                _write_split_hdf5(
                    out_hdf5,
                    required=required_out,
                    optional=optional_out,
                    train_img=train_img_out,
                    attrs=attrs,
                )
            stats["files_written"] += 1

            if bool(args.render_gif):
                try:
                    max_frames = None if int(args.gif_max_frames) <= 0 else int(args.gif_max_frames)
                    if not bool(args.dry_run):
                        gif_path = generate_episode_gif(
                            episode_hdf5_path=out_hdf5,
                            gif_root=gif_root,
                            fps=int(args.gif_fps),
                            max_frames=max_frames,
                            subsample=int(args.gif_subsample),
                            require_puck=False,
                        )
                        print(f"[trim][gif] file={rel_name} saved={gif_path}")
                    stats["gifs_written"] += 1
                except Exception as exc:
                    stats["gifs_failed"] += 1
                    print(f"[trim][gif][failed] file={rel_name} error={exc}")
        except Exception as exc:
            stats["files_failed"] += 1
            print(f"[trim][failed] file={rel_name} error={exc}")

    print(
        "[trim][summary] "
        + " ".join(f"{key}={value}" for key, value in stats.items())
    )


if __name__ == "__main__":
    main()

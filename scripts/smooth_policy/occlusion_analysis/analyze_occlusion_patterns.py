#!/usr/bin/env python3
"""Analyze occlusion patterns in processed real-world HDF5 trajectories."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np


@dataclass
class TrajectoryMetrics:
    filename: str
    num_frames: int
    duration_s: float
    occluded_count: int
    occluded_fraction: float
    num_runs: int
    mean_run_frames: float
    max_run_frames: int
    isolated_runs: int
    short_runs: int
    medium_runs: int
    long_runs: int
    window_1s_mean_count: float
    window_1s_fano: float
    window_5s_mean_count: float
    window_5s_fano: float
    has_occlusion: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze occlusion patterns across trajectories.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/data2/air_hockey/air_hockey_state_data/datastor1/calebc/public/data/mouse/state_data_all_new/",
        help="Directory containing trajectory_data*.hdf5 files.",
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=40,
        help="Number of trajectories to analyze.",
    )
    parser.add_argument(
        "--sampling-mode",
        type=str,
        choices=["first", "random"],
        default="first",
        help="Trajectory selection mode.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used when --sampling-mode=random.",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=60,
        help="2D histogram bins for position heatmaps.",
    )
    parser.add_argument(
        "--context-bins",
        type=int,
        default=8,
        help="Number of bins for context-conditioned occlusion rates.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="scripts/smooth_policy/occlusion_analysis/output",
        help="Root output directory for analysis artifacts.",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default="",
        help="Optional run tag. If empty, uses a timestamp.",
    )
    return parser.parse_args()


def robust_timestamps(cur_time: np.ndarray) -> np.ndarray:
    n = len(cur_time)
    t = np.asarray(cur_time, dtype=float).reshape(-1)
    finite = np.isfinite(t)
    if not np.any(finite):
        return np.arange(n, dtype=float) / 20.0
    if not np.all(finite):
        valid_idx = np.flatnonzero(finite)
        invalid_idx = np.flatnonzero(~finite)
        t[invalid_idx] = np.interp(invalid_idx, valid_idx, t[valid_idx])
    t = np.maximum.accumulate(t)
    t = t - t[0]
    if t[-1] <= 0.0:
        return np.arange(n, dtype=float) / 20.0
    return t


def median_dt(t: np.ndarray) -> float:
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return 1.0 / 20.0
    return float(np.median(dt))


def contiguous_true_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segs: list[tuple[int, int]] = []
    in_seg = False
    start = 0
    for i, is_true in enumerate(mask):
        if is_true and not in_seg:
            start = i
            in_seg = True
        elif (not is_true) and in_seg:
            segs.append((start, i - 1))
            in_seg = False
    if in_seg:
        segs.append((start, len(mask) - 1))
    return segs


def run_length_breakdown(occluded: np.ndarray) -> dict[str, int | list[int]]:
    segs = contiguous_true_segments(occluded)
    lengths = [(end - start + 1) for start, end in segs]
    isolated = sum(1 for x in lengths if x == 1)
    short = sum(1 for x in lengths if 2 <= x <= 5)
    medium = sum(1 for x in lengths if 6 <= x <= 20)
    long = sum(1 for x in lengths if x > 20)
    return {
        "lengths_frames": lengths,
        "isolated_runs": isolated,
        "short_runs": short,
        "medium_runs": medium,
        "long_runs": long,
    }


def sliding_window_counts(
    t: np.ndarray, mask: np.ndarray, window_s: float
) -> np.ndarray:
    if len(t) == 0:
        return np.array([], dtype=float)
    span = max(float(t[-1] - t[0]), 1e-8)
    num_windows = int(np.ceil(span / window_s))
    if num_windows <= 0:
        return np.array([float(np.sum(mask))], dtype=float)
    window_idx = np.floor((t - t[0]) / window_s).astype(int)
    window_idx = np.clip(window_idx, 0, max(num_windows - 1, 0))
    counts = np.zeros(num_windows, dtype=float)
    np.add.at(counts, window_idx, mask.astype(float))
    return counts


def safe_fano(counts: np.ndarray) -> float:
    if counts.size == 0:
        return 0.0
    mean_val = float(np.mean(counts))
    if mean_val <= 1e-12:
        return 0.0
    return float(np.var(counts) / mean_val)


def quantile_bin_occlusion_rate(
    values: np.ndarray, occluded: np.ndarray, name: str, n_bins: int
) -> list[dict[str, float | int | str]]:
    finite_mask = np.isfinite(values)
    vals = values[finite_mask]
    occ = occluded[finite_mask]
    if vals.size == 0:
        return []
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(vals, quantiles))
    if len(edges) < 2:
        return []

    rows: list[dict[str, float | int | str]] = []
    for idx in range(len(edges) - 1):
        lo = float(edges[idx])
        hi = float(edges[idx + 1])
        if idx < len(edges) - 2:
            in_bin = (vals >= lo) & (vals < hi)
        else:
            in_bin = (vals >= lo) & (vals <= hi)
        count = int(np.sum(in_bin))
        if count == 0:
            continue
        occ_count = int(np.sum(occ[in_bin]))
        rows.append(
            {
                "metric": name,
                "bin_index": idx,
                "bin_lo": lo,
                "bin_hi": hi,
                "count": count,
                "occluded_count": occ_count,
                "occlusion_rate": float(occ_count / count),
            }
        )
    return rows


def list_trajectory_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.glob("trajectory_data*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No trajectory_data*.hdf5 files found in {data_dir}")
    return files


def select_files(files: list[Path], n: int, mode: str, seed: int) -> list[Path]:
    if n <= 0:
        raise ValueError("--num-trajectories must be > 0")
    if len(files) < n:
        raise ValueError(f"Requested {n} trajectories, but only found {len(files)}")
    if mode == "first":
        return files[:n]
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(files), size=n, replace=False))
    return [files[i] for i in idx]


def compute_xy_bounds(all_xy_visible: list[np.ndarray], all_xy_occluded: list[np.ndarray]) -> tuple[float, float, float, float]:
    merged: list[np.ndarray] = []
    if all_xy_visible:
        merged.append(np.concatenate(all_xy_visible, axis=0))
    if all_xy_occluded:
        merged.append(np.concatenate(all_xy_occluded, axis=0))
    if not merged:
        return -1.0, 1.0, -0.5, 0.5
    xy = np.concatenate(merged, axis=0)
    x = xy[:, 0]
    y = xy[:, 1]
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size == 0:
        return -1.0, 1.0, -0.5, 0.5
    x_lo, x_hi = np.quantile(x, [0.01, 0.99])
    y_lo, y_hi = np.quantile(y, [0.01, 0.99])
    if not np.isfinite(x_lo) or not np.isfinite(x_hi) or x_lo == x_hi:
        x_lo, x_hi = float(np.min(x)), float(np.max(x))
    if not np.isfinite(y_lo) or not np.isfinite(y_hi) or y_lo == y_hi:
        y_lo, y_hi = float(np.min(y)), float(np.max(y))
    if x_lo == x_hi:
        x_lo -= 0.5
        x_hi += 0.5
    if y_lo == y_hi:
        y_lo -= 0.5
        y_hi += 0.5
    margin_x = 0.05 * (x_hi - x_lo)
    margin_y = 0.05 * (y_hi - y_lo)
    return x_lo - margin_x, x_hi + margin_x, y_lo - margin_y, y_hi + margin_y


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    run_tag = args.run_tag.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / run_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    all_files = list_trajectory_files(data_dir)
    selected_files = select_files(all_files, args.num_trajectories, args.sampling_mode, args.random_seed)

    per_traj_rows: list[dict] = []
    context_rows: list[dict] = []
    errors: list[dict] = []

    all_xy_visible: list[np.ndarray] = []
    all_xy_occluded: list[np.ndarray] = []
    all_xy_occ_start: list[np.ndarray] = []
    all_xy_occ_end: list[np.ndarray] = []
    all_paddle_xy_occluded: list[np.ndarray] = []
    all_run_lengths_frames: list[int] = []
    all_window_1s_counts: list[np.ndarray] = []
    all_window_5s_counts: list[np.ndarray] = []

    global_num_frames = 0
    global_occluded_frames = 0
    trajectories_with_occlusion = 0

    aggregate_puck_speed: list[np.ndarray] = []
    aggregate_paddle_speed: list[np.ndarray] = []
    aggregate_puck_paddle_dist: list[np.ndarray] = []
    aggregate_dist_to_edge: list[np.ndarray] = []
    aggregate_occluded_mask: list[np.ndarray] = []

    for file_path in selected_files:
        try:
            with h5py.File(file_path, "r") as hf:
                keys = set(hf.keys())
                if "puck" not in keys:
                    raise KeyError("Missing required key: puck")
                puck = np.asarray(hf["puck"])
                if puck.ndim != 2 or puck.shape[1] < 3:
                    raise ValueError(f"Invalid puck shape {puck.shape}, expected (N, >=3)")
                n = int(puck.shape[0])
                if n == 0:
                    raise ValueError("Trajectory has zero frames")

                if "cur_time" in keys:
                    t = robust_timestamps(np.asarray(hf["cur_time"]).reshape(-1))
                else:
                    t = np.arange(n, dtype=float) / 20.0
                if len(t) != n:
                    raise ValueError(f"cur_time length mismatch: {len(t)} vs puck length {n}")
                dt_med = median_dt(t)

                if "paddle" in keys:
                    paddle = np.asarray(hf["paddle"])
                elif "pose" in keys:
                    paddle = np.asarray(hf["pose"])
                else:
                    raise KeyError("Missing both paddle and pose; cannot derive paddle position")
                if paddle.ndim != 2 or paddle.shape[0] != n or paddle.shape[1] < 2:
                    raise ValueError(f"Invalid paddle/pose shape {paddle.shape}, expected (N, >=2)")

                speed_xy = None
                if "speed" in keys:
                    speed = np.asarray(hf["speed"])
                    if speed.ndim == 2 and speed.shape[0] == n and speed.shape[1] >= 2:
                        speed_xy = speed[:, :2].astype(float)

                puck_xy = puck[:, :2].astype(float)
                occluded = (puck[:, 2].astype(float) > 0.5)
                visible = ~occluded

                finite_puck = np.isfinite(puck_xy).all(axis=1)
                finite_paddle = np.isfinite(paddle[:, :2]).all(axis=1)
                valid = finite_puck & finite_paddle
                visible_valid = visible & valid
                occluded_valid = occluded & valid

                global_num_frames += n
                occluded_count = int(np.sum(occluded))
                global_occluded_frames += occluded_count
                if occluded_count > 0:
                    trajectories_with_occlusion += 1

                if np.any(visible_valid):
                    all_xy_visible.append(puck_xy[visible_valid])
                if np.any(occluded_valid):
                    all_xy_occluded.append(puck_xy[occluded_valid])
                    all_paddle_xy_occluded.append(paddle[occluded_valid, :2].astype(float))

                occlusion_starts = np.flatnonzero(occluded & np.concatenate([[False], ~occluded[:-1]]))
                start_ref = occlusion_starts - 1
                start_ref = start_ref[start_ref >= 0]
                if start_ref.size > 0:
                    start_vis = visible[start_ref] & valid[start_ref]
                    if np.any(start_vis):
                        all_xy_occ_start.append(puck_xy[start_ref[start_vis]])

                occlusion_ends = np.flatnonzero((~occluded) & np.concatenate([[False], occluded[:-1]]))
                if occlusion_ends.size > 0:
                    end_vis = visible[occlusion_ends] & valid[occlusion_ends]
                    if np.any(end_vis):
                        all_xy_occ_end.append(puck_xy[occlusion_ends[end_vis]])

                run_stats = run_length_breakdown(occluded)
                lengths = run_stats["lengths_frames"]
                all_run_lengths_frames.extend(lengths)

                w1 = sliding_window_counts(t, occluded, window_s=1.0)
                w5 = sliding_window_counts(t, occluded, window_s=5.0)
                all_window_1s_counts.append(w1)
                all_window_5s_counts.append(w5)

                puck_delta = np.diff(puck_xy, axis=0, prepend=puck_xy[:1])
                dt = np.diff(t, prepend=t[:1])
                dt = np.where(dt <= 1e-8, dt_med, dt)
                puck_speed = np.linalg.norm(puck_delta / dt[:, None], axis=1)

                if speed_xy is not None:
                    paddle_speed = np.linalg.norm(speed_xy, axis=1)
                else:
                    paddle_delta = np.diff(paddle[:, :2], axis=0, prepend=paddle[:1, :2])
                    paddle_speed = np.linalg.norm(paddle_delta / dt[:, None], axis=1)

                puck_paddle_dist = np.linalg.norm(puck_xy - paddle[:, :2], axis=1)

                # Temporary edge metric, refined globally after aggregate bounds are known.
                dist_to_edge_placeholder = np.zeros(n, dtype=float)

                aggregate_puck_speed.append(puck_speed[valid])
                aggregate_paddle_speed.append(paddle_speed[valid])
                aggregate_puck_paddle_dist.append(puck_paddle_dist[valid])
                aggregate_dist_to_edge.append(dist_to_edge_placeholder[valid])
                aggregate_occluded_mask.append(occluded[valid])

                row = TrajectoryMetrics(
                    filename=file_path.name,
                    num_frames=n,
                    duration_s=float(t[-1] - t[0]),
                    occluded_count=occluded_count,
                    occluded_fraction=float(occluded_count / n),
                    num_runs=len(lengths),
                    mean_run_frames=float(np.mean(lengths)) if lengths else 0.0,
                    max_run_frames=int(np.max(lengths)) if lengths else 0,
                    isolated_runs=int(run_stats["isolated_runs"]),
                    short_runs=int(run_stats["short_runs"]),
                    medium_runs=int(run_stats["medium_runs"]),
                    long_runs=int(run_stats["long_runs"]),
                    window_1s_mean_count=float(np.mean(w1)) if w1.size else 0.0,
                    window_1s_fano=safe_fano(w1),
                    window_5s_mean_count=float(np.mean(w5)) if w5.size else 0.0,
                    window_5s_fano=safe_fano(w5),
                    has_occlusion=int(occluded_count > 0),
                )
                per_traj_rows.append(row.__dict__)

        except Exception as exc:  # pylint: disable=broad-except
            errors.append({"file": file_path.name, "error": str(exc)})

    if not per_traj_rows:
        raise RuntimeError("No trajectories were processed successfully.")

    x_lo, x_hi, y_lo, y_hi = compute_xy_bounds(all_xy_visible, all_xy_occluded)
    xy_range = [[x_lo, x_hi], [y_lo, y_hi]]

    visible_xy = np.concatenate(all_xy_visible, axis=0) if all_xy_visible else np.zeros((0, 2), dtype=float)
    occluded_xy = np.concatenate(all_xy_occluded, axis=0) if all_xy_occluded else np.zeros((0, 2), dtype=float)
    occ_start_xy = np.concatenate(all_xy_occ_start, axis=0) if all_xy_occ_start else np.zeros((0, 2), dtype=float)
    occ_end_xy = np.concatenate(all_xy_occ_end, axis=0) if all_xy_occ_end else np.zeros((0, 2), dtype=float)
    paddle_occ_xy = (
        np.concatenate(all_paddle_xy_occluded, axis=0) if all_paddle_xy_occluded else np.zeros((0, 2), dtype=float)
    )

    hist_visible, x_edges, y_edges = np.histogram2d(
        visible_xy[:, 0], visible_xy[:, 1], bins=args.num_bins, range=xy_range
    )
    hist_occluded, _, _ = np.histogram2d(
        occluded_xy[:, 0], occluded_xy[:, 1], bins=args.num_bins, range=xy_range
    )
    hist_occ_start, _, _ = np.histogram2d(
        occ_start_xy[:, 0], occ_start_xy[:, 1], bins=args.num_bins, range=xy_range
    )
    hist_occ_end, _, _ = np.histogram2d(
        occ_end_xy[:, 0], occ_end_xy[:, 1], bins=args.num_bins, range=xy_range
    )
    hist_paddle_occ, _, _ = np.histogram2d(
        paddle_occ_xy[:, 0], paddle_occ_xy[:, 1], bins=args.num_bins, range=xy_range
    )

    all_window_1s = np.concatenate(all_window_1s_counts) if all_window_1s_counts else np.zeros(0, dtype=float)
    all_window_5s = np.concatenate(all_window_5s_counts) if all_window_5s_counts else np.zeros(0, dtype=float)

    # Recompute distance-to-edge with global bounds for context bins.
    dist_to_edge_segments: list[np.ndarray] = []
    for xy in all_xy_visible + all_xy_occluded:
        if xy.size == 0:
            continue
        dist = np.minimum.reduce(
            [
                xy[:, 0] - x_lo,
                x_hi - xy[:, 0],
                xy[:, 1] - y_lo,
                y_hi - xy[:, 1],
            ]
        )
        dist_to_edge_segments.append(dist)

    # Build context arrays from valid aligned samples.
    puck_speed_all = np.concatenate(aggregate_puck_speed) if aggregate_puck_speed else np.zeros(0, dtype=float)
    paddle_speed_all = np.concatenate(aggregate_paddle_speed) if aggregate_paddle_speed else np.zeros(0, dtype=float)
    puck_paddle_dist_all = (
        np.concatenate(aggregate_puck_paddle_dist) if aggregate_puck_paddle_dist else np.zeros(0, dtype=float)
    )
    occluded_all = np.concatenate(aggregate_occluded_mask) if aggregate_occluded_mask else np.zeros(0, dtype=bool)

    # Dist-to-edge uses pooled xy arrays, so we trim to a safe shared length for bin analyses.
    pooled_dist_to_edge = np.concatenate(dist_to_edge_segments) if dist_to_edge_segments else np.zeros(0, dtype=float)
    min_len = min(
        len(puck_speed_all),
        len(paddle_speed_all),
        len(puck_paddle_dist_all),
        len(occluded_all),
    )
    puck_speed_all = puck_speed_all[:min_len]
    paddle_speed_all = paddle_speed_all[:min_len]
    puck_paddle_dist_all = puck_paddle_dist_all[:min_len]
    occluded_all = occluded_all[:min_len]

    context_rows.extend(
        quantile_bin_occlusion_rate(puck_speed_all, occluded_all, "puck_speed_mps", args.context_bins)
    )
    context_rows.extend(
        quantile_bin_occlusion_rate(paddle_speed_all, occluded_all, "paddle_speed_mps", args.context_bins)
    )
    context_rows.extend(
        quantile_bin_occlusion_rate(
            puck_paddle_dist_all, occluded_all, "puck_paddle_distance_m", args.context_bins
        )
    )
    if pooled_dist_to_edge.size > 0 and occluded_xy.shape[0] > 0:
        # Dist-to-edge binning against a direct occlusion mask from pooled puck samples.
        pooled_xy = np.concatenate([visible_xy, occluded_xy], axis=0)
        pooled_occ = np.concatenate(
            [np.zeros(len(visible_xy), dtype=bool), np.ones(len(occluded_xy), dtype=bool)], axis=0
        )
        pooled_dist = np.minimum.reduce(
            [
                pooled_xy[:, 0] - x_lo,
                x_hi - pooled_xy[:, 0],
                pooled_xy[:, 1] - y_lo,
                y_hi - pooled_xy[:, 1],
            ]
        )
        context_rows.extend(
            quantile_bin_occlusion_rate(
                pooled_dist, pooled_occ, "distance_to_edge_m", args.context_bins
            )
        )

    per_traj_sorted = sorted(per_traj_rows, key=lambda row: row["filename"])
    occlusion_fractions = np.array([row["occluded_fraction"] for row in per_traj_sorted], dtype=float)
    fano_1s = np.array([row["window_1s_fano"] for row in per_traj_sorted], dtype=float)
    fano_5s = np.array([row["window_5s_fano"] for row in per_traj_sorted], dtype=float)

    global_occlusion_fraction = float(global_occluded_frames / max(global_num_frames, 1))
    trajectories_processed = len(per_traj_sorted)
    trajectories_failed = len(errors)
    trajectories_with_occlusion_fraction = float(trajectories_with_occlusion / trajectories_processed)

    summary = {
        "config": {
            "data_dir": str(data_dir),
            "num_trajectories_requested": int(args.num_trajectories),
            "sampling_mode": args.sampling_mode,
            "random_seed": int(args.random_seed),
            "num_bins": int(args.num_bins),
            "context_bins": int(args.context_bins),
            "output_dir": str(output_dir),
            "run_tag": run_tag,
        },
        "selection": {
            "trajectories_processed": trajectories_processed,
            "trajectories_failed": trajectories_failed,
            "selected_files": [path.name for path in selected_files],
            "processed_files": [row["filename"] for row in per_traj_sorted],
            "errors": errors,
        },
        "global": {
            "num_frames": int(global_num_frames),
            "occluded_frames": int(global_occluded_frames),
            "occluded_fraction": global_occlusion_fraction,
            "trajectories_with_occlusion": int(trajectories_with_occlusion),
            "trajectories_with_occlusion_fraction": trajectories_with_occlusion_fraction,
            "run_length_stats": {
                "num_runs": int(len(all_run_lengths_frames)),
                "mean_run_frames": float(np.mean(all_run_lengths_frames)) if all_run_lengths_frames else 0.0,
                "max_run_frames": int(np.max(all_run_lengths_frames)) if all_run_lengths_frames else 0,
                "isolated_runs": int(sum(1 for v in all_run_lengths_frames if v == 1)),
                "short_runs": int(sum(1 for v in all_run_lengths_frames if 2 <= v <= 5)),
                "medium_runs": int(sum(1 for v in all_run_lengths_frames if 6 <= v <= 20)),
                "long_runs": int(sum(1 for v in all_run_lengths_frames if v > 20)),
                "lengths_frames": [int(v) for v in all_run_lengths_frames],
            },
            "burstiness": {
                "window_1s_mean_count": float(np.mean(all_window_1s)) if all_window_1s.size else 0.0,
                "window_1s_fano": safe_fano(all_window_1s),
                "window_5s_mean_count": float(np.mean(all_window_5s)) if all_window_5s.size else 0.0,
                "window_5s_fano": safe_fano(all_window_5s),
                "per_trajectory_window_1s_fano_mean": float(np.mean(fano_1s)),
                "per_trajectory_window_5s_fano_mean": float(np.mean(fano_5s)),
            },
            "xy_bounds": {
                "x_lo": x_lo,
                "x_hi": x_hi,
                "y_lo": y_lo,
                "y_hi": y_hi,
            },
            "occluded_xy_mean": [
                float(np.mean(occluded_xy[:, 0])) if len(occluded_xy) else 0.0,
                float(np.mean(occluded_xy[:, 1])) if len(occluded_xy) else 0.0,
            ],
            "visible_xy_mean": [
                float(np.mean(visible_xy[:, 0])) if len(visible_xy) else 0.0,
                float(np.mean(visible_xy[:, 1])) if len(visible_xy) else 0.0,
            ],
        },
        "per_trajectory_occluded_fraction": {
            "mean": float(np.mean(occlusion_fractions)),
            "median": float(np.median(occlusion_fractions)),
            "p95": float(np.percentile(occlusion_fractions, 95)),
            "max": float(np.max(occlusion_fractions)),
        },
    }

    # Persist artifacts.
    summary_path = output_dir / "occlusion_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(output_dir / "per_trajectory_metrics.csv", per_traj_sorted)
    write_csv(output_dir / "occlusion_context_bins.csv", context_rows)

    np.savez_compressed(
        output_dir / "occlusion_arrays.npz",
        hist_visible=hist_visible,
        hist_occluded=hist_occluded,
        hist_occ_start=hist_occ_start,
        hist_occ_end=hist_occ_end,
        hist_paddle_occluded=hist_paddle_occ,
        x_edges=x_edges,
        y_edges=y_edges,
        window_1s_counts=all_window_1s,
        window_5s_counts=all_window_5s,
        run_lengths_frames=np.array(all_run_lengths_frames, dtype=int),
    )

    burst_desc = "bursty" if safe_fano(all_window_1s) > 1.0 else "mostly isolated/random"
    report_lines = [
        "# Occlusion Analysis Report",
        "",
        f"- Trajectories processed: {trajectories_processed} (requested {args.num_trajectories})",
        f"- Total frames: {global_num_frames}",
        f"- Occluded frames: {global_occluded_frames} ({100.0 * global_occlusion_fraction:.2f}%)",
        f"- Trajectories with any occlusion: {trajectories_with_occlusion}/{trajectories_processed} ({100.0 * trajectories_with_occlusion_fraction:.2f}%)",
        "",
        "## Headline Answers",
        "",
        f"- Where occlusions generally occur: mean occluded puck position is ({summary['global']['occluded_xy_mean'][0]:.3f}, {summary['global']['occluded_xy_mean'][1]:.3f}). Compare with the visible-vs-occluded heatmap for spatial concentration.",
        f"- How often occlusions occur: {100.0 * global_occlusion_fraction:.2f}% of all frames are occluded (median per-trajectory occlusion rate {100.0 * summary['per_trajectory_occluded_fraction']['median']:.2f}%).",
        f"- Are they isolated or bursty: overall windowed burstiness is {burst_desc} (1s Fano={summary['global']['burstiness']['window_1s_fano']:.3f}, 5s Fano={summary['global']['burstiness']['window_5s_fano']:.3f}).",
        "",
        "## Temporal Structure",
        "",
        f"- Number of occlusion runs: {summary['global']['run_length_stats']['num_runs']}",
        f"- Mean run length: {summary['global']['run_length_stats']['mean_run_frames']:.2f} frames",
        f"- Max run length: {summary['global']['run_length_stats']['max_run_frames']} frames",
        f"- Run classes: isolated={summary['global']['run_length_stats']['isolated_runs']}, short={summary['global']['run_length_stats']['short_runs']}, medium={summary['global']['run_length_stats']['medium_runs']}, long={summary['global']['run_length_stats']['long_runs']}",
        "",
        "## Artifacts",
        "",
        "- `occlusion_summary.json`",
        "- `per_trajectory_metrics.csv`",
        "- `occlusion_context_bins.csv`",
        "- `occlusion_arrays.npz`",
        "",
        "Generate plots with `plot_occlusion_results.py` in the same output directory.",
    ]
    (output_dir / "occlusion_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print("=== Occlusion Analysis Complete ===")
    print(f"Output directory: {output_dir}")
    print(f"Processed trajectories: {trajectories_processed}/{args.num_trajectories}")
    print(f"Global occlusion fraction: {100.0 * global_occlusion_fraction:.2f}%")
    print(f"Trajectories with occlusions: {trajectories_with_occlusion}/{trajectories_processed}")
    print(f"1s burstiness (Fano): {summary['global']['burstiness']['window_1s_fano']:.3f}")
    print(f"5s burstiness (Fano): {summary['global']['burstiness']['window_5s_fano']:.3f}")


if __name__ == "__main__":
    main()


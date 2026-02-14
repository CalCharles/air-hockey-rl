#!/usr/bin/env python3
"""
Analyze puck trajectory quality from a saved real rollout HDF5 file.

Outputs:
- In-frame time windows (raw + time-padded)
- Jump frequency metrics
- Occlusion metrics
- Noise and jitter proxies
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


# train_vals layout from scripts/real/README.md
IDX_CUR_TIME = 0
IDX_PUCK_X = 32
IDX_PUCK_Y = 33
IDX_PUCK_OCCLUDED = 34

# Match scripts/real/visualize_saved_trajectory.py constants
TABLE_LENGTH_M = 1.9304
TABLE_WIDTH_M = 0.8636


@dataclass
class TimeWindow:
    start_s: float
    end_s: float
    duration_s: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze puck position and occlusion metrics from trajectory_data*.hdf5."
    )
    parser.add_argument("input_hdf5", type=str, help="Path to trajectory_data*.hdf5")
    parser.add_argument(
        "--frame-margin-m",
        type=float,
        default=0.05,
        help="Extra spatial margin around table bounds for in-frame check (meters).",
    )
    parser.add_argument(
        "--window-pad-s",
        type=float,
        default=0.20,
        help="Time padding added to both sides of each in-frame window (seconds).",
    )
    parser.add_argument(
        "--jump-z",
        type=float,
        default=3.0,
        help="Robust jump threshold multiplier: median + jump_z * MAD.",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save JSON report next to input file (<stem>_puck_analysis.json).",
    )
    return parser.parse_args()


def load_train_vals(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        if "train_vals" not in f:
            raise ValueError(f"{path} does not contain 'train_vals'")
        vals = f["train_vals"][:]
    if vals.ndim != 2 or vals.shape[1] < 35:
        raise ValueError(f"Unexpected train_vals shape: {vals.shape}")
    return vals


def robust_timestamps(raw_t: np.ndarray, n: int) -> np.ndarray:
    raw_t = np.asarray(raw_t, dtype=float)
    if raw_t.shape[0] != n:
        raise ValueError("Timestamp length mismatch.")
    finite = np.isfinite(raw_t)
    if not np.any(finite):
        # Fallback 20 Hz if no valid times.
        return np.arange(n, dtype=float) / 20.0
    t = raw_t.copy()
    # Fill invalid timestamps by nearest valid interpolation.
    if not np.all(finite):
        valid_idx = np.flatnonzero(finite)
        invalid_idx = np.flatnonzero(~finite)
        t[invalid_idx] = np.interp(invalid_idx, valid_idx, t[valid_idx])
    # Enforce monotonic non-decreasing timestamps.
    t = np.maximum.accumulate(t)
    # If absolute epoch-like time, normalize to start at 0 for readability.
    t = t - t[0]
    if t[-1] <= 0:
        return np.arange(n, dtype=float) / 20.0
    return t


def median_dt(t: np.ndarray) -> float:
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return 1.0 / 20.0
    return float(np.median(dt))


def contiguous_true_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    if mask.size == 0:
        return []
    segs: list[tuple[int, int]] = []
    in_seg = False
    start = 0
    for i, val in enumerate(mask):
        if val and not in_seg:
            start = i
            in_seg = True
        if not val and in_seg:
            segs.append((start, i - 1))
            in_seg = False
    if in_seg:
        segs.append((start, mask.size - 1))
    return segs


def segments_to_windows(segs: Iterable[tuple[int, int]], t: np.ndarray) -> list[TimeWindow]:
    windows: list[TimeWindow] = []
    for s, e in segs:
        start_s = float(t[s])
        end_s = float(t[e])
        windows.append(TimeWindow(start_s=start_s, end_s=end_s, duration_s=end_s - start_s))
    return windows


def pad_and_merge_windows(
    windows: list[TimeWindow], pad_s: float, min_t: float, max_t: float
) -> list[TimeWindow]:
    if not windows:
        return []
    padded = [
        (max(min_t, w.start_s - pad_s), min(max_t, w.end_s + pad_s)) for w in windows
    ]
    padded.sort(key=lambda x: x[0])
    merged: list[list[float]] = [[padded[0][0], padded[0][1]]]
    for s, e in padded[1:]:
        last = merged[-1]
        if s <= last[1]:
            last[1] = max(last[1], e)
        else:
            merged.append([s, e])
    return [
        TimeWindow(start_s=s, end_s=e, duration_s=e - s)
        for s, e in merged
    ]


def run_length_stats(mask: np.ndarray, dt_s: float) -> dict:
    segs = contiguous_true_segments(mask)
    lengths_frames = np.array([(e - s + 1) for s, e in segs], dtype=float)
    if lengths_frames.size == 0:
        return {
            "num_runs": 0,
            "mean_run_frames": 0.0,
            "max_run_frames": 0.0,
            "mean_run_s": 0.0,
            "max_run_s": 0.0,
        }
    return {
        "num_runs": int(lengths_frames.size),
        "mean_run_frames": float(np.mean(lengths_frames)),
        "max_run_frames": float(np.max(lengths_frames)),
        "mean_run_s": float(np.mean(lengths_frames) * dt_s),
        "max_run_s": float(np.max(lengths_frames) * dt_s),
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_hdf5).expanduser().resolve()
    vals = load_train_vals(input_path)

    n = vals.shape[0]
    t = robust_timestamps(vals[:, IDX_CUR_TIME], n=n)
    dt_s = median_dt(t)

    puck_xy = vals[:, [IDX_PUCK_X, IDX_PUCK_Y]].astype(float)
    occluded = vals[:, IDX_PUCK_OCCLUDED] > 0.5
    finite_xy = np.isfinite(puck_xy).all(axis=1)
    visible = (~occluded) & finite_xy

    x_min = -TABLE_LENGTH_M / 2.0 - args.frame_margin_m
    x_max = TABLE_LENGTH_M / 2.0 + args.frame_margin_m
    y_min = -TABLE_WIDTH_M / 2.0 - args.frame_margin_m
    y_max = TABLE_WIDTH_M / 2.0 + args.frame_margin_m

    in_frame = (
        (puck_xy[:, 0] >= x_min)
        & (puck_xy[:, 0] <= x_max)
        & (puck_xy[:, 1] >= y_min)
        & (puck_xy[:, 1] <= y_max)
    )
    in_frame_visible = visible & in_frame

    # In-frame windows.
    raw_segments = contiguous_true_segments(in_frame_visible)
    raw_windows = segments_to_windows(raw_segments, t)
    padded_windows = pad_and_merge_windows(
        raw_windows, pad_s=max(args.window_pad_s, 0.0), min_t=float(t[0]), max_t=float(t[-1])
    )

    # Jumps and speed metrics on adjacent visible frames only.
    vis_idx = np.flatnonzero(visible)
    adj_mask = np.diff(vis_idx) == 1 if vis_idx.size >= 2 else np.array([], dtype=bool)
    i0 = vis_idx[:-1][adj_mask] if vis_idx.size >= 2 else np.array([], dtype=int)
    i1 = vis_idx[1:][adj_mask] if vis_idx.size >= 2 else np.array([], dtype=int)

    if i0.size > 0:
        disp = np.linalg.norm(puck_xy[i1] - puck_xy[i0], axis=1)
        dt_pairs = np.maximum(t[i1] - t[i0], 1e-8)
        speed = disp / dt_pairs
    else:
        disp = np.array([], dtype=float)
        speed = np.array([], dtype=float)

    if disp.size > 0:
        med_d = float(np.median(disp))
        mad_d = float(np.median(np.abs(disp - med_d)))
        jump_thr = med_d + args.jump_z * max(mad_d, 1e-6)
        jump_mask = disp > jump_thr
        jump_count = int(np.sum(jump_mask))
        total_t = float(max(t[-1] - t[0], 1e-8))
        jumps_per_s = jump_count / total_t
    else:
        med_d = 0.0
        mad_d = 0.0
        jump_thr = 0.0
        jump_count = 0
        jumps_per_s = 0.0

    # Jitter: RMS displacement under low-motion conditions.
    if speed.size > 0:
        low_thr = float(np.percentile(speed, 25))
        low_motion = speed <= low_thr
        low_disp = disp[low_motion]
        jitter_rms_m = float(np.sqrt(np.mean(low_disp ** 2))) if low_disp.size > 0 else 0.0
    else:
        low_thr = 0.0
        jitter_rms_m = 0.0

    # Noise: residual RMS to a short moving-average track, computed per contiguous
    # visible segment to avoid artifacts from occlusion gaps.
    k = 5
    kernel = np.ones(k, dtype=float) / k
    noise_residual_norms: list[np.ndarray] = []
    for s, e in contiguous_true_segments(visible):
        seg_xy = puck_xy[s : e + 1]
        if seg_xy.shape[0] < k:
            continue
        smooth_x = np.convolve(seg_xy[:, 0], kernel, mode="same")
        smooth_y = np.convolve(seg_xy[:, 1], kernel, mode="same")
        resid = seg_xy - np.stack([smooth_x, smooth_y], axis=1)
        noise_residual_norms.append(np.linalg.norm(resid, axis=1))
    if noise_residual_norms:
        noise_all = np.concatenate(noise_residual_norms)
        noise_rms_m = float(np.sqrt(np.mean(noise_all ** 2)))
    else:
        noise_rms_m = 0.0

    # Occlusion stats.
    occlusion_count = int(np.sum(occluded))
    occlusion_fraction = float(occlusion_count / max(n, 1))
    occlusion_runs = run_length_stats(occluded, dt_s=dt_s)

    # In-frame ratios.
    visible_count = int(np.sum(visible))
    in_frame_visible_count = int(np.sum(in_frame_visible))
    in_frame_visible_fraction = float(in_frame_visible_count / max(visible_count, 1))

    report = {
        "input_hdf5": str(input_path),
        "num_frames": int(n),
        "duration_s": float(t[-1] - t[0]),
        "median_dt_s": dt_s,
        "bounds": {
            "table_length_m": TABLE_LENGTH_M,
            "table_width_m": TABLE_WIDTH_M,
            "frame_margin_m": float(args.frame_margin_m),
            "window_pad_s": float(max(args.window_pad_s, 0.0)),
            "in_frame_x_range_m": [x_min, x_max],
            "in_frame_y_range_m": [y_min, y_max],
        },
        "visibility": {
            "visible_count": visible_count,
            "visible_fraction": float(visible_count / max(n, 1)),
            "occluded_count": occlusion_count,
            "occluded_fraction": occlusion_fraction,
            "occlusion_runs": occlusion_runs,
        },
        "in_frame": {
            "in_frame_visible_count": in_frame_visible_count,
            "in_frame_visible_fraction": in_frame_visible_fraction,
            "raw_windows": [asdict(w) for w in raw_windows],
            "padded_windows": [asdict(w) for w in padded_windows],
        },
        "jumps": {
            "adjacent_visible_pairs": int(disp.size),
            "displacement_median_m": med_d,
            "displacement_mad_m": mad_d,
            "jump_threshold_m": jump_thr,
            "jump_count": jump_count,
            "jumps_per_s": jumps_per_s,
        },
        "motion_quality": {
            "speed_low_motion_threshold_mps": low_thr,
            "jitter_rms_m": jitter_rms_m,
            "noise_rms_m": noise_rms_m,
        },
    }

    print("\n=== Puck Trajectory Analysis ===")
    print(f"File: {input_path}")
    print(f"Frames: {n}, Duration: {report['duration_s']:.3f}s, Median dt: {dt_s:.4f}s")

    print("\n[In-frame windows]")
    print(
        f"Visible in-frame frames: {in_frame_visible_count}/{visible_count} "
        f"({100.0 * in_frame_visible_fraction:.2f}% of visible)"
    )
    print(f"Raw windows: {len(raw_windows)} | Time-padded windows: {len(padded_windows)}")
    for idx, w in enumerate(padded_windows[:8], start=1):
        print(f"  padded_{idx}: [{w.start_s:.3f}, {w.end_s:.3f}] s (dur={w.duration_s:.3f}s)")
    if len(padded_windows) > 8:
        print(f"  ... ({len(padded_windows) - 8} more)")

    print("\n[Occlusion]")
    print(f"Occluded frames: {occlusion_count}/{n} ({100.0 * occlusion_fraction:.2f}%)")
    print(
        "Occlusion runs: "
        f"{occlusion_runs['num_runs']} "
        f"(mean={occlusion_runs['mean_run_s']:.3f}s, max={occlusion_runs['max_run_s']:.3f}s)"
    )

    print("\n[Jumpiness]")
    print(
        f"Jump threshold={jump_thr:.4f}m | jumps={jump_count} "
        f"({jumps_per_s:.3f} per second)"
    )

    print("\n[Noise & Jitter]")
    print(f"Jitter RMS (low-motion displacements): {jitter_rms_m:.5f} m")
    print(f"Noise RMS (residual to 5-sample moving average): {noise_rms_m:.5f} m")

    print("\n[Metric meanings]")
    print("- In-frame windows: periods where visible puck coords are inside table bounds + margin.")
    print("- Time-padded windows: same windows widened by --window-pad-s on both sides.")
    print("- Jumps: unusually large frame-to-frame visible displacements (robust outliers).")
    print("- Occlusion fraction/runs: how often and how long puck is marked occluded.")
    print("- Jitter RMS: small short-term position fluctuations during low-speed moments.")
    print("- Noise RMS: deviation from a short smooth track; larger means noisier trajectory.")

    if args.save_json:
        out_json = input_path.with_name(f"{input_path.stem}_puck_analysis.json")
        out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved JSON report: {out_json}")


if __name__ == "__main__":
    main()

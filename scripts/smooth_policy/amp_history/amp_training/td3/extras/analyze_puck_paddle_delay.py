#!/usr/bin/env python3
"""Analyze puck-vs-paddle timing lag from split-schema real trajectory HDF5 files.

This analysis uses timestamp channels only:
- Paddle timing channel: telemetry_read_s
- Puck timing channel: puck_detection_done_s (post-homography)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from scripts.smooth_policy.visualize_demo.visualize_real_trajectory_split import (
    load_split_optional_data,
    load_split_trajectory_data,
)


def _list_hdf5_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    files = sorted(input_path.glob("trajectory_data*.hdf5"))
    if not files:
        files = sorted(input_path.glob("*.hdf5"))
    return files


def _finite_positive_dts(timestamps: np.ndarray) -> np.ndarray:
    dts = np.diff(np.asarray(timestamps, dtype=np.float64))
    return dts[np.isfinite(dts) & (dts > 1e-6)]


def _safe_median_dt_s(timestamps: np.ndarray) -> float:
    dts = _finite_positive_dts(timestamps)
    if dts.size == 0:
        return np.nan
    return float(np.median(dts))


def _nan_stats_ms(delta_s: np.ndarray) -> Dict[str, float]:
    delta_ms = np.asarray(delta_s, dtype=np.float64) * 1000.0
    valid = np.isfinite(delta_ms)
    if np.sum(valid) == 0:
        return {
            "count": 0,
            "mean_ms": np.nan,
            "std_ms": np.nan,
            "median_ms": np.nan,
            "p10_ms": np.nan,
            "p90_ms": np.nan,
            "abs_median_ms": np.nan,
        }
    vals = delta_ms[valid]
    return {
        "count": int(vals.size),
        "mean_ms": float(np.mean(vals)),
        "std_ms": float(np.std(vals)),
        "median_ms": float(np.median(vals)),
        "p10_ms": float(np.percentile(vals, 10)),
        "p90_ms": float(np.percentile(vals, 90)),
        "abs_median_ms": float(np.median(np.abs(vals))),
    }


def _nanmedian_safe(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=np.float64)
    if not np.any(np.isfinite(arr)):
        return float("nan")
    return float(np.nanmedian(arr))


def _nanpercentile_safe(arr: np.ndarray, q: float) -> float:
    arr = np.asarray(arr, dtype=np.float64)
    if not np.any(np.isfinite(arr)):
        return float("nan")
    return float(np.nanpercentile(arr, q))


def _nanmean_safe(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=np.float64)
    if not np.any(np.isfinite(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def analyze_episode(path: Path, max_lag_s: float) -> Dict[str, float]:
    del max_lag_s  # retained for backward CLI compatibility
    train_vals = load_split_trajectory_data(path)
    optional = load_split_optional_data(path)

    timestamps = np.asarray(train_vals[:, 0], dtype=np.float64)
    dt_s = _safe_median_dt_s(timestamps)
    if not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError(f"Could not infer positive sample period from {path}")
    if "timing" not in optional:
        return {
            "timesteps": int(train_vals.shape[0]),
            "dt_s_median": float(dt_s),
            "timing_channel_present": 0.0,
            "valid_timing_steps": 0.0,
            "puck_minus_paddle_median_ms": np.nan,
            "puck_minus_paddle_p10_ms": np.nan,
            "puck_minus_paddle_p90_ms": np.nan,
            "puck_minus_paddle_abs_median_ms": np.nan,
            "puck_minus_paddle_mean_ms": np.nan,
            "puck_minus_paddle_std_ms": np.nan,
            "paddle_timing_dt_median_ms": np.nan,
            "puck_timing_dt_median_ms": np.nan,
            "timing_quality_score": 0.0,
        }

    timing = np.asarray(optional["timing"], dtype=np.float64)
    if timing.ndim != 2 or timing.shape[1] < 4:
        raise ValueError(f"Invalid timing dataset shape for {path}: {timing.shape}")
    # puck_detection_done_s lives at index 3 in timing schema.
    if timing.shape[1] < 4:
        return {
            "timesteps": int(train_vals.shape[0]),
            "dt_s_median": float(dt_s),
            "timing_channel_present": 0.0,
            "valid_timing_steps": 0.0,
            "puck_minus_paddle_median_ms": np.nan,
            "puck_minus_paddle_p10_ms": np.nan,
            "puck_minus_paddle_p90_ms": np.nan,
            "puck_minus_paddle_abs_median_ms": np.nan,
            "puck_minus_paddle_mean_ms": np.nan,
            "puck_minus_paddle_std_ms": np.nan,
            "paddle_timing_dt_median_ms": np.nan,
            "puck_timing_dt_median_ms": np.nan,
            "timing_quality_score": 0.0,
        }

    paddle_ts = timing[:, 2]  # telemetry_read_s
    puck_ts = timing[:, 3]  # puck_detection_done_s (post-homography)
    valid = (
        np.isfinite(paddle_ts)
        & np.isfinite(puck_ts)
        & (paddle_ts > 0.0)
        & (puck_ts > 0.0)
    )
    delta = puck_ts - paddle_ts
    delta_valid = delta[valid]

    paddle_dt_ms = _finite_positive_dts(paddle_ts) * 1000.0
    puck_dt_ms = _finite_positive_dts(puck_ts) * 1000.0
    delta_stats = _nan_stats_ms(delta_valid)
    timing_quality = float(min(1.0, np.sum(valid) / max(1, timing.shape[0])))

    return {
        "timesteps": int(train_vals.shape[0]),
        "dt_s_median": float(dt_s),
        "timing_channel_present": 1.0,
        "valid_timing_steps": float(np.sum(valid)),
        "puck_minus_paddle_median_ms": float(delta_stats["median_ms"]),
        "puck_minus_paddle_p10_ms": float(delta_stats["p10_ms"]),
        "puck_minus_paddle_p90_ms": float(delta_stats["p90_ms"]),
        "puck_minus_paddle_abs_median_ms": float(delta_stats["abs_median_ms"]),
        "puck_minus_paddle_mean_ms": float(delta_stats["mean_ms"]),
        "puck_minus_paddle_std_ms": float(delta_stats["std_ms"]),
        "paddle_timing_dt_median_ms": (
            float(np.median(paddle_dt_ms)) if paddle_dt_ms.size > 0 else np.nan
        ),
        "puck_timing_dt_median_ms": (
            float(np.median(puck_dt_ms)) if puck_dt_ms.size > 0 else np.nan
        ),
        "timing_quality_score": timing_quality,
    }


def _write_json(path: Path, payload: Dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _write_csv(path: Path, rows: Iterable[Dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames = ["file"] + sorted([key for key in rows[0].keys() if key != "file"])
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze puck-vs-paddle lag using timing channels "
            "(telemetry_read_s vs puck_detection_done_s)."
        )
    )
    parser.add_argument("input_path", type=str, help="HDF5 file or directory")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for report outputs (defaults next to input)",
    )
    parser.add_argument(
        "--max-lag-s",
        type=float,
        default=0.5,
        help="Unused (kept for backward compatibility).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    files = _list_hdf5_files(input_path)
    if not files:
        raise FileNotFoundError(f"No .hdf5 files found under: {input_path}")

    if args.output_dir is None:
        output_dir = input_path.parent if input_path.is_file() else input_path
    else:
        output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for file_path in files:
        metrics = analyze_episode(file_path, max_lag_s=float(args.max_lag_s))
        row = {"file": str(file_path), **metrics}
        rows.append(row)
        per_file_json = output_dir / f"{file_path.stem}_delay_report.json"
        _write_json(per_file_json, row)
        print(
            f"[delay] {file_path.name}: "
            f"puck_minus_paddle_median_ms={metrics['puck_minus_paddle_median_ms']:.2f} "
            f"p10={metrics['puck_minus_paddle_p10_ms']:.2f} "
            f"p90={metrics['puck_minus_paddle_p90_ms']:.2f} "
            f"valid_timing_steps={int(metrics['valid_timing_steps'])}/{metrics['timesteps']}"
        )

    timing_median_vals = np.asarray(
        [row["puck_minus_paddle_median_ms"] for row in rows], dtype=np.float64
    )
    timing_abs_median_vals = np.asarray(
        [row["puck_minus_paddle_abs_median_ms"] for row in rows], dtype=np.float64
    )
    quality_vals = np.asarray([row["timing_quality_score"] for row in rows], dtype=np.float64)
    summary = {
        "file_count": len(rows),
        "puck_minus_paddle_median_ms_median": _nanmedian_safe(timing_median_vals),
        "puck_minus_paddle_median_ms_p90": _nanpercentile_safe(timing_median_vals, 90),
        "puck_minus_paddle_abs_median_ms_median": _nanmedian_safe(timing_abs_median_vals),
        "timing_quality_score_mean": _nanmean_safe(quality_vals),
    }
    summary_json = output_dir / "puck_paddle_delay_summary.json"
    summary_csv = output_dir / "puck_paddle_delay_summary.csv"
    _write_json(summary_json, {"summary": summary, "episodes": rows})
    _write_csv(summary_csv, rows)
    print(f"[delay] wrote summary: {summary_json}")
    print(f"[delay] wrote csv: {summary_csv}")


if __name__ == "__main__":
    main()

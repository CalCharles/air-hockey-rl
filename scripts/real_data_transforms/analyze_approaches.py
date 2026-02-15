"""Scan inflection log files and report approach interval statistics."""

import os
import sys
import json
import argparse
from collections import defaultdict

import numpy as np


DEFAULT_LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")


def load_all_logs(log_dir):
    logs = []
    for fname in sorted(os.listdir(log_dir)):
        if fname.startswith("inflection_") and fname.endswith(".json"):
            path = os.path.join(log_dir, fname)
            with open(path) as f:
                logs.append(json.load(f))
    return logs


def analyze_approaches(logs):
    trajs_with_approaches = []
    trajs_without_approaches = []

    all_approaches = []
    free_approaches = []
    walled_approaches = []

    for log in logs:
        idx = log["traj_idx"]
        approaches = log.get("approach_intervals", [])

        if approaches:
            trajs_with_approaches.append(idx)
        else:
            trajs_without_approaches.append(idx)

        for a in approaches:
            entry = {
                "traj_idx": idx,
                "duration": a["duration"],
                "y_displacement": a["y_displacement"],
                "num_wall_hits": a["num_wall_hits"],
                "estimated_contact": a["estimated_contact"],
                "peak_pos": a["peak_puck_pos"],
                "contact_pos": a["contact_puck_pos"],
                "interval": a["interval"],
            }
            all_approaches.append(entry)
            if a["num_wall_hits"] == 0:
                free_approaches.append(entry)
            else:
                walled_approaches.append(entry)

    return {
        "trajs_with_approaches": trajs_with_approaches,
        "trajs_without_approaches": trajs_without_approaches,
        "all": all_approaches,
        "free": free_approaches,
        "walled": walled_approaches,
    }


def _compute_stats(approaches):
    if not approaches:
        return {"count": 0}

    y_disps = np.array([a["y_displacement"] for a in approaches])
    durations = np.array([a["duration"] for a in approaches])
    n_estimated = sum(1 for a in approaches if a["estimated_contact"])

    return {
        "count": len(approaches),
        "y_displacement": {
            "mean": round(float(y_disps.mean()), 4),
            "std": round(float(y_disps.std()), 4),
            "min": round(float(y_disps.min()), 4),
            "max": round(float(y_disps.max()), 4),
            "median": round(float(np.median(y_disps)), 4),
        },
        "duration": {
            "mean": round(float(durations.mean()), 1),
            "std": round(float(durations.std()), 1),
            "min": int(durations.min()),
            "max": int(durations.max()),
            "median": round(float(np.median(durations)), 0),
        },
        "estimated_contacts": n_estimated,
    }


def _print_stats(stats, label):
    if stats["count"] == 0:
        print(f"\n  {label}: 0 approaches")
        return

    s = stats
    print(f"\n  {label}: {s['count']} approaches")
    yd = s["y_displacement"]
    print(f"    y-displacement:  mean={yd['mean']:.4f}  std={yd['std']:.4f}  "
          f"min={yd['min']:.4f}  max={yd['max']:.4f}  median={yd['median']:.4f}")
    d = s["duration"]
    print(f"    duration (steps): mean={d['mean']:.1f}  std={d['std']:.1f}  "
          f"min={d['min']}  max={d['max']}  median={d['median']:.0f}")
    print(f"    estimated contacts: {s['estimated_contacts']}/{s['count']}")


def print_report(result):
    n_with = len(result["trajs_with_approaches"])
    n_without = len(result["trajs_without_approaches"])
    total = n_with + n_without

    print(f"{'='*60}")
    print(f"Approach Interval Analysis  ({total} trajectories scanned)")
    print(f"{'='*60}")
    print(f"\n  Trajectories with approaches:    {n_with}")
    print(f"  Trajectories without approaches: {n_without}")

    all_stats = _compute_stats(result["all"])
    free_stats = _compute_stats(result["free"])
    walled_stats = _compute_stats(result["walled"])

    _print_stats(all_stats, "All approaches")
    _print_stats(free_stats, "Free approaches (0 wall hits)")
    _print_stats(walled_stats, "Walled approaches (1+ wall hits)")

    free_trajs = sorted(set(a["traj_idx"] for a in result["free"]))
    walled_trajs = sorted(set(a["traj_idx"] for a in result["walled"]))
    print(f"\n  Trajectories with free approaches ({len(free_trajs)}): {free_trajs}")
    print(f"  Trajectories with walled approaches ({len(walled_trajs)}): {walled_trajs}")

    if result["free"]:
        print(f"\n  --- Free approach details ---")
        by_traj = defaultdict(list)
        for a in result["free"]:
            by_traj[a["traj_idx"]].append(a)
        for idx in sorted(by_traj):
            for a in by_traj[idx]:
                iv = a["interval"]
                print(f"    traj {idx:4d}:  t={iv['start']}->{iv['end']}  "
                      f"dur={a['duration']:3d}  dy={a['y_displacement']:.4f}  "
                      f"peak=({a['peak_pos']['x']:+.3f},{a['peak_pos']['y']:+.3f})  "
                      f"contact=({a['contact_pos']['x']:+.3f},{a['contact_pos']['y']:+.3f})"
                      f"{'  [est]' if a['estimated_contact'] else ''}")

    print(f"\n{'='*60}")


def save_report(result, output_dir):
    all_stats = _compute_stats(result["all"])
    free_stats = _compute_stats(result["free"])
    walled_stats = _compute_stats(result["walled"])

    report = {
        "num_trajectories_with_approaches": len(result["trajs_with_approaches"]),
        "num_trajectories_without_approaches": len(result["trajs_without_approaches"]),
        "trajs_with_approaches": result["trajs_with_approaches"],
        "statistics": {
            "all": all_stats,
            "free": free_stats,
            "walled": walled_stats,
        },
        "free_approaches": result["free"],
        "walled_approaches": result["walled"],
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "approach_analysis.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved analysis to {out_path}")


DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "approach_analysis")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze approach intervals from inflection logs")
    parser.add_argument("--log-dir", type=str, default=DEFAULT_LOG_DIR)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    logs = load_all_logs(args.log_dir)
    if not logs:
        print(f"No inflection logs found in {args.log_dir}")
        sys.exit(1)

    result = analyze_approaches(logs)
    print_report(result)
    save_report(result, args.output_dir)

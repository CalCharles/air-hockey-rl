"""Audit saved physical trajectories without resetting or stepping a simulator."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def audit(results: Path, output: Path):
    rows = []
    for name, scenario in json.loads(results.read_text())["scenarios"].items():
        params = scenario["config"]["simulator_params"]
        threshold = params["paddle_radius"] + params["puck_radius"] + .02
        for episode in scenario["episodes"]:
            with np.load(results.parent / episode["trajectory"]) as saved:
                puck, paddle = saved["true_puck_xy"], saved["true_paddle_xy"]
            expected = (episode["steps"], 2)
            if puck.shape != expected or paddle.shape != expected:
                raise ValueError(f"Body trajectory shape mismatch: {name}, {episode['seed']}")
            if not np.isfinite(puck).all() or not np.isfinite(paddle).all():
                raise ValueError("Body positions must be finite")
            separation = np.linalg.norm(puck - paddle, axis=1)
            rows.append(dict(scenario=name, seed=episode["seed"], steps=episode["steps"],
                             reached_horizon=int(episode["reached_horizon"]),
                             physical_juggles=episode["true_state_juggles"],
                             puck_x_span_m=float(np.ptp(puck[:, 0])),
                             puck_y_span_m=float(np.ptp(puck[:, 1])),
                             puck_path_length_m=float(np.linalg.norm(np.diff(puck, axis=0), axis=1).sum()),
                             separated_fraction=float(np.mean(separation > threshold)),
                             separation_threshold_m=threshold))
    if not rows:
        raise ValueError("No episodes to audit")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    caps = [row for row in rows if row["reached_horizon"]]
    summary = dict(episodes=len(rows), capped_episodes=len(caps))
    if caps:
        for field in ("puck_x_span_m", "puck_y_span_m", "puck_path_length_m", "separated_fraction"):
            summary["minimum_cap_" + field] = min(row[field] for row in caps)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit(args.results, args.out)

"""Summarize saved noisy-juggling evaluations without running environments.

Example: python -m scripts.analysis.juggle_progress --roots runs/td3/noisy_* --out docs/juggling-noisy-progress.md
Only roots with saved training evaluation results are included. Trajectories and
policies are never modified. Final-start results remain separate from validation.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def collect(root):
    rows = []
    for result in (root / "train").glob("checkpoint_*/survival_eval/results.json"):
        payload = json.loads(result.read_text())
        if "training" not in payload["scenarios"]:
            continue
        scenario = payload["scenarios"]["training"]
        summary = scenario["summary"]
        rows.append(dict(
            run=root.name,
            step=int(result.parents[1].name.split("_")[-1]),
            mean_steps=summary["restricted_mean_steps"],
            mean_juggles=summary["mean_juggles"],
            mean_true_state_juggles=summary.get("mean_true_state_juggles"),
            cap_fraction=summary["sustained_juggling_fraction"],
            episodes=payload["episodes_per_scenario"],
            seed_base=payload["seed_base"],
            horizon=payload["horizon"],
            source=str(result),
        ))
    return sorted(rows, key=lambda row: row["step"])


def write_report(roots, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    series = [(root, collect(root)) for root in roots if root.is_dir()]
    figure = out.with_suffix(".png")
    table = out.with_suffix(".csv")
    rows = [row for _, values in series for row in values]
    if rows:
        with table.open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for root, values in series:
        if not values:
            continue
        steps = [v["step"] / 1e6 for v in values]
        label = root.name.replace("noisy_iteration2_20261029_", "iteration 2: ")
        axes[0].plot(steps, [v["mean_steps"] for v in values], marker=".", label=label)
        axes[1].plot(steps, [v["mean_juggles"] for v in values], marker=".", label=label)
    for ax, ylabel in zip(axes, ["Mean survival (steps)", "Mean observed juggles"]):
        ax.set(xlabel="Training steps (millions)", ylabel=ylabel)
        ax.grid(alpha=.25)
    if rows:
        axes[0].legend(fontsize=7)
    fig.suptitle("Canonical noisy simulator: checkpoint validation (one seed per recipe)")
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    link = lambda path: os.path.relpath(path, out.parent)
    lines = ["# Noisy juggling progress", "",
             "Generated " + datetime.now(timezone.utc).isoformat() + " from saved evaluations; no new rollouts.", "",
             "Validation curves use repeatedly evaluated starts, not independent acceptance tests. "
             "Each point is an episode mean for one training seed. Check the CSV for episode counts, seeds and caps. "
             "Observed juggle counts use noisy camera positions and a direction-reversal filter. "
             "Contact timesteps are not distinct juggles.", "",
             f"![Checkpoint validation]({link(figure)})", "",
             f"[Raw validation table]({link(table)})" if rows else "No saved validation results yet.", "",
             "| Run | Runner status | Latest validation step | Mean survival | Mean juggles | Genuine cap fraction |",
             "|---|---|---:|---:|---:|---:|"]
    for root, values in series:
        status_file = root / "status.json"
        status = json.loads(status_file.read_text()).get("status", "unknown") if status_file.exists() else "unknown"
        if values:
            v = values[-1]
            lines.append(f"| {root.name} | {status} | {v['step']:,} | {v['mean_steps']:.1f} | {v['mean_juggles']:.2f} | {v['cap_fraction']:.1%} |")
        else:
            lines.append(f"| {root.name} | {status} | No saved validation | — | — | — |")
    lines += ["", "Separate-start evaluations (selection and benchmark phases labeled):", ""]
    for root, _ in series:
        for directory, phase in (("final_eval", "final actor benchmark"),
                                 ("selected_eval", "validation-selected checkpoint benchmark"),
                                 ("selected_paired_eval", "paired selected checkpoint"),
                                 ("final_paired_eval", "paired final actor")):
            result = root / directory / "results.json"
            if result.exists():
                payload = json.loads(result.read_text())
                s = payload["scenarios"]["training"]["summary"]
                true_count = (f", {s['mean_true_state_juggles']:.2f} physical-trajectory juggles"
                              if "mean_true_state_juggles" in s else "")
                lines.append(f"- [{root.name}, {phase}]({link(result)}): {s['restricted_mean_steps']:.1f} mean steps, "
                             f"{s['mean_juggles']:.2f} observed juggles{true_count}, {s['sustained_juggling_fraction']:.1%} cap completion; "
                             f"{payload['episodes_per_scenario']} starts from seed {payload['seed_base']}, cap {payload['horizon']}.")
    lines += ["", "The final-start benchmark is reused across recipes. A selected policy requires untouched seeds "
              "and longer horizons before acceptance. Bounded survival does not prove infinite survival."]
    out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    cli = parser.parse_args()
    write_report(cli.roots, cli.out)

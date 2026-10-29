"""Leaderboard over every evaluation written by ``scripts.ceiling.eval_target``.

Run with ``python -m scripts.ceiling.leaderboard [--protocol selection]``.
Per policy: mean ± std err over its episodes. Per label (a recipe, usually one
policy per training seed): mean ± std err over the policies' means, so the
error bar reflects training-seed variation. Writes ``runs/ceiling/evals/<protocol>/leaderboard.md``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.ceiling.eval_target import EVAL_ROOT


def mean_se(values):
    values = np.asarray(values, dtype=float)
    se = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else float("nan")
    return values.mean(), se


def fmt(values, digits=1):
    m, se = mean_se(values)
    return f"{m:.{digits}f}" if np.isnan(se) else f"{m:.{digits}f} ± {se:.{digits}f}"


def policy_row(results_path):
    data = json.loads(results_path.read_text())
    target = data["scenarios"]["target"]
    rows, summary = target["episodes"], target["summary"]
    steps = [r["steps"] for r in rows]
    return {
        "policy": results_path.parent.name,
        "run_dir": data["run_dir"],
        "steps": steps,
        "return_per_step": [r["return"] / r["steps"] for r in rows],
        "juggles": [r["juggles"] for r in rows],
        "reached_cap": [r["reached_horizon"] for r in rows],
        "survival_at": summary["survival_at_steps"],
    }


def build(protocol):
    root = EVAL_ROOT / protocol
    labels = {}
    for results in sorted(root.glob("*/*/results.json")):
        labels.setdefault(results.parent.parent.name, []).append(policy_row(results))
    if not labels:
        raise SystemExit(f"No evaluations under {root}")
    horizon = json.loads(next(root.glob("*/protocol.json")).read_text())["horizon"]
    marks = [t for t in ("250", "1000", "2000", "10000") if int(t) <= horizon]

    def recipe_key(item):
        return -np.mean([np.mean(p["steps"]) for p in item[1]])

    lines = [f"# Ceiling search leaderboard ({protocol}, cap {horizon} steps)", "",
             "Recipes: mean ± std err over policies (training seeds) of each policy's episode mean.", "",
             "| Recipe | Policies | Mean steps | Median steps | Reached cap | Juggles | Reward / step | "
             + " | ".join(f"Alive @{t}" for t in marks) + " |",
             "|---|---:|---:|---:|---:|---:|---:|" + "---:|" * len(marks)]
    for label, policies in sorted(labels.items(), key=recipe_key):
        lines.append(
            f"| {label} | {len(policies)} | {fmt([np.mean(p['steps']) for p in policies])} | "
            f"{fmt([np.median(p['steps']) for p in policies], 0)} | "
            f"{fmt([100 * np.mean(p['reached_cap']) for p in policies])} % | "
            f"{fmt([np.mean(p['juggles']) for p in policies])} | "
            f"{fmt([np.mean(p['return_per_step']) for p in policies], 3)} | "
            + " | ".join(fmt([100 * p['survival_at'][t] for p in policies]) + " %" for t in marks) + " |")

    lines += ["", "## Individual policies", "",
              "Mean ± std err over episodes.", "",
              "| Recipe | Policy | Mean steps | Median | Reached cap | Juggles |", "|---|---|---:|---:|---:|---:|"]
    for label, policies in sorted(labels.items(), key=recipe_key):
        for p in sorted(policies, key=lambda p: -np.mean(p["steps"])):
            lines.append(f"| {label} | {p['policy']} | {fmt(p['steps'])} | {np.median(p['steps']):.0f} | "
                         f"{100 * np.mean(p['reached_cap']):.1f} % | {fmt(p['juggles'])} |")
    out = root / "leaderboard.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="selection")
    build(parser.parse_args().protocol)


if __name__ == "__main__":
    main()

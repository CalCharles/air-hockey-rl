#!/usr/bin/env python3
"""
analyze_direct_force_vs_baseline.py

Compares the `td3_baseline_direct_force_1250_exp_on_ID_seed_*` run against
the `runs/td3/baseline` seeds.

For each seed folder in both groups:
  1. Find the best checkpoint (by multi_env_eval.json aggregate
     mean_return_across_envs) — same selection logic as
     analyze_ood_oracle_step_sweep / analyze_transformer_context_sweep.
  2. Read that checkpoint's mean_return_across_envs and
     mean_success_across_envs.

Then aggregate across seeds within each group and plot a single grouped
bar chart (2 bars: "direct_force_1250_exp_on_ID" vs "baseline") with
±stderr error bars.

No re-evaluation is performed — this only reads existing
multi_env_eval.json files already on disk.

Run from the repo root, e.g.:

    python -m scripts.td3.analysis.analyze_direct_force \
        --runs-dir runs/td3 \
        --target-prefix td3_baseline_direct_force_1250_exp_on_ID_seed \
        --baseline-dir runs/td3/baseline
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Best checkpoint per seed folder (same logic as the step-sweep script)
# ---------------------------------------------------------------------------

@dataclass
class CheckpointResult:
    seed_folder: str
    checkpoint_dir: str
    checkpoint_step: int
    mean_return_across_envs: float
    mean_success_across_envs: float | None


def _read_multi_env_eval(checkpoint_dir: str) -> dict | None:
    path = os.path.join(checkpoint_dir, "multi_env_eval.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [WARN] could not parse {path}: {e}")
        return None
    aggregate = data.get("aggregate")
    if not isinstance(aggregate, dict) or "mean_return_across_envs" not in aggregate:
        print(
            f"  [WARN] {path} missing aggregate.mean_return_across_envs "
            f"(top-level keys: {list(data.keys())}); skipping."
        )
        return None
    return aggregate


def find_best_checkpoint_per_seed(seed_folder: str) -> CheckpointResult | None:
    """Return the CheckpointResult with the highest mean_return_across_envs
    across all checkpoint_* subdirs of seed_folder. Returns None if no
    usable checkpoint is found.
    """
    if not os.path.isdir(seed_folder):
        return None

    candidates: list[CheckpointResult] = []

    for entry in sorted(os.listdir(seed_folder)):
        if not entry.startswith("checkpoint_"):
            continue
        ckpt_dir = os.path.join(seed_folder, entry)
        if not os.path.isdir(ckpt_dir):
            continue

        aggregate = _read_multi_env_eval(ckpt_dir)
        if aggregate is None:
            continue

        step_match = re.match(r"checkpoint_(\d+)$", entry)
        step = int(step_match.group(1)) if step_match else -1

        candidates.append(CheckpointResult(
            seed_folder=seed_folder,
            checkpoint_dir=ckpt_dir,
            checkpoint_step=step,
            mean_return_across_envs=float(aggregate["mean_return_across_envs"]),
            mean_success_across_envs=(
                float(aggregate["mean_success_across_envs"])
                if "mean_success_across_envs" in aggregate else None
            ),
        ))

    # ------------------------------------------------------------------
    # DEBUG CHECKPOINT: per-seed candidate selection
    #
    #   print(f"[BEST-CKPT] seed_folder={seed_folder}")
    #   for c in candidates:
    #       print(f"    step={c.checkpoint_step:>8}  "
    #             f"mean_return={c.mean_return_across_envs:8.2f}")
    #   breakpoint()
    # ------------------------------------------------------------------

    if not candidates:
        return None
    # return max(candidates, key=lambda c: c.mean_return_across_envs)
    return max(candidates, key=lambda c: c.checkpoint_step)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_matching_seed_folders(runs_dir: str, prefix: str) -> list[str]:
    """Find direct children of runs_dir whose name starts with `prefix`,
    e.g. td3_baseline_direct_force_1250_exp_on_ID_seed_41, _42, ...
    """
    if not os.path.isdir(runs_dir):
        raise FileNotFoundError(f"runs_dir does not exist: {runs_dir}")

    folders = []
    for entry in sorted(os.listdir(runs_dir)):
        full = os.path.join(runs_dir, entry)
        if os.path.isdir(full) and entry.startswith(prefix):
            folders.append(full)

    # ------------------------------------------------------------------
    # DEBUG CHECKPOINT: discovered target seed folders
    #
    #   print(f"[DISCOVER] prefix={prefix}")
    #   for f in folders:
    #       print(f"    {f}")
    #   breakpoint()
    # ------------------------------------------------------------------

    return folders


def discover_baseline_seed_folders(baseline_dir: str) -> list[str]:
    """baseline_dir's direct subdirectories are treated as seed folders."""
    if not os.path.isdir(baseline_dir):
        raise FileNotFoundError(f"baseline_dir does not exist: {baseline_dir}")
    return [
        os.path.join(baseline_dir, entry)
        for entry in sorted(os.listdir(baseline_dir))
        if os.path.isdir(os.path.join(baseline_dir, entry))
    ]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def stderr_of_mean(values: list[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(n))


def aggregate_group(results: list[CheckpointResult]) -> dict:
    returns = [r.mean_return_across_envs for r in results]
    successes = [r.mean_success_across_envs for r in results
                 if r.mean_success_across_envs is not None]
    agg = {
        "mean_return": float(np.mean(returns)),
        "stderr_return": stderr_of_mean(returns),
        "n_seeds": len(results),
        "per_seed": [
            {
                "seed_folder": os.path.basename(r.seed_folder),
                "checkpoint_step": r.checkpoint_step,
                "mean_return_across_envs": r.mean_return_across_envs,
                "mean_success_across_envs": r.mean_success_across_envs,
            }
            for r in results
        ],
    }
    if successes:
        agg["mean_success_rate"] = float(np.mean(successes))
        agg["stderr_success_rate"] = stderr_of_mean(successes)
    return agg


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(group_aggregates: dict[str, dict]) -> None:
    print(f"\n{'='*90}")
    print(f"{'baseline_no_exploration vs baseline':^90}")
    print(f"{'='*90}")
    header = f"{'Group':<32} | {'MeanReturn':>11} ± {'StdErr':<7} | {'n_seeds':>7}"
    print(header)
    print("-" * 90)
    for name, agg in group_aggregates.items():
        print(f"{name:<32} | {agg['mean_return']:>11.2f} ± {agg['stderr_return']:<7.2f} "
              f"| {agg['n_seeds']:>7}")
        for p in agg["per_seed"]:
            print(f"    {p['seed_folder']:<28} checkpoint_{p['checkpoint_step']:<8} "
                  f"mean_return={p['mean_return_across_envs']:.2f}")
    print("-" * 90)


def plot_bar(group_aggregates: dict[str, dict], out_dir: str) -> None:
    names = list(group_aggregates.keys())
    means = [group_aggregates[n]["mean_return"] for n in names]
    errs = [group_aggregates[n]["stderr_return"] for n in names]
    colors = ["#2196F3", "#2b2b2b"]

    fig, ax = plt.subplots(figsize=(6, 5.5))
    x = np.arange(len(names))
    bars = ax.bar(x, means, width=0.5, yerr=errs, capsize=5,
                   color=colors[:len(names)], alpha=0.88)

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Mean Return ± StdErr", fontsize=10)
    ax.set_title("direct_force_1250_exp_on_ID vs baseline — ID Mean Return", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "direct_force_vs_baseline.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved figure -> {fig_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyze_direct_force_vs_baseline(
    runs_dir: str = "runs/td3",
    target_prefix: str = "td3_baseline_direct_force_1250_exp_on_ID_seed",
    baseline_dir: str = "runs/td3/baseline",
    out_dir: str = "results/direct_force_vs_baseline",
) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    # --- discover seed folders ---
    target_seed_folders = discover_matching_seed_folders(runs_dir, target_prefix)
    if not target_seed_folders:
        raise RuntimeError(f"No folders matching prefix '{target_prefix}' found under {runs_dir}")
    print(f"Found {len(target_seed_folders)} target seed folder(s):")
    for f in target_seed_folders:
        print(f"  {f}")

    baseline_seed_folders = discover_baseline_seed_folders(baseline_dir)
    if not baseline_seed_folders:
        raise RuntimeError(f"No seed folders found under {baseline_dir}")
    print(f"\nFound {len(baseline_seed_folders)} baseline seed folder(s):")
    for f in baseline_seed_folders:
        print(f"  {f}")

    # --- best checkpoint per seed ---
    target_results = []
    for sf in target_seed_folders:
        best = find_best_checkpoint_per_seed(sf)
        if best is None:
            print(f"  [SKIP] {sf}: no usable checkpoints")
            continue
        target_results.append(best)

    baseline_results = []
    for sf in baseline_seed_folders:
        best = find_best_checkpoint_per_seed(sf)
        if best is None:
            print(f"  [SKIP] {sf}: no usable checkpoints")
            continue
        baseline_results.append(best)

    if not target_results:
        raise RuntimeError("No usable checkpoints found for target group.")
    if not baseline_results:
        raise RuntimeError("No usable checkpoints found for baseline group.")

    # --- aggregate ---
    group_aggregates = {
        "direct_force_1250_exp_on_ID": aggregate_group(target_results),
        "baseline": aggregate_group(baseline_results),
    }

    # --- save + report ---
    summary_path = os.path.join(out_dir, "direct_force_vs_baseline_summary.json")
    with open(summary_path, "w") as f:
        json.dump(group_aggregates, f, indent=2)
    print(f"\nSaved results -> {summary_path}")

    print_table(group_aggregates)
    plot_bar(group_aggregates, out_dir)

    return group_aggregates


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-dir", default="runs/td3",
                   help="Root directory containing the target seed folders.")
    p.add_argument("--target-prefix", default="td3_baseline_direct_force_1250_exp_on_ID_seed",
                   help="Folder name prefix to match under runs-dir.")
    p.add_argument("--baseline-dir", default="runs/td3/baseline",
                   help="Directory containing baseline seed subfolders.")
    p.add_argument("--out-dir", default="results/direct_force_vs_baseline")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    analyze_direct_force_vs_baseline(
        runs_dir=args.runs_dir,
        target_prefix=args.target_prefix,
        baseline_dir=args.baseline_dir,
        out_dir=args.out_dir,
    )
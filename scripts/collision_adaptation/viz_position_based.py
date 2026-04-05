"""
Visualize position-based collision adaptation results.

Usage:
    python scripts/collision_adaptation/viz_position_based.py \
        --runs runs/collision_adaptation_position_based runs/collision_adaptation_position_based_extreme \
        --labels "oracle=[0.7,1.0,1.2]" "oracle=[0.5,1.0,2.0]" \
        --output-dir runs/collision_adaptation_viz
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

TIER_COLORS = {"low": "#4c9be8", "mid": "#f5a623", "high": "#e85c5c"}
TIER_NAMES = ("low", "mid", "high")


def load_history(run_dir: str) -> list[dict]:
    path = os.path.join(run_dir, "adaptation_history.json")
    with open(path) as f:
        return json.load(f)


def extract_scale_series(history: list[dict]) -> dict[str, list[float]]:
    """Extract learner scales over iterations (using scale_after)."""
    series: dict[str, list[float]] = {t: [] for t in TIER_NAMES}
    for entry in history:
        for i, tier in enumerate(TIER_NAMES):
            series[tier].append(entry["learner_scales_after"][i])
    return series


def extract_convergence_series(history: list[dict]) -> list[float]:
    return [e["convergence_max_ratio_minus_one"] for e in history]


def extract_count_series(history: list[dict], sim: str = "oracle") -> dict[str, list[int]]:
    """Extract per-tier collision counts for oracle or learner across iterations."""
    series: dict[str, list[int]] = {t: [] for t in TIER_NAMES}
    for entry in history:
        stats = entry[f"{sim}_stats"]["paddle"]
        for tier in TIER_NAMES:
            series[tier].append(stats[tier]["count"])
    return series


def extract_speed_series(history: list[dict], sim: str, speed: str = "out") -> dict[str, list[float]]:
    """Extract mean_speed_in or mean_speed_out per tier per iteration."""
    key = f"mean_speed_{speed}"
    series: dict[str, list[float]] = {t: [] for t in TIER_NAMES}
    for entry in history:
        stats = entry[f"{sim}_stats"]["paddle"]
        for tier in TIER_NAMES:
            series[tier].append(stats[tier].get(key, 0.0))
    return series


def extract_ratio_series(history: list[dict]) -> dict[str, list[float | None]]:
    """Extract oracle/learner speed_out ratio per tier (None if skipped)."""
    series: dict[str, list[float | None]] = {t: [] for t in TIER_NAMES}
    for entry in history:
        info = entry["update_info"]
        for tier in TIER_NAMES:
            if info[tier].get("skipped", True):
                series[tier].append(None)
            else:
                series[tier].append(info[tier]["ratio"])
    return series


def _oracle_targets(history: list[dict]) -> dict[str, float]:
    """Infer oracle scales from scale trajectory (scale converges toward oracle)."""
    # Read from final entry's update_info ratio × scale_after if not skipped.
    # Better: just read oracle_scales from the first iteration's update_info if available.
    # We'll compute oracle target = initial learner scale * product of all ratio steps,
    # but that's complex. Instead: infer from oracle mean_speed_out / learner initial mean_speed_out.
    targets = {}
    first = history[0]
    oracle_stats = first["oracle_stats"]["paddle"]
    learner_stats = first["learner_stats"]["paddle"]
    for tier in TIER_NAMES:
        o = oracle_stats[tier]["mean_speed_out"]
        l = learner_stats[tier]["mean_speed_out"]
        if l > 1e-6:
            targets[tier] = o / l  # effective ratio target (≈ oracle_scale / 1.0 initial)
        else:
            targets[tier] = None
    return targets


def plot_run(
    history: list[dict],
    label: str,
    out_dir: str,
    run_tag: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    iters = list(range(1, len(history) + 1))

    scales = extract_scale_series(history)
    convergence = extract_convergence_series(history)
    oracle_counts = extract_count_series(history, "oracle")
    learner_counts = extract_count_series(history, "learner")
    oracle_out = extract_speed_series(history, "oracle", "out")
    learner_out = extract_speed_series(history, "learner", "out")
    ratios = extract_ratio_series(history)

    # ---- Figure 1: Scale trajectories + convergence -------------------------
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"Position-based adaptation — {label}", fontsize=13)

    ax = axes[0]
    for tier in TIER_NAMES:
        ax.plot(iters, scales[tier], color=TIER_COLORS[tier], lw=2, label=tier)
    ax.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax.set_ylabel("Learner paddle scale")
    ax.legend(loc="upper right")
    ax.set_title("Learner paddle restitution scales over iterations")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(iters, convergence, color="black", lw=2)
    ax.set_ylabel("max|ratio − 1|")
    ax.set_xlabel("Iteration")
    ax.set_title("Convergence metric (max|oracle_out/learner_out − 1| across tiers)")
    ax.axhline(0.0, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{run_tag}_scales_convergence.png"), dpi=150)
    plt.close(fig)

    # ---- Figure 2: Per-tier mean_speed_out oracle vs learner ----------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    fig.suptitle(f"Per-tier mean outgoing speed — {label}", fontsize=13)

    for ax, tier in zip(axes, TIER_NAMES):
        ax.plot(iters, oracle_out[tier], color=TIER_COLORS[tier], lw=2, label="oracle", ls="--")
        ax.plot(iters, learner_out[tier], color=TIER_COLORS[tier], lw=2, label="learner", alpha=0.7)
        ax.set_title(f"{tier} tier")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Mean speed out (m/s)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{run_tag}_speed_out.png"), dpi=150)
    plt.close(fig)

    # ---- Figure 3: Collision counts per tier --------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    fig.suptitle(f"Per-tier collision counts — {label}", fontsize=13)

    for ax, tier in zip(axes, TIER_NAMES):
        ax.plot(iters, oracle_counts[tier], color=TIER_COLORS[tier], lw=2, label="oracle", ls="--")
        ax.plot(iters, learner_counts[tier], color=TIER_COLORS[tier], lw=2, label="learner", alpha=0.7)
        ax.set_title(f"{tier} tier")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Collision count")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{run_tag}_counts.png"), dpi=150)
    plt.close(fig)

    # ---- Figure 4: Speed ratio oracle/learner per tier ----------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    fig.suptitle(f"Speed ratio (oracle_out / learner_out) — {label}", fontsize=13)

    for ax, tier in zip(axes, TIER_NAMES):
        xs = [iters[i] for i, r in enumerate(ratios[tier]) if r is not None]
        ys = [r for r in ratios[tier] if r is not None]
        ax.plot(xs, ys, color=TIER_COLORS[tier], lw=2, marker="o", ms=4)
        ax.axhline(1.0, color="gray", lw=1.0, ls="--")
        ax.set_title(f"{tier} tier")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Ratio")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{run_tag}_ratios.png"), dpi=150)
    plt.close(fig)

    print(f"Saved 4 plots to {out_dir}/ [{run_tag}_*.png]")


def print_summary(history: list[dict], label: str) -> None:
    print(f"\n{'='*60}")
    print(f"Summary: {label}")
    print(f"{'='*60}")
    n = len(history)
    final = history[-1]
    scales_after = final["learner_scales_after"]
    print(f"Iterations: {n}")
    print(f"Final learner scales: low={scales_after[0]:.4f}  mid={scales_after[1]:.4f}  high={scales_after[2]:.4f}")
    print(f"Final max|ratio-1|: {final['convergence_max_ratio_minus_one']:.4f}")

    print("\nPer-tier stats (final iteration):")
    for tier in TIER_NAMES:
        info = final["update_info"][tier]
        if info.get("skipped"):
            print(f"  {tier}: SKIPPED — {info['reason']}")
        else:
            print(
                f"  {tier}: oracle_out={info['oracle_mean_out']:.3f}  learner_out={info['learner_mean_out']:.3f}"
                f"  ratio={info['ratio']:.4f}  scale={info['scale_after']:.4f}"
            )

    print("\nCollision counts (oracle / learner) across all iters [mean ± std]:")
    for tier in TIER_NAMES:
        oc = [e["oracle_stats"]["paddle"][tier]["count"] for e in history]
        lc = [e["learner_stats"]["paddle"][tier]["count"] for e in history]
        print(
            f"  {tier}: oracle {np.mean(oc):.1f}±{np.std(oc):.1f}  "
            f"learner {np.mean(lc):.1f}±{np.std(lc):.1f}"
        )


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--output-dir", default="runs/collision_adaptation_viz")
    return p.parse_args()


def main():
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    labels = args.labels or [os.path.basename(r) for r in args.runs]

    for run_dir, label in zip(args.runs, labels):
        history = load_history(run_dir)
        run_tag = os.path.basename(run_dir)
        print_summary(history, label)
        plot_run(history, label, args.output_dir, run_tag)


if __name__ == "__main__":
    main()

"""Aggregate per-checkpoint deterministic eval metrics for the paddle50 sim2sim
campaign.

Reads `eval_combined_ckpt_<step>/metrics.json` files under each run dir,
prints a compact comparative table + per-run trajectory shape.

Usage:
    .venv/bin/python notes/scratch/aggregate_paddle50_results.py [run_root]

Default run_root: runs/td3/sim2sim/hist2_motion0_to_paddle50/

Each direct subdir is treated as a "variant" (e.g. residual_v1_canonical/),
and the seed0/ subdir of that holds the eval_combined_ckpt_*/ dirs.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys


ZERO_SHOT = 67.54  # paddle -50% mass-preserved, n=50 seed=0
RUN_ROOT_DEFAULT = "runs/td3/sim2sim/hist2_motion0_to_paddle50"


def load_runs(root: str):
    out = []
    for variant_dir in sorted(glob.glob(os.path.join(root, "*"))):
        name = os.path.basename(variant_dir)
        if not os.path.isdir(variant_dir):
            continue
        # Discover seeds under the variant dir.
        for seed_dir in sorted(glob.glob(os.path.join(variant_dir, "seed*"))):
            seed = os.path.basename(seed_dir)
            ckpt_metrics = []
            for f in sorted(
                glob.glob(os.path.join(seed_dir, "eval_combined_ckpt_*", "metrics.json")),
                key=lambda p: int(re.search(r"ckpt_([0-9]+)", p).group(1)),
            ):
                step = int(re.search(r"ckpt_([0-9]+)", f).group(1))
                d = json.load(open(f))
                ckpt_metrics.append((step, d["mean_return"], d.get("tail10", float("nan"))))
            final_path = os.path.join(seed_dir, "eval_combined_final", "metrics.json")
            final_mean = None
            if os.path.exists(final_path):
                final_mean = json.load(open(final_path))["mean_return"]
            if ckpt_metrics:
                out.append((f"{name}/{seed}", ckpt_metrics, final_mean))
    return out


def summarize(name, ckpts, final):
    means = [m for _, m, _ in ckpts]
    peak = max(ckpts, key=lambda t: t[1])
    above = sum(1 for _, m, _ in ckpts if m > ZERO_SHOT)
    last_5 = ckpts[-5:]
    last5_mean = sum(m for _, m, _ in last_5) / max(1, len(last_5))
    print(f"\n=== {name} ===")
    print(
        f"n={len(ckpts)} | peak={peak[1]:.2f} @ {peak[0]} | "
        f"final={final if final is None else f'{final:.2f}'} | "
        f"mean(all)={sum(means)/len(means):.2f} | "
        f"last5_mean={last5_mean:.2f} | drift={peak[1]-last5_mean:+.2f} | "
        f">zs={above}/{len(ckpts)}"
    )
    print("step | mean | tail10")
    for step, m, t in ckpts:
        flag = " <-- PEAK" if step == peak[0] else ""
        marker = ">" if m > ZERO_SHOT else " "
        t_str = f"{t:>6.2f}" if t == t else "    nan"
        print(f"  {marker} {step:>6} | {m:>6.2f} | {t_str}{flag}")


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else RUN_ROOT_DEFAULT
    print("# paddle50 sim2sim — per-checkpoint deterministic eval (n=50, seed=0)")
    print(f"# zero-shot reference: {ZERO_SHOT}  (sim2sim_combined.yaml = paddle -50% mass-preserved)")
    print(f"# run root: {root}")
    print()

    runs = load_runs(root)

    if not runs:
        print("(no eval_combined_ckpt_* metrics found)")
        return

    print("=== Compact table ===")
    print(
        "| run | n | peak | @step | final | mean(all) | last5_mean | >zs | drift(peak-last5) |"
    )
    print(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for name, ckpts, final in runs:
        means = [m for _, m, _ in ckpts]
        peak = max(ckpts, key=lambda t: t[1])
        above = sum(1 for _, m, _ in ckpts if m > ZERO_SHOT)
        last5 = ckpts[-5:]
        last5_mean = sum(m for _, m, _ in last5) / max(1, len(last5))
        final_s = "?" if final is None else f"{final:.1f}"
        print(
            f"| {name} | {len(ckpts)} | {peak[1]:.1f} | {peak[0]} | {final_s} | "
            f"{sum(means)/len(means):.1f} | {last5_mean:.1f} | "
            f"{above}/{len(ckpts)} | {peak[1]-last5_mean:+.1f} |"
        )
    print()

    for name, ckpts, final in runs:
        summarize(name, ckpts, final)


if __name__ == "__main__":
    main()

"""One-off analysis: simplified-exploration ablation vs canonical 4-primitive DR.

Parses per-checkpoint multi_env_eval.json across the 4 expl_compare runs,
produces a trajectory plot and prints back-half / peak / per-env stats.
"""

from __future__ import annotations

import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "runs/td3/zeroshot_paramrand/expl_compare"
ARMS = {
    "baseline": ["baseline_seed0", "baseline_seed1"],
    "simple": ["simple_seed0", "simple_seed1"],
}
BACK_HALF_MIN = 1_500_000


def load_run(run_dir: str):
    """Return (steps, mean_return, per_env_returns array) sorted by step."""
    rows = []
    for path in glob.glob(os.path.join(run_dir, "checkpoint_*", "multi_env_eval.json")):
        m = re.search(r"checkpoint_(\d+)", path)
        if not m:
            continue
        step = int(m.group(1))
        with open(path) as f:
            data = json.load(f)
        agg = data["aggregate"]
        rows.append(
            (
                step,
                agg["mean_return_across_envs"],
                agg["mean_success_across_envs"],
                agg["per_env_mean_return"],
            )
        )
    rows.sort(key=lambda r: r[0])
    steps = np.array([r[0] for r in rows])
    mret = np.array([r[1] for r in rows])
    msucc = np.array([r[2] for r in rows])
    per_env = np.array([r[3] for r in rows])  # (n_ckpt, 5)
    return steps, mret, msucc, per_env


def main():
    runs = {}
    for arm, run_names in ARMS.items():
        for rn in run_names:
            runs[rn] = load_run(os.path.join(ROOT, rn))

    # ---- trajectory plot ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    colors = {"baseline": "tab:blue", "simple": "tab:red"}
    for arm, run_names in ARMS.items():
        for i, rn in enumerate(run_names):
            steps, mret, msucc, _ = runs[rn]
            ax1.plot(
                steps / 1e6,
                mret,
                color=colors[arm],
                alpha=0.7,
                lw=1.3,
                ls="-" if i == 0 else "--",
                label=rn,
            )
            ax2.plot(steps / 1e6, msucc, color=colors[arm], alpha=0.7, lw=1.3,
                     ls="-" if i == 0 else "--", label=rn)
    # arm-mean (interp to baseline_seed0 grid)
    grid = runs["baseline_seed0"][0]
    for arm, run_names in ARMS.items():
        stacked = []
        for rn in run_names:
            steps, mret, _, _ = runs[rn]
            stacked.append(np.interp(grid, steps, mret))
        arm_mean = np.mean(stacked, axis=0)
        ax1.plot(grid / 1e6, arm_mean, color=colors[arm], lw=3.0, alpha=0.95,
                 label=f"{arm} (seed-mean)")
    ax1.axvspan(BACK_HALF_MIN / 1e6, 2.0, color="gray", alpha=0.08)
    ax1.set_xlabel("training step (M)")
    ax1.set_ylabel("multi-env mean_return (5 envs x 4 eps)")
    ax1.set_title("Exploration ablation: mean_return trajectory")
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(alpha=0.3)
    ax2.axvspan(BACK_HALF_MIN / 1e6, 2.0, color="gray", alpha=0.08)
    ax2.set_xlabel("training step (M)")
    ax2.set_ylabel("multi-env mean_success")
    ax2.set_title("Success rate trajectory")
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(alpha=0.3)
    out_png = os.path.join(ROOT, "expl_compare_trajectory.png")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"[saved] {out_png}")

    # ---- back-half + peak stats ----
    def back_half(rn):
        steps, mret, msucc, per_env = runs[rn]
        mask = steps >= BACK_HALF_MIN
        return mret[mask], msucc[mask], per_env[mask], steps, mret

    print("\n=== Per-run summary ===")
    print(f"{'run':<16}{'back-half mean':>16}{'back-half succ':>16}{'peak (step)':>22}{'final ckpt':>12}")
    arm_backhalf = {a: [] for a in ARMS}
    arm_backhalf_perenv = {a: [] for a in ARMS}
    for arm, run_names in ARMS.items():
        for rn in run_names:
            bh_ret, bh_succ, bh_perenv, steps, mret = back_half(rn)
            peak_i = int(np.argmax(mret))
            arm_backhalf[arm].append(bh_ret.mean())
            arm_backhalf_perenv[arm].append(bh_perenv.mean(axis=0))
            print(f"{rn:<16}{bh_ret.mean():>10.1f}±{bh_ret.std():<4.0f}"
                  f"{bh_succ.mean():>16.2f}"
                  f"{mret[peak_i]:>14.1f} ({steps[peak_i]/1e6:.2f}M)"
                  f"{mret[-1]:>12.1f}")

    print("\n=== Arm-level (back-half 1.5M-2M, n=2 seeds) ===")
    for arm in ARMS:
        vals = np.array(arm_backhalf[arm])
        se = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else float("nan")
        print(f"{arm:<10} back-half mean_return = {vals.mean():.1f}  "
              f"(seeds: {', '.join(f'{v:.1f}' for v in vals)}; SE={se:.1f})")

    b = np.array(arm_backhalf["baseline"])
    s = np.array(arm_backhalf["simple"])
    diff = s.mean() - b.mean()
    pooled_se = np.sqrt(b.std(ddof=1)**2 / 2 + s.std(ddof=1)**2 / 2)
    print(f"\nsimple - baseline = {diff:+.1f}  (pooled SE ~ {pooled_se:.1f})")

    print("\n=== Per-env back-half mean (averaged over 2 seeds) ===")
    print(f"{'arm':<10}" + "".join(f"{'env'+str(i):>10}" for i in range(5)))
    for arm in ARMS:
        pe = np.mean(arm_backhalf_perenv[arm], axis=0)
        print(f"{arm:<10}" + "".join(f"{v:>10.1f}" for v in pe))
    base_pe = np.mean(arm_backhalf_perenv["baseline"], axis=0)
    simp_pe = np.mean(arm_backhalf_perenv["simple"], axis=0)
    print(f"{'diff':<10}" + "".join(f"{v:>10.1f}" for v in (simp_pe - base_pe)))


if __name__ == "__main__":
    main()

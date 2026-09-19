"""Collate RMA baseline results into markdown tables.

    .venv/bin/python -m scripts.rma.summarize \
        --phase1-dirs runs/rma/juggle_dr_phase1_seed0 runs/rma/juggle_dr_phase1_seed1 \
        --baseline-dirs runs/td3/tasks_20260904/dr/juggle_dr \
        --phase2-dirs runs/rma/juggle_dr_phase1_seed0/phase2 ... \
        --out runs/rma/summary.md

Tables:
  A. phase-1 learning curve — privileged RMA policy vs the plain-DR TD3 baseline on the
     fixed 5-env eval set (per-checkpoint ``multi_env_eval.json``), plus the back-half mean;
  B. phase-2 latent fit per iteration (validation MSE / R^2, parameter-probe R^2, on-policy
     MSE before the update, rollout return);
  C. final paired policy evaluation (adapted / privileged / nominal / td3_dr on ID and OOD envs).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional

import numpy as np


def _load_json(path: str) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def checkpoint_curve(run_dir: str) -> Dict[int, float]:
    """step -> mean return across eval envs, from every checkpoint_*/multi_env_eval.json."""
    curve: Dict[int, float] = {}
    for path in glob.glob(os.path.join(run_dir, "checkpoint_*", "multi_env_eval.json")):
        step_str = os.path.basename(os.path.dirname(path)).split("_")[-1]
        data = _load_json(path)
        if data is None or not step_str.isdigit():
            continue
        curve[int(step_str)] = float(data["aggregate"]["mean_return_across_envs"])
    # The final (post-training) eval lives at the run-dir level; file it at the
    # run's total_timesteps so the curve ends on the final policy.
    final = _load_json(os.path.join(run_dir, "multi_env_eval.json"))
    if final is not None:
        total = None
        try:
            import yaml

            with open(os.path.join(run_dir, "args.yaml")) as f:
                total = int(yaml.safe_load(f).get("total_timesteps"))
        except Exception:
            total = (max(curve) if curve else 0) + 1
        curve[int(total)] = float(final["aggregate"]["mean_return_across_envs"])
    return dict(sorted(curve.items()))


def _fmt(x) -> str:
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.1f}"


def table_phase1(phase1_dirs: List[str], baseline_dirs: List[str], every: int, total: int) -> str:
    def _prefix(d: str) -> str:
        meta = _load_json(os.path.join(d, "rma_meta.json")) or {}
        return "history" if meta.get("mode") == "history" else "rma_priv"

    runs = [(f"{_prefix(d)}:{os.path.basename(d.rstrip('/'))}", checkpoint_curve(d)) for d in phase1_dirs]
    runs += [(f"td3_dr:{os.path.basename(d.rstrip('/'))}", checkpoint_curve(d)) for d in baseline_dirs]
    if not runs:
        return "_no phase-1 / baseline runs_\n"
    steps = [s for s in range(every, total + 1, every)]
    header = "| Run | " + " | ".join(f"{s // 1000}k" for s in steps) + " | back-half mean | max |"
    sep = "|---|" + "---:|" * (len(steps) + 2)
    lines = [header, sep]
    for name, curve in runs:
        cells = [_fmt(curve.get(s)) for s in steps]
        back = [v for s, v in curve.items() if s > total // 2 and s <= total]
        lines.append(
            f"| {name} | " + " | ".join(cells) + f" | {_fmt(float(np.mean(back)) if back else None)} | "
            f"{_fmt(max(curve.values()) if curve else None)} |"
        )
    lines.append("")
    lines.append(
        "Mean return of the privileged RMA policy (rma_priv) / the long-history control (history) / the plain-DR actor (td3_dr) on the fixed 5-env eval set (eval_param_seed 12345, "
        f"4 episodes per env per checkpoint; the {total // 1000}k column is the final policy); "
        f"back-half = mean over all 25k-checkpoints in ({total // 2 // 1000}k, {total // 1000}k]."
    )
    return "\n".join(lines) + "\n"


def table_phase2_fit(phase2_dir: str) -> str:
    rows = []
    path = os.path.join(phase2_dir, "phase2_metrics.jsonl")
    try:
        with open(path) as f:
            rows = [json.loads(ln) for ln in f if ln.strip()]
    except OSError:
        return f"_no phase2_metrics.jsonl in {phase2_dir}_\n"
    lines = [
        f"**{phase2_dir}**",
        "",
        "| iter | rollouts | env steps (train+val) | rollout return | on-policy MSE (pre-update) | val MSE | val R² | probe R² (e from ẑ) | val MSE t<10 | t≥50 | ID return (adapted) | OOD return |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['iteration']} | {r.get('rollout_policy', '')} | {int(r.get('fit/train_steps', 0))}+{int(r.get('fit/val_steps', 0))} | "
            f"{_fmt(r.get('collect/mean_return'))} | {r.get('collect/online_latent_mse', float('nan')):.4f} | "
            f"{r.get('fit/val_mse', float('nan')):.4f} | {r.get('fit/val_r2_mean', float('nan')):.3f} | "
            f"{r.get('probe/val_r2_mean', float('nan')):.3f} | {r.get('fit/val_mse_t0_10', float('nan')):.4f} | "
            f"{r.get('fit/val_mse_t50_end', float('nan')):.4f} | {_fmt(r.get('eval/id_adapted_return'))} | {_fmt(r.get('eval/ood_adapted_return'))} |"
        )
    last = rows[-1] if rows else {}
    if last:
        lines.append("")
        lines.append(
            f"Final per-parameter probe R² (validation): "
            + ", ".join(f"{k.split('probe/val_r2_')[-1]} {v:.3f}" for k, v in last.items() if k.startswith("probe/val_r2_") and k not in ("probe/val_r2_mean", "probe/val_r2_mean_from_true_z"))
            + f"; from the true z: {last.get('probe/val_r2_mean_from_true_z', float('nan')):.3f}. "
            f"Validation target variance (mean over latent dims): {last.get('fit/val_target_var', float('nan')):.4f}."
        )
    return "\n".join(lines) + "\n"


def table_phase2_policy(phase2_dir: str) -> str:
    summary = _load_json(os.path.join(phase2_dir, "phase2_summary.json"))
    if summary is None:
        return f"_no phase2_summary.json in {phase2_dir}_\n"
    lines = [f"**{phase2_dir}** (final eval, {summary.get('total_env_steps', 0)} phase-2 env steps, {summary.get('wall_s', 0) / 60:.0f} min)", ""]
    lines.append("| split | agent | mean return ± SEM | success | ep length | latent MSE |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for split, agents in summary.get("policy_eval", {}).items():
        for name, v in agents.items():
            lines.append(
                f"| {split} | {name} | {v['mean_return']:.1f} ± {v['return_sem']:.1f} | {v['mean_success']:.2f} | {v['mean_ep_length']:.0f} | "
                f"{v['latent_mse']:.4f} |" if "latent_mse" in v else
                f"| {split} | {name} | {v['mean_return']:.1f} ± {v['return_sem']:.1f} | {v['mean_success']:.2f} | {v['mean_ep_length']:.0f} | — |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase1-dirs", nargs="*", default=[])
    ap.add_argument("--baseline-dirs", nargs="*", default=[])
    ap.add_argument("--phase2-dirs", nargs="*", default=[])
    ap.add_argument("--every", type=int, default=250000, help="checkpoint stride for the curve table")
    ap.add_argument("--total", type=int, default=2000000)
    ap.add_argument("--out", default=None)
    cli = ap.parse_args()

    parts = ["## A. Phase 1 — privileged RMA policy vs plain-DR TD3 (fixed eval envs)", "", table_phase1(cli.phase1_dirs, cli.baseline_dirs, cli.every, cli.total)]
    if cli.phase2_dirs:
        parts += ["## B. Phase 2 — adaptation-module fit", ""]
        parts += [table_phase2_fit(d) for d in cli.phase2_dirs]
        parts += ["## C. Final paired policy evaluation", ""]
        parts += [table_phase2_policy(d) for d in cli.phase2_dirs]
    text = "\n".join(parts)
    print(text)
    if cli.out:
        os.makedirs(os.path.dirname(os.path.abspath(cli.out)), exist_ok=True)
        with open(cli.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()

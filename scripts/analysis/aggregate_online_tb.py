"""Aggregate sim2sim online fine-tuning TensorBoard logs over seeds: mean ± std err per env.

Reads <online_dir>/env*/hist*_seed*/online_tb and writes one TensorBoard run per env
(plus `all_envs`, the per-seed average over envs). Each scalar is written as the seed
mean under its original tag, plus `<tag>/lower` and `<tag>/upper` (mean ∓ SE) for the
Custom Scalars margin charts (shaded ±SE band).

One online dir -> runs `env*` in <online_dir>_agg/. Several (e.g. an actor-LR sweep)
-> runs `<online dir name>/env*` in --out (default <parent>/online_agg/), so the
sweeps sit side by side in one TensorBoard. A dir that holds a single env (e.g.
the data-recycling trials of `sim2sim_online_finetune --recycle-data`) becomes
one run named after the dir (`trial0`, `trial1`, ...), no `all_envs`: one
mean ± SE curve per trial.

    python scripts/analysis/aggregate_online_tb.py runs/td3/sim2sim_step6_direct_force/online_recycle_alpha0.5/trial*

    python scripts/analysis/aggregate_online_tb.py runs/td3/sim2sim_step6_direct_force/online
    python scripts/analysis/aggregate_online_tb.py runs/td3/sim2sim_step6_direct_force/online_policy_lr*
    tensorboard --logdir runs/td3/sim2sim_step6_direct_force/online_agg
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
from collections import defaultdict

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

# (custom-scalars category, chart title, tag) — the charts shown in the Custom Scalars tab
CHARTS = [
    ("return", "online episode return (w/ exploration noise)", "online_finetune/return"),
    ("return", "deterministic eval return (every 10 eps)", "online_finetune_eval/return"),
    ("return", "deterministic eval juggles", "online_finetune_eval/juggles"),
    ("losses", "critic loss (round mean)", "online_finetune/critic_loss"),
    ("losses", "actor loss (round mean)", "online_finetune/actor_loss"),
    ("losses", "Q1 mean", "online_finetune_loss/q1_mean"),
    ("critic error", "critic error mean  (Q_φ − Q_MC)", "online_finetune_critic_error/error_mean"),
    ("critic error", "critic error median", "online_finetune_critic_error/error_median"),
    ("critic error", "critic |error| mean", "online_finetune_critic_error/error_abs_mean"),
    ("critic error", "critic error std (within episode)", "online_finetune_critic_error/error_std"),
    ("critic error", "Q_φ mean (h-inverted)", "online_finetune_critic_error/q_mean"),
    ("critic error", "Q_MC mean (discounted reward-to-go)", "online_finetune_critic_error/mc_return_mean"),
    ("episode", "episode juggles", "online_finetune_episode/juggles"),
    ("episode", "episode length", "online_finetune_episode/length"),
    ("recycle", "α (online share of each batch; data-recycling trials ≥ 1)", "online_finetune_recycle/alpha"),
]


def load_scalars(tb_dir: str) -> dict[str, dict[int, float]]:
    ea = EventAccumulator(tb_dir, size_guidance={"scalars": 0})
    ea.Reload()
    return {t: {e.step: e.value for e in ea.Scalars(t)} for t in ea.Tags()["scalars"]}


def write_run(out_dir: str, per_seed: list[dict[str, dict[int, float]]], layout: dict) -> None:
    w = SummaryWriter(out_dir)
    w.add_custom_scalars(layout)
    for tag in sorted(set.intersection(*(set(s) for s in per_seed))):
        steps = sorted(set.intersection(*(set(s[tag]) for s in per_seed)))
        for step in steps:
            v = np.array([s[tag][step] for s in per_seed])
            m = v.mean()
            se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
            w.add_scalar(tag, m, step)
            w.add_scalar(f"{tag}/lower", m - se, step)
            w.add_scalar(f"{tag}/upper", m + se, step)
    w.close()


def aggregate(online_dir: str, out_root: str, layout: dict) -> None:
    by_env: dict[str, list] = defaultdict(list)
    for tb in sorted(glob.glob(os.path.join(online_dir, "env*", "*seed*", "online_tb"))):
        env = os.path.basename(os.path.dirname(os.path.dirname(tb)))
        by_env[env].append(load_scalars(tb))
    if not by_env:
        print(f"[skip] no */*seed*/online_tb under {online_dir}")
        return

    if len(by_env) == 1:
        # One env (e.g. the data-recycling trials): one run per online dir, named after it
        # (trial0, trial1, ...); all_envs would only duplicate it.
        (env, seeds), = by_env.items()
        write_run(out_root, seeds, layout)
        print(f"{online_dir}: {env} {len(seeds)} seeds -> {out_root}")
        return

    for env, seeds in sorted(by_env.items(), key=lambda kv: int(re.sub(r"\D", "", kv[0]) or 0)):
        write_run(os.path.join(out_root, env), seeds, layout)
        print(f"{online_dir}: {env} {len(seeds)} seeds")

    # all_envs: per seed index, average each scalar over envs, then mean ± SE over seeds
    n_seeds = min(len(s) for s in by_env.values())
    all_seeds = []
    for i in range(n_seeds):
        runs = [seeds[i] for seeds in by_env.values()]
        tags = set.intersection(*(set(r) for r in runs))
        avg = {}
        for tag in tags:
            steps = set.intersection(*(set(r[tag]) for r in runs))
            avg[tag] = {st: float(np.mean([r[tag][st] for r in runs])) for st in steps}
        all_seeds.append(avg)
    write_run(os.path.join(out_root, "all_envs"), all_seeds, layout)
    print(f"{online_dir}: all_envs {n_seeds} seeds x {len(by_env)} envs")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("online_dirs", nargs="+", help="e.g. runs/td3/sim2sim_step6_direct_force/online_policy_lr*")
    ap.add_argument("--out", default=None,
                    help="output logdir (default: <online_dir>_agg for one dir, <parent>/online_agg for several)")
    cli = ap.parse_args()
    online_dirs = [d.rstrip("/") for d in cli.online_dirs if os.path.isdir(d)]
    if not online_dirs:
        raise SystemExit("no online dirs found")
    single = len(online_dirs) == 1
    out_root = cli.out or (online_dirs[0] + "_agg" if single
                           else os.path.join(os.path.dirname(online_dirs[0]), "online_agg"))

    layout: dict = defaultdict(dict)
    for cat, title, tag in CHARTS:
        layout[cat][title] = ["Margin", [tag, f"{tag}/lower", f"{tag}/upper"]]
    layout = dict(layout)

    if os.path.isdir(out_root):
        shutil.rmtree(out_root)
    for d in online_dirs:
        aggregate(d, out_root if single else os.path.join(out_root, os.path.basename(d)), layout)
    print(f"tensorboard --logdir {out_root}")


if __name__ == "__main__":
    main()

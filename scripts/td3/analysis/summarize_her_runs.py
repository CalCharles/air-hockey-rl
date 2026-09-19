"""Summarise HER runs: per-checkpoint eval success and the training success curve.

    python -m scripts.td3.analysis.summarize_her_runs runs/her/round1_20260910 [more roots...]

For every run directory under each root (one that has ``args.yaml``) prints a
row per checkpoint (``checkpoint_<step>/goal_eval.json``: success rate,
contacts, end reasons) plus the final ``goal_eval.json``, and the rolling
training success (``charts/avg_success_rate``) at matching steps.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np


def _tb_success(run_dir):
    try:
        from tensorboard.backend.event_processing import event_accumulator
        ea = event_accumulator.EventAccumulator(run_dir, size_guidance={"scalars": 0})
        ea.Reload()
        if "charts/avg_success_rate" not in ea.Tags()["scalars"]:
            return None
        ev = ea.Scalars("charts/avg_success_rate")
        return np.array([e.step for e in ev]), np.array([e.value for e in ev])
    except Exception:
        return None


def summarise(run_dir: str) -> None:
    ckpts = sorted(glob.glob(os.path.join(run_dir, "checkpoint_*", "goal_eval.json")),
                   key=lambda p: int(p.split("checkpoint_")[1].split("/")[0]))
    tb = _tb_success(run_dir)
    print(f"\n== {run_dir}")
    print(f"{'step':>8} | {'eval succ':>9} | {'contacts':>8} | {'len':>5} | {'train succ':>10} | ends")
    rows = []
    for p in ckpts:
        step = int(p.split("checkpoint_")[1].split("/")[0])
        d = json.load(open(p))
        train = float("nan")
        if tb is not None:
            m = (tb[0] >= step - 25000) & (tb[0] <= step)
            if m.any():
                train = float(np.mean(tb[1][m]))
        rows.append((step, d["success_rate"], d["mean_contacts"], d["mean_length"], train, d["end_reasons"]))
    for step, s, c, l, t, e in rows:
        print(f"{step:>8} | {s:>9.2f} | {c:>8.2f} | {l:>5.0f} | {t:>10.2f} | {e}")
    final = os.path.join(run_dir, "goal_eval.json")
    if os.path.exists(final):
        d = json.load(open(final))
        print(f"{'final':>8} | {d['success_rate']:>9.2f} | {d['mean_contacts']:>8.2f} | {d['mean_length']:>5.0f} | {'':>10} | {d['end_reasons']} (n={d['n_eps']})")
    if rows:
        succ = np.array([r[1] for r in rows])
        print(f"best ckpt eval succ {succ.max():.2f} @ {rows[int(succ.argmax())][0]}; last-4 mean {succ[-4:].mean():.2f}")


def main(roots):
    for root in roots:
        runs = sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d) and os.path.exists(os.path.join(d, "args.yaml")))
        if not runs and os.path.exists(os.path.join(root, "args.yaml")):
            runs = [root]
        for r in runs:
            summarise(r)


if __name__ == "__main__":
    main(sys.argv[1:])

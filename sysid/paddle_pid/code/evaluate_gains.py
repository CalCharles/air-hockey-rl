#!/usr/bin/env python3
"""Score one or more PID gain sets on a paddle-motion session (same replay + metric as the fit).

    python sysid/paddle_pid/code/evaluate_gains.py --input-dir <session> \
        --gains canonical 9000 0 50 --gains fitted 12000 300 80 [--paddle-density 3000] [--split-seed 0]

Prints train / val / all mean per-step position error per gain set and writes ``evaluations.json``
plus ``per_trial.csv`` (and optional trajectory plots) to ``--out``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sysid.paddle_pid.code.dataset import load_session, split_train_val, session_attrs
from sysid.paddle_pid.code.replay import (DEFAULT_BASE_CONFIG, PaddleReplayer, PlantParams, build_replay_sim_config,
                                         evaluate_trials, load_base_config)
from sysid.paddle_pid.code import report
from sysid.common import link_data


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", required=True, nargs="+", help="one or more session directories")
    p.add_argument("--out", default=None)
    p.add_argument("--base-config", default=str(DEFAULT_BASE_CONFIG))
    p.add_argument("--gains", nargs=4, action="append", metavar=("LABEL", "KP", "KI", "KD"), required=True)
    p.add_argument("--paddle-density", type=float, default=None)
    p.add_argument("--hist-len", type=int, default=None)
    p.add_argument("--action-delay-steps", type=int, default=0)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--val-repeat", type=int, default=None)
    p.add_argument("--plots", action="store_true", help="write per-trial trajectory plots of the val trials")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    trials = load_session(a.input_dir)
    train, val, split_info = split_train_val(trials, seed=a.split_seed, val_repeat=a.val_repeat)
    base = load_base_config(a.base_config)
    sim_cfg = build_replay_sim_config(base, session_attrs(trials), hist_len=a.hist_len)
    density = a.paddle_density if a.paddle_density is not None else float(base["air_hockey"]["simulator_params"]["paddle_density"])
    rep = PaddleReplayer(sim_cfg)
    out = Path(a.out) if a.out else _REPO_ROOT / "sysid/paddle_pid/results" / ("eval_" + "+".join(Path(d).name for d in a.input_dir))
    for d in a.input_dir:
        link_data("paddle_pid", Path(d), out)
    out.mkdir(parents=True, exist_ok=True)
    evals, traj = {}, {}
    print(f"{'gains':<14} {'kp':>8} {'ki':>8} {'kd':>7} | {'train mm':>9} {'val mm':>9} {'all mm':>9}")
    for label, kp, ki, kd in a.gains:
        p = PlantParams(float(kp), float(ki), float(kd), density)
        e_tr = evaluate_trials(rep, train, p, a.action_delay_steps)
        e_va = evaluate_trials(rep, val, p, a.action_delay_steps, keep_trajectories=True)
        traj[label] = e_va.pop("trajectories")
        e_all = evaluate_trials(rep, trials, p, a.action_delay_steps)
        evals[f"{label}_train"], evals[f"{label}_val"], evals[f"{label}_all"] = e_tr, e_va, e_all
        print(f"{label:<14} {p.kp:8.0f} {p.ki:8.0f} {p.kd:7.1f} | {e_tr['mean_pos_err_mm']:9.2f} {e_va['mean_pos_err_mm']:9.2f} {e_all['mean_pos_err_mm']:9.2f}")
    with open(out / "evaluations.json", "w") as fh:
        json.dump({"split": split_info, "paddle_density": density, "evaluations": evals}, fh, indent=1)
    report.write_per_trial_csv(evals, out / "per_trial.csv")
    if a.plots:
        keys = list(traj)
        sets = {"baseline": traj[keys[0]]}
        if len(keys) > 1:
            sets["fitted"] = traj[keys[1]]
        if len(keys) > 2:
            sets["extra"] = traj[keys[2]]
        report.plot_trajectories(val, sets, out / "val_trajectories.png", extra_label=keys[2] if len(keys) > 2 else None)
        report.plot_step_errors(val, sets, out / "val_step_error_profile.png")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

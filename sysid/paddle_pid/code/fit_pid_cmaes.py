#!/usr/bin/env python3
"""Fit the paddle PID gains (kp, ki, kd; mass fixed) to a scripted paddle-motion session with CMA-ES.

    python sysid/paddle_pid/code/fit_pid_cmaes.py \
        --input-dir /data2/air_hockey/vertical_horizontal_diagonal_arc_paddle_motion_20260909_2024 \
        --out sysid/paddle_pid/results/cmaes_20260909
    # several sessions pooled (paddle_motion lines / arcs + reversal_jerk):
    python sysid/paddle_pid/code/fit_pid_cmaes.py --input-dir <session A> <session B> --out …

Pipeline: load trials → hold out one trial per condition → evaluate the canonical gains (x0) →
CMA-ES on the training trials (objective: mean per-step position error, mm) → evaluate the
best gains on validation → percentile validation of the search → plots + summary.md.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sysid.paddle_pid.code.dataset import load_session, split_train_val, session_attrs, sessions_of
from sysid.paddle_pid.code.replay import (DEFAULT_BASE_CONFIG, PaddleReplayer, PlantParams, build_replay_sim_config,
                                         evaluate_trials, load_base_config)
from sysid.paddle_pid.code.cmaes_fit import CandidateEvaluator, GainBounds, run_cmaes
from sysid.paddle_pid.code import report
from sysid.common.fit_validation import percentile_report
from sysid.common import link_data


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", required=True, nargs="+", help="session directory / directories with traj_* or jerk_* trial files")
    p.add_argument("--out", default=None, help="output directory (default sysid/paddle_pid/results/cmaes_<session>)")
    p.add_argument("--base-config", default=str(DEFAULT_BASE_CONFIG), help="sim YAML providing plant + x0 gains")
    p.add_argument("--split-seed", type=int, default=0, help="seed for the per-condition held-out trial")
    p.add_argument("--val-repeat", type=int, default=None, help="hold out this repeat everywhere instead of a random one")
    p.add_argument("--paddle-density", type=float, default=None, help="fixed paddle density (default: base config)")
    p.add_argument("--hist-len", type=int, default=None, help="sim PID-target smoothing window (default: the recording's)")
    p.add_argument("--action-delay-steps", type=int, default=0, help="diagnostic: delay the replayed actions by n steps")
    p.add_argument("--x0", type=float, nargs=3, metavar=("KP", "KI", "KD"), default=None, help="start gains (default: base config)")
    p.add_argument("--kp-range", type=float, nargs=2, default=(500.0, 1.0e5))
    p.add_argument("--ki-max", type=float, default=1.0e5)
    p.add_argument("--kd-max", type=float, default=5.0e3)
    p.add_argument("--sigma0", type=float, default=0.3, help="initial CMA-ES step in the unit cube")
    p.add_argument("--popsize", type=int, default=16)
    p.add_argument("--max-iter", type=int, default=60)
    p.add_argument("--restarts", type=int, default=1)
    p.add_argument("--seed", type=int, default=0, help="CMA-ES seed")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    a = parse_args(argv)
    t_start = time.time()
    in_dirs = [Path(d) for d in a.input_dir]
    in_dir = in_dirs[0]
    out = Path(a.out) if a.out else _REPO_ROOT / "sysid/paddle_pid/results" / ("cmaes_" + "+".join(d.name for d in in_dirs))
    for d in in_dirs:
        link_data("paddle_pid", d, out)
    plots = out / "plots"
    out.mkdir(parents=True, exist_ok=True)
    plots.mkdir(exist_ok=True)

    trials = load_session(in_dirs)
    train, val, split_info = split_train_val(trials, seed=a.split_seed, val_repeat=a.val_repeat)
    attrs = session_attrs(trials)
    sessions = sessions_of(trials)
    split_info["sessions"] = {s: {"n_trials": sum(t.session == s for t in trials),
                                  "n_train": sum(t.session == s for t in train),
                                  "n_val": sum(t.session == s for t in val)} for s in sessions}
    print(f"{len(trials)} trials from {len(sessions)} session(s), {len(split_info['conditions'])} conditions → "
          f"{len(train)} train / {len(val)} val; recording hist_len={attrs['hist_len']} move_lims={attrs['move_lims']} "
          f"mean dt={np.mean([t.dt for t in trials]):.4f} s")
    for s_name, cnt in split_info["sessions"].items():
        print(f"  {s_name}: {cnt['n_trials']} trials ({cnt['n_train']} train / {cnt['n_val']} val)")
    with open(out / "split.json", "w") as fh:
        json.dump(split_info, fh, indent=1)

    base = load_base_config(a.base_config)
    sim_cfg = build_replay_sim_config(base, attrs, hist_len=a.hist_len)
    sp = base["air_hockey"]["simulator_params"]
    density = float(a.paddle_density if a.paddle_density is not None else sp["paddle_density"])
    x0 = PlantParams(*(a.x0 if a.x0 else (float(sp["pid_kp"]), float(sp.get("pid_ki", 0.0)), float(sp["pid_kd"]))), density)
    with open(out / "sim_config_replay.yaml", "w") as fh:
        yaml.safe_dump({"air_hockey": sim_cfg}, fh, sort_keys=False)

    rep = PaddleReplayer(sim_cfg)
    print(f"sim dt {rep.dt:.4f} s, hist_len {rep.sim.hist_len}, paddle density {density:.0f} → mass "
          f"{density * np.pi * rep.sim.paddle_radius ** 2:.2f} kg; x0 = kp {x0.kp:.0f} ki {x0.ki:.0f} kd {x0.kd:.1f}")
    evals = {"baseline_train": evaluate_trials(rep, train, x0, a.action_delay_steps, keep_trajectories=True),
             "baseline_val": evaluate_trials(rep, val, x0, a.action_delay_steps, keep_trajectories=True)}
    print(f"canonical gains: train {evals['baseline_train']['mean_pos_err_mm']:.2f} mm, val {evals['baseline_val']['mean_pos_err_mm']:.2f} mm")

    bounds = GainBounds(kp=tuple(a.kp_range), ki_max=a.ki_max, kd_max=a.kd_max)
    evaluator = CandidateEvaluator(sim_cfg, train, val, density, a.action_delay_steps, workers=a.workers)
    try:
        res = run_cmaes(evaluator, x0, bounds=bounds, sigma0=a.sigma0, popsize=a.popsize, max_iter=a.max_iter,
                        seed=a.seed, restarts=a.restarts)
    finally:
        evaluator.close()
    fitted = PlantParams(res.best.kp, res.best.ki, res.best.kd, density)
    evals["fitted_train"] = evaluate_trials(rep, train, fitted, a.action_delay_steps, keep_trajectories=True)
    evals["fitted_val"] = evaluate_trials(rep, val, fitted, a.action_delay_steps, keep_trajectories=True)
    print(f"CMA-ES best: kp {fitted.kp:.1f} ki {fitted.ki:.1f} kd {fitted.kd:.2f} → train "
          f"{evals['fitted_train']['mean_pos_err_mm']:.2f} mm, val {evals['fitted_val']['mean_pos_err_mm']:.2f} mm "
          f"({res.n_evaluations} candidates, {res.wall_seconds:.0f} s)")
    per_session = {}
    if len(sessions) > 1:
        for s_name in sessions:
            s_train = [t for t in train if t.session == s_name]
            s_val = [t for t in val if t.session == s_name]
            per_session[s_name] = {
                "n_train": len(s_train), "n_val": len(s_val),
                "canonical_train": evaluate_trials(rep, s_train, x0, a.action_delay_steps)["mean_pos_err_mm"] if s_train else None,
                "canonical_val": evaluate_trials(rep, s_val, x0, a.action_delay_steps)["mean_pos_err_mm"] if s_val else None,
                "fitted_train": evaluate_trials(rep, s_train, fitted, a.action_delay_steps)["mean_pos_err_mm"] if s_train else None,
                "fitted_val": evaluate_trials(rep, s_val, fitted, a.action_delay_steps)["mean_pos_err_mm"] if s_val else None}
            print(f"  {s_name}: canonical val {per_session[s_name]['canonical_val']} mm → fitted val {per_session[s_name]['fitted_val']} mm")

    # -- validation of the search: where does the selected point sit among all candidates on val?
    pr = None
    if val:
        vals = np.array([c.val_err_mm for c in res.candidates])
        trs = np.array([c.train_err_mm for c in res.candidates])
        pr = percentile_report("paddle per-step position error", "mm", vals, evals["fitted_val"]["mean_pos_err_mm"],
                               lambda i: {"kp": res.candidates[i].kp, "ki": res.candidates[i].ki, "kd": res.candidates[i].kd},
                               canonical_err=evals["baseline_val"]["mean_pos_err_mm"], train_errors=trs,
                               selected_train_err=evals["fitted_train"]["mean_pos_err_mm"],
                               canonical_train_err=evals["baseline_train"]["mean_pos_err_mm"])

    # -- outputs
    fit_json = res.to_json()
    fit_json.update({"paddle_density": density, "action_delay_steps": a.action_delay_steps, "split_seed": a.split_seed,
                     "input_dir": str(in_dir), "input_dirs": [str(d) for d in in_dirs], "sessions": sessions,
                     "per_session_err_mm": per_session,
                     "base_config": str(a.base_config), "hist_len_sim": int(rep.sim.hist_len),
                     "dt_sim": rep.dt, "percentile_validation": pr.to_json() if pr else None,
                     "train_err_mm": {"canonical": evals["baseline_train"]["mean_pos_err_mm"], "fitted": evals["fitted_train"]["mean_pos_err_mm"]},
                     "val_err_mm": {"canonical": evals["baseline_val"]["mean_pos_err_mm"], "fitted": evals["fitted_val"]["mean_pos_err_mm"]}})
    with open(out / "fit_result.json", "w") as fh:
        json.dump(fit_json, fh, indent=1)
    report.write_candidates_csv(res.candidates, out / "candidates.csv")
    traj = {k: v.pop("trajectories") for k, v in evals.items()}
    with open(out / "evaluations.json", "w") as fh:
        json.dump(evals, fh, indent=1)
    report.write_per_trial_csv(evals, out / "per_trial.csv")
    fitted_cfg = copy.deepcopy(base)
    fsp = fitted_cfg["air_hockey"]["simulator_params"]
    fsp["pid_kp"], fsp["pid_ki"], fsp["pid_kd"], fsp["paddle_density"] = round(fitted.kp, 1), round(fitted.ki, 1), round(fitted.kd, 2), density
    with open(out / "sim_config_fitted.yaml", "w") as fh:
        fh.write(f"# Paddle PID gains fitted by sysid/paddle_pid/code/fit_pid_cmaes.py on {' + '.join(d.name for d in in_dirs)}\n"
                 f"# val per-step error {evals['baseline_val']['mean_pos_err_mm']:.2f} mm (canonical) -> "
                 f"{evals['fitted_val']['mean_pos_err_mm']:.2f} mm (fitted); mass fixed (paddle_density {density:.0f}).\n")
        yaml.safe_dump(fitted_cfg, fh, sort_keys=False)

    if not a.no_plots:
        report.plot_convergence(res.generation_best, plots / "convergence.png")
        report.plot_candidates(res.candidates, fitted, x0, plots / "candidates.png")
        report.plot_per_condition(evals, plots / "per_condition.png")
        vsets = {"baseline": traj["baseline_val"], "fitted": traj["fitted_val"]}
        report.plot_trajectories(val, vsets, plots / "val_trajectories.png")
        report.plot_step_errors(val, vsets, plots / "val_step_error_profile.png")

    ctx = {"dataset": " + ".join(str(d) for d in in_dirs), "split_info": split_info, "baseline": x0, "fitted": fitted,
           "evals": evals, "cma": fit_json, "percentile": pr, "per_session": per_session, "sim": {"base_config": a.base_config, "dt": rep.dt, "hist_len": int(rep.sim.hist_len),
                                     "paddle_density": density, "paddle_mass": rep.paddle_mass,
                                     "action_delay_steps": a.action_delay_steps}}
    with open(out / "summary.md", "w") as fh:
        fh.write(report.summary_markdown(ctx))
    print(f"wrote {out} ({time.time() - t_start:.0f} s)")
    return fit_json


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fit the paddle–puck restitution and the paddle / puck mass ratio to a scripted collision session with CMA-ES.

    python sysid/paddle_puck_collision/code/fit_collision_cmaes.py \
        --input-dir /data2/air_hockey/robot_data_collection_puck_collision_20260910_1719 \
        --out sysid/paddle_puck_collision/results/cmaes_20260910

Pipeline: load trials → measure every collision (incoming / outgoing puck speed from free-flight
model fits, paddle speed from the robot) → calibrate the camera lag → keep the best three trials
per condition (the canonical dataset) → hold out one trial per condition → evaluate the canonical
sim (e, r) → CMA-ES on the training collisions (objective: RMS outgoing-speed error, m/s) →
validation + percentile report → (e, r) landscape + ridge → plots + summary.md.
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

from sysid.paddle_puck_collision.code.dataset import load_session, split_train_val, session_attrs
from sysid.paddle_puck_collision.code.speeds import (SpeedConfig, measure_all, calibrate_camera_lag, select_canonical,
                                              selected_trials, write_measurements_csv)
from sysid.paddle_puck_collision.code.sim_collision import (DEFAULT_BASE_CONFIG, CollisionParams, HeadOnCollider,
                                                     build_collision_sim_config, evaluate_measurements, load_base_config,
                                                     params_from_config, puck_density_for_ratio)
from sysid.paddle_puck_collision.code.cmaes_fit import CandidateEvaluator, ParamBounds, run_cmaes, grid_scan
from sysid.paddle_puck_collision.code import report
from sysid.common.fit_validation import percentile_report
from sysid.common import link_data


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", required=True, help="session directory with collision_*.hdf5")
    p.add_argument("--out", default=None, help="output directory (default sysid/paddle_puck_collision/results/cmaes_<session>)")
    p.add_argument("--base-config", default=str(DEFAULT_BASE_CONFIG), help="sim YAML providing the plant and the canonical (e, r)")
    p.add_argument("--gravity", type=float, default=0.661, help="puck acceleration down the table (m/s²) for the free-flight fits")
    p.add_argument("--damping", type=float, default=0.178, help="puck linear damping (1/s) for the free-flight fits")
    p.add_argument("--pre-frames", type=int, default=6)
    p.add_argument("--post-frames", type=int, default=6)
    p.add_argument("--max-out-angle", type=float, default=40.0, help="validity gate on the outgoing direction (deg)")
    p.add_argument("--camera-lag", type=float, default=None, help="s; default: calibrated from the moving-paddle trials")
    p.add_argument("--per-condition", type=int, default=3, help="trials kept per condition (the canonical dataset)")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--val-repeat", type=int, default=None, help="hold out this rank (1..per-condition) of every condition instead of a random one")
    p.add_argument("--e-range", type=float, nargs=2, default=(0.0, 1.5))
    p.add_argument("--r-range", type=float, nargs=2, default=(0.25, 1000.0))
    p.add_argument("--x0", type=float, nargs=2, metavar=("E", "R"), default=None, help="start point (default: base config)")
    p.add_argument("--sigma0", type=float, default=0.3)
    p.add_argument("--popsize", type=int, default=12)
    p.add_argument("--max-iter", type=int, default=40)
    p.add_argument("--restarts", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--grid", type=int, default=31, help="landscape grid size per axis (0 = skip)")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args(argv)


def _ridge_points(collider, fitted: CollisionParams, train, val, base: CollisionParams) -> list[dict]:
    k = fitted.gain()
    pts = [("rigid paddle", 1000.0), ("real weights 58 g / 13 g", 58.0 / 13.0), ("canonical sim", base.mass_ratio), ("CMA-ES best", fitted.mass_ratio)]
    rows = []
    for label, r in pts:
        e = k * (r + 1.0) / r - 1.0
        p = CollisionParams(e, r)
        rows.append({"label": label, "mass_ratio": r, "restitution": e,
                     "train_err": evaluate_measurements(collider, train, p)["rms_err"],
                     "val_err": evaluate_measurements(collider, val, p)["rms_err"] if val else float("nan")})
    return rows


def main(argv=None) -> dict:
    a = parse_args(argv)
    t_start = time.time()
    in_dir = Path(a.input_dir)
    out = Path(a.out) if a.out else _REPO_ROOT / "sysid/paddle_puck_collision/results" / f"cmaes_{in_dir.name}"
    link_data("paddle_puck_collision", in_dir, out)
    plots = out / "plots"
    out.mkdir(parents=True, exist_ok=True)
    plots.mkdir(exist_ok=True)

    # -- 1. measure every collision, calibrate the camera lag, select the canonical dataset
    trials = load_session(in_dir)
    attrs = session_attrs(trials)
    cfg = SpeedConfig(gravity_x=a.gravity, damping=a.damping, pre_frames=a.pre_frames, post_frames=a.post_frames,
                      max_out_angle_deg=a.max_out_angle)
    ms = measure_all(trials, cfg)
    lag = calibrate_camera_lag(trials, ms, cfg)
    cfg.camera_lag_s = float(a.camera_lag) if a.camera_lag is not None else lag["lag_s"]
    lag["used_lag_s"] = cfg.camera_lag_s
    ms = measure_all(trials, cfg)
    selection = select_canonical(ms, per_condition=a.per_condition)
    canon = selected_trials(trials, ms)
    by_name = {m.name: m for m in ms}
    print(f"{len(trials)} files, {len(selection)} conditions; camera lag {cfg.camera_lag_s * 1000:.0f} ms "
          f"(calibrated {lag['lag_s'] * 1000:.0f} ± {lag['lag_std_s'] * 1000:.0f} ms on {lag['n_trials']} trials; "
          f"stationary contact gap {lag['static_contact_gap_mean'] * 1000:.1f} mm vs {cfg.contact_distance * 1000:.1f} mm expected)")
    short = {c: (v["n_valid"], v["n_files"]) for c, v in selection.items()}
    print("valid / files per condition:", short)
    print(f"canonical dataset: {len(canon)} trials")
    write_measurements_csv(ms, out / "all_trials.csv")
    write_measurements_csv([by_name[t.name] for t in canon], out / "canonical_dataset.csv")
    with open(out / "lag_calibration.json", "w") as fh:
        json.dump(lag, fh, indent=1)
    with open(out / "selection.json", "w") as fh:
        json.dump(selection, fh, indent=1)

    train_t, val_t, split_info = split_train_val(canon, seed=a.split_seed, val_repeat=a.val_repeat)
    train = [by_name[t.name] for t in train_t]
    val = [by_name[t.name] for t in val_t]
    for m in train + val:                       # the repeat rank among the selected trials
        m.repeat = next(t.repeat for t in canon if t.name == m.name)
    with open(out / "split.json", "w") as fh:
        json.dump(split_info, fh, indent=1)
    print(f"{len(train)} train / {len(val)} val collisions")

    # -- 2. sim
    base_cfg = load_base_config(a.base_config)
    sim_cfg = build_collision_sim_config(base_cfg, attrs)
    with open(out / "sim_config_replay.yaml", "w") as fh:
        yaml.safe_dump({"air_hockey": sim_cfg}, fh, sort_keys=False)
    collider = HeadOnCollider(sim_cfg)
    base_params = params_from_config(base_cfg)
    x0 = CollisionParams(*a.x0) if a.x0 else base_params
    print(f"sim dt {collider.dt:.3f} s, paddle mass {collider.paddle_mass:.2f} kg, max paddle speed {collider.max_speed():.2f} m/s; "
          f"canonical e {base_params.restitution:.4f} r {base_params.mass_ratio:.3f} (gain {base_params.gain():.4f})")
    evals = {"baseline_train": evaluate_measurements(collider, train, base_params, keep_details=True),
             "baseline_val": evaluate_measurements(collider, val, base_params, keep_details=True)}
    print(f"canonical sim: train RMS {evals['baseline_train']['rms_err']:.4f} m/s (mean err {evals['baseline_train']['mean_err']:+.4f}), "
          f"val RMS {evals['baseline_val']['rms_err']:.4f} m/s")

    # -- 3. CMA-ES
    bounds = ParamBounds(restitution=tuple(a.e_range), mass_ratio=tuple(a.r_range))
    evaluator = CandidateEvaluator(sim_cfg, train, val, workers=a.workers)
    try:
        res = run_cmaes(evaluator, x0, bounds=bounds, sigma0=a.sigma0, popsize=a.popsize, max_iter=a.max_iter,
                        seed=a.seed, restarts=a.restarts)
        grid = grid_scan(evaluator, bounds, a.grid, a.grid) if a.grid > 0 else None
    finally:
        evaluator.close()
    fitted = res.best
    evals["fitted_train"] = evaluate_measurements(collider, train, fitted, keep_details=True)
    evals["fitted_val"] = evaluate_measurements(collider, val, fitted, keep_details=True)
    print(f"CMA-ES best: e {fitted.restitution:.4f} r {fitted.mass_ratio:.3f} gain {fitted.gain():.4f} → train RMS "
          f"{evals['fitted_train']['rms_err']:.4f} m/s, val {evals['fitted_val']['rms_err']:.4f} m/s "
          f"({res.n_evaluations} candidates, {res.wall_seconds:.0f} s)")
    ridge = _ridge_points(collider, fitted, train, val, base_params)
    for row in ridge:
        print(f"  ridge @ r={row['mass_ratio']:8.3f} ({row['label']}): e={row['restitution']:.4f} → train {row['train_err']:.4f} val {row['val_err']:.4f}")

    pr = None
    if val:
        vals = np.array([c.val_err for c in res.candidates])
        trs = np.array([c.train_err for c in res.candidates])
        pr = percentile_report("outgoing puck speed RMS error", "m/s", vals, evals["fitted_val"]["rms_err"],
                               lambda i: {"e": res.candidates[i].restitution, "r": res.candidates[i].mass_ratio},
                               canonical_err=evals["baseline_val"]["rms_err"], train_errors=trs,
                               selected_train_err=evals["fitted_train"]["rms_err"],
                               canonical_train_err=evals["baseline_train"]["rms_err"])

    # -- 4. outputs
    fit_json = res.to_json()
    fit_json.update({"input_dir": str(in_dir), "base_config": str(a.base_config), "camera_lag_s": cfg.camera_lag_s,
                     "speed_config": {k: getattr(cfg, k) for k in ("gravity_x", "damping", "pre_frames", "post_frames", "max_out_angle_deg",
                                                                    "min_separation", "min_speed_fraction", "recontact_margin", "camera_lag_s")},
                     "canonical": base_params.as_dict(), "canonical_gain": base_params.gain(), "ridge": ridge,
                     "grid": {k: v for k, v in grid.items() if k in ("best", "best_train_err", "best_val_err")} if grid else None,
                     "percentile_validation": pr.to_json() if pr else None,
                     "train_rms": {"canonical": evals["baseline_train"]["rms_err"], "fitted": evals["fitted_train"]["rms_err"]},
                     "val_rms": {"canonical": evals["baseline_val"]["rms_err"], "fitted": evals["fitted_val"]["rms_err"]},
                     "n_files": len(trials), "n_canonical": len(canon), "n_train": len(train), "n_val": len(val)})
    with open(out / "fit_result.json", "w") as fh:
        json.dump(fit_json, fh, indent=1)
    if grid:
        with open(out / "landscape_grid.json", "w") as fh:
            json.dump(grid, fh)
    report.write_candidates_csv(res.candidates, out / "candidates.csv")
    with open(out / "evaluations.json", "w") as fh:
        json.dump(evals, fh, indent=1)
    report.write_per_trial_csv(evals, out / "per_trial.csv")
    fitted_cfg = copy.deepcopy(base_cfg)
    fsp = fitted_cfg["air_hockey"]["simulator_params"]
    fsp["puck_restitution"] = round(fitted.restitution, 5)
    fsp["paddle_restitution"] = round(fitted.restitution, 5)
    fsp["puck_density"] = round(puck_density_for_ratio(fitted.mass_ratio, fsp["paddle_density"], fsp["paddle_radius"], fsp["puck_radius"]), 2)
    with open(out / "sim_config_fitted.yaml", "w") as fh:
        fh.write(f"# Paddle-puck restitution + mass ratio fitted by sysid/paddle_puck_collision/code/fit_collision_cmaes.py on {in_dir.name}\n"
                 f"# e = {fitted.restitution:.4f} (both fixtures; the listener uses max(puck, paddle)), m_paddle/m_puck = {fitted.mass_ratio:.3f} "
                 f"-> puck_density {fsp['puck_density']} with paddle_density {fsp['paddle_density']} fixed; head-on gain {fitted.gain():.4f}.\n"
                 f"# val outgoing-speed RMS {evals['baseline_val']['rms_err']:.3f} m/s (canonical) -> {evals['fitted_val']['rms_err']:.3f} m/s (fitted).\n"
                 f"# NOTE: head-on data only identify the gain (1+e)*r/(r+1); every (e, r) on that ridge fits equally (see summary.md).\n")
        yaml.safe_dump(fitted_cfg, fh, sort_keys=False)

    if not a.no_plots:
        report.plot_trial_fits(trials, ms, cfg, plots / "trial_fits.png")
        report.plot_gain_vs_speed(ms, plots / "gain_vs_speed.png", fitted=fitted, baseline=base_params)
        if grid:
            report.plot_landscape(grid, plots / "landscape.png", fitted=fitted, baseline=base_params, candidates=res.candidates)
        report.plot_sim_vs_real(evals, plots / "sim_vs_real.png")
        report.plot_convergence(res.generation_best, plots / "convergence.png")

    ctx = {"dataset": str(in_dir), "baseline": base_params, "fitted": fitted, "evals": evals, "lag": lag, "selection": selection,
           "cma": fit_json, "percentile": pr, "grid": grid, "ridge": ridge, "speed_cfg": cfg, "measurements": ms,
           "n_selected": len(canon), "n_files": len(trials), "per_condition": a.per_condition, "n_train": len(train), "n_val": len(val),
           "sim": {"base_config": a.base_config, "dt": collider.dt, "hist_len": int(collider.sim.hist_len),
                   "paddle_density": collider.paddle_density, "paddle_mass": collider.paddle_mass}}
    with open(out / "summary.md", "w") as fh:
        fh.write(report.summary_markdown(ctx))
    print(f"wrote {out} ({time.time() - t_start:.0f} s)")
    return fit_json


if __name__ == "__main__":
    main()

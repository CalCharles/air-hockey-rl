#!/usr/bin/env python3
"""Side-by-side videos of every real collision and its sim replication (camera | real scene | sim scene).

    python sysid/paddle_puck_collision/code/render_collisions.py \
        --input-dir /data2/air_hockey/robot_data_collection_puck_collision_20260910_1719 \
        --fit-dir sysid/paddle_puck_collision/results/cmaes_20260910_puck_collision            # → <fit-dir>/videos/

Per trial: ``gifs/<trial>.gif`` and ``mp4/<trial>.mp4`` (one frame per real 20 Hz step, played at
``--fps``), ``png/<trial>.png`` (the last frame: full trails) and ``mosaic_<condition>.png``;
``render_summary.csv`` lists the real vs sim outgoing speed of every rendered collision next to the
value the fit obtained with its own (teleport) timeline, so the rendered run can be checked to be
the same collision. The sim uses the fit's parameters by default (``--params canonical`` or
``--params E R`` for others); measurement settings (camera lag, gravity, damping, gates) come from
the fit's ``fit_result.json`` so the trials, speeds and selection are exactly the fit's.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sysid.paddle_puck_collision.code.dataset import load_session, session_attrs, group_by_condition
from sysid.paddle_puck_collision.code.speeds import SpeedConfig, measure_all, calibrate_camera_lag, select_canonical
from sysid.paddle_puck_collision.code.sim_collision import (DEFAULT_BASE_CONFIG, CollisionParams, HeadOnCollider,
                                                     build_collision_sim_config, load_base_config, params_from_config)
from sysid.paddle_puck_collision.code.overlay import CollisionRenderer, write_gif, write_mp4, write_png, mosaic


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--fit-dir", default=None, help="a fit_collision_cmaes.py output dir: parameters + measurement settings")
    p.add_argument("--out", default=None, help="default <fit-dir>/videos or sysid/paddle_puck_collision/results/videos_<session>")
    p.add_argument("--params", nargs="+", default=["fitted"], help="'fitted' | 'canonical' | E R")
    p.add_argument("--base-config", default=None)
    p.add_argument("--trials", default="selected", help="'selected' (the canonical dataset) | 'all' | comma-separated names")
    p.add_argument("--fps", type=int, default=10, help="playback rate; the data are 20 Hz, so 10 = half speed")
    p.add_argument("--panel-height", type=int, default=240)
    p.add_argument("--tail-steps", type=int, default=12, help="real steps rendered after the contact")
    p.add_argument("--no-camera", action="store_true", help="skip the stored camera images")
    p.add_argument("--no-mp4", action="store_true")
    p.add_argument("--no-gif", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    in_dir = Path(a.input_dir)
    fit = json.load(open(Path(a.fit_dir) / "fit_result.json")) if a.fit_dir else None
    out = Path(a.out) if a.out else (Path(a.fit_dir) / "videos" if a.fit_dir else _REPO_ROOT / "sysid/paddle_puck_collision/results" / f"videos_{in_dir.name}")
    for sub in ("gifs", "mp4", "png"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    # -- the same measurements / selection as the fit
    trials = load_session(in_dir)
    sc = fit["speed_config"] if fit else {}
    cfg = SpeedConfig(**{k: v for k, v in sc.items() if k != "camera_lag_s"})
    ms = measure_all(trials, cfg)
    cfg.camera_lag_s = float(sc["camera_lag_s"]) if fit else calibrate_camera_lag(trials, ms, cfg)["lag_s"]
    ms = measure_all(trials, cfg)
    select_canonical(ms, per_condition=3)
    by_name = {m.name: m for m in ms}
    if a.trials == "selected":
        subset = [t for t in trials if by_name[t.name].selected]
    elif a.trials == "all":
        subset = [t for t in trials if np.isfinite(by_name[t.name].t_c)]
    else:
        names = set(a.trials.split(","))
        subset = [t for t in trials if t.name in names or t.name.replace("collision_", "") in names]

    # -- sim
    base_cfg = load_base_config(a.base_config or (fit["base_config"] if fit else DEFAULT_BASE_CONFIG))
    sim_cfg = build_collision_sim_config(base_cfg, session_attrs(trials), x_max_lim=0.0,
                                         gravity=float(base_cfg["air_hockey"]["simulator_params"]["gravity"]))
    collider = HeadOnCollider(sim_cfg)
    if a.params[0] == "fitted":
        if not fit:
            raise SystemExit("--params fitted needs --fit-dir")
        params, label = CollisionParams(**fit["best"]), "fitted"
    elif a.params[0] == "canonical":
        params, label = params_from_config(base_cfg), "canonical"
    else:
        params, label = CollisionParams(float(a.params[0]), float(a.params[1])), "custom"
    fit_sim = {}
    if fit and label == "fitted" and (Path(a.fit_dir) / "evaluations.json").exists():
        ev = json.load(open(Path(a.fit_dir) / "evaluations.json"))
        fit_sim = {p["name"]: p["speed_out_sim"] for key in ("fitted_train", "fitted_val") for p in ev[key]["per_trial"]}
    renderer = CollisionRenderer(collider, cfg, params, label, panel_height=a.panel_height, tail_steps=a.tail_steps)
    print(f"{len(subset)} trials → {out}; params {label} e {params.restitution:.4f} r {params.mass_ratio:.2f}; camera lag {cfg.camera_lag_s * 1000:.0f} ms")

    rows, pngs = [], {}
    for t in subset:
        m = by_name[t.name]
        cam = None
        if not a.no_camera:
            with h5py.File(t.path, "r") as f:
                if "train_img" in f:
                    cam = f["train_img"][()]
        frames, sim = renderer.frames(t, m, cam)
        if not a.no_gif:
            write_gif(frames, out / "gifs" / f"{t.name}.gif", fps=a.fps)
        if not a.no_mp4:
            write_mp4(frames, out / "mp4" / f"{t.name}.mp4", fps=a.fps)
        pngs[t.name] = frames[-1]
        write_png(frames[-1], out / "png" / f"{t.name}.png")
        rows.append({"name": t.name, "condition": t.condition, "selected": m.selected, "u_p": m.u_p, "speed_in": m.speed_in,
                     "speed_out_real": m.speed_out, "speed_out_sim_rendered": sim["speed_out"],
                     "speed_out_sim_fit": fit_sim.get(t.name, float("nan")), "sim_u_p_actual": sim["u_p_actual"],
                     "sim_u_k_actual": sim["u_k_actual"],
                     "sim_contacts": sim["n_contacts"], "sim_first_contact_step": sim["first_contact_step"],
                     "sim_pre_roll_steps": sim["pre_roll"], "real_first_post_step": sim["first_post"]})
        print(f"  {t.name}: in {m.speed_in:.3f} (sim {sim['u_k_actual']:.3f})  real out {m.speed_out:.3f}  sim out {sim['speed_out']:.3f} (fit run {fit_sim.get(t.name, float('nan')):.3f})  "
              f"contact step {sim['first_contact_step'] + sim['pre_roll']} vs real first post step {sim['first_post']}  contacts {sim['n_contacts']}")
    for cond, group in group_by_condition(subset).items():
        write_png(mosaic([pngs[t.name] for t in group], ncols=1), out / f"mosaic_{cond}.png")
    with open(out / "render_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    d = np.array([r["speed_out_sim_rendered"] - r["speed_out_sim_fit"] for r in rows if np.isfinite(r["speed_out_sim_fit"])])
    if d.size:
        print(f"rendered vs fit-run sim outgoing speed: max |diff| {np.abs(d).max():.4f} m/s over {d.size} trials "
              f"(the rendered puck decays under gravity / damping for up to one step after the contact)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

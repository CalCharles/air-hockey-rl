#!/usr/bin/env python
"""Evaluate the fitted paddle–puck parameters on the *offset* collision session (no fitting).

The ``puck_collision_change_angle`` session repeats one condition (release ``1/2``, paddle action
0.66) while the paddle is commanded to y offsets of −3 … −11 cm, so the puck — which slides
straight down the table — hits the paddle off-centre and leaves at an angle. This tool replays
every trial in the Box2D env with the parameters identified on the head-on session
(``--fit-dir``: e, m_paddle/m_puck, camera lag, speed-estimation settings) under one assumption:

    the puck comes straight down the table (vy = 0) at the measured incoming speed and meets
    the paddle at the *measured* contact position — the same lateral offset ``dy`` between the
    puck lane and the paddle centre (``HeadOnCollider.run(..., dy=dy)``), the paddle moving at
    the measured speed.

Per trial the real and the simulated outgoing puck are compared on **exit speed** and **exit
angle** (signed, from the paddle normal); per offset condition and overall the mean / std / RMS
of both errors are reported. The canonical parameters and the other ridge point (r = 2.56,
e = 1.26) are scored alongside for reference — head-on speeds could not tell them apart, oblique
hits can. Videos (camera · real on the table · sim replay, as ``render_collisions.py``) are
written for every trial with the fitted parameters.

    python sysid/paddle_puck_collision/code/evaluate_offset_collisions.py \\
        --input-dir /data2/air_hockey/robot_data_collection_puck_collision_change_angle_20260910_1818 \\
        --fit-dir sysid/paddle_puck_collision/results/cmaes_20260910_puck_collision
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

from sysid.common import link_data  # noqa: E402
from sysid.paddle_puck_collision.code.dataset import load_session, session_attrs  # noqa: E402
from sysid.paddle_puck_collision.code.speeds import SpeedConfig, measure_all, write_measurements_csv  # noqa: E402
from sysid.paddle_puck_collision.code.sim_collision import (  # noqa: E402
    DEFAULT_BASE_CONFIG, CollisionParams, HeadOnCollider, build_collision_sim_config, load_base_config, params_from_config,
)
from sysid.paddle_puck_collision.code.overlay import CollisionRenderer, mosaic, write_gif, write_mp4, write_png  # noqa: E402
from sysid.paddle_puck_collision.code.report import plot_trial_fits  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", required=True, help="the offset session (collision_*.hdf5 with varying y offsets)")
    p.add_argument("--fit-dir", required=True, help="fit_collision_cmaes.py output: parameters, camera lag, speed settings")
    p.add_argument("--out", default=None, help="default sysid/paddle_puck_collision/results/offset_<session>")
    p.add_argument("--base-config", default=None)
    p.add_argument("--max-out-angle", type=float, default=90.0, help="validity gate on the outgoing direction (deg); oblique exits are the point here")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--panel-height", type=int, default=240)
    p.add_argument("--tail-steps", type=int, default=12)
    p.add_argument("--no-videos", action="store_true")
    p.add_argument("--no-camera", action="store_true")
    p.add_argument("--no-mp4", action="store_true")
    return p.parse_args(argv)


def signed_angle(vy: float, vx_away: float) -> float:
    return float(np.degrees(np.arctan2(vy, vx_away)))


def score(collider: HeadOnCollider, params: CollisionParams, ms) -> list[dict]:
    """Replay every measurement straight down its real lane; per-trial exit speed / angle vs real."""
    collider.set_params(params)
    rows = []
    for m in ms:
        r = collider.run(m.u_p, m.speed_in, dy=m.dy)
        real_ang, sim_ang = signed_angle(m.vy_out, m.vx_out_away), signed_angle(r["vy_out"], r["vx_out_away"])
        rows.append({"name": m.name, "y_offset": float(m.name.split("_y")[-1]), "dy_cm": 100 * m.dy, "u_p": m.u_p, "speed_in": m.speed_in,
                     "angle_in_deg": signed_angle(m.vy_in, m.vx_in),
                     "speed_out_real": m.speed_out, "speed_out_sim": r["speed_out"], "speed_err": r["speed_out"] - m.speed_out,
                     "speed_rel_err": (r["speed_out"] - m.speed_out) / m.speed_out if m.speed_out > 1e-6 else float("nan"),
                     "angle_out_real": real_ang, "angle_out_sim": sim_ang, "angle_err": sim_ang - real_ang,
                     "vx_out_real": m.vx_out_away, "vx_out_sim": r["vx_out_away"], "vy_out_real": m.vy_out, "vy_out_sim": r["vy_out"],
                     "sim_contacts": r["n_contacts"], "sim_paddle_vy_post": r["paddle_vy_post"]})
    return rows


def required_offset(collider: HeadOnCollider, m, target_angle: float, tol: float = 0.3) -> float:
    """Diagnostic: the lateral offset the sim would need to reproduce the real exit angle (bisection on
    |dy|; the sim exit angle grows monotonically with the offset). NaN if unreachable."""
    lo, hi = 0.0, 0.98 * collider.contact_distance
    sgn = 1.0 if m.dy >= 0 else -1.0
    f = lambda d: signed_angle(*(lambda r: (r["vy_out"], r["vx_out_away"]))(collider.run(m.u_p, m.speed_in, dy=sgn * d))) * sgn
    if f(hi) < abs(target_angle) - tol:
        return float("nan")
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        if f(mid) < abs(target_angle):
            lo = mid
        else:
            hi = mid
        if hi - lo < 2e-4:
            break
    return sgn * 0.5 * (lo + hi)


def aggregate(rows: list[dict]) -> dict:
    a = {}
    for key in ("speed_err", "speed_rel_err", "angle_err"):
        v = np.array([r[key] for r in rows], dtype=float)
        v = v[np.isfinite(v)]
        a[key] = {"mean": float(v.mean()), "std": float(v.std()), "rms": float(np.sqrt(np.mean(v ** 2))),
                  "mean_abs": float(np.abs(v).mean()), "max_abs": float(np.abs(v).max()), "n": int(v.size)} if v.size else {}
    return a


def main(argv=None):
    a = parse_args(argv)
    in_dir = Path(a.input_dir)
    fit_dir = Path(a.fit_dir)
    fit = json.load(open(fit_dir / "fit_result.json"))
    out = Path(a.out) if a.out else _REPO_ROOT / "sysid/paddle_puck_collision/results" / f"offset_{in_dir.name}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)
    link_data("paddle_puck_collision", in_dir, out)

    # -- measurements with the head-on fit's settings (camera lag included), oblique exits allowed
    trials = load_session(in_dir)
    sc = dict(fit["speed_config"])
    sc["max_out_angle_deg"] = float(a.max_out_angle)
    sc["oblique"] = True                      # 2-D separation / re-contact gates, |dy| < contact distance
    cfg = SpeedConfig(**sc)
    ms_all = measure_all(trials, cfg)
    write_measurements_csv(ms_all, out / "all_trials.csv")
    ms = [m for m in ms_all if m.valid]
    dropped = [(m.name, m.reason) for m in ms_all if not m.valid]
    by_name = {m.name: m for m in ms_all}
    print(f"{len(trials)} trials, {len(ms)} valid collisions (camera lag {cfg.camera_lag_s * 1000:.0f} ms from the head-on fit); dropped: {dropped}")

    # -- sim: the head-on fit's env, three parameter sets
    base_cfg = load_base_config(a.base_config or fit["base_config"])
    collider = HeadOnCollider(build_collision_sim_config(base_cfg, session_attrs(trials)))
    fitted = CollisionParams(**fit["best"])
    canonical = params_from_config(base_cfg)
    ridge_alt = None
    for pt in fit.get("ridge", []):
        if isinstance(pt, dict) and abs(float(pt.get("mass_ratio", -1)) - canonical.mass_ratio) < 1e-6:
            ridge_alt = CollisionParams(float(pt["restitution"]), float(pt["mass_ratio"]))
    if ridge_alt is None:
        g = fitted.gain()
        ridge_alt = CollisionParams(g * (canonical.mass_ratio + 1.0) / canonical.mass_ratio - 1.0, canonical.mass_ratio)
    param_sets = {"fitted": fitted, "ridge_r_canonical": ridge_alt, "canonical": canonical}
    results = {}
    collider.set_params(fitted)
    dy_req = {m.name: required_offset(collider, m, signed_angle(m.vy_out, m.vx_out_away)) for m in ms}
    for label, params in param_sets.items():
        rows = score(collider, params, ms)
        for r in rows:
            r["dy_required_cm"] = 100 * dy_req[r["name"]]
            r["dy_required_ratio"] = dy_req[r["name"]] / (r["dy_cm"] / 100) if abs(r["dy_cm"]) > 0.3 else float("nan")
        results[label] = {"params": params.as_dict(), "gain": params.gain(), "per_trial": rows, "overall": aggregate(rows),
                          "per_offset": {str(y): aggregate([r for r in rows if r["y_offset"] == y]) for y in sorted({r["y_offset"] for r in rows})}}
        o = results[label]["overall"]
        print(f"[{label}] e {params.restitution:.3f} r {params.mass_ratio:.1f} gain {params.gain():.3f}: "
              f"speed err mean {o['speed_err']['mean']:+.3f} rms {o['speed_err']['rms']:.3f} m/s ({100 * o['speed_rel_err']['mean_abs']:.1f} % abs) | "
              f"angle err mean {o['angle_err']['mean']:+.1f} rms {o['angle_err']['rms']:.1f} deg")
    with open(out / "evaluations.json", "w") as f:
        json.dump({"input_dir": str(in_dir), "fit_dir": str(fit_dir), "speed_config": sc, "n_trials": len(trials), "n_valid": len(ms),
                   "dropped": dropped, "results": results}, f, indent=1)
    req = np.array([r["dy_required_ratio"] for r in results["fitted"]["per_trial"]]); req = req[np.isfinite(req)]
    if req.size:
        print(f"[diagnostic] offset the sim needs for the real exit angle / measured offset: median {np.median(req):.2f}, "
              f"IQR {np.percentile(req, 25):.2f}–{np.percentile(req, 75):.2f} (n={req.size})")
    with open(out / "per_trial.csv", "w", newline="") as fh:
        cols = list(results["fitted"]["per_trial"][0])
        w = csv.DictWriter(fh, fieldnames=cols + ["speed_out_sim_ridge_r_canonical", "angle_out_sim_ridge_r_canonical", "speed_out_sim_canonical", "angle_out_sim_canonical"])
        w.writeheader()
        for i, r in enumerate(results["fitted"]["per_trial"]):
            extra = {f"{k}_{lab}": results[lab]["per_trial"][i][k] for lab in ("ridge_r_canonical", "canonical") for k in ("speed_out_sim", "angle_out_sim")}
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in (r | extra).items()})

    # -- plots: exit angle and speed vs offset (real vs sim), error vs offset, per-trial fits
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = results["fitted"]["per_trial"]
    dy = np.array([r["dy_cm"] for r in rows])
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    ax[0].scatter(dy, [r["angle_out_real"] for r in rows], c="k", s=18, label="real")
    for lab, c, mk in (("fitted", "#2a78d6", "o"), ("ridge_r_canonical", "#eb6834", "^"), ("canonical", "#888888", "x")):
        ax[0].scatter(dy, [r["angle_out_sim"] for r in results[lab]["per_trial"]], c=c, s=18, marker=mk, label=f"sim {lab}")
    ax[0].set_xlabel("lateral offset at contact dy (cm)"); ax[0].set_ylabel("exit angle from the paddle normal (deg)"); ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)
    ax[1].scatter(dy, [r["speed_out_real"] for r in rows], c="k", s=18, label="real")
    for lab, c, mk in (("fitted", "#2a78d6", "o"), ("ridge_r_canonical", "#eb6834", "^"), ("canonical", "#888888", "x")):
        ax[1].scatter(dy, [r["speed_out_sim"] for r in results[lab]["per_trial"]], c=c, s=18, marker=mk, label=f"sim {lab}")
    ax[1].set_xlabel("lateral offset at contact dy (cm)"); ax[1].set_ylabel("exit speed (m/s)"); ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3)
    ax[2].scatter(dy, [r["angle_err"] for r in rows], c="#2a78d6", s=18, label="angle err (deg), fitted")
    ax2 = ax[2].twinx()
    ax2.scatter(dy, [100 * r["speed_rel_err"] for r in rows], c="#eb6834", s=18, marker="^", label="speed err (%), fitted")
    ax[2].axhline(0, color="k", lw=0.6); ax[2].set_xlabel("lateral offset at contact dy (cm)"); ax[2].set_ylabel("sim − real exit angle (deg)", color="#2a78d6"); ax2.set_ylabel("sim − real exit speed (%)", color="#eb6834"); ax[2].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "plots" / "exit_vs_offset.png", dpi=150); plt.close(fig)
    plot_trial_fits(trials, ms_all, cfg, out / "plots" / "trial_fits.png")

    # -- videos with the fitted parameters, puck launched straight in the real lane
    video_rows = []
    if not a.no_videos:
        vid = out / "videos"
        for sub in ("gifs", "mp4", "png"):
            (vid / sub).mkdir(parents=True, exist_ok=True)
        rcfg = build_collision_sim_config(base_cfg, session_attrs(trials), x_max_lim=0.0,
                                          gravity=float(base_cfg["air_hockey"]["simulator_params"]["gravity"]))
        renderer = CollisionRenderer(HeadOnCollider(rcfg), cfg, fitted, "fitted", panel_height=a.panel_height, tail_steps=a.tail_steps, offset=True)
        pngs = {}
        for t in trials:
            m = by_name[t.name]
            if not m.valid:
                continue
            cam = None
            if not a.no_camera:
                with h5py.File(t.path, "r") as f:
                    if "train_img" in f:
                        cam = f["train_img"][()]
            frames, sim = renderer.frames(t, m, cam)
            write_gif(frames, vid / "gifs" / f"{t.name}.gif", fps=a.fps)
            if not a.no_mp4:
                write_mp4(frames, vid / "mp4" / f"{t.name}.mp4", fps=a.fps)
            pngs[t.name] = frames[-1]
            write_png(frames[-1], vid / "png" / f"{t.name}.png")
            video_rows.append({"name": t.name, "speed_out_sim_rendered": sim["speed_out"], "angle_out_sim_rendered": sim["angle_out_deg"],
                               "sim_contacts": sim["n_contacts"], "sim_first_contact_step": sim["first_contact_step"] + sim["pre_roll"], "real_first_post_step": sim["first_post"]})
        for y in sorted({r["y_offset"] for r in rows}):
            names = [r["name"] for r in rows if r["y_offset"] == y]
            write_png(mosaic([pngs[n] for n in names if n in pngs], ncols=1), vid / f"mosaic_y{y:+.3f}.png")
        with open(vid / "render_summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(video_rows[0])); w.writeheader()
            for r in video_rows:
                w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    # -- summary
    o = results["fitted"]["overall"]
    lines = [f"# Offset paddle–puck collisions replayed with the head-on fit (no fitting) — `{in_dir.name}`", "",
             f"{len(trials)} trials (one condition: release 1/2, action 0.66; commanded paddle y −3 … −11 cm), {len(ms)} valid collisions. "
             f"Parameters from `{fit_dir.name}`: **e = {fitted.restitution:.3f}, m_paddle/m_puck = {fitted.mass_ratio:.1f}** (gain {fitted.gain():.3f}); "
             f"speed estimation and camera lag ({cfg.camera_lag_s * 1000:.0f} ms) as in that fit, outgoing-angle gate {a.max_out_angle:.0f}°.", "",
             "**Assumption.** The sim puck travels straight down the table (vy = 0) at the measured incoming speed and meets the paddle at the "
             "measured contact position: the same lateral offset `dy` between the puck lane and the paddle centre (`x`-gap `sqrt(d² − dy²)` at contact), "
             "the paddle moving at the measured speed under its own PID. Box2D's paddle–puck contact is frictionless, so the tangential velocity is "
             "preserved and the exit angle is set by the normal impulse (e, and the paddle recoil through r) alone.", "",
             "## Errors, all valid trials (sim − real)", "",
             "| parameters | e | r | gain | exit-speed err mean ± std (m/s) | rms | mean abs rel | exit-angle err mean ± std (deg) | rms | max abs |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for lab, res in results.items():
        p_, oo = res["params"], res["overall"]
        lines.append(f"| {lab} | {p_['restitution']:.3f} | {p_['mass_ratio']:.1f} | {res['gain']:.3f} | {oo['speed_err']['mean']:+.3f} ± {oo['speed_err']['std']:.3f} | {oo['speed_err']['rms']:.3f} | "
                     f"{100 * oo['speed_rel_err']['mean_abs']:.1f} % | {oo['angle_err']['mean']:+.1f} ± {oo['angle_err']['std']:.1f} | {oo['angle_err']['rms']:.1f} | {oo['angle_err']['max_abs']:.1f} |")
    lines += ["", "## Per offset (fitted parameters)", "",
              "| commanded y | n | dy at contact (cm) | in (m/s) | paddle (m/s) | exit speed real → sim (m/s) | speed err mean ± std | exit angle real → sim (deg) | angle err mean ± std |",
              "|---|---|---|---|---|---|---|---|---|"]
    for y, agg in results["fitted"]["per_offset"].items():
        rr = [r for r in rows if str(r["y_offset"]) == y]
        f_ = lambda k: np.mean([r[k] for r in rr])
        lines.append(f"| {float(y):+.2f} | {len(rr)} | {f_('dy_cm'):+.1f} | {f_('speed_in'):.2f} | {f_('u_p'):.2f} | {f_('speed_out_real'):.2f} → {f_('speed_out_sim'):.2f} | "
                     f"{agg['speed_err']['mean']:+.3f} ± {agg['speed_err']['std']:.3f} | {f_('angle_out_real'):+.1f} → {f_('angle_out_sim'):+.1f} | {agg['angle_err']['mean']:+.1f} ± {agg['angle_err']['std']:.1f} |")
    lines += ["", "## Per trial (fitted parameters)", "",
              "| trial | dy (cm) | in | paddle | angle in | exit speed real / sim | err | exit angle real / sim | err | sim contacts | dy the sim needs for the real angle (cm) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| `{r['name'].replace('collision_', '')}` | {r['dy_cm']:+.1f} | {r['speed_in']:.2f} | {r['u_p']:.2f} | {r['angle_in_deg']:+.1f}° | {r['speed_out_real']:.2f} / {r['speed_out_sim']:.2f} | "
                     f"{r['speed_err']:+.2f} ({100 * r['speed_rel_err']:+.0f} %) | {r['angle_out_real']:+.1f}° / {r['angle_out_sim']:+.1f}° | {r['angle_err']:+.1f}° | {r['sim_contacts']} | {r['dy_required_cm']:+.1f} |")
    if req.size:
        lines += ["", f"**Diagnostic.** For the sim to reproduce the real exit angle it would need a lateral offset of {np.median(req):.2f} × the measured one "
                  f"(median over trials with |dy| > 3 mm, IQR {np.percentile(req, 25):.2f}–{np.percentile(req, 75):.2f}). A constant ratio points at a physical effect the frictionless "
                  "contact lacks (tangential impulse from friction / rolling reduces the deflection); a constant *difference* would point at a y offset between the puck and the paddle frame."]
    if dropped:
        lines += ["", "Dropped (no valid collision): " + "; ".join(f"`{n.replace('collision_', '')}` ({why})" for n, why in dropped)]
    lines += ["", "Files: `per_trial.csv`, `all_trials.csv` (every file with its fits and gates), `evaluations.json` (all three parameter sets), "
              "`plots/exit_vs_offset.png`, `plots/trial_fits.png`" + ("" if a.no_videos else ", `videos/{gifs,mp4,png}/` + `videos/mosaic_y*.png` (camera · real · sim, fitted parameters, puck launched straight in the real lane)") + ".", ""]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

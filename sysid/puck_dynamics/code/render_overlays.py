#!/usr/bin/env python
"""Overlay real free-flight windows with the Box2D puck under the identified (g, γ).

For verification and visualisation of a ``fit_puck.py`` run: picks representative validation
windows (spread over the fit-error distribution: 10 / 30 / 50 / 70 / 90 % for ``--n 5``),
starts the Box2D puck (env built from the run's ``sim_config_fitted.yaml``, paddle parked in the
corner furthest from the path) at the fitted position / velocity of the first sample, steps it
once per real sample **with that sample's real time interval** (the sim reads ``time_per_step``
at every step; real spacing is ≈ 48.5 ms, not 50), and draws both on the table:

    real puck    the env's puck sprite + black trail (tracker samples)
    Box2D puck   blue ghost + trail (the simulator under the fitted parameters)
    model        thin grey line — the analytic damped model the grid search scores

Windows whose real path comes within ``--wall-margin`` of one of the sim's wall contact lines are
not candidates: the puck frame is offset from the sim table (y− apex 0.442 vs 0.400 m, x+ 0.865 vs
0.933 — see the wall fit's "apparent wall lines"), so the sim puck would bounce off a wall the real
puck never reached. The number excluded is reported.

Outputs in ``<results>/overlays/``: one GIF + last-frame PNG per example, ``mosaic.png``,
``overlay_summary.md`` / ``.csv`` with per-example errors (Box2D vs real, model vs real).

    python sysid/puck_dynamics/code/render_overlays.py --results-dir sysid/puck_dynamics/results/mouse_dataset
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sysid.common.trajectory_segmentation import SegmentationConfig, fit_damped, model_state  # noqa: E402
from sysid.common.sysid_dataset import FreeFallWindow, load_manifest, make_free_fall_windows  # noqa: E402
from sysid.common.table_scene import (  # noqa: E402
    C_MODEL, C_REAL, C_SIM, TableScene, legend_strip, mosaic, pick_percentiles, sim_to_base, write_gif, write_png,
)
from sysid.wall_collision.code.wall_restitution_fit import build_env  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=Path, required=True, help="a fit_puck.py output folder (results.json, sim_config_fitted.yaml)")
    p.add_argument("--sections-dir", type=Path, default=None, help="default: the one recorded in results.json")
    p.add_argument("--split", choices=["val", "train", "all"], default="val", help="which recordings to draw examples from")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--out", type=Path, default=None, help="default <results-dir>/overlays")
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--fps", type=int, default=10, help="GIF frame rate (real data are 20 Hz; 10 = half speed)")
    p.add_argument("--hold-last", type=int, default=8)
    p.add_argument("--wall-margin", type=float, default=0.01, help="m; windows closer than this to a sim wall contact line are not candidates")
    p.add_argument("--integrator-compensation", action="store_true",
                   help="start the Box2D puck at v0 − g·h/2 to cancel the semi-implicit-Euler position bias (diagnostic; default: the sim as it is)")
    return p.parse_args(argv)


def park_paddle(env, path_sim: np.ndarray) -> np.ndarray:
    """Corner of the paddle workspace (base frame) furthest from the real puck path."""
    x_lo, x_hi, y_lo, y_hi = env.get_paddle_workspace_bounds()
    path_base = np.stack([sim_to_base(p) for p in path_sim])
    corners = [np.array(c) for c in ((x_hi, y_lo), (x_hi, y_hi), (x_lo, y_lo), (x_lo, y_hi))]
    return max(corners, key=lambda c: np.min(np.linalg.norm(path_base - c, axis=1)))


def replay_window(env, w: FreeFallWindow, cfg: SegmentationConfig, integrator_compensation: bool = False) -> dict:
    """Box2D replay from the fitted state at the first sample; one env step per real sample.

    Box2D integrates with semi-implicit Euler (v += g h, then x += v h), so over k steps its
    position leads the exact solution by ½ g h² k = ½ g h t along gravity — a constant velocity
    bias of ½ g h (≈ 1.8 cm/s at h = 48.5 ms, g = 0.73). ``integrator_compensation`` starts the
    puck at v0 − ½ g h so the replay matches the analytic model (diagnostic only)."""
    fit = fit_damped(w.t, w.xy, cfg)
    p0, v0 = model_state(fit, 0.0, cfg)
    v_start = v0 - 0.5 * np.array([cfg.gravity_x, cfg.gravity_y]) * float(np.mean(np.diff(w.t))) if integrator_compensation else v0
    pad = park_paddle(env, w.xy)
    env.reset_from_state(np.concatenate([pad, np.zeros(2), sim_to_base(p0), sim_to_base(v_start)]))
    sim = [np.asarray(p0, dtype=np.float64)]
    zero = np.zeros(2, dtype=np.float32)
    dt_nominal = env.simulator.time_per_step
    try:
        for k in range(1, len(w.t)):
            env.simulator.time_per_step = float(w.t[k] - w.t[k - 1])     # step exactly to the next real sample
            env.step(zero)
            pk = env.current_state["pucks"][0]
            p = np.asarray(pk["position"][:2], dtype=np.float64)
            sim.append(np.array([-p[0], p[1]]))
    finally:
        env.simulator.time_per_step = dt_nominal
    sim = np.asarray(sim)
    model = np.asarray([model_state(fit, float(tau), cfg)[0] for tau in w.t])
    d_sim, d_model = np.linalg.norm(sim - w.xy, axis=1), np.linalg.norm(model - w.xy, axis=1)
    d_sm = np.linalg.norm(sim - model, axis=1)
    dist = float(np.sum(np.linalg.norm(np.diff(w.xy, axis=0), axis=1)))
    return {"sim": sim, "model": model, "fit_rms_cm": 100 * float(fit["rms"]), "speed0": float(np.linalg.norm(v0)),
            "sim_rms_cm": 100 * float(np.sqrt(np.mean(d_sim ** 2))), "sim_final_cm": 100 * float(d_sim[-1]), "sim_max_cm": 100 * float(d_sim.max()),
            "model_rms_cm": 100 * float(np.sqrt(np.mean(d_model ** 2))), "model_final_cm": 100 * float(d_model[-1]),
            "sim_vs_model_final_cm": 100 * float(d_sm[-1]), "sim_vs_model_max_cm": 100 * float(d_sm.max()),
            "final_rel": float(d_sim[-1] / max(dist, 1e-6)), "distance_m": dist, "dt_real": float(np.mean(np.diff(w.t))),
            "err_sim_cm": 100 * d_sim, "paddle_base": pad}


def main(argv=None):
    a = parse_args(argv)
    res = json.load(open(a.results_dir / "results.json"))
    sections_dir = a.sections_dir or Path(res["sections_dir"])
    g, gam = float(res["gravity_x"]), float(res["damping"])
    cfg = SegmentationConfig(gravity_x=g, damping=gam)
    window_frames = int(res["args"]["window_frames"])
    sim_cfg = a.results_dir / "sim_config_fitted.yaml"
    out = a.out or (a.results_dir / "overlays")
    for sub in ("gifs", "png"):                      # a re-render replaces the previous example set
        (out / sub).mkdir(parents=True, exist_ok=True)
        for old_file in (out / sub).glob("example*"):
            old_file.unlink()

    manifest = load_manifest(sections_dir)
    sources = set(res["split_sources"]["train"] + res["split_sources"]["val"]) if a.split == "all" else set(res["split_sources"][a.split])
    rows = [r for r in manifest["sections"] if r["kind"] == "free_fall" and r["source"] in sources]
    all_windows = make_free_fall_windows(rows, sections_dir, cfg, window_frames)
    env = build_env(sim_cfg)
    sim_ = env.simulator
    lim_x, lim_y = 0.5 * float(sim_.length) - float(sim_.puck_radius) - a.wall_margin, 0.5 * float(sim_.width) - float(sim_.puck_radius) - a.wall_margin
    windows = [w for w in all_windows if np.all(np.abs(w.xy[:, 0]) < lim_x) and np.all(np.abs(w.xy[:, 1]) < lim_y)]
    n_excl = len(all_windows) - len(windows)
    rms = np.array([100 * fit_damped(w.t, w.xy, cfg)["rms"] for w in windows])
    picks = pick_percentiles(rms, a.n)
    print(f"{len(all_windows)} {a.split} windows of {window_frames} samples, {n_excl} excluded (within {100 * a.wall_margin:.0f} cm of a sim wall line); "
          f"fit rms cm p10/p50/p90 = {np.percentile(rms, 10):.2f}/{np.percentile(rms, 50):.2f}/{np.percentile(rms, 90):.2f}; picked {len(picks)} at spread percentiles")

    scene = TableScene(env)
    legend = legend_strip([("real (tracker)", C_REAL), (f"Box2D  g={g:+.3f} gamma={gam:.3f}", C_SIM), ("analytic model", C_MODEL)], a.width)
    summaries, table = [], []
    for i, k in enumerate(picks, 1):
        w = windows[k]
        r = replay_window(env, w, cfg, a.integrator_compensation)
        pct = 100 * (np.searchsorted(np.sort(rms), rms[k]) / max(len(rms) - 1, 1))
        name = f"example{i}_{Path(w.clip).stem}_w{w.start_in_clip}"
        head = f"#{i}  {Path(w.clip).stem} @{w.start_in_clip}  |  speed {r['speed0']:.2f} m/s  fit rms {r['fit_rms_cm']:.2f} cm (p{pct:.0f} of {a.split})"
        frames = []
        for j in range(len(w.t)):
            f = scene.table()
            scene.draw_trail(f, r["model"][: j + 1], C_MODEL, 1, dots=False)
            scene.draw_trail(f, w.xy[: j + 1], C_REAL, 2)
            scene.draw_trail(f, r["sim"][: j + 1], C_SIM, 1, dots=False)
            scene.draw_sprite(f, w.xy[j], "puck")
            scene.draw_ghost(f, r["sim"][j], C_SIM)
            labels = [(head, C_REAL),
                      (f"step {j:2d}/{len(w.t) - 1}  t={w.t[j]:.2f} s   Box2D - real: {r['err_sim_cm'][j]:4.1f} cm   (rms {r['sim_rms_cm']:.2f}, final {r['sim_final_cm']:.1f} cm = {100 * r['final_rel']:.1f} % of {100 * r['distance_m']:.0f} cm)", C_SIM)]
            frames.append(np.vstack([legend, scene.finish(f, a.width, labels)]))
        frames += [frames[-1]] * a.hold_last
        write_gif(frames, out / "gifs" / f"{name}.gif", a.fps)
        write_png(frames[-1], out / "png" / f"{name}.png")
        summaries.append(frames[-1])
        row = {"example": i, "clip": w.clip, "start_in_clip": w.start_in_clip, "source": Path(w.source).name, "n_samples": len(w.t),
               "percentile_of_split": round(pct), **{k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in r.items() if k_ not in ("sim", "model", "err_sim_cm", "paddle_base")}}
        table.append(row)
        print(f"  #{i} {w.clip}@{w.start_in_clip}: speed0 {r['speed0']:.2f} m/s, fit rms {r['fit_rms_cm']:.2f} cm | Box2D rms {r['sim_rms_cm']:.2f} final {r['sim_final_cm']:.1f} max {r['sim_max_cm']:.1f} cm | model rms {r['model_rms_cm']:.2f} final {r['model_final_cm']:.1f} cm | Box2D vs model final {r['sim_vs_model_final_cm']:.1f} cm")
    write_png(mosaic(summaries, ncols=1 if len(summaries) <= 3 else 2), out / "mosaic.png")
    with open(out / "overlay_summary.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(table[0])); wr.writeheader(); wr.writerows(table)
    lines = [f"# Puck free flight — real vs Box2D overlays ({a.split} windows, fitted g = {g:+.3f}, γ = {gam:.3f})", "",
             f"{len(picks)} windows of {window_frames} samples picked at spread percentiles of the fit rms over the {len(windows)} {a.split} windows "
             f"(p10/p50/p90 = {np.percentile(rms, 10):.2f}/{np.percentile(rms, 50):.2f}/{np.percentile(rms, 90):.2f} cm); {n_excl} of {len(all_windows)} windows "
             f"were not candidates because the real path comes within {100 * a.wall_margin:.0f} cm of a sim wall contact line (the puck frame is offset from the sim table: "
             "y− apex 0.442 vs 0.400 m, x+ 0.865 vs 0.933 — the sim puck would bounce where the real one did not). "
             f"Box2D env from `{sim_cfg.name}` (noise / delays off, paddle parked in the far corner), puck started at the fitted state of the "
             f"first sample and stepped to every real sample with its real interval (mean {1000 * np.mean([r['dt_real'] for r in table]):.1f} ms; the sim's nominal step is 50 ms). "
             "`model` = the analytic damped model the grid search scores. Box2D integrates the same law (linear damping + constant acceleration) with "
             "semi-implicit Euler at one step per frame, whose position leads the exact solution by ½·g·h per second along gravity "
             f"(≈ {100 * 0.5 * abs(g) * np.mean([r['dt_real'] for r in table]):.1f} cm/s here) — the `Box2D vs model` column is that integrator bias, not a dynamics difference"
             + (" (**cancelled in this run** by starting the puck at v0 − ½·g·h, `--integrator-compensation`)" if a.integrator_compensation else "") + ".", "",
             "| # | clip @ start | speed (m/s) | fit rms (cm) | percentile | Box2D rms (cm) | Box2D final (cm) | Box2D max (cm) | final / distance | model rms (cm) | model final (cm) | Box2D vs model final (cm) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        lines.append(f"| {r['example']} | `{Path(r['clip']).stem}` @{r['start_in_clip']} | {r['speed0']:.2f} | {r['fit_rms_cm']:.2f} | p{r['percentile_of_split']} | {r['sim_rms_cm']:.2f} | {r['sim_final_cm']:.1f} | {r['sim_max_cm']:.1f} | {100 * r['final_rel']:.1f} % | {r['model_rms_cm']:.2f} | {r['model_final_cm']:.1f} | {r['sim_vs_model_final_cm']:.1f} |")
    lines += ["", "Files: `gifs/example*.gif` (real sprite + black trail, Box2D blue ghost + trail, model grey line; half speed), `png/` last frames, `mosaic.png`, `overlay_summary.csv`.", ""]
    (out / "overlay_summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Qualitative + per-trial view of a paddle PID fit: sim-vs-real overlays on the Box2D scene.

    python sysid/paddle_pid/code/render_overlays.py --input-dir <session> \
        --fit-dir sysid/paddle_pid/results/cmaes_20260909_paddle_motion [--trials all|val] [--width 360] [--fps 10]

Gain sets: the canonical (x0) and CMA-ES gains of ``--fit-dir`` (``fit_result.json``), plus any
``--gains LABEL KP KI KD``. Without ``--fit-dir`` pass ``--gains`` (first = reference).

Writes under ``--out`` (default ``<fit-dir>/overlays``):
  gifs/<trial>.gif               real paddle + sim ghosts + trails, one frame per step
  png/<trial>.png                static overlay with full trails
  mosaic_<condition>.png         the repeats of one condition side by side
  mosaic_val.png / mosaic_all.png
  per_trial_errors.md / .csv     every trial: split, per gain set mean / max / final error
  plots/per_trial_errors.png     dot plot per condition × repeat (val trials ringed)
  plots/per_trial_step_errors.png  per-step error curves, one panel per condition
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

from sysid.paddle_pid.code.dataset import load_session, split_train_val, session_attrs, group_by_condition
from sysid.paddle_pid.code.replay import (DEFAULT_BASE_CONFIG, PaddleReplayer, PlantParams, build_replay_sim_config,
                                         evaluate_trials, load_base_config)
from sysid.paddle_pid.code.overlay import (SceneRenderer, SIM_COLORS, COLOR_REAL, _hex_to_rgb, write_gif, write_png,
                                          mosaic, legend_strip)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", required=True, nargs="+", help="one or more session directories")
    p.add_argument("--fit-dir", default=None, help="output dir of fit_pid_cmaes.py (gains, density, split, config)")
    p.add_argument("--gains", nargs=4, action="append", metavar=("LABEL", "KP", "KI", "KD"), default=[])
    p.add_argument("--out", default=None)
    p.add_argument("--base-config", default=None)
    p.add_argument("--paddle-density", type=float, default=None)
    p.add_argument("--hist-len", type=int, default=None)
    p.add_argument("--action-delay-steps", type=int, default=None)
    p.add_argument("--split-seed", type=int, default=None)
    p.add_argument("--trials", choices=("all", "val"), default="all")
    p.add_argument("--width", type=int, default=360, help="frame width in px (native 360); 160 = repo GIF convention")
    p.add_argument("--fps", type=int, default=10, help="GIF playback (real time is 20)")
    p.add_argument("--no-gifs", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    fit = None
    if a.fit_dir:
        fit = json.load(open(Path(a.fit_dir) / "fit_result.json"))
    base_config = a.base_config or (fit["base_config"] if fit else str(DEFAULT_BASE_CONFIG))
    density = a.paddle_density if a.paddle_density is not None else (fit["paddle_density"] if fit else None)
    delay = a.action_delay_steps if a.action_delay_steps is not None else (fit["action_delay_steps"] if fit else 0)
    split_seed = a.split_seed if a.split_seed is not None else (fit["split_seed"] if fit else 0)
    hist_len = a.hist_len if a.hist_len is not None else (fit["hist_len_sim"] if fit else None)
    out = Path(a.out) if a.out else (Path(a.fit_dir) / "overlays" if a.fit_dir else _REPO_ROOT / "sysid/paddle_pid/results/overlays")
    for sub in ("gifs", "png", "plots"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    trials = load_session(a.input_dir)
    train, val, _ = split_train_val(trials, seed=split_seed)
    val_names = {t.name for t in val}
    base = load_base_config(base_config)
    sp = base["air_hockey"]["simulator_params"]
    if density is None:
        density = float(sp["paddle_density"])
    gain_sets: list[tuple[str, PlantParams]] = []
    if fit:
        x0 = fit["x0"]
        gain_sets.append(("canonical", PlantParams(x0["kp"], x0["ki"], x0["kd"], density)))
        b = fit["best"]
        gain_sets.append(("cmaes", PlantParams(b["kp"], b["ki"], b["kd"], density)))
    for label, kp, ki, kd in a.gains:
        gain_sets.append((label, PlantParams(float(kp), float(ki), float(kd), density)))
    if not gain_sets:
        raise SystemExit("pass --fit-dir and/or --gains")
    if len(gain_sets) > len(SIM_COLORS):
        raise SystemExit(f"at most {len(SIM_COLORS)} gain sets")
    names = [n for n, _ in gain_sets]

    rep = PaddleReplayer(build_replay_sim_config(base, session_attrs(trials), hist_len=hist_len))
    evals = {n: evaluate_trials(rep, trials, p, delay, keep_trajectories=True) for n, p in gain_sets}
    scene = SceneRenderer(rep)
    subset = [t for t in trials if a.trials == "all" or t.name in val_names]

    # -- per-trial products
    pngs = {}
    for t in subset:
        results = {n: evals[n]["trajectories"][t.name] for n in names}
        img = scene.trial_summary_image(t, results, a.width)
        pngs[t.name] = img
        write_png(img, out / "png" / f"{t.name}.png")
        if not a.no_gifs:
            write_gif(scene.trial_frames(t, results, a.width), out / "gifs" / f"{t.name}.gif", fps=a.fps)
    legend = legend_strip(names, width=max(im.shape[1] for im in pngs.values()) * 3 + 16)
    for cond, group in group_by_condition(subset).items():
        m = mosaic([pngs[t.name] for t in group], ncols=3)
        write_png(np.vstack([legend[:, : m.shape[1]] if legend.shape[1] >= m.shape[1] else legend, m]) if legend.shape[1] >= m.shape[1] else m,
                  out / f"mosaic_{cond}.png")
    if val:
        m = mosaic([pngs[t.name] for t in subset if t.name in val_names], ncols=3)
        write_png(m, out / "mosaic_val.png")
    if a.trials == "all":
        write_png(mosaic([pngs[t.name] for t in subset], ncols=3), out / "mosaic_all.png")

    # -- per-trial error table
    rows = []
    for t in trials:
        r = {"name": t.name, "condition": t.condition, "repeat": t.repeat, "split": "val" if t.name in val_names else "train"}
        for n in names:
            m = next(m for m in evals[n]["per_trial"] if m["name"] == t.name)
            r[f"{n}_mean"], r[f"{n}_max"], r[f"{n}_final"] = m["mean_pos_err_mm"], m["max_pos_err_mm"], m["final_pos_err_mm"]
        rows.append(r)
    with open(out / "per_trial_errors.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    L = ["# Per-trial paddle position error (mm)", "",
         "Gain sets: " + ", ".join(f"**{n}** kp {p.kp:.0f} / ki {p.ki:.0f} / kd {p.kd:.1f}" for n, p in gain_sets)
         + f"; paddle_density {density:.0f}; action delay {delay} steps. Mean / max / final over steps k ≥ 1.", ""]
    hdr = "| trial | split | " + " | ".join(f"{n} mean | {n} max | {n} final" for n in names)
    if len(names) >= 2:
        hdr += f" | Δ mean ({names[1]} − {names[0]})"
    L += [hdr + " |", "|" + "---|" * (hdr.count("|"))]
    for r in rows:
        line = f"| {r['name']} | {r['split']} | " + " | ".join(f"{r[f'{n}_mean']:.1f} | {r[f'{n}_max']:.1f} | {r[f'{n}_final']:.1f}" for n in names)
        if len(names) >= 2:
            line += f" | {r[f'{names[1]}_mean'] - r[f'{names[0]}_mean']:+.1f}"
        L.append(line + " |")
    for split in ("train", "val"):
        sub = [r for r in rows if r["split"] == split]
        L.append(f"| **{split} mean** ({len(sub)}) | | " + " | ".join(
            f"**{np.mean([r[f'{n}_mean'] for r in sub]):.1f}** | {np.mean([r[f'{n}_max'] for r in sub]):.1f} | {np.mean([r[f'{n}_final'] for r in sub]):.1f}" for n in names)
            + (f" | {np.mean([r[f'{names[1]}_mean'] - r[f'{names[0]}_mean'] for r in sub]):+.1f}" if len(names) >= 2 else "") + " |")
    (out / "per_trial_errors.md").write_text("\n".join(L) + "\n")

    # -- plots
    plot_per_trial_errors(rows, names, out / "plots" / "per_trial_errors.png")
    plot_per_trial_step_errors(trials, evals, names, val_names, out / "plots" / "per_trial_step_errors.png")
    print(f"wrote {out}: {len(subset)} trials ({'gifs, ' if not a.no_gifs else ''}png, mosaics), per-trial table + plots")


def plot_per_trial_errors(rows, names, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conds = list(dict.fromkeys(r["condition"] for r in rows))
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(conds) + 1.5))
    colors = [_hex_to_rgb(c) for c in SIM_COLORS]
    for ci, cond in enumerate(conds):
        group = [r for r in rows if r["condition"] == cond]
        for ri, r in enumerate(group):
            y = ci + (ri - (len(group) - 1) / 2) * 0.22
            for ni, n in enumerate(names):
                ax.plot(r[f"{n}_mean"], y, marker="o", ms=5, color=colors[ni], ls="none",
                        mec="black" if r["split"] == "val" else colors[ni], mew=1.2 if r["split"] == "val" else 0.5,
                        label=n if (ci == 0 and ri == 0) else None)
            if len(names) >= 2:
                ax.plot([r[f"{names[0]}_mean"], r[f"{names[1]}_mean"]], [y, y], color="#c3c2b7", lw=0.8, zorder=0)
    ax.set_yticks(range(len(conds))); ax.set_yticklabels(conds); ax.invert_yaxis()
    ax.set_xlabel("mean per-step position error (mm) — each dot one trial; black ring = validation trial")
    ax.grid(axis="x", color="#e6e5e1"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_per_trial_step_errors(trials, evals, names, val_names, path, ncols=3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = group_by_condition(trials)
    n = len(groups); nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 2.3 * nrows), squeeze=False, sharey=True)
    colors = [_hex_to_rgb(c) for c in SIM_COLORS]
    for i, (cond, group) in enumerate(groups.items()):
        ax = axes[i // ncols][i % ncols]
        for t in group:
            isval = t.name in val_names
            for ni, nm in enumerate(names):
                e = evals[nm]["trajectories"][t.name].pos_err_mm
                ax.plot(np.arange(len(e)), e, color=colors[ni], lw=2.0 if isval else 1.0, alpha=1.0 if isval else 0.55,
                        label=f"{nm}{' (val)' if isval else ''}" if (i == 0 and (isval or t.repeat == 1)) else None)
        ax.set_title(cond, fontsize=9, loc="left"); ax.grid(color="#e6e5e1")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        if i % ncols == 0:
            ax.set_ylabel("error (mm)")
        if i // ncols == nrows - 1:
            ax.set_xlabel("step")
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False)
    fig.suptitle("Per-step position error, every trial (thick = validation trial, thin = training trials)", fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98)); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()

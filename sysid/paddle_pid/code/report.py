"""Plots and markdown summary for a paddle PID fit."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

# Fixed categorical order (dataviz reference palette): real, sim baseline, sim fitted, extra.
C_REAL, C_BASE, C_FIT, C_EXTRA = "#0b0b0b", "#eb6834", "#2a78d6", "#1baf7a"
C_MUTED = "#52514e"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "font.size": 9})
    return plt


def write_candidates_csv(records, path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["restart", "generation", "kp", "ki", "kd", "train_err_mm", "val_err_mm"])
        for r in records:
            w.writerow([r.restart, r.generation, f"{r.kp:.4f}", f"{r.ki:.4f}", f"{r.kd:.4f}",
                        f"{r.train_err_mm:.4f}", f"{r.val_err_mm:.4f}"])


def write_per_trial_csv(evals: dict[str, dict], path):
    """``evals``: {"baseline_train": eval dict, "fitted_val": ...} → one row per (label, trial)."""
    keys = ("mean_pos_err_mm", "rms_pos_err_mm", "max_pos_err_mm", "final_pos_err_mm",
            "active_mean_pos_err_mm", "mean_delta_err_mm", "n_steps")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "split", "name", "condition", "repeat", "trial_type", *keys])
        for label, ev in evals.items():
            gains, split = label.rsplit("_", 1)
            for m in ev["per_trial"]:
                w.writerow([gains, split, m["name"], m["condition"], m["repeat"], m["trial_type"],
                            *[f"{m[k]:.4f}" if isinstance(m[k], float) else m[k] for k in keys]])


def plot_convergence(gen_log, out_path):
    plt = _plt()
    if not gen_log:
        return
    x = [g["n_evaluations"] for g in gen_log]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(x, [g["gen_median_train_err_mm"] for g in gen_log], color=C_MUTED, lw=1.2, label="generation median (train)")
    ax.plot(x, [g["gen_best_train_err_mm"] for g in gen_log], color=C_FIT, lw=1.2, alpha=0.5, label="generation best (train)")
    ax.plot(x, [g["best_so_far_train_err_mm"] for g in gen_log], color=C_FIT, lw=2, label="best so far (train)")
    ax.plot(x, [g["best_so_far_val_err_mm"] for g in gen_log], color=C_BASE, lw=2, ls="--", label="best so far (val)")
    restarts = sorted({g["restart"] for g in gen_log})
    for r in restarts[1:]:
        ax.axvline(min(g["n_evaluations"] for g in gen_log if g["restart"] == r), color=C_MUTED, lw=0.8, ls=":")
    ax.set_xlabel("candidates evaluated")
    ax.set_ylabel("mean per-step position error (mm)")
    ax.set_title("CMA-ES convergence")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_candidates(records, best, baseline, out_path):
    """Every evaluated candidate in (kp, kd) and (kp, ki), coloured by training error."""
    plt = _plt()
    if not records:
        return
    kp = np.array([r.kp for r in records]); ki = np.array([r.ki for r in records]); kd = np.array([r.kd for r in records])
    tr = np.array([r.train_err_mm for r in records])
    ok = np.isfinite(tr)
    vmax = np.percentile(tr[ok], 90) if ok.any() else None
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, y, ylab, ylog, by, bb in ((axes[0], kd, "kd", False, best.kd, baseline.kd),
                                      (axes[1], ki, "ki", False, best.ki, baseline.ki)):
        sc = ax.scatter(kp[ok], y[ok], c=tr[ok], cmap="Blues_r", s=12, vmin=tr[ok].min(), vmax=vmax, edgecolors="none")
        ax.scatter([baseline.kp], [bb], marker="s", s=70, facecolors="none", edgecolors=C_BASE, lw=1.8, label="canonical")
        ax.scatter([best.kp], [by], marker="*", s=140, color=C_FIT, edgecolors="white", lw=0.6, label="CMA-ES best")
        ax.set_xscale("log"); ax.set_xlabel("kp")
        if ylog:
            ax.set_yscale("log")
        else:
            ax.set_yscale("symlog", linthresh=10.0)
        ax.set_ylim(bottom=0.0)
        ax.set_ylabel(ylab)
        ax.legend(frameon=False, loc="best")
    cb = fig.colorbar(sc, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("train error (mm), clipped at p90")
    fig.suptitle("Evaluated candidates")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_condition(evals: dict[str, dict], out_path):
    """Grouped bars: per-condition mean error, baseline vs fitted, train and val side by side."""
    plt = _plt()
    conds = list(evals["baseline_train"]["per_condition"].keys())
    for k in evals.values():
        for c in k["per_condition"]:
            if c not in conds:
                conds.append(c)
    fig, axes = plt.subplots(2, 1, figsize=(max(8, 0.42 * len(conds)), 6.5), sharex=True)
    for ax, split in zip(axes, ("train", "val")):
        b = [evals[f"baseline_{split}"]["per_condition"].get(c, {}).get("mean_pos_err_mm", np.nan) for c in conds]
        f = [evals[f"fitted_{split}"]["per_condition"].get(c, {}).get("mean_pos_err_mm", np.nan) for c in conds]
        x = np.arange(len(conds)); w = 0.38
        ax.bar(x - w / 2, b, w, color=C_BASE, label="canonical gains")
        ax.bar(x + w / 2, f, w, color=C_FIT, label="CMA-ES gains")
        ax.set_ylabel("mean per-step error (mm)")
        ax.set_title(f"{split}: canonical {evals[f'baseline_{split}']['mean_pos_err_mm']:.1f} mm → "
                     f"fitted {evals[f'fitted_{split}']['mean_pos_err_mm']:.1f} mm", loc="left")
        ax.legend(frameon=False)
    axes[1].set_xticks(np.arange(len(conds)))
    axes[1].set_xticklabels(conds, rotation=60, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_trajectories(trials, traj_sets: dict[str, dict], out_path, ncols: int = 3, extra_label=None):
    """One panel per trial: real x(k), y(k) vs the sim replays (baseline / fitted / extra)."""
    plt = _plt()
    n = len(trials)
    if n == 0:
        return
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.6 * nrows), squeeze=False)
    colors = {"baseline": C_BASE, "fitted": C_FIT, "extra": C_EXTRA}
    labels = {"baseline": "sim, canonical gains", "fitted": "sim, CMA-ES gains", "extra": extra_label or "sim, extra"}
    for i, t in enumerate(trials):
        ax = axes[i // ncols][i % ncols]
        k = np.arange(t.n_steps)
        ax.plot(k, t.pose[:, 0], color=C_REAL, lw=2, label="real x")
        ax.plot(k, t.pose[:, 1], color=C_REAL, lw=2, ls="--", label="real y")
        for key, res_map in traj_sets.items():
            r = res_map.get(t.name)
            if r is None:
                continue
            ax.plot(k, r.sim_pose[:, 0], color=colors[key], lw=1.4, label=f"{labels[key]} x")
            ax.plot(k, r.sim_pose[:, 1], color=colors[key], lw=1.4, ls="--", label=f"{labels[key]} y")
        errs = " / ".join(f"{key[:4]} {np.mean(res_map[t.name].pos_err_mm[1:]):.0f}" for key, res_map in traj_sets.items()
                          if t.name in res_map)
        ax.set_title(f"{t.condition} (trial {t.repeat}) — err mm: {errs}", fontsize=8, loc="left")
        ax.set_xlabel("step"); ax.set_ylabel("robot-frame position (m)")
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    handles, labels_ = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=min(6, len(handles)), frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_step_errors(trials, traj_sets: dict[str, dict], out_path):
    """Mean per-step error vs step index, averaged over the given trials, per gain set."""
    plt = _plt()
    colors = {"baseline": C_BASE, "fitted": C_FIT, "extra": C_EXTRA}
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for key, res_map in traj_sets.items():
        rows = [res_map[t.name].pos_err_mm for t in trials if t.name in res_map]
        if not rows:
            continue
        # Trials may differ in length (e.g. 20-step and 44-step jerk trials): pad with NaN and use nanmean.
        n = max(len(r) for r in rows)
        errs = np.full((len(rows), n), np.nan)
        for i, r in enumerate(rows):
            errs[i, :len(r)] = r
        ax.plot(np.arange(n), np.nanmean(errs, axis=0), color=colors[key], lw=2,
                label=f"{key} (mean {np.nanmean(errs[:, 1:]):.1f} mm)")
    ax.set_xlabel("step k"); ax.set_ylabel("position error (mm), mean over trials")
    ax.set_title("Per-step error profile")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def summary_markdown(ctx: dict) -> str:
    """``ctx`` keys: dataset, split_info, sim, baseline (PlantParams), fitted, evals, cma (json),
    percentile (PercentileReport or None), extra (optional dict label→(params, evals))."""
    e = ctx["evals"]
    b, f = ctx["baseline"], ctx["fitted"]
    L = [f"# Paddle PID sysid — CMA-ES fit", "",
         f"- **Data**: `{ctx['dataset']}` — {ctx['split_info']['n_train']} train / {ctx['split_info']['n_val']} val trials "
         f"({len(ctx['split_info']['conditions'])} conditions, one trial per condition held out; split seed {ctx['split_info']['seed']}"
         + (f", val repeat {ctx['split_info']['val_repeat']}" if ctx['split_info'].get('val_repeat') else "") + ")",
         f"- **Sim**: `{ctx['sim']['base_config']}`, dt {ctx['sim']['dt']:.4f} s, hist_len {ctx['sim']['hist_len']}, "
         f"paddle_density {ctx['sim']['paddle_density']:.0f} (mass {ctx['sim']['paddle_mass']:.2f} kg, fixed), "
         f"action delay {ctx['sim']['action_delay_steps']} steps",
         f"- **Metric**: mean per-step paddle position error ‖sim − real‖ (mm), steps k ≥ 1, mean over trials",
         f"- **Search**: CMA-ES popsize {ctx['cma']['popsize']}, sigma0 {ctx['cma']['sigma0']}, {ctx['cma']['restarts']} restart(s), "
         f"{ctx['cma']['n_evaluations']} candidates, {ctx['cma']['wall_seconds']:.0f} s wall; bounds kp {ctx['cma']['bounds']['kp']}, "
         f"ki [0, {ctx['cma']['bounds']['ki_max']}], kd [0, {ctx['cma']['bounds']['kd_max']}]", "",
         "## Result", "",
         "| gains | kp | ki | kd | train err (mm) | val err (mm) | val rms | val max | val final | val Δ-step |",
         "|---|---|---|---|---|---|---|---|---|---|"]

    def row(label, p, tr, va):
        return (f"| {label} | {p.kp:.0f} | {p.ki:.0f} | {p.kd:.1f} | {tr['mean_pos_err_mm']:.2f} | **{va['mean_pos_err_mm']:.2f}** | "
                f"{va['rms_pos_err_mm']:.2f} | {va['max_pos_err_mm']:.1f} | {va['final_pos_err_mm']:.1f} | {va['mean_delta_err_mm']:.2f} |")
    L.append(row("canonical (x0)", b, e["baseline_train"], e["baseline_val"]))
    L.append(row("**CMA-ES**", f, e["fitted_train"], e["fitted_val"]))
    for label, (p, ev_tr, ev_va) in (ctx.get("extra") or {}).items():
        L.append(row(label, p, ev_tr, ev_va))
    gain = 100 * (1 - e["fitted_val"]["mean_pos_err_mm"] / e["baseline_val"]["mean_pos_err_mm"])
    L += ["", f"Validation error {e['baseline_val']['mean_pos_err_mm']:.2f} → {e['fitted_val']['mean_pos_err_mm']:.2f} mm "
              f"(**{gain:+.1f} %** relative to canonical; negative = worse).", ""]
    pr = ctx.get("percentile")
    if pr is not None and pr.n_candidates > 0:
        L += ["## Is the selection validated?", "",
              f"Validation error of the selected gains vs every candidate CMA-ES evaluated ({pr.n_candidates}): the selection beats "
              f"**{100 * pr.selected_beats:.0f} %** of the candidates and reaches {100 * pr.selected_fraction_of_best:.0f} % of the "
              f"validation oracle ({pr.oracle_err:.2f} mm at kp={pr.oracle_params['kp']:.0f}, ki={pr.oracle_params['ki']:.0f}, "
              f"kd={pr.oracle_params['kd']:.1f}); the canonical gains beat {100 * (pr.canonical_beats or 0):.0f} %. "
              f"Median candidate {pr.median_err:.2f} mm, p90 candidate {pr.percentiles[90]['err']:.2f} mm "
              f"({100 * pr.percentiles[90]['fraction_of_best']:.0f} % of oracle).", ""]
    ps = ctx.get("per_session") or {}
    if ps:
        L += ["## Per session (mean per-step error, mm)", "",
              "| session | trials train / val | canonical train | canonical val | CMA-ES train | CMA-ES val |",
              "|---|---|---|---|---|---|"]
        fmt = lambda v: "–" if v is None else f"{v:.2f}"
        for s_name, r in ps.items():
            L.append(f"| `{s_name}` | {r['n_train']} / {r['n_val']} | {fmt(r['canonical_train'])} | {fmt(r['canonical_val'])} | "
                     f"{fmt(r['fitted_train'])} | **{fmt(r['fitted_val'])}** |")
        L.append("")
    L += ["## Per condition (val trials, mm)", "", "| condition | canonical | CMA-ES | Δ |", "|---|---|---|---|"]
    for c in e["baseline_val"]["per_condition"]:
        bv = e["baseline_val"]["per_condition"][c]["mean_pos_err_mm"]
        fv = e["fitted_val"]["per_condition"].get(c, {}).get("mean_pos_err_mm", np.nan)
        L.append(f"| {c} | {bv:.1f} | {fv:.1f} | {fv - bv:+.1f} |")
    L += ["", "## Files", "", "- `fit_result.json` — best gains, search settings, per-generation log",
          "- `candidates.csv` — every evaluated (kp, ki, kd) with train and val error",
          "- `evaluations.json` / `per_trial.csv` — baseline and fitted, train and val, per trial and per condition",
          "- `sim_config_fitted.yaml` — the base sim config with the fitted gains (hist_len restored to the base value)",
          "- `plots/` — convergence, candidates, per-condition bars, validation trajectories, per-step error profile"]
    return "\n".join(L) + "\n"

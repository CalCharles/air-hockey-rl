"""CSV outputs, plots and the markdown summary for a paddle–puck collision fit."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from sysid.common.trajectory_segmentation import model_state
from sysid.paddle_pid.code.report import C_REAL, C_BASE, C_FIT, C_EXTRA, C_MUTED, _plt
from .speeds import CollisionMeasurement, SpeedConfig, measurements_table
from .sim_collision import CollisionParams

# fixed categorical order for the three release heights (identity, never cycled)
C_HEIGHT = {"top": "#2a78d6", "3/4": "#eb6834", "1/2": "#1baf7a"}
M_DELTA = {0.0: "s", 0.33: "^", 0.66: "D", 1.0: "o"}


def write_candidates_csv(records, path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["restart", "generation", "restitution", "mass_ratio", "gain", "train_err", "val_err"])
        for r in records:
            w.writerow([r.restart, r.generation, f"{r.restitution:.5f}", f"{r.mass_ratio:.4f}", f"{r.gain:.5f}",
                        f"{r.train_err:.5f}", f"{r.val_err:.5f}"])


def write_per_trial_csv(evals: dict[str, dict], path):
    keys = ("u_p", "speed_in", "speed_out_real", "speed_out_sim", "err", "rel_err", "closed_form", "n_contacts", "paddle_v_post_sim")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "split", "name", "condition", "repeat", *keys])
        for label, ev in evals.items():
            params, split = label.rsplit("_", 1)
            for m in ev["per_trial"]:
                w.writerow([params, split, m["name"], m["condition"], m["repeat"],
                            *[f"{m[k]:.5f}" if isinstance(m[k], float) else m[k] for k in keys]])


# -- plots ----------------------------------------------------------------------------------
def plot_trial_fits(trials, measurements: list[CollisionMeasurement], cfg: SpeedConfig, out_path, ncols: int = 6):
    """Puck x(t) of every file with the pre / post model, the contact time and the (lag-corrected)
    paddle pose; selected trials framed in blue, rejected ones in orange."""
    plt = _plt()
    by_name = {m.name: m for m in measurements}
    n = len(trials)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.5 * nrows), squeeze=False)
    seg = cfg.seg_cfg()
    for ax, tr in zip(axes.ravel(), trials):
        m = by_name[tr.name]
        t = tr.t - tr.t0
        ok = tr.puck_valid
        ax.plot(t[ok], tr.puck_xy[ok, 0], ".", color=C_REAL, ms=4, label="puck (usable)")
        ax.plot(t[~ok], tr.puck_xy[~ok, 0], "x", color=C_MUTED, ms=4, label="occluded / stale")
        ax.plot(tr.pad_t - tr.t0 - cfg.camera_lag_s, tr.pad_xy[:, 0], "-", color=C_MUTED, lw=1.2, label="paddle (lag-corrected)")
        ax.plot(tr.pad_t - tr.t0 - cfg.camera_lag_s, tr.pad_xy[:, 0] - cfg.contact_distance, ":", color=C_MUTED, lw=0.9)
        if np.isfinite(m.t_c):
            t0 = m.pre_fit["t0"]
            fpre = {"p0": np.array(m.pre_fit["p0"]), "u": np.array(m.pre_fit["u"])}
            fpost = {"p0": np.array(m.post_fit["p0"]), "u": np.array(m.post_fit["u"])}
            tt = np.linspace(tr.t[m.pre_idx[0]], tr.t0 + m.t_c, 40)
            ax.plot(tt - tr.t0, [model_state(fpre, s - t0, seg)[0][0] for s in tt], "-", color=C_FIT, lw=2, label="pre fit")
            tt = np.linspace(tr.t0 + m.t_c, tr.t[m.post_idx[-1]], 40)
            ax.plot(tt - tr.t0, [model_state(fpost, s - t0, seg)[0][0] for s in tt], "-", color=C_BASE, lw=2, label="post fit")
            ax.axvline(m.t_c, color=C_EXTRA, lw=1, ls="--")
            ax.set_title(f"{tr.name.replace('collision_', '')}\n"
                         f"in {m.speed_in:.2f} out {m.speed_out:.2f} pad {m.u_p:.2f} m/s · gain {m.gain:.2f} · {m.angle_out_deg:.0f}°",
                         fontsize=7.5)
        else:
            ax.set_title(f"{tr.name.replace('collision_', '')}\n{m.reason}", fontsize=7.5)
        ax.set_xlim(-0.5, 1.2)
        ax.set_ylim(-0.95, 0.95)
        color = C_FIT if m.selected else (C_BASE if m.valid else "#b02020")
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor(color)
            sp.set_linewidth(1.6)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=6.5, loc="lower left", frameon=False)
    fig.supxlabel("time since first scripted step (s)", fontsize=9)
    fig.supylabel("puck x (observation frame, m)", fontsize=9)
    fig.suptitle("Collision fits — blue frame = selected, orange = valid but not selected, red = invalid", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_gain_vs_speed(measurements: list[CollisionMeasurement], out_path, fitted: CollisionParams = None,
                       baseline: CollisionParams = None):
    """Measured head-on speed gain (v_out + v_in) / (u_p + v_in) against the approach speed."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in measurements:
        if not np.isfinite(m.gain):
            continue
        mk = M_DELTA.get(round(m.action_delta, 2), "o")
        c = C_HEIGHT.get(m.height, C_MUTED)
        if m.selected:
            ax.plot(m.approach_speed, m.gain, mk, color=c, ms=7, mec="white", mew=0.8)
        else:
            ax.plot(m.approach_speed, m.gain, mk, color="none", mec=c, ms=7, mew=1.2)
    if fitted is not None:
        ax.axhline(fitted.gain(), color=C_FIT, lw=2, label=f"fitted gain {fitted.gain():.3f}")
    if baseline is not None:
        ax.axhline(baseline.gain(), color=C_BASE, lw=2, ls="--", label=f"canonical sim gain {baseline.gain():.3f}")
    for h, c in C_HEIGHT.items():
        ax.plot([], [], "o", color=c, label=f"release {h}")
    for d, mk in M_DELTA.items():
        ax.plot([], [], mk, color=C_MUTED, label=f"delta {d:.2f}")
    ax.plot([], [], "o", color="none", mec=C_MUTED, label="not selected")
    ax.set_xlabel("approach speed u_p + v_in (m/s)")
    ax.set_ylabel("gain (v_out + v_in) / (u_p + v_in)")
    ax.set_ylim(0.8, 2.2)
    ax.legend(fontsize=7, ncol=3, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_landscape(grid: dict, out_path, fitted: CollisionParams = None, baseline: CollisionParams = None,
                   candidates=None):
    """Train RMS over the (restitution, mass-ratio) grid with the constant-gain ridge."""
    plt = _plt()
    e = np.array(grid["restitution"]); r = np.array(grid["mass_ratio"]); z = np.array(grid["train_err"])
    fig, ax = plt.subplots(figsize=(7, 4.6))
    zc = np.clip(z, np.nanmin(z), np.nanmin(z) + 0.6)
    cs = ax.contourf(e, r, zc, levels=24, cmap="Blues_r")
    fig.colorbar(cs, ax=ax, label="train RMS of outgoing speed (m/s), clipped at min + 0.6")
    if candidates:
        ax.plot([c.restitution for c in candidates], [c.mass_ratio for c in candidates], ".", color=C_MUTED, ms=2.5, alpha=0.5, label="CMA-ES candidates")
    if fitted is not None:
        k = fitted.gain()
        ee = np.linspace(e.min(), e.max(), 200)
        rr = k / (1.0 + ee - k)
        okm = (rr > 0) & (rr >= r.min()) & (rr <= r.max())
        ax.plot(ee[okm], rr[okm], "-", color=C_FIT, lw=2, label=f"ridge (1+e)·r/(r+1) = {k:.3f}")
        ax.plot(fitted.restitution, fitted.mass_ratio, "*", color=C_FIT, ms=14, mec="white", label="CMA-ES best")
    if baseline is not None:
        ax.plot(baseline.restitution, baseline.mass_ratio, "o", color=C_BASE, ms=9, mec="white", label="canonical sim")
    ax.axhline(58.0 / 13.0, color=C_EXTRA, lw=1, ls=":", label="real weights 58 g / 13 g")
    ax.set_yscale("log")
    ax.set_xlabel("paddle–puck restitution e")
    ax.set_ylabel("mass ratio m_paddle / m_puck")
    ax.legend(fontsize=7, frameon=True, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_sim_vs_real(evals: dict[str, dict], out_path):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
    for ax, split in zip(axes, ("train", "val")):
        for label, color, mk in (("baseline", C_BASE, "o"), ("fitted", C_FIT, "o")):
            ev = evals.get(f"{label}_{split}")
            if not ev or not ev["per_trial"]:
                continue
            xs = [p["speed_out_real"] for p in ev["per_trial"]]
            ys = [p["speed_out_sim"] for p in ev["per_trial"]]
            ax.plot(xs, ys, mk, color=color, ms=6, mec="white", mew=0.6, label=f"{label} (RMS {ev['rms_err']:.3f} m/s)")
        lim = (0, 2.8)
        ax.plot(lim, lim, "-", color=C_MUTED, lw=1)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_title(f"{split} collisions", fontsize=10)
        ax.set_xlabel("real outgoing speed (m/s)")
        ax.legend(fontsize=8, frameon=False, loc="upper left")
    axes[0].set_ylabel("sim outgoing speed (m/s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_convergence(gen_log, out_path):
    plt = _plt()
    if not gen_log:
        return
    x = [g["n_evaluations"] for g in gen_log]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(x, [g["gen_median_train_err"] for g in gen_log], color=C_MUTED, lw=1.2, label="generation median (train)")
    ax.plot(x, [g["best_so_far_train_err"] for g in gen_log], color=C_FIT, lw=2, label="best so far (train)")
    ax.plot(x, [g["best_so_far_val_err"] for g in gen_log], color=C_BASE, lw=2, ls="--", label="best so far (val)")
    for r in sorted({g["restart"] for g in gen_log})[1:]:
        ax.axvline(min(g["n_evaluations"] for g in gen_log if g["restart"] == r), color=C_MUTED, lw=0.8, ls=":")
    ax.set_xlabel("candidates evaluated")
    ax.set_ylabel("RMS outgoing-speed error (m/s)")
    ax.set_yscale("log")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# -- summary --------------------------------------------------------------------------------
def summary_markdown(ctx: dict) -> str:
    base, fit, ev, lag, sel = ctx["baseline"], ctx["fitted"], ctx["evals"], ctx["lag"], ctx["selection"]
    cma, pr, grid, cfg = ctx["cma"], ctx.get("percentile"), ctx.get("grid"), ctx["speed_cfg"]
    f3 = lambda v: f"{v:.3f}" if v is not None and np.isfinite(v) else "–"
    L = ["# Paddle–puck restitution + mass ratio — CMA-ES fit", "",
         f"- dataset: `{ctx['dataset']}`", f"- sim base config: `{ctx['sim']['base_config']}` (dt {ctx['sim']['dt']:.3f} s, "
         f"paddle density {ctx['sim']['paddle_density']:.0f} → mass {ctx['sim']['paddle_mass']:.2f} kg, hist_len {ctx['sim']['hist_len']})",
         f"- speed estimation: free-flight model with gravity {cfg.gravity_x} m/s² / damping {cfg.damping} 1/s, "
         f"{cfg.pre_frames} + {cfg.post_frames} usable frames, camera lag {lag['lag_s'] * 1000:.0f} ms "
         f"(median of {lag['n_trials']} moving-paddle trials, std {lag['lag_std_s'] * 1000:.0f} ms); stationary-paddle contact gap "
         f"{lag['static_contact_gap_mean'] * 1000:.1f} ± {lag['static_contact_gap_std'] * 1000:.1f} mm vs r_paddle + r_puck = {lag['expected_contact_distance'] * 1000:.1f} mm",
         f"- canonical dataset: {ctx['n_selected']} of {ctx['n_files']} files (best {ctx['per_condition']} per condition, "
         f"{len(sel)} conditions; {ctx['n_train']} train / {ctx['n_val']} val)", ""]
    L += ["## Result", "", "| | restitution e | mass ratio r | gain (1+e)·r/(r+1) | train RMS (m/s) | val RMS (m/s) | mean err train (m/s) | mean |rel err| train |",
          "|---|---|---|---|---|---|---|---|",
          f"| canonical sim | {base.restitution:.4f} | {base.mass_ratio:.3f} | {base.gain():.4f} | {f3(ev['baseline_train']['rms_err'])} | {f3(ev['baseline_val']['rms_err'])} | {f3(ev['baseline_train']['mean_err'])} | {100 * ev['baseline_train']['mean_abs_rel_err']:.1f} % |",
          f"| **CMA-ES best** | **{fit.restitution:.4f}** | **{fit.mass_ratio:.3f}** | **{fit.gain():.4f}** | **{f3(ev['fitted_train']['rms_err'])}** | **{f3(ev['fitted_val']['rms_err'])}** | {f3(ev['fitted_train']['mean_err'])} | {100 * ev['fitted_train']['mean_abs_rel_err']:.1f} % |",
          "", f"CMA-ES: {cma['n_evaluations']} candidates, {cma['wall_seconds']:.0f} s, sigma0 {cma['sigma0']}, popsize {cma['popsize']}, restarts {cma['restarts']}.", ""]
    if ctx.get("ridge"):
        rg = ctx["ridge"]
        L += ["### The (e, r) degeneracy", "",
              "For a head-on hit the sim's collision listener produces `v_out = −v_in + (1+e)·r/(r+1)·(u_p + v_in)`, so the",
              "data only pin the **gain** `(1+e)·r/(r+1)`; every (e, r) on that ridge reproduces the puck speeds equally well.",
              "Points on the fitted ridge:", "", "| mass ratio r | restitution e on the ridge | train RMS (m/s) | val RMS (m/s) |", "|---|---|---|---|"]
        for row in rg:
            L.append(f"| {row['mass_ratio']:.3f} ({row['label']}) | {row['restitution']:.4f} | {f3(row['train_err'])} | {f3(row['val_err'])} |")
        L.append("")
    if grid:
        L += [f"Grid scan ({len(grid['restitution'])} × {len(grid['mass_ratio'])}): best e {grid['best']['restitution']:.3f}, r {grid['best']['mass_ratio']:.2f} "
              f"→ train {f3(grid['best_train_err'])}, val {f3(grid['best_val_err'])} m/s (`plots/landscape.png`).", ""]
    if pr is not None:
        from sysid.common.fit_validation import summarize_reports
        L += ["## Validation of the search", ""] + summarize_reports([pr]) + [""]
    L += ["## Per-condition RMS (m/s)", "", "| condition | n train | canonical train | fitted train | n val | canonical val | fitted val |", "|---|---|---|---|---|---|---|"]
    conds = sorted(set(ev["baseline_train"]["per_condition"]) | set(ev["baseline_val"]["per_condition"]))
    for c in conds:
        g = lambda k: ev[k]["per_condition"].get(c)
        row = [c]
        for split in ("train", "val"):
            b, f_ = g(f"baseline_{split}"), g(f"fitted_{split}")
            row += [str(b["n"]) if b else "0", f3(b["rms"]) if b else "–", f3(f_["rms"]) if f_ else "–"]
        L.append("| " + " | ".join(row) + " |")
    L += ["", "## Canonical dataset — every file", "",
          "Selected = best-3 per condition by the quality score (outgoing angle + 5/cm lateral offset + 2/10 mm fit residual + acceleration penalty); "
          "invalid trials show the gate that rejected them.", ""]
    L += measurements_table(ctx["measurements"])
    L += ["", "## Files", "", "- `canonical_dataset.csv` — the selected trials with all measured speeds; `all_trials.csv` — every file with quality / gate",
          "- `fit_result.json`, `candidates.csv`, `evaluations.json`, `per_trial.csv`, `sim_config_fitted.yaml`, `lag_calibration.json`, `selection.json`",
          "- `plots/trial_fits.png` (every fit), `plots/gain_vs_speed.png`, `plots/landscape.png`, `plots/sim_vs_real.png`, `plots/convergence.png`"]
    return "\n".join(L) + "\n"

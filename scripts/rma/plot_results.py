"""Figure for the RMA baseline report (matplotlib, PNG).

    .venv/bin/python -m scripts.rma.plot_results \
        --phase1-dirs runs/rma/juggle_dr_phase1_seed0 runs/rma/juggle_dr_phase1_seed1 \
        --baseline-dirs runs/td3/tasks_20260904/dr/juggle_dr \
        --phase2-dirs runs/rma/juggle_dr_phase1_seed0/phase2 runs/rma/juggle_dr_phase1_seed1/phase2 \
        --out runs/rma/rma_results.png

Panels: (a) phase-1 eval return per checkpoint, privileged RMA vs plain-DR TD3;
(b) phase-2 latent MSE per iteration (validation vs pre-update on-policy);
(c) phase-2 validation R² of ẑ and of the parameter probe; (d) final paired
policy evaluation, ID and OOD. The tables of `scripts/rma/summarize.py` are
the table view of the same numbers.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from scripts.rma.summarize import _load_json, checkpoint_curve  # noqa: E402
import json as _json  # noqa: E402

# Validated light categorical palette (dataviz skill reference instance), fixed order.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
MAGENTA = "#e87ba4"
AGENT_COLORS = {"adapted": BLUE, "privileged": AQUA, "nominal": YELLOW, "td3_dr": ORANGE, "history": MAGENTA}
AGENT_LABELS = {"adapted": "RMA adapted (deployable)", "privileged": "RMA privileged (oracle z)", "nominal": "RMA nominal (z = μ(0))", "td3_dr": "plain-DR TD3 (5-frame obs)", "history": "long-history TD3 (50-step window, end-to-end)"}


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_title(title, loc="left", fontsize=10, color=INK, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=8, color=INK2)
    ax.set_ylabel(ylabel, fontsize=8, color=INK2)


def _rows(phase2_dir: str) -> List[dict]:
    try:
        with open(os.path.join(phase2_dir, "phase2_metrics.jsonl")) as f:
            return [json.loads(ln) for ln in f if ln.strip()]
    except OSError:
        return []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase1-dirs", nargs="*", default=[])
    ap.add_argument("--baseline-dirs", nargs="*", default=[])
    ap.add_argument("--phase2-dirs", nargs="*", default=[])
    ap.add_argument("--paired-eval", default=None, help="dir with paired_eval.json / paired_eval_ood.json (scripts.rma.eval_paired); replaces panel (d)")
    ap.add_argument("--out", required=True)
    cli = ap.parse_args()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6), facecolor=SURFACE)
    fig.subplots_adjust(wspace=0.35, left=0.05, right=0.99, bottom=0.36, top=0.88)
    below = dict(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)

    # (a) phase-1 curves
    ax = axes[0]
    styles = ["-", "--", ":"]
    n_priv = n_hist = 0
    for d in cli.phase1_dirs:
        c = checkpoint_curve(d)
        if not c:
            continue
        meta = _load_json(os.path.join(d, "rma_meta.json")) or {}
        if meta.get("mode") == "history":
            ax.plot(np.array(list(c)) / 1e6, list(c.values()), styles[n_hist % 3], color=MAGENTA, linewidth=2, label=f"long-history TD3 ({os.path.basename(d.rstrip('/'))})")
            n_hist += 1
        else:
            ax.plot(np.array(list(c)) / 1e6, list(c.values()), styles[n_priv % 3], color=BLUE, linewidth=2, label=f"RMA privileged ({os.path.basename(d.rstrip('/'))})")
            n_priv += 1
    for i, d in enumerate(cli.baseline_dirs):
        c = checkpoint_curve(d)
        if c:
            ax.plot(np.array(list(c)) / 1e6, list(c.values()), styles[i % 3], color=ORANGE, linewidth=2, label=f"plain-DR TD3 ({os.path.basename(d.rstrip('/'))})")
    _style(ax, "(a) Phase 1: eval return per checkpoint", "env steps (M)", "mean return, 5 fixed DR envs")
    ax.legend(**below)

    # (b) + (c) phase-2 fit
    ax_mse, ax_r2 = axes[1], axes[2]
    for i, d in enumerate(cli.phase2_dirs):
        rows = _rows(d)
        if not rows:
            continue
        it = [r["iteration"] for r in rows]
        tag = os.path.basename(os.path.dirname(d.rstrip("/"))) if os.path.basename(d.rstrip("/")) == "phase2" else os.path.basename(d.rstrip("/"))
        ax_mse.plot(it, [r.get("fit/val_mse", np.nan) for r in rows], styles[i % 3], color=BLUE, linewidth=2, label=f"validation MSE ({tag})")
        ax_mse.plot(it, [r.get("collect/online_latent_mse", np.nan) for r in rows], styles[i % 3], color=ORANGE, linewidth=2, label=f"on-policy MSE, pre-update ({tag})")
        ax_r2.plot(it, [r.get("fit/val_r2_mean", np.nan) for r in rows], styles[i % 3], color=BLUE, linewidth=2, label=f"R² of ẑ vs z ({tag})")
        ax_r2.plot(it, [r.get("probe/val_r2_mean", np.nan) for r in rows], styles[i % 3], color=AQUA, linewidth=2, label=f"probe R², physics params from ẑ ({tag})")
    _style(ax_mse, "(b) Phase 2: latent regression error", "iteration (20k on-policy steps each)", "MSE in z units")
    _style(ax_r2, "(c) Phase 2: explained variance", "iteration", "R² (validation episodes)")
    ax_r2.set_ylim(min(-0.2, ax_r2.get_ylim()[0]), 1.02)
    ax_mse.legend(**below)
    ax_r2.legend(**below)

    # (d) final paired eval
    ax = axes[3]
    summaries = [(d, _load_json(os.path.join(d, "phase2_summary.json"))) for d in cli.phase2_dirs]
    summaries = [(d, s) for d, s in summaries if s]
    if cli.paired_eval:
        # One synthetic "summary" per agent kind built from the paired eval (agents named kind:label).
        pe = {"id": _load_json(os.path.join(cli.paired_eval, "paired_eval.json")), "ood": _load_json(os.path.join(cli.paired_eval, "paired_eval_ood.json"))}
        pe = {k: v for k, v in pe.items() if v}
        kinds = sorted({n.split(":")[0] for n in next(iter(pe.values()))["agents"]}, key=lambda k: ["adapted", "privileged", "nominal", "history", "td3_dr"].index(k) if k in AGENT_COLORS else 99)
        summaries = []
        max_seeds = max(sum(n.startswith(k + ":") for n in next(iter(pe.values()))["agents"]) for k in kinds)
        for si in range(max_seeds):
            fake = {"policy_eval": {}}
            for split, res in pe.items():
                fake["policy_eval"][split] = {}
                for k in kinds:
                    names = [n for n in res["agents"] if n.startswith(k + ":")]
                    if si < len(names):
                        agg = res["agents"][names[si]]["aggregate"]
                        fake["policy_eval"][split][k] = {"mean_return": agg["mean_return_across_envs"], "return_sem": agg["return_sem_across_episodes"]}
            summaries.append((f"seed{si}", fake))
    if summaries:
        agents = [a for a in ("adapted", "privileged", "nominal", "history", "td3_dr") if any(a in s["policy_eval"].get("id", {}) for _, s in summaries)]
        splits = [sp for sp in ("id", "ood") if any(sp in s["policy_eval"] for _, s in summaries)]
        width = 0.8 / max(len(agents), 1)
        for ai, agent in enumerate(agents):
            for si, split in enumerate(splits):
                vals = [s["policy_eval"][split][agent]["mean_return"] for _, s in summaries if agent in s["policy_eval"].get(split, {})]
                sems = [s["policy_eval"][split][agent]["return_sem"] for _, s in summaries if agent in s["policy_eval"].get(split, {})]
                if not vals:
                    continue
                if not vals:
                    continue
                x = si + (ai - (len(agents) - 1) / 2) * width
                m = float(np.mean(vals))
                err = float(np.sqrt(np.mean(np.square(sems)))) if len(vals) == 1 else float(np.std(vals))
                ax.bar(x, m, width * 0.9, color=AGENT_COLORS[agent], label=AGENT_LABELS[agent] if si == 0 else None, edgecolor=SURFACE, linewidth=1)
                ax.errorbar(x, m, yerr=err, color=INK2, linewidth=1, capsize=2)
                ax.text(x, m + err + 1, f"{m:.0f}", ha="center", va="bottom", fontsize=7, color=INK)
        ax.set_xticks(range(len(splits)))
        ax.set_xticklabels([{"id": "in-distribution (±25 %)", "ood": "out-of-distribution (2–2.5×)"}[s] for s in splits])
        ax.legend(**{**below, "ncol": 2})
    _style(ax, "(d) Final paired policy eval", "", "mean return (paired episodes, 5 envs)")
    if len(summaries) > 1:
        ax.text(0.0, -0.13, "bars = mean over seeds, error = std across seeds (SEM if one seed)", transform=ax.transAxes, fontsize=7, color=INK2)

    os.makedirs(os.path.dirname(os.path.abspath(cli.out)), exist_ok=True)
    fig.savefig(cli.out, dpi=150, facecolor=SURFACE)
    print(f"wrote {cli.out}")


if __name__ == "__main__":
    main()

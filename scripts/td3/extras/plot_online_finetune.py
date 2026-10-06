"""Plot online real-robot fine-tuning curves (return and critic loss vs episodes).

Reads every ``hist<H>/seed<S>/online_progress.jsonl`` under an experiment folder
written by ``scripts/td3/td3_online_real_finetune.py`` (rows from resumed
launches are already appended in order) and draws two panels:

* **Return** at x = i: the return of the rollout made by the policy after i
  training rounds (x = 0 is the unchanged sim policy). Warm-start episodes, if
  any, and episodes excluded for an e-stop are not on this curve.
* **Critic loss** at x = i: the mean per-critic TD loss (h-transformed MSE) over
  the K updates of training round i, the round after rollout i - 1.

One thin line per hist / seed (colour = hist, marker = seed); once a hist has
two or more seeds, a bold mean ± standard-error band is added.

    python -m scripts.td3.extras.plot_online_finetune \\
        --exp-dir real_runs/td3_online_real_finetune/no_warmup_no_sim_data

Writes ``<exp-dir>/plots/online_finetune.png`` (override with ``--out``) and
prints a per-curve summary table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Categorical slots 1 / 2 (blue, orange); further hist values take the next slots.
_HIST_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
_SEED_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
_INK = "#3d3d3a"
_GRID = "#e5e4de"


def load_curves(exp_dir: Path) -> dict[tuple[int, int], list[dict]]:
    curves: dict[tuple[int, int], list[dict]] = {}
    for path in sorted(exp_dir.glob("hist*/seed*/online_progress.jsonl")):
        with open(path, "r") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        if rows:
            hist = int(path.parent.parent.name.removeprefix("hist"))
            seed = int(path.parent.name.removeprefix("seed"))
            curves[(hist, seed)] = rows
    return curves


def return_series(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    points = [(r["policy_episodes_trained"], r["episode_return"]) for r in rows
              if not r.get("warm_start") and not r.get("excluded")]
    return _as_xy(points)


def loss_series(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    points = [(r["episodes_trained_after"], r["critic_loss_mean"]) for r in rows if r.get("trained")]
    return _as_xy(points)


def _as_xy(points: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    # A resumed launch can re-log an x already on disk (crash between the log
    # and the checkpoint); keep the latest value per x.
    by_x = {int(x): float(y) for x, y in points}
    xs = np.array(sorted(by_x), dtype=float)
    return xs, np.array([by_x[int(x)] for x in xs], dtype=float)


def _mean_se(series: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean ± SE over the seeds that reached each x."""
    xs = sorted({int(x) for sx, _ in series for x in sx})
    means, ses, kept_x = [], [], []
    for x in xs:
        values = [sy[sx == x][0] for sx, sy in series if np.any(sx == x)]
        if len(values) < 2:
            continue
        kept_x.append(x)
        means.append(np.mean(values))
        ses.append(np.std(values, ddof=1) / np.sqrt(len(values)))
    return np.array(kept_x, float), np.array(means), np.array(ses)


def plot(curves: dict[tuple[int, int], list[dict]], out_path: Path, title: str) -> None:
    hists = sorted({h for h, _ in curves})
    seeds = sorted({s for _, s in curves})
    color = {h: _HIST_COLORS[i % len(_HIST_COLORS)] for i, h in enumerate(hists)}
    marker = {s: _SEED_MARKERS[i % len(_SEED_MARKERS)] for i, s in enumerate(seeds)}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    panels = [
        (axes[0], return_series, "Return", "Episodes trained on (policy after i rounds)"),
        (axes[1], loss_series, "Critic loss (mean over round's K updates)", "Training rounds completed"),
    ]
    for ax, series_fn, ylabel, xlabel in panels:
        for hist in hists:
            per_seed = []
            for (h, seed), rows in sorted(curves.items()):
                if h != hist:
                    continue
                xs, ys = series_fn(rows)
                if xs.size == 0:
                    continue
                per_seed.append((xs, ys))
                ax.plot(
                    xs, ys,
                    color=color[hist], linewidth=1.2, alpha=0.55 if len(seeds) > 1 else 0.9,
                    marker=marker[seed], markersize=5, markeredgecolor="white", markeredgewidth=0.8,
                    label=f"hist{hist} seed{seed}",
                )
            if len(per_seed) >= 2:
                mx, my, se = _mean_se(per_seed)
                if mx.size:
                    ax.fill_between(mx, my - se, my + se, color=color[hist], alpha=0.15, linewidth=0)
                    ax.plot(mx, my, color=color[hist], linewidth=2.5, label=f"hist{hist} mean ± SE")
        ax.set_xlabel(xlabel, color=_INK)
        ax.set_ylabel(ylabel, color=_INK)
        ax.grid(True, color=_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(_GRID)
        ax.tick_params(colors=_INK)
        ax.xaxis.get_major_locator().set_params(integer=True)
    axes[1].set_yscale("log")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, fontsize=8, loc="outside lower center", ncol=min(len(labels), 4))
    fig.suptitle(title, color=_INK)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def summarize(curves: dict[tuple[int, int], list[dict]]) -> str:
    lines = [
        "| curve | rollouts | rounds trained | return @0 (sim) | mean return first 5 | mean return last 5 | critic loss last |",
        "|---|---|---|---|---|---|---|",
    ]
    for (hist, seed), rows in sorted(curves.items()):
        rx, ry = return_series(rows)
        lx, ly = loss_series(rows)
        r0 = f"{ry[0]:.1f}" if rx.size and rx[0] == 0 else "–"
        first = f"{ry[:5].mean():.1f}" if ry.size else "–"
        last = f"{ry[-5:].mean():.1f}" if ry.size else "–"
        loss = f"{ly[-1]:.4f}" if ly.size else "–"
        lines.append(f"| hist{hist} seed{seed} | {rx.size} | {int(lx[-1]) if lx.size else 0} | {r0} | {first} | {last} | {loss} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp-dir", required=True, type=Path, help="<data_root_dir>/<experiment_name>")
    parser.add_argument("--out", type=Path, default=None, help="PNG path (default <exp-dir>/plots/online_finetune.png)")
    cli = parser.parse_args()
    curves = load_curves(cli.exp_dir)
    if not curves:
        raise SystemExit(f"No hist*/seed*/online_progress.jsonl under {cli.exp_dir}")
    out_path = cli.out or cli.exp_dir / "plots" / "online_finetune.png"
    plot(curves, out_path, title=f"Online fine-tuning on the real robot — {cli.exp_dir.name}")
    print(summarize(curves))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

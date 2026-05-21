#!/usr/bin/env python3
"""
Fine 3D grid search (kp x kd x density) with windowed reset every 20 frames.

Instead of running the entire 100-frame segment open-loop, resets the sim to
the real paddle state every 20 frames and seeds PID state to avoid transients.
This measures per-step fidelity rather than long-horizon drift.
"""

from __future__ import annotations

import copy
import itertools
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.visualization.replay_real_in_sim import (
    load_sim_config,
    replay_errors_windowed,
)

from scripts.sysid._sysid_paths import (
    DEFAULT_CONFIG,
    SYSID_DIR,
    load_subset,
)


SUBSET_TRAJECTORIES_DEFAULT = [
    "circle_fast/frames_2235_2335/segment_2235_2335.hdf5",
    "circle_slow/frames_2680_2780/segment_2680_2780.hdf5",
    "side_to_side_dynamic/frames_650_750/segment_650_750.hdf5",
    "side_to_side_slow/frames_325_425/segment_325_425.hdf5",
    "up_and_down_dynamic/frames_1590_1690/segment_1590_1690.hdf5",
    "up_and_down_slow/frames_1275_1375/segment_1275_1375.hdf5",
    "diagonal_fast/frames_1890_1990/segment_1890_1990.hdf5",
    "random/frames_3225_3325/segment_3225_3325.hdf5",
]

SUBSET_TRAJECTORIES = load_subset(SUBSET_TRAJECTORIES_DEFAULT)

RESET_INTERVAL = 10


def run_3d_grid_search(kp_values, kd_values, density_values):
    episode_paths = [str(SYSID_DIR / t) for t in SUBSET_TRAJECTORIES]
    for p in episode_paths:
        if not Path(p).exists():
            raise FileNotFoundError(p)

    base_cfg = load_sim_config(str(DEFAULT_CONFIG), enable_noise=False)
    combos = list(itertools.product(kp_values, kd_values, density_values))
    n_combos = len(combos)
    print(f"Fine grid (windowed reset every {RESET_INTERVAL} frames):")
    print(f"  {len(kp_values)} kp x {len(kd_values)} kd x {len(density_values)} density = {n_combos} combos")
    print(f"  Trajectories: {len(episode_paths)}")
    print(f"  Total runs: {n_combos * len(episode_paths)}\n")

    results = {}
    t0 = time.time()
    for idx, (kp, kd, density) in enumerate(combos):
        cfg = copy.deepcopy(base_cfg)
        cfg["simulator_params"]["pid_kp"] = kp
        cfg["simulator_params"]["pid_kd"] = kd
        cfg["simulator_params"]["paddle_density"] = density

        paddle_errs = []
        for ep_path in episode_paths:
            r = replay_errors_windowed(
                ep_path, cfg,
                reset_interval=RESET_INTERVAL,
                park_puck=True,
            )
            paddle_errs.append(r["mean_paddle_err"])

        avg_paddle = float(np.mean(paddle_errs))
        results[(kp, kd, density)] = {
            "mean_paddle_err": avg_paddle,
            "per_traj_paddle": paddle_errs,
        }
        elapsed = time.time() - t0
        print(
            f"[{idx+1:3d}/{n_combos}] kp={kp:6.0f} kd={kd:6.0f} density={density:5.0f}  "
            f"paddle={avg_paddle:.4f} m  ({elapsed:.1f}s)"
        )

    return results


def print_leaderboard(results: dict, top_n: int = 20):
    ranked = sorted(results.items(), key=lambda x: x[1]["mean_paddle_err"])
    print(f"\n{'='*80}")
    print(f"{'RANK':>4}  {'kp':>6}  {'kd':>6}  {'density':>7}  {'paddle_err':>12}")
    print(f"{'='*80}")
    for i, ((kp, kd, d), v) in enumerate(ranked[:top_n]):
        marker = " <-- BEST" if i == 0 else ""
        print(
            f"{i+1:4d}  {kp:6.0f}  {kd:6.0f}  {d:7.0f}  "
            f"{v['mean_paddle_err']:12.6f}{marker}"
        )
    best_kp, best_kd, best_d = ranked[0][0]
    print(f"\nBest: kp={best_kp:.0f}, kd={best_kd:.0f}, density={best_d:.0f}  "
          f"(mean paddle error = {ranked[0][1]['mean_paddle_err']:.6f} m)")
    return ranked


def plot_heatmaps_per_density(results, kp_values, kd_values, density_values, out_dir, metric, label):
    n_d = len(density_values)
    fig, axes = plt.subplots(1, n_d, figsize=(5 * n_d, 5), squeeze=False)
    global_min = min(v[metric] for v in results.values())
    global_max = max(v[metric] for v in results.values())

    for di, density in enumerate(density_values):
        ax = axes[0, di]
        grid = np.full((len(kd_values), len(kp_values)), np.nan)
        for j, kp in enumerate(kp_values):
            for i, kd in enumerate(kd_values):
                key = (kp, kd, density)
                if key in results:
                    grid[i, j] = results[key][metric]

        im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis",
                        vmin=global_min, vmax=global_max)
        ax.set_xticks(range(len(kp_values)))
        ax.set_xticklabels([f"{v:.0f}" for v in kp_values], rotation=45, fontsize=7)
        ax.set_yticks(range(len(kd_values)))
        ax.set_yticklabels([f"{v:.0f}" for v in kd_values])
        ax.set_xlabel("kp")
        if di == 0:
            ax.set_ylabel("kd")
        ax.set_title(f"density={density:.0f}")

        for j2 in range(len(kp_values)):
            for i2 in range(len(kd_values)):
                val = grid[i2, j2]
                if np.isnan(val):
                    continue
                text_color = "white" if val > (global_min + global_max) / 2 else "black"
                ax.text(j2, i2, f"{val:.4f}", ha="center", va="center",
                        fontsize=7, color=text_color)

    fig.colorbar(im, ax=axes.ravel().tolist(), label="Error (m)", shrink=0.8)
    fig.suptitle(f"Fine Grid (windowed) — Mean {label.title()} Error by density", fontsize=13)
    fig.tight_layout()
    path = out_dir / f"fine_{label}_error_by_density.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    print(f"Heatmap saved to: {path}")


def plot_density_vs_kp(results, kp_values, kd_values, density_values, out_dir, metric, label):
    n_kd = len(kd_values)
    fig, axes = plt.subplots(1, n_kd, figsize=(5 * n_kd, 5), squeeze=False)
    global_min = min(v[metric] for v in results.values())
    global_max = max(v[metric] for v in results.values())

    for ki, kd in enumerate(kd_values):
        ax = axes[0, ki]
        grid = np.full((len(density_values), len(kp_values)), np.nan)
        for j, kp in enumerate(kp_values):
            for i, density in enumerate(density_values):
                key = (kp, kd, density)
                if key in results:
                    grid[i, j] = results[key][metric]

        im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis",
                        vmin=global_min, vmax=global_max)
        ax.set_xticks(range(len(kp_values)))
        ax.set_xticklabels([f"{v:.0f}" for v in kp_values], rotation=45, fontsize=7)
        ax.set_yticks(range(len(density_values)))
        ax.set_yticklabels([f"{v:.0f}" for v in density_values])
        ax.set_xlabel("kp")
        if ki == 0:
            ax.set_ylabel("density")
        ax.set_title(f"kd={kd:.0f}")

        for j2 in range(len(kp_values)):
            for i2 in range(len(density_values)):
                val = grid[i2, j2]
                if np.isnan(val):
                    continue
                text_color = "white" if val > (global_min + global_max) / 2 else "black"
                ax.text(j2, i2, f"{val:.4f}", ha="center", va="center",
                        fontsize=7, color=text_color)

    fig.colorbar(im, ax=axes.ravel().tolist(), label="Error (m)", shrink=0.8)
    fig.suptitle(f"Fine Grid (windowed) — Mean {label.title()} Error: density vs kp", fontsize=13)
    fig.tight_layout()
    path = out_dir / f"fine_{label}_error_density_vs_kp.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    print(f"Heatmap saved to: {path}")


def run_ki_sweep(ki_values, fixed_kp, fixed_kd, fixed_density):
    episode_paths = [str(SYSID_DIR / t) for t in SUBSET_TRAJECTORIES]
    for p in episode_paths:
        if not Path(p).exists():
            raise FileNotFoundError(p)

    base_cfg = load_sim_config(str(DEFAULT_CONFIG), enable_noise=False)
    print(f"Ki sweep (windowed reset every {RESET_INTERVAL} frames):")
    print(f"  Fixed: kp={fixed_kp}, kd={fixed_kd}, density={fixed_density}")
    print(f"  Ki values: {ki_values}")
    print(f"  Trajectories: {len(episode_paths)}")
    print(f"  Total runs: {len(ki_values) * len(episode_paths)}\n")

    results = {}
    t0 = time.time()
    for idx, ki in enumerate(ki_values):
        cfg = copy.deepcopy(base_cfg)
        cfg["simulator_params"]["pid_kp"] = fixed_kp
        cfg["simulator_params"]["pid_kd"] = fixed_kd
        cfg["simulator_params"]["pid_ki"] = ki
        cfg["simulator_params"]["paddle_density"] = fixed_density

        paddle_errs = []
        for ep_path in episode_paths:
            r = replay_errors_windowed(
                ep_path, cfg,
                reset_interval=RESET_INTERVAL,
                park_puck=True,
            )
            paddle_errs.append(r["mean_paddle_err"])

        avg_paddle = float(np.mean(paddle_errs))
        results[ki] = {
            "mean_paddle_err": avg_paddle,
            "per_traj_paddle": paddle_errs,
        }
        elapsed = time.time() - t0
        print(
            f"[{idx+1:3d}/{len(ki_values)}] ki={ki:6.0f}  "
            f"paddle={avg_paddle:.6f} m  ({elapsed:.1f}s)"
        )

    ranked = sorted(results.items(), key=lambda x: x[1]["mean_paddle_err"])
    print(f"\n{'='*50}")
    print(f"{'RANK':>4}  {'ki':>6}  {'paddle_err':>12}")
    print(f"{'='*50}")
    for i, (ki, v) in enumerate(ranked):
        marker = " <-- BEST" if i == 0 else ""
        print(f"{i+1:4d}  {ki:6.0f}  {v['mean_paddle_err']:12.6f}{marker}")
    print(f"\nBest ki={ranked[0][0]:.0f}  (mean paddle error = {ranked[0][1]['mean_paddle_err']:.6f} m)")
    return results, ranked


def main():
    # NOTE (2026-05-21): main() previously invoked run_ki_sweep with hardcoded
    # kp=9000/kd=50/density=3000 (hist2 canonical values), so this "fine
    # windowed PID/density" script was actually producing a Ki sweep written
    # to grid_search_results_ki_windowed_{RESET_INTERVAL}/. That contradicted
    # the script name and the step-6b protocol documented in
    # notes/docs/environments/real-world/teleop-system-id.md.
    #
    # The original Ki-sweep main() body is preserved below under
    # `_legacy_main_ki_sweep()` — call that directly if you need to reproduce
    # the previous behavior. Revert by replacing the body of this function
    # with `_legacy_main_ki_sweep()`.
    # Follow-up refine grid around hist4 windowed-10 best (kp=6500, kd=100, d=3250).
    # Previous pass: grid_search_results_3d_fine_windowed_10/ (mean paddle err ≈ 0.057 m).
    kp_values = [6500, 7500, 8500, 9000]
    kd_values = [50, 75, 100]
    density_values = [3250, 3500, 3750]

    results = run_3d_grid_search(kp_values, kd_values, density_values)
    ranked = print_leaderboard(results, top_n=20)

    out_dir = SYSID_DIR / f"grid_search_results_3d_fine_windowed_{RESET_INTERVAL}_refine"
    out_dir.mkdir(exist_ok=True)

    plot_heatmaps_per_density(
        results, kp_values, kd_values, density_values, out_dir,
        metric="mean_paddle_err", label="paddle",
    )
    plot_density_vs_kp(
        results, kp_values, kd_values, density_values, out_dir,
        metric="mean_paddle_err", label="paddle",
    )

    summary = {
        "reset_interval": RESET_INTERVAL,
        "kp_values": kp_values,
        "kd_values": kd_values,
        "density_values": density_values,
        "puck": "parked at [-0.9, 0.0] with zero velocity (no interaction)",
        "protocol": f"windowed reset every {RESET_INTERVAL} frames with PID state seeding",
        "best": {
            "kp": ranked[0][0][0],
            "kd": ranked[0][0][1],
            "density": ranked[0][0][2],
            "mean_paddle_err": ranked[0][1]["mean_paddle_err"],
        },
        "trajectories": SUBSET_TRAJECTORIES,
        "results": {
            f"kp{kp}_kd{kd}_d{d}": v for (kp, kd, d), v in results.items()
        },
    }
    out_path = out_dir / "grid_search_3d_fine_windowed_refine_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {out_path}")


def _legacy_main_ki_sweep():
    """Original main() body — runs a Ki sweep at hardcoded hist2 canonical
    (kp=9000, kd=50, density=3000). Kept for one-line revert; not part of the
    documented step-6b protocol."""
    fixed_kp = 9000
    fixed_kd = 50
    fixed_density = 3000
    ki_values = [0, 5, 10, 25, 50, 100, 150, 200, 300, 500]

    results, ranked = run_ki_sweep(ki_values, fixed_kp, fixed_kd, fixed_density)

    out_dir = SYSID_DIR / f"grid_search_results_ki_windowed_{RESET_INTERVAL}"
    out_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    kis = sorted(results.keys())
    errs = [results[k]["mean_paddle_err"] for k in kis]
    ax.plot(kis, errs, "o-", markersize=6, linewidth=2)
    for k, e in zip(kis, errs):
        ax.annotate(f"{e:.4f}", (k, e), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=7)
    ax.set_xlabel("ki")
    ax.set_ylabel("Mean Paddle Error (m)")
    ax.set_title(f"Ki Sweep (kp={fixed_kp}, kd={fixed_kd}, density={fixed_density}, "
                 f"window={RESET_INTERVAL})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = out_dir / "ki_sweep.png"
    fig.savefig(str(plot_path), dpi=150)
    plt.close(fig)
    print(f"Plot saved to: {plot_path}")

    summary = {
        "reset_interval": RESET_INTERVAL,
        "fixed_kp": fixed_kp,
        "fixed_kd": fixed_kd,
        "fixed_density": fixed_density,
        "ki_values": ki_values,
        "puck": "parked at [-0.9, 0.0] with zero velocity (no interaction)",
        "protocol": f"windowed reset every {RESET_INTERVAL} frames with PID state seeding",
        "best_ki": ranked[0][0],
        "best_err": ranked[0][1]["mean_paddle_err"],
        "trajectories": SUBSET_TRAJECTORIES,
        "results": {f"ki{k}": v for k, v in results.items()},
    }
    with open(out_dir / "ki_sweep_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {out_dir / 'ki_sweep_results.json'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
3D grid search over paddle PID (kp, kd) + paddle_density to minimise
sim-vs-real tracking error.
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
    load_real_episode,
    load_sim_config,
    reconstruct_actions,
    initial_state_vector,
)
from airhockey import AirHockeyEnv

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


def replay_errors_only(episode_path: str, sim_cfg: dict) -> dict[str, float]:
    episode = load_real_episode(episode_path)
    n_replay = episode["num_steps"]

    env = AirHockeyEnv(copy.deepcopy(sim_cfg))
    move_lims = np.asarray(
        getattr(env.simulator, "move_lims", (0.26, 0.12)), dtype=np.float64
    ).reshape(-1)[:2]
    actions = reconstruct_actions(
        episode["pose_xy"], episode["desired_xy"], move_lims
    )
    state0 = initial_state_vector(episode, start_frame=0)
    env.reset_from_state(state0)

    paddle_err_sum = 0.0
    puck_err_sum = 0.0
    for offset in range(n_replay):
        sim_paddle = np.asarray(
            env.current_state["paddles"]["paddle_ego"]["position"][:2], dtype=np.float64
        )
        sim_puck = np.asarray(
            env.current_state["pucks"][0]["position"][:2], dtype=np.float64
        )
        paddle_err_sum += float(np.linalg.norm(sim_paddle - episode["pose_xy"][offset]))
        puck_err_sum += float(np.linalg.norm(sim_puck - episode["puck_xy"][offset]))

        if offset < n_replay - 1:
            _, _, terminated, truncated, _ = env.step(actions[offset + 1])
            if terminated or truncated:
                break

    return {
        "mean_paddle_err": paddle_err_sum / n_replay,
        "mean_puck_err": puck_err_sum / n_replay,
    }


def run_3d_grid_search(
    kp_values: list[float],
    kd_values: list[float],
    density_values: list[float],
) -> dict:
    episode_paths = [str(SYSID_DIR / t) for t in SUBSET_TRAJECTORIES]
    for p in episode_paths:
        if not Path(p).exists():
            raise FileNotFoundError(p)

    base_cfg = load_sim_config(str(DEFAULT_CONFIG), enable_noise=False)
    combos = list(itertools.product(kp_values, kd_values, density_values))
    n_combos = len(combos)
    print(f"Grid: {len(kp_values)} kp × {len(kd_values)} kd × {len(density_values)} density = {n_combos} combos")
    print(f"Trajectories: {len(episode_paths)}")
    print(f"Total runs: {n_combos * len(episode_paths)}\n")

    results = {}
    t0 = time.time()
    for idx, (kp, kd, density) in enumerate(combos):
        cfg = copy.deepcopy(base_cfg)
        cfg["simulator_params"]["pid_kp"] = kp
        cfg["simulator_params"]["pid_kd"] = kd
        cfg["simulator_params"]["paddle_density"] = density

        paddle_errs = []
        puck_errs = []
        for ep_path in episode_paths:
            r = replay_errors_only(ep_path, cfg)
            paddle_errs.append(r["mean_paddle_err"])
            puck_errs.append(r["mean_puck_err"])

        avg_paddle = float(np.mean(paddle_errs))
        avg_puck = float(np.mean(puck_errs))
        results[(kp, kd, density)] = {
            "mean_paddle_err": avg_paddle,
            "mean_puck_err": avg_puck,
            "per_traj_paddle": paddle_errs,
            "per_traj_puck": puck_errs,
        }
        elapsed = time.time() - t0
        print(
            f"[{idx+1:3d}/{n_combos}] kp={kp:6.0f} kd={kd:6.0f} density={density:5.0f}  "
            f"paddle={avg_paddle:.4f} m  puck={avg_puck:.4f} m  "
            f"({elapsed:.1f}s)"
        )

    return results


def print_leaderboard(results: dict, top_n: int = 15):
    ranked = sorted(results.items(), key=lambda x: x[1]["mean_paddle_err"])
    print(f"\n{'='*80}")
    print(f"{'RANK':>4}  {'kp':>6}  {'kd':>6}  {'density':>7}  {'paddle_err':>12}  {'puck_err':>12}")
    print(f"{'='*80}")
    for i, ((kp, kd, d), v) in enumerate(ranked[:top_n]):
        marker = " <-- BEST" if i == 0 else ""
        print(
            f"{i+1:4d}  {kp:6.0f}  {kd:6.0f}  {d:7.0f}  "
            f"{v['mean_paddle_err']:12.6f}  {v['mean_puck_err']:12.6f}{marker}"
        )
    best_kp, best_kd, best_d = ranked[0][0]
    print(f"\nBest: kp={best_kp:.0f}, kd={best_kd:.0f}, density={best_d:.0f}  "
          f"(mean paddle error = {ranked[0][1]['mean_paddle_err']:.6f} m)")
    return ranked


def plot_heatmaps_per_density(
    results: dict,
    kp_values: list[float],
    kd_values: list[float],
    density_values: list[float],
    out_dir: Path,
    metric: str = "mean_paddle_err",
    label: str = "paddle",
):
    """One kp×kd heatmap per density slice, plus a combined figure."""
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
                ax.text(j2, i2, f"{val:.3f}", ha="center", va="center",
                        fontsize=6, color=text_color)

    fig.colorbar(im, ax=axes.ravel().tolist(), label="Error (m)", shrink=0.8)
    fig.suptitle(f"Mean {label.title()} Error by density slice", fontsize=13)
    fig.tight_layout()
    path = out_dir / f"{label}_error_by_density.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    print(f"Heatmap saved to: {path}")


def plot_density_vs_kp(
    results: dict,
    kp_values: list[float],
    kd_values: list[float],
    density_values: list[float],
    out_dir: Path,
    metric: str = "mean_paddle_err",
    label: str = "paddle",
):
    """One density×kp heatmap per kd slice."""
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
                ax.text(j2, i2, f"{val:.3f}", ha="center", va="center",
                        fontsize=6, color=text_color)

    fig.colorbar(im, ax=axes.ravel().tolist(), label="Error (m)", shrink=0.8)
    fig.suptitle(f"Mean {label.title()} Error: density vs kp (per kd)", fontsize=13)
    fig.tight_layout()
    path = out_dir / f"{label}_error_density_vs_kp.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    print(f"Heatmap saved to: {path}")


def main():
    kp_values = [2000, 3000, 4000, 5000, 6000, 7000, 8000]
    kd_values = [200, 600, 1000]
    density_values = [1000, 1500, 2000, 2500, 3000]

    results = run_3d_grid_search(kp_values, kd_values, density_values)

    ranked = print_leaderboard(results, top_n=20)

    out_dir = SYSID_DIR / "grid_search_results_3d"
    out_dir.mkdir(exist_ok=True)

    for metric, label in [("mean_paddle_err", "paddle"), ("mean_puck_err", "puck")]:
        plot_heatmaps_per_density(
            results, kp_values, kd_values, density_values, out_dir,
            metric=metric, label=label,
        )
        plot_density_vs_kp(
            results, kp_values, kd_values, density_values, out_dir,
            metric=metric, label=label,
        )

    summary = {
        "kp_values": kp_values,
        "kd_values": kd_values,
        "density_values": density_values,
        "best": {
            "kp": ranked[0][0][0],
            "kd": ranked[0][0][1],
            "density": ranked[0][0][2],
            "mean_paddle_err": ranked[0][1]["mean_paddle_err"],
            "mean_puck_err": ranked[0][1]["mean_puck_err"],
        },
        "trajectories": SUBSET_TRAJECTORIES,
        "results": {
            f"kp{kp}_kd{kd}_d{d}": v
            for (kp, kd, d), v in results.items()
        },
    }
    with open(out_dir / "grid_search_3d_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Full results saved to: {out_dir / 'grid_search_3d_results.json'}")


if __name__ == "__main__":
    main()

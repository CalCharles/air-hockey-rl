#!/usr/bin/env python3
"""
Grid search over paddle PID (kp, kd) to minimise sim-vs-real tracking error.

Runs a render-free replay for each (kp, kd) pair on a subset of system_id3
trajectories, collects mean paddle position error, and produces:
  1. A printed leaderboard of the best parameter combos.
  2. A 2-D heatmap (kp × kd) saved as a PNG.
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
import yaml

import importlib.util

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.visualization.replay_real_in_sim import (        # noqa: E402
    load_real_episode,
    load_sim_config,
    reconstruct_actions,
    initial_state_vector,
)
from airhockey import AirHockeyEnv      # noqa: E402

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


def replay_errors_only(
    episode_path: str,
    sim_cfg: dict,
) -> dict[str, float]:
    """Run sim replay with NO rendering and return mean paddle/puck error."""
    episode = load_real_episode(episode_path)
    total = episode["num_steps"]
    n_replay = total

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
        i = offset
        sim_paddle = np.asarray(
            env.current_state["paddles"]["paddle_ego"]["position"][:2], dtype=np.float64
        )
        sim_puck = np.asarray(
            env.current_state["pucks"][0]["position"][:2], dtype=np.float64
        )
        paddle_err_sum += float(np.linalg.norm(sim_paddle - episode["pose_xy"][i]))
        puck_err_sum += float(np.linalg.norm(sim_puck - episode["puck_xy"][i]))

        if offset < n_replay - 1:
            _, _, terminated, truncated, _ = env.step(actions[i + 1])
            if terminated or truncated:
                break

    return {
        "mean_paddle_err": paddle_err_sum / n_replay,
        "mean_puck_err": puck_err_sum / n_replay,
        "cum_paddle_err": paddle_err_sum,
        "cum_puck_err": puck_err_sum,
        "n_frames": n_replay,
    }


def run_grid_search(
    kp_values: list[float],
    kd_values: list[float],
    config_path: str | Path = DEFAULT_CONFIG,
    trajectories: list[str] | None = None,
) -> dict:
    if trajectories is None:
        trajectories = SUBSET_TRAJECTORIES
    episode_paths = [str(SYSID_DIR / t) for t in trajectories]
    for p in episode_paths:
        if not Path(p).exists():
            raise FileNotFoundError(p)

    base_cfg = load_sim_config(str(config_path), enable_noise=False)
    n_combos = len(kp_values) * len(kd_values)
    print(f"Grid search: {len(kp_values)} kp × {len(kd_values)} kd = {n_combos} combos")
    print(f"Trajectories: {len(episode_paths)}")
    print(f"Total runs: {n_combos * len(episode_paths)}\n")

    results = {}
    t0 = time.time()
    for idx, (kp, kd) in enumerate(itertools.product(kp_values, kd_values)):
        cfg = copy.deepcopy(base_cfg)
        cfg["simulator_params"]["pid_kp"] = kp
        cfg["simulator_params"]["pid_kd"] = kd

        paddle_errs = []
        puck_errs = []
        for ep_path in episode_paths:
            r = replay_errors_only(ep_path, cfg)
            paddle_errs.append(r["mean_paddle_err"])
            puck_errs.append(r["mean_puck_err"])

        avg_paddle = float(np.mean(paddle_errs))
        avg_puck = float(np.mean(puck_errs))
        results[(kp, kd)] = {
            "mean_paddle_err": avg_paddle,
            "mean_puck_err": avg_puck,
            "per_traj_paddle": paddle_errs,
            "per_traj_puck": puck_errs,
        }
        elapsed = time.time() - t0
        print(
            f"[{idx+1:3d}/{n_combos}] kp={kp:6.0f} kd={kd:6.0f}  "
            f"paddle={avg_paddle:.4f} m  puck={avg_puck:.4f} m  "
            f"({elapsed:.1f}s)"
        )

    return results


def print_leaderboard(results: dict, top_n: int = 10) -> tuple[float, float]:
    ranked = sorted(results.items(), key=lambda x: x[1]["mean_paddle_err"])
    print(f"\n{'='*70}")
    print(f"{'RANK':>4}  {'kp':>6}  {'kd':>6}  {'paddle_err':>12}  {'puck_err':>12}")
    print(f"{'='*70}")
    for i, ((kp, kd), v) in enumerate(ranked[:top_n]):
        marker = " <-- BEST" if i == 0 else ""
        print(
            f"{i+1:4d}  {kp:6.0f}  {kd:6.0f}  "
            f"{v['mean_paddle_err']:12.6f}  {v['mean_puck_err']:12.6f}{marker}"
        )
    best_kp, best_kd = ranked[0][0]
    print(f"\nBest: kp={best_kp:.0f}, kd={best_kd:.0f}  "
          f"(mean paddle error = {ranked[0][1]['mean_paddle_err']:.6f} m)")
    return best_kp, best_kd


def plot_heatmap(
    results: dict,
    kp_values: list[float],
    kd_values: list[float],
    out_path: str | Path,
    metric: str = "mean_paddle_err",
    title: str = "Mean Paddle Position Error (m)",
) -> None:
    grid = np.full((len(kd_values), len(kp_values)), np.nan)
    for j, kp in enumerate(kp_values):
        for i, kd in enumerate(kd_values):
            grid[i, j] = results[(kp, kd)][metric]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(kp_values)))
    ax.set_xticklabels([f"{v:.0f}" for v in kp_values])
    ax.set_yticks(range(len(kd_values)))
    ax.set_yticklabels([f"{v:.0f}" for v in kd_values])
    ax.set_xlabel("kp")
    ax.set_ylabel("kd")
    ax.set_title(title)

    for j in range(len(kp_values)):
        for i in range(len(kd_values)):
            val = grid[i, j]
            text_color = "white" if val > (grid.min() + grid.max()) / 2 else "black"
            ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                    fontsize=8, color=text_color)

    fig.colorbar(im, ax=ax, label="Error (m)")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"Heatmap saved to: {out_path}")


def load_previous_results(path: Path) -> dict:
    """Load a previous grid_search_results.json and return the (kp, kd)->metrics dict."""
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    out = {}
    for key, v in data.get("results", {}).items():
        parts = key.replace("kp", "").split("_kd")
        kp, kd = float(parts[0]), float(parts[1])
        out[(kp, kd)] = v
    return out


def main():
    kp_values = [2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]
    kd_values = [200, 400, 600, 800, 1000]

    out_dir = SYSID_DIR / "grid_search_results"
    out_dir.mkdir(exist_ok=True)

    prev = load_previous_results(out_dir / "grid_search_results.json")
    needed_kp = [kp for kp in kp_values
                 if not all((kp, kd) in prev for kd in kd_values)]
    if needed_kp:
        print(f"Running grid search for new kp values: {needed_kp}")
        new_results = run_grid_search(needed_kp, kd_values)
        prev.update(new_results)
    else:
        print("All (kp, kd) combos already cached.")

    results = {k: v for k, v in prev.items()
               if k[0] in kp_values and k[1] in kd_values}

    best_kp, best_kd = print_leaderboard(results)

    plot_heatmap(
        results, kp_values, kd_values,
        out_dir / "paddle_error_heatmap.png",
        metric="mean_paddle_err",
        title="Mean Paddle Position Error (m)",
    )
    plot_heatmap(
        results, kp_values, kd_values,
        out_dir / "puck_error_heatmap.png",
        metric="mean_puck_err",
        title="Mean Puck Position Error (m)",
    )

    summary = {
        "kp_values": kp_values,
        "kd_values": kd_values,
        "best_kp": best_kp,
        "best_kd": best_kd,
        "trajectories": SUBSET_TRAJECTORIES,
        "results": {
            f"kp{kp}_kd{kd}": v
            for (kp, kd), v in results.items()
        },
    }
    with open(out_dir / "grid_search_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Full results saved to: {out_dir / 'grid_search_results.json'}")


if __name__ == "__main__":
    main()

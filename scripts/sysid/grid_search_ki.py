#!/usr/bin/env python3
"""
Grid search over Ki with the best (kp, kd, density) fixed.
"""

from __future__ import annotations

import copy
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

BEST_KP = 6500 # 7500
BEST_KD = 100
BEST_DENSITY = 3250 # 2750


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


def main():
    ki_values = [0, 10, 25, 50, 100, 150, 200, 300, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]

    episode_paths = [str(SYSID_DIR / t) for t in SUBSET_TRAJECTORIES]
    for p in episode_paths:
        if not Path(p).exists():
            raise FileNotFoundError(p)

    base_cfg = load_sim_config(str(DEFAULT_CONFIG), enable_noise=False)
    base_cfg["simulator_params"]["pid_kp"] = BEST_KP
    base_cfg["simulator_params"]["pid_kd"] = BEST_KD
    base_cfg["simulator_params"]["paddle_density"] = BEST_DENSITY

    print(f"Ki search with fixed kp={BEST_KP}, kd={BEST_KD}, density={BEST_DENSITY}")
    print(f"Ki values: {ki_values}")
    print(f"Trajectories: {len(episode_paths)}")
    print(f"Total runs: {len(ki_values) * len(episode_paths)}\n")

    results = {}
    t0 = time.time()
    for idx, ki in enumerate(ki_values):
        cfg = copy.deepcopy(base_cfg)
        cfg["simulator_params"]["pid_ki"] = ki

        paddle_errs = []
        puck_errs = []
        for ep_path in episode_paths:
            r = replay_errors_only(ep_path, cfg)
            paddle_errs.append(r["mean_paddle_err"])
            puck_errs.append(r["mean_puck_err"])

        avg_paddle = float(np.mean(paddle_errs))
        avg_puck = float(np.mean(puck_errs))
        results[ki] = {
            "mean_paddle_err": avg_paddle,
            "mean_puck_err": avg_puck,
            "per_traj_paddle": paddle_errs,
            "per_traj_puck": puck_errs,
        }
        elapsed = time.time() - t0
        print(
            f"[{idx+1:2d}/{len(ki_values)}] ki={ki:5.0f}  "
            f"paddle={avg_paddle:.6f} m  puck={avg_puck:.6f} m  "
            f"({elapsed:.1f}s)"
        )

    # Leaderboard
    ranked = sorted(results.items(), key=lambda x: x[1]["mean_paddle_err"])
    print(f"\n{'='*70}")
    print(f"{'RANK':>4}  {'ki':>6}  {'paddle_err':>14}  {'puck_err':>14}")
    print(f"{'='*70}")
    for i, (ki, v) in enumerate(ranked):
        marker = " <-- BEST" if i == 0 else ""
        print(
            f"{i+1:4d}  {ki:6.0f}  "
            f"{v['mean_paddle_err']:14.6f}  {v['mean_puck_err']:14.6f}{marker}"
        )
    best_ki = ranked[0][0]
    print(f"\nBest: ki={best_ki:.0f}  "
          f"(mean paddle error = {ranked[0][1]['mean_paddle_err']:.6f} m)")
    print(f"Full best config: kp={BEST_KP}, kd={BEST_KD}, ki={best_ki}, density={BEST_DENSITY}")

    # Plot
    out_dir = SYSID_DIR / "grid_search_results_ki"
    out_dir.mkdir(exist_ok=True)

    ki_arr = np.array(ki_values, dtype=float)
    paddle_arr = np.array([results[ki]["mean_paddle_err"] for ki in ki_values])
    puck_arr = np.array([results[ki]["mean_puck_err"] for ki in ki_values])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(ki_arr, paddle_arr, "o-", color="tab:blue", linewidth=2, markersize=6)
    ax1.set_xlabel("ki")
    ax1.set_ylabel("Mean Paddle Error (m)")
    ax1.set_title(f"Paddle Error vs Ki\n(kp={BEST_KP}, kd={BEST_KD}, density={BEST_DENSITY})")
    ax1.grid(True, alpha=0.3)
    best_idx_p = int(np.argmin(paddle_arr))
    ax1.annotate(f"best: ki={ki_values[best_idx_p]:.0f}\n{paddle_arr[best_idx_p]:.5f} m",
                 xy=(ki_arr[best_idx_p], paddle_arr[best_idx_p]),
                 xytext=(10, 15), textcoords="offset points",
                 arrowprops=dict(arrowstyle="->"), fontsize=9)

    ax2.plot(ki_arr, puck_arr, "o-", color="tab:orange", linewidth=2, markersize=6)
    ax2.set_xlabel("ki")
    ax2.set_ylabel("Mean Puck Error (m)")
    ax2.set_title(f"Puck Error vs Ki\n(kp={BEST_KP}, kd={BEST_KD}, density={BEST_DENSITY})")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = out_dir / "ki_sweep.png"
    fig.savefig(str(plot_path), dpi=150)
    plt.close(fig)
    print(f"\nPlot saved to: {plot_path}")

    summary = {
        "fixed_params": {"kp": BEST_KP, "kd": BEST_KD, "density": BEST_DENSITY},
        "ki_values": ki_values,
        "best_ki": best_ki,
        "trajectories": SUBSET_TRAJECTORIES,
        "results": {f"ki{ki}": v for ki, v in results.items()},
    }
    with open(out_dir / "ki_sweep_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to: {out_dir / 'ki_sweep_results.json'}")


if __name__ == "__main__":
    main()

"""Grid search over paddle system identification parameters.

Evaluates all combinations of parameters from the sys-id YAML config on
trajectory segments sampled from multiple real-world episodes. 

Usage:
    python scripts/domain_adaptation/system_identification/paddle/grid_search_paddle.py \
        --data-dir /data2/air_hockey/air_hockey_state_data/datastor1/calebc/public/data/mouse/state_data_all_new/ \
        --num-trajectories 50 \
        --sys-id-path-paddle scripts/domain_adaptation/sys_id_configs/real2sim/paddle_id_params.yaml \
        --num-resamples 200 --traj-length 10 --grid-points 7
"""

import argparse
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

from airhockey import AirHockeyEnv
from scripts.domain_adaptation.encode_env_params import assign_values

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../real_data_transforms")
)
from scripts.real_data_transforms.real_to_sim_observations import (
    load_multiple_real_trajectories_for_sysid,
)
from data_loading import (
    BOX2D_TABLE_LENGTH,
    BOX2D_TABLE_WIDTH,
    BOX2D_PADDLE_RADIUS,
    BOX2D_PUCK_RADIUS,
)


def compare_trajectories(a_traj, b_traj, comp_type="posl1"):
    if comp_type == "l2":
        diffs = a_traj - b_traj
        step_rmse = np.sqrt(np.mean(diffs**2, axis=2))
        traj_loss = np.mean(step_rmse, axis=1)
        return np.mean(traj_loss)
    if comp_type == "posl2":
        diffs = a_traj[..., :2] - b_traj[..., :2]
        step_rmse = np.sqrt(np.mean(diffs**2, axis=2))
        traj_loss = np.mean(step_rmse, axis=1)
        return np.mean(traj_loss)
    if comp_type == "l1":
        diffs = a_traj - b_traj
        step_l1 = np.sum(np.abs(diffs), axis=2)
        traj_loss = np.mean(step_l1, axis=1)
        return np.mean(traj_loss)
    if comp_type == "posl1":
        pos_diffs = a_traj[..., :2] - b_traj[..., :2]
        step_l1 = np.sum(np.abs(pos_diffs), axis=2)
        traj_loss = np.mean(step_l1, axis=1)
        return np.mean(traj_loss)
    raise ValueError(f"Unknown comp_type: {comp_type}")


def build_sim_config():
    """Build a Box2D env config from known table constants."""
    return {
        "simulator": "box2d",
        "task": "puck_juggle",
        "obs_type": "vel",
        "seed": 0,
        "n_training_steps": 0,
        "max_timesteps": 10000,
        "num_paddles": 1,
        "num_pucks": 1,
        "num_blocks": 0,
        "num_obstacles": 0,
        "num_targets": 0,
        "return_goal_obs": False,
        "terminate_on_enemy_goal": False,
        "terminate_on_out_of_bounds": False,
        "terminate_on_puck_hit_bottom": False,
        "terminate_on_puck_pass_paddle": False,
        "terminate_on_puck_stop": False,
        "diagonal_motion_rew": 0,
        "direction_change_rew": 0,
        "horizontal_vel_rew": 0,
        "stand_still_rew": 0,
        "truncate_rew": 0,
        "wall_bumping_rew": 0,
        "goal_max_x_velocity": 1,
        "goal_max_y_velocity": 5,
        "goal_min_y_velocity": 1,
        "simulator_params": {
            "length": BOX2D_TABLE_LENGTH,
            "width": BOX2D_TABLE_WIDTH,
            "paddle_radius": BOX2D_PADDLE_RADIUS,
            "puck_radius": BOX2D_PUCK_RADIUS,
            "block_width": 0.0254,
            "render_size": 360,
            "absorb_target": False,
            "max_force_timestep": 10000,
            "wall_bounce_scale": 0.02,
            # Placeholders -- grid search will overwrite these
            "force_scaling": 1000,
            "paddle_damping": 3,
            "paddle_density": 1000,
            "puck_damping": 0.5,
            "puck_density": 250,
            "gravity": -0.67,
        },
    }


def sample_trajectory_segments(episodes, num_samples, traj_length, rng):
    """Sample random trajectory segments across episodes.

    Uses the same paddle-style simple random sampling as the CMAPlanner
    (no hits_array filtering -- we want full-episode coverage for paddle).

    Args:
        episodes: list of episode dicts with 'observations' and 'actions'.
        num_samples: number of segments to sample.
        traj_length: length of each segment.
        rng: numpy RandomState for reproducibility.

    Returns:
        dict with 'observations', 'actions', 'traj_length' keys.
    """
    sampled_obs = []
    sampled_act = []
    for _ in range(num_samples):
        ep_idx = rng.randint(0, len(episodes))
        ep = episodes[ep_idx]
        N = len(ep["observations"])
        if N <= traj_length:
            continue
        start = rng.randint(0, N - traj_length)
        sampled_obs.append(ep["observations"][start : start + traj_length])
        sampled_act.append(ep["actions"][start : start + traj_length])

    return {
        "observations": np.stack(sampled_obs, axis=0),
        "actions": np.stack(sampled_act, axis=0),
        "traj_length": traj_length,
    }


def evaluate_params(param_vector, param_names, sim_config, trajectories,
                    object_name="paddle", comp_type="posl1"):
    """Evaluate a single parameter combo on trajectory segments.

    Mirrors RealDataPaddlePipeline.get_value: for each segment, resets env
    to initial state, replays actions, and compares resulting trajectory.
    """
    candidate_config = assign_values(param_vector, param_names, sim_config)
    env = AirHockeyEnv(candidate_config)

    num_segments = trajectories["observations"].shape[0]
    traj_length = trajectories["traj_length"]

    all_eval_segments = []
    failures = 0
    for i in range(num_segments):
        state_vector = trajectories["observations"][i, 0]
        try:
            obs, _ = env.reset_from_state(state_vector)
        except Exception:
            failures += 1
            continue

        segment_obs = [obs.copy()]
        for j in range(traj_length - 1):
            action = trajectories["actions"][i, j]
            obs, _, _, _, _ = env.step(action)
            segment_obs.append(obs.copy())

        all_eval_segments.append(np.array(segment_obs))

    if len(all_eval_segments) == 0:
        return 1e6

    evaluated_states = np.stack(all_eval_segments, axis=0)

    # Subset to paddle columns for comparison
    if object_name == "paddle":
        eval_sub = evaluated_states[:, :, :4]
        # Only use the segments that didn't fail
        target_sub = trajectories["observations"][:num_segments - failures, :, :4] \
            if failures == 0 else None
    else:
        eval_sub = evaluated_states
        target_sub = trajectories["observations"][:num_segments - failures] \
            if failures == 0 else None

    if failures > 0:
        # Rebuild target to match only successful segments
        print("getting to failures")
        success_indices = []
        idx = 0
        for i in range(num_segments):
            state_vector = trajectories["observations"][i, 0]
            try:
                env.reset_from_state(state_vector)
                success_indices.append(i)
            except Exception:
                pass
        if object_name == "paddle":
            target_sub = trajectories["observations"][success_indices, :, :4]
        else:
            target_sub = trajectories["observations"][success_indices]

    if target_sub is None:
        target_sub = trajectories["observations"][:, :, :4] if object_name == "paddle" \
            else trajectories["observations"]

    return compare_trajectories(eval_sub, target_sub, comp_type=comp_type)


def make_grid(param_config, grid_points):
    """Build parameter grid from sys-id YAML config.

    Args:
        param_config: dict from YAML with param_name -> {lower, upper, start}.
        grid_points: number of points per parameter.

    Returns:
        param_names: sorted list of parameter names.
        grid_values: dict mapping param_name -> array of values.
        all_combos: list of tuples, one per grid point.
    """
    param_names = sorted(param_config.keys())
    grid_values = {}
    for name in param_names:
        lower = param_config[name]["lower"]
        upper = param_config[name]["upper"]
        grid_values[name] = np.linspace(lower, upper, grid_points)

    all_combos = list(itertools.product(*(grid_values[n] for n in param_names)))
    return param_names, grid_values, all_combos


def main():
    parser = argparse.ArgumentParser(description="Grid search paddle system identification")

    # Data
    parser.add_argument("--cfg", type=str, default=None,help="Optional YAML config (air_hockey key expected)")
    parser.add_argument("--data-dir", type=str,default="/data2/air_hockey/air_hockey_state_data/datastor1/calebc/public/data/mouse/state_data_all_new/",help="Directory containing trajectory HDF5 files")
    parser.add_argument("--num-trajectories", type=int, default=50,help="Number of trajectory files to load (-1 for all)")
    parser.add_argument("--sys-id-path-paddle", type=str,default="scripts/domain_adaptation/sys_id_configs/real2sim/paddle_id_params.yaml",help="System ID parameter bounds YAML")

    # Comparison
    parser.add_argument("--object-name", type=str, default="paddle",help="Object to compare: paddle, puck, or all")
    parser.add_argument("--comp-type", type=str, default="posl2",help="Comparison metric: l2, posl2, l1, posl1")
    parser.add_argument("--dt", type=float, default=0.05,help="Timestep for velocity finite differences")

    # Sampling
    parser.add_argument("--num-resamples", type=int, default=500,help="Number of trajectory segments per evaluation")
    parser.add_argument("--traj-length", type=int, default=10,help="Length of each trajectory segment")
    parser.add_argument("--num-eval-seeds", type=int, default=3,help="Number of random seeds to average over per grid point")

    # Grid
    parser.add_argument("--grid-points", type=int, default=20,help="Number of grid points per parameter")

    # Output
    parser.add_argument("--run-dir", type=str,default="scripts/domain_adaptation/system_identification/paddle/grid_search_runs",help="Directory for results")
    parser.add_argument("--run-id", type=str, default="6",help="Run identifier")

    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────
    if args.cfg is not None:
        with open(args.cfg, "r") as f:
            cfg = yaml.safe_load(f)
        sim_config = cfg["air_hockey"]
    else:
        sim_config = build_sim_config()

    with open(args.sys_id_path_paddle, "r") as f:
        param_config = yaml.safe_load(f)

    # ── Build grid ───────────────────────────────────────────────
    param_names, grid_values, all_combos = make_grid(param_config, args.grid_points)
    total_combos = len(all_combos)
    print(f"Parameters: {param_names}")
    for name in param_names:
        print(f"  {name}: {grid_values[name]}")
    print(f"Total grid points: {total_combos}")
    print(f"Eval seeds per point: {args.num_eval_seeds}")
    print(f"Total evaluations: {total_combos * args.num_eval_seeds}")

    # ── Load trajectories ────────────────────────────────────────
    print(f"\nLoading trajectories from {args.data_dir} (num={args.num_trajectories})...")
    episodes, _ = load_multiple_real_trajectories_for_sysid(
        args.data_dir, num_load=args.num_trajectories, dt=args.dt,
    )
    total_obs = sum(len(ep["observations"]) for ep in episodes)
    print(f"Loaded {len(episodes)} episodes, {total_obs} total timesteps")

    # ── Pre-sample trajectory segments ───────────────────────────
    # Use fixed seeds so each grid point is evaluated on the same segments
    print(f"\nPre-sampling {args.num_eval_seeds} sets of {args.num_resamples} segments "
          f"(traj_length={args.traj_length})...")
    sampled_sets = []
    for seed in range(args.num_eval_seeds):
        rng = np.random.RandomState(seed=seed * 1000)
        traj_set = sample_trajectory_segments(
            episodes, args.num_resamples, args.traj_length, rng
        )
        print(f"  Seed {seed}: {traj_set['observations'].shape[0]} segments sampled")
        sampled_sets.append(traj_set)

    # ── Run grid search ──────────────────────────────────────────
    run_dir = os.path.join(args.run_dir, str(args.run_id))
    os.makedirs(run_dir, exist_ok=True)

    # Save run config
    run_config = {
        "data_dir": args.data_dir,
        "num_trajectories": args.num_trajectories,
        "num_episodes_loaded": len(episodes),
        "sys_id_path_paddle": args.sys_id_path_paddle,
        "cfg": args.cfg,
        "object_name": args.object_name,
        "comp_type": args.comp_type,
        "dt": args.dt,
        "num_resamples": args.num_resamples,
        "traj_length": args.traj_length,
        "num_eval_seeds": args.num_eval_seeds,
        "grid_points": args.grid_points,
        "total_combos": total_combos,
        "param_names": param_names,
        "grid_values": {k: v.tolist() for k, v in grid_values.items()},
    }
    with open(os.path.join(run_dir, "run_config.yaml"), "w") as f:
        yaml.dump(run_config, f, default_flow_style=False)

    # Results storage
    columns = param_names + ["mean_loss", "std_loss"] + [f"loss_seed{s}" for s in range(args.num_eval_seeds)]
    results = []

    print(f"\nStarting grid search ({total_combos} combinations)...")
    t_start = time.time()

    for combo_idx, combo in enumerate(all_combos):
        param_vector = list(combo)

        # Evaluate on each pre-sampled set
        losses = []
        for seed_idx, traj_set in enumerate(sampled_sets):
            loss = evaluate_params(
                param_vector, param_names, sim_config, traj_set,
                object_name=args.object_name, comp_type=args.comp_type,
            )
            losses.append(loss)

        mean_loss = np.mean(losses)
        std_loss = np.std(losses)
        row = list(combo) + [mean_loss, std_loss] + losses
        results.append(row)

        # Progress
        elapsed = time.time() - t_start
        rate = (combo_idx + 1) / elapsed if elapsed > 0 else 0
        eta = (total_combos - combo_idx - 1) / rate if rate > 0 else 0
        param_str = ", ".join(f"{n}={v:.2f}" for n, v in zip(param_names, combo))
        print(f"  [{combo_idx+1}/{total_combos}] {param_str} -> "
              f"loss={mean_loss:.6f} +/- {std_loss:.6f}  "
              f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

        # Periodically save intermediate results
        if (combo_idx + 1) % 50 == 0 or combo_idx == total_combos - 1:
            df = pd.DataFrame(results, columns=columns)
            df.to_csv(os.path.join(run_dir, "grid_results.csv"), index=False)

    # ── Save final results ───────────────────────────────────────
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(os.path.join(run_dir, "grid_results.csv"), index=False)

    # Find best
    best_idx = df["mean_loss"].idxmin()  # lowest = best
    best_row = df.iloc[best_idx]
    print(f"\n{'='*60}")
    print("GRID SEARCH COMPLETE")
    print(f"{'='*60}")
    print(f"Best parameters:")
    for name in param_names:
        print(f"  {name}: {best_row[name]:.4f}")
    print(f"Best mean loss: {best_row['mean_loss']:.6f} +/- {best_row['std_loss']:.6f}")

    # Save best params
    best_params = {name: float(best_row[name]) for name in param_names}
    best_params["best_loss"] = float(best_row["mean_loss"])
    best_params["best_loss_std"] = float(best_row["std_loss"])
    with open(os.path.join(run_dir, "best_params.yaml"), "w") as f:
        yaml.dump(best_params, f, default_flow_style=False)

    # Save full config with best parameters applied
    best_vector = [float(best_row[name]) for name in param_names]
    best_config = assign_values(best_vector, param_names, sim_config)
    # Convert any numpy types for clean YAML serialization
    def _to_native(obj):
        if isinstance(obj, dict):
            return {k: _to_native(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_native(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    with open(os.path.join(run_dir, "best_config.yaml"), "w") as f:
        yaml.dump({"air_hockey": _to_native(best_config)}, f, default_flow_style=False)

    # ── Plots ────────────────────────────────────────────────────
    # 2D heatmaps for each pair of parameters (marginalizing over the third)
    if len(param_names) >= 2:
        for i, pi in enumerate(param_names):
            for j, pj in enumerate(param_names):
                if j <= i:
                    continue
                # The remaining parameter(s) to marginalize over
                other_params = [p for p in param_names if p not in (pi, pj)]

                fig, ax = plt.subplots(figsize=(8, 6))
                pivot = df.groupby([pi, pj])["mean_loss"].min().reset_index()
                pivot_table = pivot.pivot(index=pi, columns=pj, values="mean_loss")

                im = ax.imshow(
                    pivot_table.values,
                    aspect="auto",
                    origin="lower",
                    extent=[
                        grid_values[pj][0], grid_values[pj][-1],
                        grid_values[pi][0], grid_values[pi][-1],
                    ],
                    cmap="viridis",
                )
                plt.colorbar(im, ax=ax, label=f"Min loss (over {', '.join(other_params)})")
                ax.set_xlabel(pj)
                ax.set_ylabel(pi)
                ax.set_title(f"Grid Search: {pi} vs {pj}")

                # Mark the global best
                ax.plot(best_row[pj], best_row[pi], "r*", markersize=15, label="Best")
                ax.legend()

                fig.savefig(
                    os.path.join(run_dir, f"heatmap_{pi}_vs_{pj}.png"),
                    dpi=150, bbox_inches="tight",
                )
                plt.close(fig)

    # 1D marginal plots: best loss vs each parameter
    fig, axes = plt.subplots(1, len(param_names), figsize=(6 * len(param_names), 5))
    if len(param_names) == 1:
        axes = [axes]
    for ax, name in zip(axes, param_names):
        marginal = df.groupby(name)["mean_loss"].agg(["min", "mean", "std"]).reset_index()
        ax.plot(marginal[name], marginal["min"], "o-", label="best (min error)")
        ax.plot(marginal[name], marginal["mean"], "s--", alpha=0.5, label="avg error")
        ax.set_xlabel(name)
        ax.set_ylabel("Error (lower = better)")
        ax.set_title(f"Marginal: {name}")
        ax.legend()
        ax.axvline(best_row[name], color="r", linestyle=":", alpha=0.7, label="best")
    fig.suptitle(f"Grid Search Marginals ({args.comp_type})")
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "marginal_plots.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Results saved to {run_dir}")


if __name__ == "__main__":
    main()

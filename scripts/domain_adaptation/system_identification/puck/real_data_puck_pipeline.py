"""
Usage (multi-trajectory, recommended):
    python scripts/domain_adaptation/system_identification/puck/real_data_puck_pipeline.py \
        --data-dir /path/to/state_data/ \
        --num-trajectories 50 \
        --sys-id-path-puck scripts/domain_adaptation/sys_id_configs/real2sim/puck_id_params.yaml \
        --paddle-params-path scripts/domain_adaptation/system_identification/paddle/real2sim_runs/15/optimal_params.yaml

Usage (single trajectory, legacy):
    python scripts/domain_adaptation/system_identification/puck/real_data_puck_pipeline.py \
        --data-dir /path/to/state_data/ \
        --num-trajectories 0 --trajectory-file trajectory_data253.hdf5 \
        --sys-id-path-puck scripts/domain_adaptation/sys_id_configs/real2sim/puck_id_params.yaml \
        --paddle-params-path scripts/domain_adaptation/system_identification/paddle/real2sim_runs/15/optimal_params.yaml
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

from airhockey import AirHockeyEnv
from scripts.domain_adaptation.planners import CMAPlanner
from scripts.domain_adaptation.encode_env_params import assign_values

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../real_data_transforms")
)
from scripts.real_data_transforms.real_to_sim_observations import (
    load_real_trajectory_for_sysid,
    load_multiple_real_trajectories_for_sysid,
)
from data_loading import (
    BOX2D_TABLE_LENGTH,
    BOX2D_TABLE_WIDTH,
    BOX2D_PADDLE_RADIUS,
    BOX2D_PUCK_RADIUS,
)


def compare_trajectories(a_traj, b_traj, comp_type="l2"):
    """Compare two trajectory arrays. Returns negative loss (higher = better)."""
    diffs = a_traj - b_traj
    if comp_type == "l2":
        step_l2 = np.linalg.norm(diffs**2, axis=2)
        traj_loss = np.mean(step_l2, axis=1)
        return -np.mean(traj_loss)
    if comp_type == "posl2":
        pos_diffs = a_traj[..., :2] - b_traj[..., :2]
        step_dist = np.sqrt(np.sum(pos_diffs**2, axis=2))
        traj_loss = np.mean(step_dist, axis=1) 
        return -np.mean(traj_loss)
    if comp_type == "l1":
        return -np.linalg.norm(a_traj - b_traj, ord=1)
    if comp_type == "posl1":
        return -np.linalg.norm(a_traj[..., :2] - b_traj[..., :2], ord=1)
    if comp_type == "last":
        return -np.linalg.norm(a_traj[-1] - b_traj[-1])


def build_sim_config(paddle_params=None):
    """Build a Box2D env config from known table constants.

    Optionally injects optimal paddle parameters so they remain fixed
    while puck parameters are optimized by CMA-ES.
    """
    config = {
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
            "max_force_timestep": 100,
            "wall_bounce_scale": 0.02,
            "force_scaling": 1000,
            "paddle_damping": 3,
            "paddle_density": 2500,
            "puck_damping": 0.5,
            "puck_density": 250,
            "gravity": -0.5,
        },
    }
    if paddle_params:
        for k, v in paddle_params.items():
            if k != "best_error":
                config["simulator_params"][k] = v
    return config


class RealDataPuckPipeline:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.paddle_params = self._load_paddle_params()

        if getattr(self, "cfg", None) is not None:
            with open(self.cfg, "r") as f:
                cfg = yaml.safe_load(f)
            self.sim_config = cfg["air_hockey"]
            if self.paddle_params:
                for k, v in self.paddle_params.items():
                    if k != "best_error":
                        self.sim_config["simulator_params"][k] = v
        else:
            self.sim_config = build_sim_config(self.paddle_params)

        self.param_names, self.lower_bounds, self.upper_bounds, self.param_start_values = (
            self._load_sys_id_yaml()
        )
        self.dataframe_columns = self._make_dataframe_columns()
        self.df = pd.DataFrame(columns=self.dataframe_columns)
        self.ground_truth_episodes = None

    def _load_paddle_params(self):
        path = getattr(self, "paddle_params_path", None)
        if not path:
            print("No paddle params path provided -- using default paddle parameters")
            return None
        with open(path, "r") as f:
            params = yaml.safe_load(f)
        print(f"Loaded paddle params from {path}:")
        for k, v in params.items():
            if k != "best_error":
                print(f"  {k}: {v}")
        return params

    def _load_sys_id_yaml(self):
        with open(self.sys_id_path_puck, "r") as f:
            id_config = yaml.safe_load(f)
        param_names = sorted(id_config.keys())
        lower_bounds = [id_config[n]["lower"] for n in param_names]
        upper_bounds = [id_config[n]["upper"] for n in param_names]
        initial_params = [id_config[n]["start"] for n in param_names]
        return param_names, lower_bounds, upper_bounds, initial_params

    def _make_dataframe_columns(self):
        columns = []
        for s in self.param_names:
            columns.append(f"initial_{s}_param_value")
        for s in self.param_names:
            columns.append(f"best_{s}_param_value")
        columns.append("puck_traj_loss")
        return columns

    def load_real_data(self):
        num_trajectories = getattr(self, "num_trajectories", None)
        dt = getattr(self, "dt", 0.05)

        if num_trajectories is not None and num_trajectories != 0:
            # Multi-trajectory mode
            print(f"Loading multiple trajectories from {self.data_dir} "
                  f"(num_load={num_trajectories})")
            self.ground_truth_episodes = load_multiple_real_trajectories_for_sysid(
                self.data_dir, num_load=num_trajectories, dt=dt,
            )
        else:
            # Legacy single-trajectory mode
            traj_file = getattr(self, "trajectory_file", "trajectory_data.hdf5")
            print(f"Loading single trajectory: {traj_file}")
            result = load_real_trajectory_for_sysid(
                self.data_dir, traj_file, dt=dt,
            )
            self.ground_truth_episodes = [result[0]]

        total_obs = sum(len(ep["observations"]) for ep in self.ground_truth_episodes)
        total_hits = sum(sum(ep["hits_array"]) for ep in self.ground_truth_episodes)
        print(f"Loaded {len(self.ground_truth_episodes)} episodes, "
              f"{total_obs} total timesteps, {total_hits} hit timesteps")

    def get_value(self, param_vector, trajectories, sample_idx=0, iteration=0):
        candidate_config = assign_values(
            param_vector, self.param_names, self.sim_config
        )
        env = AirHockeyEnv(candidate_config)

        num_segments = trajectories["observations"].shape[0]
        traj_length = trajectories["traj_length"]
        dt = getattr(self, "dt", 0.05)

        all_eval_segments = []
        for i in range(num_segments):
            state_vector = trajectories["observations"][i, 0].copy()

            #finite-differences being used here
            if traj_length > 1:
                puck_pos_curr = trajectories["observations"][i, 0, 4:6]
                puck_pos_next = trajectories["observations"][i, 1, 4:6]
                state_vector[6:8] = (puck_pos_next - puck_pos_curr) / dt

            try:
                obs, _ = env.reset_from_state(state_vector)
            except Exception:
                return -1e6, None

            segment_obs = [obs.copy()]
            for j in range(traj_length - 1):
                action = trajectories["actions"][i, j]
                obs, _, _, _, _ = env.step(action)
                segment_obs.append(obs.copy())

            all_eval_segments.append(np.array(segment_obs))

        evaluated_states = np.stack(all_eval_segments, axis=0)
        value = self._compute_comparison(
            evaluated_states, trajectories, candidate_config
        )
        return value, None

    def _compute_comparison(self, evaluated_states, trajectories, config):
        """Compare puck trajectories (obs cols 4:8) between sim and real."""
        eval_sub = evaluated_states[:, :, 4:8]
        target_sub = trajectories["observations"][:, :, 4:8]

        if "simulator_params" in config:
            if "x_offset" in config["simulator_params"]:
                eval_sub = eval_sub.copy()
                eval_sub[:, :, 0] += config["simulator_params"]["x_offset"]
            if "y_offset" in config["simulator_params"]:
                eval_sub = eval_sub.copy()
                eval_sub[:, :, 1] += config["simulator_params"]["y_offset"]

        return compare_trajectories(eval_sub, target_sub, comp_type=self.comp_type)

    def _perturb_initial_params(self):
        initial = np.array(self.param_start_values)
        lower = np.array(self.lower_bounds)
        upper = np.array(self.upper_bounds)
        values = []
        for i in range(len(initial)):
            max_delta = min(initial[i] - lower[i], upper[i] - initial[i])
            perturbation = np.random.uniform(-max_delta, max_delta)
            values.append(initial[i] + perturbation)
        return np.array(values)

    def _save_run_config(self, run_dir):
        config = {
            "data_dir": self.data_dir,
            "trajectory_file": getattr(self, "trajectory_file", None),
            "num_trajectories": getattr(self, "num_trajectories", None),
            "num_episodes_loaded": len(self.ground_truth_episodes),
            "sys_id_path_puck": self.sys_id_path_puck,
            "paddle_params_path": getattr(self, "paddle_params_path", None),
            "paddle_params": (
                {k: float(v) for k, v in self.paddle_params.items()}
                if self.paddle_params else None
            ),
            "cfg": getattr(self, "cfg", None),
            "comp_type": self.comp_type,
            "dt": getattr(self, "dt", 0.05),
            "num_resamples": self.num_resamples,
            "traj_length": self.traj_length,
            "num_population": self.num_population,
            "num_iterations": self.num_iterations,
            "variance": self.variance,
            "simulation_iterations": self.simulation_iterations,
            "run_dir": self.run_dir,
            "run_id": self.run_id,
        }
        path = os.path.join(run_dir, "run_config.yaml")
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Saved run config to {path}")

    def _save_optimal_params(self, run_dir, optimal_params, best_error):
        result = {name: float(val) for name, val in zip(self.param_names, optimal_params)}
        result["best_error"] = float(best_error)
        path = os.path.join(run_dir, "optimal_params.yaml")
        with open(path, "w") as f:
            yaml.dump(result, f, default_flow_style=False)
        print(f"Saved optimal puck params to {path}")

    def run_simulation(self):
        run_dir = os.path.join(self.run_dir, str(self.run_id))
        os.makedirs(run_dir, exist_ok=True)
        self._save_run_config(run_dir)

        fig, ax = plt.subplots(figsize=(8, 5))

        overall_best_error = -1e10
        overall_best_params = None

        for i in range(self.simulation_iterations):
            print(f"\n{'='*60}")
            print(f"Puck optimization iteration {i}")
            print(f"{'='*60}")

            initial_params = self._perturb_initial_params()
            print(f"Initial params: {dict(zip(self.param_names, initial_params))}")

            log_file = os.path.join(run_dir, f"reward_log_iter{i}.csv")

            planner = CMAPlanner(
                eval_fn=lambda params, trajs, idx=0, it=0: self.get_value(
                    params, trajs, idx, it
                ),
                trajectories=self.ground_truth_episodes,
                n_samples=self.num_population,
                n_iterations=self.num_iterations,
                variance=self.variance,
                n_starts=self.num_resamples,
                traj_length=self.traj_length,
                lower_bounds=self.lower_bounds,
                upper_bounds=self.upper_bounds,
                param_names=self.param_names,
                wdb_logging=len(getattr(self, "wdb_entity", "")) > 0,
                log_file=log_file,
            )

            optimal_params, opt_iter, best_error, best_error_idx = planner.optimize(
                initial_params
            )

            print(f"Best params: {dict(zip(self.param_names, optimal_params))}")
            print(f"Best error:  {best_error:.6f} (iteration {best_error_idx})")

            if best_error > overall_best_error:
                overall_best_error = best_error
                overall_best_params = optimal_params

            df = pd.DataFrame(columns=self.dataframe_columns)
            row = list(initial_params) + list(optimal_params) + [float(best_error)]
            df.loc[0] = row

            ax.plot(planner.best_values_history, alpha=0.5)
            ax.set_title("Puck Loss Curves Across Trials")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Loss")
            fig.savefig(
                os.path.join(run_dir, "loss_curves.png"), dpi=150, bbox_inches="tight"
            )

            mode = "w" if i == 0 else "a"
            df.to_csv(
                os.path.join(run_dir, "data.csv"),
                index=False,
                mode=mode,
                header=(i == 0),
            )

        plt.close(fig)

        if overall_best_params is not None:
            self._save_optimal_params(run_dir, overall_best_params, overall_best_error)

        print(f"\nPuck results saved to {run_dir}")
        return overall_best_params, overall_best_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real data puck system identification")

    parser.add_argument("--cfg", type=str,default = "/home/air-hockey/air-hockey-rl/scripts/domain_adaptation/realworld_paddle_config/estimate.yaml",help="Optional YAML config (air_hockey key expected)")
    parser.add_argument("--data-dir", type=str,default="/data2/air_hockey/air_hockey_state_data/datastor1/calebc/public/data/mouse/state_data_all_new/",help="Directory containing trajectory HDF5 files")
    parser.add_argument("--trajectory-file", type=str, default=None,help="Single trajectory HDF5 filename (legacy mode). ""Ignored if --num-trajectories is set.")
    parser.add_argument("--num-trajectories", type=int, default=20,help="Number of trajectory files to load from data-dir. ""-1 for all. Set to 0 to use --trajectory-file instead.")
    parser.add_argument("--sys-id-path-puck", type=str,default="scripts/domain_adaptation/sys_id_configs/real2sim/puck_id_params.yaml",help="Puck system ID parameter bounds YAML")
    parser.add_argument("--paddle-params-path", type=str, default=None,help="Path to optimal paddle params YAML from paddle pipeline")

    parser.add_argument("--comp-type", type=str, default="posl2",help="Comparison metric: l2, posl2, l1, posl1, last")
    parser.add_argument("--dt", type=float, default=0.05,help="Timestep for puck velocity finite differences")

    parser.add_argument("--num-resamples", type=int, default=400,help="Number of free-space segments to sample per evaluation")
    parser.add_argument("--traj-length", type=int, default=20,help="Length of each trajectory segment")
    parser.add_argument("--num-population", type=int, default=17,help="CMA-ES population size")
    parser.add_argument("--num-iterations", type=int, default=100,help="CMA-ES iterations per trial")
    parser.add_argument("--variance", type=float, default=0.35,help="CMA-ES initial step size (sigma)")
    parser.add_argument("--simulation-iterations", type=int, default=1,help="Number of restarts with perturbed initial params")

    parser.add_argument("--run-dir", type=str,default="scripts/domain_adaptation/system_identification/puck/real2sim_runs",help="Directory for run results")
    parser.add_argument("--run-id", type=str, default="14")
    parser.add_argument("--wdb-entity", type=str, default="")

    args = parser.parse_args()

    pipeline = RealDataPuckPipeline(**vars(args))
    pipeline.load_real_data()
    pipeline.run_simulation()

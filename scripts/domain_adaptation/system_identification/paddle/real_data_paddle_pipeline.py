"""Real-data paddle system identification pipeline.

Finds Box2D physics parameters that best reproduce a real robot trajectory
when the recorded actions are replayed in simulation. Uses CMA-ES optimization
with per-segment rollouts (reset env to each segment's initial state).

Unlike the sim2sim PaddlePipeline, there is NO source/base environment here.
We only have the real trajectory data (observations, actions). The Box2D
environment is constructed from known table constants and CMA-ES candidate
parameters -- the entire point is to discover the unknown physics parameters.

Usage:
    python scripts/domain_adaptation/system_identification/paddle/real_data_paddle_pipeline.py \
        --data-dir /data2/air_hockey/air_hockey_state_data/datastor1/calebc/public/data/mouse/state_data_all_new/ \
        --trajectory-file trajectory_data100.hdf5 \
        --sys-id-path-paddle scripts/domain_adaptation/sys_id_configs/real2sim/paddle_id_params.yaml \
        --num-population 20 --num-iterations 100 --traj-length 50 --num-resamples 20
"""

import argparse
import os
import sys
import copy

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

from airhockey import AirHockeyEnv
from scripts.domain_adaptation.planners import CMAPlanner
from scripts.domain_adaptation.encode_env_params import assign_values

# Add real_data_transforms to path for the conversion utility
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../real_data_transforms")
)
from scripts.real_data_transforms.real_to_sim_observations import load_real_trajectory_for_sysid
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
        step_rmse = np.sqrt(np.mean(diffs**2, axis=2))  # RMSE per step across features
        traj_loss = np.mean(step_rmse, axis=1)            # average across timesteps
        return -np.mean(traj_loss)    
    if comp_type == "posl2":
        diffs = a_traj[..., :2] - b_traj[..., :2]
        step_rmse = np.sqrt(np.mean(diffs**2, axis=2))  # RMSE per step across features
        traj_loss = np.mean(step_rmse, axis=1)            # average across timesteps
        return -np.mean(traj_loss)
    if comp_type == "l1":
        step_l1 = np.sum(np.abs(diffs), axis=2)
        traj_loss = np.mean(step_l1, axis=1)
        return -np.mean(traj_loss)
    if comp_type == "posl1":
        pos_diffs = a_traj[..., :2] - b_traj[..., :2]
        step_l1 = np.sum(np.abs(pos_diffs), axis=2)
        traj_loss = np.mean(step_l1, axis=1)
        return -np.mean(traj_loss)
    if comp_type == "last":
        return -np.linalg.norm(a_traj[-1] - b_traj[-1])


def build_sim_config():
    """Build a Box2D env config from known table constants.

    There is no source/base environment for real2sim -- we only know the
    physical table dimensions and structural parameters. All physics
    parameters (force_scaling, paddle_damping, paddle_density, etc.) are
    placeholders that CMA-ES will overwrite via assign_values().

    Returns:
        dict: Minimal config sufficient to create AirHockeyEnv(config).
    """
    return {
        # Env-level params required by AirHockeyBaseEnv.__init__
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
        # Disable all termination -- we are replaying recorded actions
        "terminate_on_enemy_goal": False,
        "terminate_on_out_of_bounds": False,
        "terminate_on_puck_hit_bottom": False,
        "terminate_on_puck_pass_paddle": False,
        "terminate_on_puck_stop": False,
        # Reward params (unused during sys-id, but env requires them)
        "diagonal_motion_rew": 0,
        "direction_change_rew": 0,
        "horizontal_vel_rew": 0,
        "stand_still_rew": 0,
        "truncate_rew": 0,
        "wall_bumping_rew": 0,
        "goal_max_x_velocity": 1,
        "goal_max_y_velocity": 5,
        "goal_min_y_velocity": 1,
        # Known structural parameters of the real table
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
            # Physics placeholders -- CMA-ES will overwrite these
            "force_scaling": 1000,
            "paddle_damping": 3,
            "paddle_density": 2500,
            "puck_damping": 0.5,
            "puck_density": 250,
            "gravity": -0.5,
        },
    }


class RealDataPaddlePipeline:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        # Load sim config from YAML if provided, otherwise build from constants
        if getattr(self, "cfg", None) is not None:
            with open(self.cfg, "r") as f:
                cfg = yaml.safe_load(f)
            self.sim_config = cfg["air_hockey"]
        else:
            self.sim_config = build_sim_config()

        self.param_names, self.lower_bounds, self.upper_bounds, self.param_start_values = (
            self._load_sys_id_yaml()
        )
        self.dataframe_columns = self._make_dataframe_columns()
        self.df = pd.DataFrame(columns=self.dataframe_columns)
        self.ground_truth_results = None

    # ── Config loading ────────────────────────────────────────────

    def _load_sys_id_yaml(self):
        """Load parameter names, bounds, and initial values from sys-id config."""
        with open(self.sys_id_path_paddle, "r") as f:
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
        columns.append(f"{self.object_name}_traj_loss")
        return columns

    # ── Real data loading ─────────────────────────────────────────

    def load_real_data(self):
        """Load real trajectory data and convert to pipeline format.

        The real data provides observations and actions -- there is no
        source environment. CMA-ES will search for Box2D parameters that
        reproduce these observations when the actions are replayed.
        """
        print(f"Loading real trajectory: {self.trajectory_file}")
        print(f"Data directory: {self.data_dir}")

        self.ground_truth_results = load_real_trajectory_for_sysid(
            self.data_dir,
            self.trajectory_file,
            dt=getattr(self, "dt", 0.05),
        )

        ep_data = self.ground_truth_results[0]
        print(f"Loaded trajectory: {len(ep_data['observations'])} timesteps")
        print(f"  Observation shape: {ep_data['observations'].shape}")
        print(f"  Action shape:      {ep_data['actions'].shape}")
        n_hits = sum(ep_data["hits_array"])
        print(f"  Hit timesteps:     {n_hits} / {len(ep_data['hits_array'])}")

    # ── Per-segment evaluation ────────────────────────────────────

    def get_value(self, param_vector, trajectories, sample_idx=0, iteration=0):
        """Evaluate candidate parameters on sampled trajectory segments.

        For each sampled segment:
          1. Create a Box2D env with the candidate physics params
          2. Reset to the segment's initial observation state
          3. Replay segment actions
          4. Compare resulting paddle trajectory vs real
        """
        # Overlay candidate physics params onto the sim config
        candidate_config = assign_values(
            param_vector, self.param_names, self.sim_config
        )
        env = AirHockeyEnv(candidate_config)

        num_segments = trajectories["observations"].shape[0]
        traj_length = trajectories["traj_length"]

        all_eval_segments = []
        for i in range(num_segments):
            state_vector = trajectories["observations"][i, 0]
            try:
                obs, _ = env.reset_from_state(state_vector)
            except Exception:
                return -1e6, None

            segment_obs = [obs.copy()]
            for j in range(traj_length - 1):
                action = trajectories["actions"][i, j]
                obs, _, terminated, truncated, _ = env.step(action)
                segment_obs.append(obs.copy())
                # if terminated or truncated:
                #     while len(segment_obs) < traj_length:
                #         segment_obs.append(obs.copy())
                #     break

            all_eval_segments.append(np.array(segment_obs))

        evaluated_states = np.stack(all_eval_segments, axis=0)

        # Convert sim observations from Box2D coords back to base coords.
        # Box2D applies (x,y) -> (y,-x) on spawn; inverse is (bx,by) -> (-by, bx).
        # evaluated_states = self._box2d_obs_to_base(evaluated_states)

        value = self._compute_comparison(
            evaluated_states, trajectories, candidate_config
        )
        return value, None

    @staticmethod
    def _box2d_obs_to_base(obs):
        """Convert observations from Box2D coords back to base coords.

        Box2D internal coords are rotated 90 degrees from base coords:
            base_to_box2d: (x, y) -> (y, -x)
            box2d_to_base: (bx, by) -> (-by, bx)

        Applies to paddle pos (cols 0-1), paddle vel (cols 2-3),
        puck pos (cols 4-5), and puck vel (cols 6-7).
        """
        out = obs.copy()
        for col_start in (0, 2, 4, 6):
            if obs.shape[-1] > col_start + 1:
                bx = obs[..., col_start]
                by = obs[..., col_start + 1]
                out[..., col_start] = -by
                out[..., col_start + 1] = bx
        return out

    def _compute_comparison(self, evaluated_states, trajectories, config):
        """Compare evaluated sim states with ground truth segments."""
        if self.object_name == "paddle":
            eval_sub = evaluated_states[:, :, :4]
            target_sub = trajectories["observations"][:, :, :4]
        elif self.object_name == "puck":
            eval_sub = evaluated_states[:, :, 4:8]
            target_sub = trajectories["observations"][:, :, 4:8]
        else:  # "all"
            eval_sub = evaluated_states
            target_sub = trajectories["observations"]

        # Apply offsets if CMA-ES is also optimizing offset params
        if "simulator_params" in config:
            if "x_offset" in config["simulator_params"]:
                eval_sub = eval_sub.copy()
                eval_sub[:, :, 0] += config["simulator_params"]["x_offset"]
            if "y_offset" in config["simulator_params"]:
                eval_sub = eval_sub.copy()
                eval_sub[:, :, 1] += config["simulator_params"]["y_offset"]

        return compare_trajectories(eval_sub, target_sub, comp_type=self.comp_type)

    # ── CMA-ES optimization loop ─────────────────────────────────

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
        """Save the CMA-ES / argparse parameters used for this run."""
        config = {
            "data_dir": self.data_dir,
            "trajectory_file": self.trajectory_file,
            "sys_id_path_paddle": self.sys_id_path_paddle,
            "cfg": getattr(self, "cfg", None),
            "object_name": self.object_name,
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
        """Save the best parameters found by CMA-ES."""
        result = {name: float(val) for name, val in zip(self.param_names, optimal_params)}
        result["best_error"] = float(best_error)
        path = os.path.join(run_dir, "optimal_params.yaml")
        with open(path, "w") as f:
            yaml.dump(result, f, default_flow_style=False)
        print(f"Saved optimal params to {path}")

    def run_simulation(self):
        """Run CMA-ES optimization to find best Box2D parameters."""
        run_dir = os.path.join(self.run_dir, str(self.run_id))
        os.makedirs(run_dir, exist_ok=True)

        self._save_run_config(run_dir)

        fig, ax = plt.subplots(figsize=(8, 5))

        overall_best_error = -1e10
        overall_best_params = None

        for i in range(self.simulation_iterations):
            print(f"\n{'='*60}")
            print(f"Simulation iteration {i}")
            print(f"{'='*60}")

            initial_params = self._perturb_initial_params()
            print(f"Initial params: {dict(zip(self.param_names, initial_params))}")

            planner = CMAPlanner(
                eval_fn=lambda params, trajs, idx=0, it=0: self.get_value(
                    params, trajs, idx, it
                ),
                trajectories=self.ground_truth_results,
                n_samples=self.num_population,
                n_iterations=self.num_iterations,
                variance=self.variance,
                n_starts=self.num_resamples,
                traj_length=self.traj_length,
                lower_bounds=self.lower_bounds,
                upper_bounds=self.upper_bounds,
                param_names=self.param_names,
                wdb_logging=len(getattr(self, "wdb_entity", "")) > 0,
            )

            optimal_params, opt_iter, best_error, best_error_idx = planner.optimize(
                initial_params
            )

            print(f"Best params: {dict(zip(self.param_names, optimal_params))}")
            print(f"Best error:  {best_error:.6f} (iteration {best_error_idx})")

            # Track best across all simulation iterations
            if best_error > overall_best_error:
                overall_best_error = best_error
                overall_best_params = optimal_params

            # Log to dataframe
            df = pd.DataFrame(columns=self.dataframe_columns)
            row = list(initial_params) + list(optimal_params) + [float(best_error)]
            df.loc[0] = row

            # Plot loss curve
            ax.plot(planner.best_values_history, alpha=0.5)
            ax.set_title(f"Loss Curves: {self.comp_type}")
            
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Loss")
            fig.savefig(
                os.path.join(run_dir, "loss_curves.png"), dpi=150, bbox_inches="tight"
            )

            # Append to CSV
            mode = "w" if i == 0 else "a"
            header = i == 0
            df.to_csv(
                os.path.join(run_dir, "data.csv"),
                index=False,
                mode=mode,
                header=header,
            )

        plt.close(fig)

        if overall_best_params is not None:
            self._save_optimal_params(run_dir, overall_best_params, overall_best_error)

        print(f"\nResults saved to {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real data paddle system identification")

    # Data arguments
    parser.add_argument("--cfg",type=str,default=None,help="Path to a YAML config file (air_hockey key expected). If provided, uses this instead of build_sim_config().",)
    parser.add_argument("--data-dir",type=str,default="/data2/air_hockey/air_hockey_state_data/datastor1/calebc/public/data/mouse/state_data_all_new/",help="Directory containing trajectory HDF5 files",)
    parser.add_argument("--trajectory-file",type=str,default="trajectory_data.hdf5",help="Trajectory HDF5 filename",)
    parser.add_argument("--sys-id-path-paddle",type=str,default="scripts/domain_adaptation/sys_id_configs/real2sim/paddle_id_params.yaml",help="System ID parameter bounds YAML",)

    # Comparison
    parser.add_argument("--object-name",type=str,default="paddle",help="Object to compare: paddle, puck, or all",)
    parser.add_argument("--comp-type",type=str,default="posl1",help="Comparison metric: l2, posl2, l1, posl1, last, dtw",)
    parser.add_argument("--dt",type=float,default=0.05,help="Timestep for puck velocity finite differences",)

    # CMA-ES parameters
    parser.add_argument("--num-resamples",type=int,default=262,help="Number of trajectory segments to sample per evaluation",)
    parser.add_argument("--traj-length", type=int, default=10, help="Length of each trajectory segment")
    parser.add_argument("--num-population", type=int, default=18, help="CMA-ES population size")
    parser.add_argument("--num-iterations", type=int, default=100, help="CMA-ES iterations per trial")
    parser.add_argument("--variance",type=float,default=0.4,help="CMA-ES initial step size (sigma)",)
    parser.add_argument("--simulation-iterations",type=int,default=1,help="Number of restarts with perturbed initial params",)

    # Logging
    parser.add_argument("--run-dir",type=str,default="scripts/domain_adaptation/system_identification/paddle/real2sim_runs",help="Directory for run results",)
    parser.add_argument("--run-id", type=str, default="15", help="Run identifier")
    parser.add_argument("--wdb-entity", type=str, default="", help="W&B entity for logging")

    args = parser.parse_args()

    pipeline = RealDataPaddlePipeline(**vars(args))
    pipeline.load_real_data()
    pipeline.run_simulation()

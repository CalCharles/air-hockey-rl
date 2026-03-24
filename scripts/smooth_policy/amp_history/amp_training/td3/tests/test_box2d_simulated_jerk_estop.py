import unittest
from pathlib import Path

import numpy as np
import yaml

from airhockey.airhockey_base import AirHockeyBaseEnv
from airhockey.sims.airhockey_box2d import AirHockeyBox2D


class _BaseEnvShell(AirHockeyBaseEnv):
    @staticmethod
    def from_dict(state_dict):
        raise NotImplementedError

    def initialize_spaces(self, obs_type):
        raise NotImplementedError

    def create_world_objects(self):
        raise NotImplementedError

    def validate_configuration(self):
        raise NotImplementedError

    def get_observation(self, state_info, obs_type="vel", **kwargs):
        raise NotImplementedError


class SimulatedJerkEstopTests(unittest.TestCase):
    @staticmethod
    def _default_simulator_params() -> dict:
        repo_root = Path(__file__).resolve().parents[6]
        config_path = (
            repo_root
            / "scripts"
            / "smooth_policy"
            / "amp_history"
            / "configs"
            / "new_juggle"
            / "pid_noise_constant_upper_half_custom_sim_params.yaml"
        )
        with config_path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        simulator_params = dict(loaded["air_hockey"]["simulator_params"])
        simulator_params["seed"] = int(loaded["air_hockey"].get("seed", 0))
        return simulator_params

    def test_box2d_simulated_jerk_estop_latches(self):
        sim_params = self._default_simulator_params()
        sim_params.update(
            {
                "simulate_jerk_estop": True,
                "jerk_estop_consecutive_steps": 3,
                "jerk_estop_consecutive_threshold": 0.5,
                "jerk_estop_avg_window_steps": 10,
                "jerk_estop_avg_threshold": 100.0,
                "enable_random_occlusions": False,
                "puck_noise": False,
            }
        )
        sim = AirHockeyBox2D(**sim_params)
        sim.spawn_paddle(pos=(-0.6, 0.0), vel=(0.0, 0.0), name="paddle_ego")

        def _high_jerk(initial_vel, final_vel):
            return np.zeros(2, dtype=float), np.array([1.0, 0.0], dtype=float)

        sim._update_motion_derivatives = _high_jerk

        state = None
        for _ in range(3):
            state = sim.get_transition(np.zeros(2, dtype=float))
        self.assertIsNotNone(state)
        self.assertTrue(bool(state.get("protective_stop", False)))

        def _low_jerk(initial_vel, final_vel):
            return np.zeros(2, dtype=float), np.zeros(2, dtype=float)

        sim._update_motion_derivatives = _low_jerk
        state_after = sim.get_transition(np.zeros(2, dtype=float))
        self.assertTrue(bool(state_after.get("protective_stop", False)))

    def test_has_finished_terminates_on_protective_stop(self):
        env = object.__new__(_BaseEnvShell)
        env.current_timestep = 0
        env.max_timesteps = 100
        env.terminate_on_out_of_bounds = False
        env.terminate_on_enemy_goal = False
        env.terminate_on_puck_stop = False
        env.terminate_on_puck_hit_bottom = False
        env.terminate_on_puck_pass_paddle = False
        env.terminate_on_puck_hit_paddle = False
        env._puck_pass_paddle_score = 0
        env._last_done_reasons = {"terminated": [], "truncated": []}
        env.table_x_top = -0.5
        env.table_x_bot = 0.5
        env.table_y_left = -0.3
        env.table_y_right = 0.3
        env.puck_radius = 0.03
        env.puck_hit_bottom_boundary_m = 0.03

        state_info = {
            "protective_stop": True,
            "paddles": {
                "paddle_ego": {
                    "position": (0.0, 0.0),
                    "velocity": (0.0, 0.0),
                    "acceleration": (0.0, 0.0),
                    "jerk": (0.0, 0.0),
                }
            },
            "pucks": [{"position": (0.0, 0.0), "history": [(0.0, 0.0, 0.0)]}],
        }

        terminated, truncated, _, _, _, _ = env.has_finished(state_info)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertIn("protective_stop", env._last_done_reasons["terminated"])


if __name__ == "__main__":
    unittest.main()

"""Reach / reach-velocity goals must land inside the reachable paddle workspace.

The env-level ``paddle_bounds`` used for action masking are ~3x wider than what
the paddle can actually reach: the simulator re-clips every PID target to the
robot workspace (``x_min_lim`` / ``x_max_lim`` / ``y_min`` / ``y_max``).  Goals
used to be sampled from the former, which made most of them unreachable.
"""

import unittest
from pathlib import Path

import numpy as np
import yaml

from airhockey import AirHockeyEnv


REPO_ROOT = Path(__file__).resolve().parents[3]


class ReachGoalWorkspaceTests(unittest.TestCase):
    def _make_env(self, config_name, **overrides):
        config_path = REPO_ROOT / "configs" / "new_juggle" / "throughput_bench" / config_name
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)["air_hockey"]
        cfg["seed"] = 7
        cfg.update(overrides)
        return AirHockeyEnv(cfg)

    def _workspace(self, env):
        """Reachable paddle-centre box, straight from the simulator clip."""
        sim = env.simulator
        offset = sim.center_offset_constant
        return (
            sim.x_min_lim + offset,
            sim.x_max_lim + offset,
            sim.y_min,
            sim.y_max,
        )

    def _sample_goals(self, env, n=2000):
        goals = []
        for _ in range(n):
            env.set_goals(env.goal_radius_type)
            goals.append(env.get_desired_goal())
        return np.asarray(goals)

    def test_reach_goals_inside_workspace(self):
        env = self._make_env("sim_nodr_reach.yaml")
        try:
            x_lo, x_hi, y_lo, y_hi = self._workspace(env)
            goals = self._sample_goals(env)
            margin = env.goal_position_margin
            self.assertGreaterEqual(goals[:, 0].min(), x_lo + margin - 1e-9)
            self.assertLessEqual(goals[:, 0].max(), x_hi - margin + 1e-9)
            self.assertGreaterEqual(goals[:, 1].min(), y_lo + margin - 1e-9)
            self.assertLessEqual(goals[:, 1].max(), y_hi - margin + 1e-9)
            # The whole goal disc, not just its centre, must be reachable.
            self.assertGreaterEqual(margin, env.goal_radius)
        finally:
            env.close()

    def test_reach_velocity_goals_inside_workspace_with_runway(self):
        env = self._make_env("sim_nodr_reach_vel.yaml")
        try:
            x_lo, x_hi, y_lo, y_hi = self._workspace(env)
            goals = self._sample_goals(env)
            margin = env.goal_position_margin
            self.assertGreaterEqual(goals[:, 0].min(), x_lo + margin - 1e-9)
            self.assertLessEqual(goals[:, 0].max(), x_hi - margin + 1e-9)
            self.assertGreaterEqual(goals[:, 1].min(), y_lo + margin - 1e-9)
            self.assertLessEqual(goals[:, 1].max(), y_hi - margin + 1e-9)
            # Goals need room to accelerate to / decelerate from the target
            # velocity; ~0.12 m at the 2 m/s speed clamp.
            self.assertGreaterEqual(margin, 0.10)
        finally:
            env.close()

    def test_reach_velocity_goal_speed_within_paddle_clamp(self):
        env = self._make_env("sim_nodr_reach_vel.yaml")
        try:
            goals = self._sample_goals(env)
            speeds = np.linalg.norm(goals[:, 2:], axis=1)
            self.assertLessEqual(speeds.max(), env.max_paddle_vel + 1e-9)
            # The clamp must not flatten the distribution onto the cap.
            self.assertLess(np.mean(speeds >= env.max_paddle_vel - 1e-9), 0.2)
        finally:
            env.close()

    def test_goal_observation_space_contains_sampled_goals(self):
        for config_name in ("sim_nodr_reach.yaml", "sim_nodr_reach_vel.yaml"):
            with self.subTest(config=config_name):
                env = self._make_env(config_name)
                try:
                    goals = self._sample_goals(env, n=500)
                    low = env.observation_space.low[-goals.shape[1]:]
                    high = env.observation_space.high[-goals.shape[1]:]
                    self.assertTrue(np.all(goals >= low - 1e-9))
                    self.assertTrue(np.all(goals <= high + 1e-9))
                finally:
                    env.close()


if __name__ == "__main__":
    unittest.main()

"""Episode-start randomization: one puck spawn scheme, workspace paddle spawns.

The puck spawns uniformly over the top ``puck_spawn_top_fraction`` of the table
with a random heading (the old 85 % "linear top" / 15 % "near paddle" split is
gone), and every task that randomizes its paddle start does so inside the
workspace the paddle can actually reach — not the much looser ``paddle_bounds``.
"""

import math
import unittest
from pathlib import Path

import numpy as np
import yaml

from airhockey import AirHockeyEnv


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load(path, **overrides):
    with (REPO_ROOT / path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["air_hockey"]
    cfg["seed"] = 11
    cfg.update(overrides)
    return cfg


def _spawns(env, n=250):
    paddle, puck_pos, puck_vel = [], [], []
    for _ in range(n):
        env.reset()
        paddle.append(env.current_state["paddles"]["paddle_ego"]["position"])
        pucks = env.current_state.get("pucks") or []
        if pucks:
            puck_pos.append(pucks[0]["position"])
            puck_vel.append(pucks[0]["velocity"])
    return np.array(paddle), np.array(puck_pos), np.array(puck_vel)


class PuckSpawnTests(unittest.TestCase):
    def test_uniform_over_the_top_fraction_with_random_heading(self):
        cfg = _load("configs/new_juggle/sysid_best_params_hist2.yaml")
        cfg["simulator_params"]["puck_noise"] = False
        cfg["simulator_params"]["enable_random_occlusions"] = False
        cfg["simulator_params"]["enable_puck_delay_interpolation"] = False
        env = AirHockeyEnv(cfg)
        try:
            _, pos, vel = _spawns(env)
            x_low = env.table_x_top + env.puck_radius
            x_high = (
                env.table_x_top
                + env.puck_spawn_top_fraction * env.length
                - env.puck_radius
            )
            self.assertGreaterEqual(pos[:, 0].min(), x_low - 1e-9)
            self.assertLessEqual(pos[:, 0].max(), x_high + 1e-9)
            # Spread over the allowed band rather than pinned to the top edge.
            self.assertLess(pos[:, 0].min(), x_low + 0.1 * (x_high - x_low))
            self.assertGreater(pos[:, 0].max(), x_high - 0.1 * (x_high - x_low))

            self.assertGreaterEqual(pos[:, 1].min(), env.table_y_left + env.puck_radius - 1e-9)
            self.assertLessEqual(pos[:, 1].max(), env.table_y_right - env.puck_radius + 1e-9)

            speed = np.linalg.norm(vel, axis=1)
            self.assertGreaterEqual(speed.min(), env.puck_spawn_speed_min - 1e-9)
            self.assertLessEqual(speed.max(), env.puck_spawn_speed_max + 1e-9)

            # Headings cover the circle: the mean unit vector is near zero and
            # every quadrant is used.
            heading = np.arctan2(vel[:, 1], vel[:, 0])
            self.assertLess(abs(np.mean(np.exp(1j * heading))), 0.25)
            quadrants = {int((h + math.pi) // (math.pi / 2)) for h in heading}
            self.assertGreaterEqual(len(quadrants), 4)
        finally:
            env.close()

    def test_puck_velocity_task_uses_the_same_spawn(self):
        cfg = _load("configs/new_juggle/throughput_bench/sim_nodr_puck_vel.yaml")
        cfg["simulator_params"]["puck_noise"] = False
        cfg["simulator_params"]["enable_random_occlusions"] = False
        cfg["simulator_params"]["enable_puck_delay_interpolation"] = False
        env = AirHockeyEnv(cfg)
        try:
            _, pos, vel = _spawns(env)
            x_high = (
                env.table_x_top
                + env.puck_spawn_top_fraction * env.length
                - env.puck_radius
            )
            self.assertLessEqual(pos[:, 0].max(), x_high + 1e-9)
            self.assertLessEqual(np.linalg.norm(vel, axis=1).max(), env.puck_spawn_speed_max + 1e-9)
            # Not the old fixed straight-down launch.
            self.assertGreater(np.std(np.arctan2(vel[:, 1], vel[:, 0])), 0.5)
        finally:
            env.close()

    def test_top_fraction_is_configurable(self):
        cfg = _load(
            "configs/new_juggle/sysid_best_params_hist2.yaml",
            puck_spawn_top_fraction=0.25,
            puck_spawn_speed_max=0.1,
        )
        cfg["simulator_params"]["puck_noise"] = False
        cfg["simulator_params"]["enable_random_occlusions"] = False
        cfg["simulator_params"]["enable_puck_delay_interpolation"] = False
        env = AirHockeyEnv(cfg)
        try:
            _, pos, vel = _spawns(env, n=100)
            x_high = env.table_x_top + 0.25 * env.length - env.puck_radius
            self.assertLessEqual(pos[:, 0].max(), x_high + 1e-9)
            self.assertLessEqual(np.linalg.norm(vel, axis=1).max(), 0.1 + 1e-9)
        finally:
            env.close()


class PaddleSpawnTests(unittest.TestCase):
    RANDOM_SPAWN_CONFIGS = [
        "configs/new_juggle/sysid_best_params_hist2.yaml",
        "configs/new_juggle/throughput_bench/sim_nodr_touch.yaml",
        "configs/new_juggle/throughput_bench/sim_nodr_reach.yaml",
        "configs/new_juggle/throughput_bench/sim_nodr_reach_vel.yaml",
    ]

    def test_spawns_are_random_and_inside_the_reachable_workspace(self):
        for path in self.RANDOM_SPAWN_CONFIGS:
            with self.subTest(config=path):
                env = AirHockeyEnv(_load(path))
                try:
                    self.assertTrue(env.random_paddle_spawn)
                    paddle, _, _ = _spawns(env)
                    x_low, x_high, y_low, y_high = env.get_paddle_workspace_bounds()
                    self.assertGreaterEqual(paddle[:, 0].min(), x_low - 1e-9)
                    self.assertLessEqual(paddle[:, 0].max(), x_high + 1e-9)
                    self.assertGreaterEqual(paddle[:, 1].min(), y_low - 1e-9)
                    self.assertLessEqual(paddle[:, 1].max(), y_high + 1e-9)
                    # Actually spread out, not a constant pose.
                    self.assertGreater(paddle[:, 0].std(), 0.05)
                    self.assertGreater(paddle[:, 1].std(), 0.05)
                finally:
                    env.close()

    def test_can_be_disabled_from_the_config(self):
        env = AirHockeyEnv(
            _load(
                "configs/new_juggle/sysid_best_params_hist2.yaml",
                random_paddle_spawn=False,
            )
        )
        try:
            paddle, _, _ = _spawns(env, n=20)
            self.assertAlmostEqual(paddle[:, 0].std(), 0.0, places=9)
            self.assertAlmostEqual(paddle[:, 1].std(), 0.0, places=9)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()

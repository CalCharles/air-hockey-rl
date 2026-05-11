import math
import unittest
from pathlib import Path

import yaml

from airhockey import AirHockeyEnv


class NearPaddleSpawnTests(unittest.TestCase):
    def _load_base_cfg(self):
        config_path = (
            Path(__file__).resolve().parents[3]
            / "configs"
            / "new_juggle"
            / "sysid_best_params.yaml"
        )
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)["air_hockey"]
        return cfg

    def _make_env(self, **overrides):
        cfg = self._load_base_cfg()

        cfg["task"] = "puck_juggle_linear_top"
        cfg["seed"] = 123
        cfg["num_pucks"] = 1
        cfg["obs_type"] = "vel"
        # current_state['pucks'][i]['position'] is the *observed* (occlusion-aware)
        # position; disable occlusion so spawn-bounds assertions read the true puck
        # position rather than the (-0.8, 0.0) fallback used when occluded.
        cfg.setdefault("simulator_params", {})["enable_random_occlusions"] = False
        cfg.update(overrides)
        return AirHockeyEnv(cfg), cfg

    def test_linear_top_spawn_x_cutoff_and_speed_bounds(self):
        env, cfg = self._make_env(
            puck_spawn_near_paddle_prob=0.0,
            puck_linear_top_spawn_goal_cutoff_x=-0.60,
            puck_linear_top_spawn_center_cutoff_x=-0.30,
            puck_linear_top_spawn_speed_min=0.4,
            puck_linear_top_spawn_speed_max=0.45,
        )
        try:
            for seed in range(10):
                env.reset(seed=seed)
                puck = env.current_state["pucks"][0]
                puck_x = puck["position"][0]
                min_x = cfg["puck_linear_top_spawn_goal_cutoff_x"]
                max_x = (
                    cfg["puck_linear_top_spawn_center_cutoff_x"]
                    - cfg["simulator_params"]["puck_radius"]
                )

                self.assertGreaterEqual(puck_x, min_x - 1e-9)
                self.assertLessEqual(puck_x, max_x + 1e-9)

                vx, vy = puck["velocity"]
                speed = math.hypot(vx, vy)
                self.assertGreaterEqual(
                    speed, cfg["puck_linear_top_spawn_speed_min"] - 1e-9
                )
                self.assertLessEqual(
                    speed, cfg["puck_linear_top_spawn_speed_max"] + 1e-9
                )
        finally:
            env.close()

    def test_near_paddle_spawn_offset_and_speed_bounds(self):
        env, cfg = self._make_env(
            paddle_bounds=[0.2, 0.8, -0.2, 0.2],
            puck_spawn_near_paddle_prob=1.0,
            puck_near_paddle_offset_min_m=0.025,
            puck_near_paddle_offset_max_m=0.05,
            puck_near_paddle_horizontal_std_m=0.015,
            puck_near_paddle_speed_min_m_s=0.1,
            puck_near_paddle_speed_max_m_s=0.2,
        )
        try:
            env.reset(seed=123)
            state = env.current_state
            puck = state["pucks"][0]
            paddle = state["paddles"]["paddle_ego"]

            puck_x, puck_y = puck["position"]
            paddle_x, paddle_y = paddle["position"]
            dx = paddle_x - puck_x
            dy = puck_y - paddle_y

            self.assertGreaterEqual(
                dx, cfg["puck_near_paddle_offset_min_m"] - 1e-9
            )
            self.assertLessEqual(
                dx, cfg["puck_near_paddle_offset_max_m"] + 1e-9
            )

            std = cfg["puck_near_paddle_horizontal_std_m"]
            self.assertLessEqual(abs(dy), 6.0 * std)

            vx, vy = puck["velocity"]
            speed = math.hypot(vx, vy)
            self.assertGreaterEqual(speed, cfg["puck_near_paddle_speed_min_m_s"] - 1e-9)
            self.assertLessEqual(speed, cfg["puck_near_paddle_speed_max_m_s"] + 1e-9)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()

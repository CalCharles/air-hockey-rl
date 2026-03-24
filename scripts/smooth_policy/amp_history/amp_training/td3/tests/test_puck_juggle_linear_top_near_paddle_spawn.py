import math
import unittest
from pathlib import Path

import yaml

from airhockey import AirHockeyEnv


class NearPaddleSpawnTests(unittest.TestCase):
    def _make_env(self):
        config_path = (
            Path(__file__).resolve().parents[3]
            / "configs"
            / "new_juggle"
            / "pid_noise_constant_upper_half_custom_sim_params.yaml"
        )
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)["air_hockey"]

        cfg["task"] = "puck_juggle_linear_top"
        cfg["seed"] = 123
        cfg["num_pucks"] = 1
        cfg["obs_type"] = "vel"
        cfg["paddle_bounds"] = [0.2, 0.8, -0.2, 0.2]
        cfg["puck_spawn_near_paddle_prob"] = 1.0
        cfg["puck_near_paddle_offset_min_m"] = 0.025
        cfg["puck_near_paddle_offset_max_m"] = 0.05
        cfg["puck_near_paddle_horizontal_std_m"] = 0.015
        cfg["puck_near_paddle_speed_max_m_s"] = 0.2
        return AirHockeyEnv(cfg), cfg

    def test_near_paddle_spawn_offset_and_speed_bounds(self):
        env, cfg = self._make_env()
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
            self.assertLessEqual(speed, cfg["puck_near_paddle_speed_max_m_s"] + 1e-9)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()

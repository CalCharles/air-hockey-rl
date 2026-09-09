"""The three simple tasks use sparse rewards.

``paddle_reach_position`` / ``paddle_reach_position_velocity``: +1 on the step
the goal is met (which ends the episode), 0 on every other step.  ``puck_touch``:
+1 on the touch that ends the episode, 0 everywhere else.
"""

import unittest
from pathlib import Path

import numpy as np
import yaml

from airhockey import AirHockeyEnv


REPO_ROOT = Path(__file__).resolve().parents[3]


def _make_env(config_name, **overrides):
    config_path = REPO_ROOT / "configs" / "new_juggle" / "tasks" / config_name
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["air_hockey"]
    cfg["seed"] = 5
    cfg.update(overrides)
    return AirHockeyEnv(cfg)


class SparseReachRewardTests(unittest.TestCase):
    def test_zero_until_goal_then_plus_one_and_terminate(self):
        env = _make_env("sim_sysid_reach.yaml")
        try:
            env.reset()
            paddle = np.array(env.current_state["paddles"]["paddle_ego"]["position"])

            # Goal far from the paddle: 0, episode continues.
            env.goal_pos = np.array([paddle[0] - 0.3, paddle[1]])
            _, reward, done, truncated, info = env.step(np.zeros(2, dtype=np.float32))
            self.assertEqual(reward, 0.0)
            self.assertFalse(info["success"])
            self.assertFalse(done or truncated)

            # Goal on top of the paddle: +1, episode terminates as "goal_reached".
            paddle = np.array(env.current_state["paddles"]["paddle_ego"]["position"])
            env.goal_pos = paddle.copy()
            _, reward, done, truncated, info = env.step(np.zeros(2, dtype=np.float32))
            self.assertEqual(reward, 10.0)
            self.assertTrue(info["success"])
            self.assertTrue(done)
            self.assertIn("goal_reached", info["termination_reasons"])
        finally:
            env.close()

    def test_return_is_one_when_the_goal_is_reached(self):
        env = _make_env("sim_sysid_reach.yaml")
        try:
            env.reset()
            total, steps = 0.0, 0
            while True:
                paddle = np.array(env.current_state["paddles"]["paddle_ego"]["position"])
                err = np.array(env.goal_pos) - paddle
                action = np.array(
                    [np.clip(err[0] / 0.26, -1, 1), np.clip(err[1] / 0.12, -1, 1)],
                    dtype=np.float32,
                )
                _, reward, done, truncated, info = env.step(action)
                total += reward
                steps += 1
                if done or truncated:
                    break
            self.assertTrue(info["success"])
            # +1 on the terminal step, 0 on each step before it.
            self.assertEqual(total, 10.0)
            self.assertLessEqual(steps, env.max_timesteps + 1)
        finally:
            env.close()


class SparseReachVelocityRewardTests(unittest.TestCase):
    def test_position_alone_is_not_enough(self):
        env = _make_env("sim_sysid_reach_vel.yaml")
        try:
            env.reset()
            paddle = env.current_state["paddles"]["paddle_ego"]
            # Paddle parked on the goal position, but the goal asks for speed.
            env.goal_pos = np.array(paddle["position"][:2])
            env.goal_vel = np.array([-2.0, 0.0])
            _, reward, done, truncated, info = env.step(np.zeros(2, dtype=np.float32))
            self.assertEqual(reward, 0.0)
            self.assertFalse(info["success"])
            self.assertFalse(done or truncated)

            # Same position, and a velocity the stationary paddle does match.
            paddle = env.current_state["paddles"]["paddle_ego"]
            env.goal_pos = np.array(paddle["position"][:2])
            env.goal_vel = np.array(paddle["velocity"][:2])
            _, reward, done, truncated, info = env.step(np.zeros(2, dtype=np.float32))
            self.assertEqual(reward, 10.0)
            self.assertTrue(info["success"])
            self.assertTrue(done)
        finally:
            env.close()

    def test_tolerances_come_from_config(self):
        env = _make_env(
            "sim_sysid_reach_vel.yaml",
            base_goal_radius=0.07,
            base_goal_velocity_radius=0.25,
        )
        try:
            env.reset()
            self.assertAlmostEqual(env.goal_radius, 0.07)
            self.assertAlmostEqual(env.goal_velocity_radius, 0.25)
        finally:
            env.close()


class SparsePuckTouchRewardTests(unittest.TestCase):
    def test_plus_one_only_on_the_touch(self):
        env = _make_env("sim_sysid_touch.yaml")
        try:
            self.assertTrue(env.terminate_on_puck_hit_paddle)
            touched = 0
            for _ in range(20):
                env.reset()
                rewards = []
                while True:
                    paddle = np.array(env.current_state["paddles"]["paddle_ego"]["position"])
                    puck = np.array(env.current_state["pucks"][0]["position"])
                    err = puck - paddle
                    action = np.array(
                        [np.clip(err[0] / 0.26, -1, 1), np.clip(err[1] / 0.12, -1, 1)],
                        dtype=np.float32,
                    )
                    _, reward, done, truncated, info = env.step(action)
                    rewards.append(reward)
                    if done or truncated:
                        break
                if "puck_hit_paddle" in info["termination_reasons"]:
                    touched += 1
                    self.assertEqual(rewards[-1], 1.0)
                    self.assertEqual(sum(rewards), 1.0)
                    self.assertTrue(info["success"])
                else:
                    self.assertEqual(sum(rewards), 0.0)
            self.assertGreater(touched, 0)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()

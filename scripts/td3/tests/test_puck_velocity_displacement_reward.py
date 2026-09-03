"""``puck_velocity`` pays the puck's upward displacement, measured from positions.

The real robot reports puck positions and not velocities, so the reward must be
reproducible from two consecutive position readings, must be zero while the puck
travels downward, and must not invent a number for steps it cannot measure (the
first step of an episode, or a step where the puck was occluded).
"""

import unittest
from pathlib import Path

import numpy as np
import yaml

from airhockey import AirHockeyEnv


REPO_ROOT = Path(__file__).resolve().parents[3]


def _make_env(**sim_overrides):
    config_path = (
        REPO_ROOT / "configs" / "new_juggle" / "throughput_bench" / "sim_nodr_puck_vel.yaml"
    )
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["air_hockey"]
    cfg["seed"] = 4
    cfg["simulator_params"].update(sim_overrides)
    return AirHockeyEnv(cfg)


def _puck(env):
    puck = env.current_state["pucks"][0]
    occluded = float(np.asarray(puck.get("occluded", 0)).reshape(-1)[0]) > 0.5
    return float(puck["position"][0]), occluded


class PuckVelocityDisplacementRewardTests(unittest.TestCase):
    def test_reward_is_the_observed_upward_displacement(self):
        env = _make_env(puck_noise=False, enable_random_occlusions=False)
        try:
            for episode in range(5):
                env.reset()
                prev_x = None
                step = 0
                while True:
                    _, reward, done, truncated, _ = env.step(
                        env.action_space.sample()
                    )
                    puck_x, _ = _puck(env)
                    if step == 0:
                        # Nothing to difference against yet.
                        expected = 0.0
                    else:
                        expected = max(prev_x - puck_x, 0.0)
                    self.assertAlmostEqual(reward, expected, places=9)
                    prev_x = puck_x
                    step += 1
                    if done or truncated:
                        break
        finally:
            env.close()

    def test_downward_motion_pays_nothing(self):
        env = _make_env(puck_noise=False, enable_random_occlusions=False)
        try:
            env.reset()
            # The puck spawns at the top moving down; idle so it keeps falling.
            prev_x, _ = _puck(env)
            fell = 0
            for _ in range(10):
                _, reward, done, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
                puck_x, _ = _puck(env)
                if puck_x > prev_x:  # moved toward the agent, i.e. downward
                    fell += 1
                    self.assertEqual(reward, 0.0)
                prev_x = puck_x
                if done or truncated:
                    break
            self.assertGreater(fell, 0)
        finally:
            env.close()

    def test_no_velocity_signal_is_used(self):
        """A puck teleported upward pays, even with its velocity pointing down."""
        env = _make_env(puck_noise=False, enable_random_occlusions=False)
        try:
            env.reset()
            env.step(np.zeros(2, dtype=np.float32))
            before_x, _ = _puck(env)
            reward, success = env.reward.get_base_reward(
                {
                    "pucks": [{"position": (before_x - 0.07, 0.0), "velocity": (5.0, 0.0)}],
                    "paddles": env.current_state["paddles"],
                }
            )
            self.assertAlmostEqual(reward, 0.07, places=9)
        finally:
            env.close()

    def test_occluded_steps_score_zero(self):
        env = _make_env(random_occlusion_rate=0.5)
        try:
            occluded_steps = 0
            for _ in range(20):
                env.reset()
                while True:
                    _, reward, done, truncated, _ = env.step(env.action_space.sample())
                    _, occluded = _puck(env)
                    if occluded:
                        occluded_steps += 1
                        self.assertEqual(reward, 0.0)
                    if done or truncated:
                        break
            self.assertGreater(occluded_steps, 0)
        finally:
            env.close()

    def test_first_step_after_reset_scores_zero(self):
        env = _make_env(puck_noise=False, enable_random_occlusions=False)
        try:
            for _ in range(5):
                env.reset()
                _, reward, _, _, _ = env.step(np.zeros(2, dtype=np.float32))
                self.assertEqual(reward, 0.0)
                # Drain a couple of steps so the next reset crosses a non-zero
                # timestep, which is what the episode-boundary guard keys on.
                for _ in range(3):
                    env.step(np.zeros(2, dtype=np.float32))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()

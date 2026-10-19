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
from airhockey.airhockey_rewards.airhockey_simple_task_rewards import AirHockeyPuckVelReward


REPO_ROOT = Path(__file__).resolve().parents[3]


def _make_env(task_overrides=None, **sim_overrides):
    config_path = (
        REPO_ROOT / "configs" / "new_juggle" / "tasks" / "sim_sysid_puck_vel.yaml"
    )
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["air_hockey"]
    cfg["seed"] = 4
    cfg.update(task_overrides or {})
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
                        expected = AirHockeyPuckVelReward.DISPLACEMENT_SCALE * max(prev_x - puck_x, 0.0)
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
            self.assertAlmostEqual(reward, AirHockeyPuckVelReward.DISPLACEMENT_SCALE * 0.07, places=9)
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


class PuckVelocityAfterContactTests(unittest.TestCase):
    """``puck_vel_reward_after_contact``: nothing pays before the first paddle
    contact, which on the real robot excludes the reset policy's strike."""

    @staticmethod
    def _state(env, puck_x, contact=False, policy_in_control=True):
        return {
            "pucks": [{"position": (puck_x, 0.0), "velocity": (0.0, 0.0), "occluded": 0}],
            "paddles": env.current_state["paddles"],
            "paddle_puck_collision_count": int(contact),
            "policy_in_control": policy_in_control,
        }

    def _reward(self, env, state):
        # Consecutive timesteps, as env.step would give them.
        env.current_timestep += 1
        return env.reward.get_base_reward(state)

    def test_off_by_default(self):
        env = _make_env(puck_noise=False, enable_random_occlusions=False)
        try:
            self.assertFalse(env.puck_vel_reward_after_contact)
        finally:
            env.close()

    def test_rising_puck_pays_only_after_a_policy_contact(self):
        env = _make_env({"puck_vel_reward_after_contact": True},
                        puck_noise=False, enable_random_occlusions=False)
        try:
            env.reset()
            self._reward(env, self._state(env, -0.10))
            # Rising from the reset strike: no contact yet.
            reward, success = self._reward(env, self._state(env, -0.20))
            self.assertEqual(reward, 0.0)
            # Contact during the post-reset hold is not the policy's.
            reward, _ = self._reward(env, self._state(env, -0.30, contact=True, policy_in_control=False))
            self.assertEqual(reward, 0.0)
            reward, _ = self._reward(env, self._state(env, -0.40))
            self.assertEqual(reward, 0.0)
            # The policy's contact step and every step after it pay.
            reward, _ = self._reward(env, self._state(env, -0.45, contact=True))
            self.assertAlmostEqual(reward, AirHockeyPuckVelReward.DISPLACEMENT_SCALE * 0.05, places=9)
            reward, _ = self._reward(env, self._state(env, -0.55))
            self.assertAlmostEqual(reward, AirHockeyPuckVelReward.DISPLACEMENT_SCALE * 0.10, places=9)
            # A soft reset (the real-robot hand-off) starts a fresh episode.
            env.soft_reset()
            self._reward(env, self._state(env, -0.10))
            reward, _ = self._reward(env, self._state(env, -0.20))
            self.assertEqual(reward, 0.0)
        finally:
            env.close()

    def test_success_needs_a_contact(self):
        env = _make_env({"puck_vel_reward_after_contact": True},
                        puck_noise=False, enable_random_occlusions=False)
        try:
            env.reset()
            env.current_timestep = 30
            _, success = self._reward(env, self._state(env, -0.70))
            self.assertFalse(success)
            _, success = self._reward(env, self._state(env, -0.75, contact=True))
            self.assertTrue(success)
        finally:
            env.close()

    def test_sim_rollouts_match_ungated_from_first_contact(self):
        gated = _make_env({"puck_vel_reward_after_contact": True},
                          puck_noise=False, enable_random_occlusions=False)
        plain = _make_env(puck_noise=False, enable_random_occlusions=False)
        rng = np.random.default_rng(0)
        contacted_episodes = 0
        try:
            for episode in range(30):
                gated.reset(seed=episode)
                plain.reset(seed=episode)
                contacted = False
                while True:
                    action = rng.uniform(-1, 1, size=2).astype(np.float32)
                    _, r_gated, done, truncated, info = gated.step(action)
                    _, r_plain, _, _, _ = plain.step(action)
                    contacted = contacted or info.get("paddle_puck_collision_count", 0) > 0
                    self.assertAlmostEqual(r_gated, r_plain if contacted else 0.0, places=9)
                    if done or truncated:
                        break
                contacted_episodes += int(contacted)
            self.assertGreater(contacted_episodes, 0)
        finally:
            gated.close()
            plain.close()


if __name__ == "__main__":
    unittest.main()

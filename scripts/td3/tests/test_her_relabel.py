"""Hindsight relabelling (scripts/td3/helper/td3_her.py) and the sparse
goal-conditioned puck tasks it trains on."""

import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from airhockey import AirHockeyEnv
from scripts.td3.helper.replay_buffer import TD3ReplayBuffer
from scripts.td3.helper.td3_her import (
    GoalEnvVector,
    HEREpisodeTrajectory,
    HERRelabeler,
    make_goal_functions,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
OBS_DIM = 3
GOAL_DIM = 2
RADIUS = 0.1


def _toy_goal_met(achieved, desired):
    # achieved = (x, y, vx): met when within RADIUS and moving "up" (vx < 0).
    dist = np.linalg.norm(achieved[:, :2] - desired[:, :2], axis=1)
    return (dist <= RADIUS) & (achieved[:, 2] < 0)


def _toy_reward(achieved, desired):
    return np.where(_toy_goal_met(achieved, desired), 10.0, 0.0)


def _toy_episode(T=6, seed=0):
    rng = np.random.default_rng(seed)
    obs = rng.normal(size=(T, OBS_DIM + GOAL_DIM)).astype(np.float32)
    next_obs = rng.normal(size=(T, OBS_DIM + GOAL_DIM)).astype(np.float32)
    actions = rng.normal(size=(T, 2)).astype(np.float32)
    prev_actions = rng.normal(size=(T, 2)).astype(np.float32)
    # Achieved goals spread far apart so only t' == t gives a hit.
    achieved = np.zeros((T, 3))
    achieved[:, 0] = np.arange(T) * 1.0
    achieved[:, 2] = -1.0  # all moving up -> all valid
    hard = np.zeros(T, dtype=bool)
    return obs, next_obs, actions, prev_actions, achieved, hard


class RelabelerTests(unittest.TestCase):
    def _relabeler(self, **kw):
        base = dict(observation_dim=OBS_DIM, goal_dim=GOAL_DIM, compute_reward=_toy_reward,
                    goal_met=_toy_goal_met, k=4, strategy="future", seed=1)
        base.update(kw)
        return HERRelabeler(**base)

    def test_future_goals_come_from_later_steps_and_rewrite_goal_slice(self):
        obs, next_obs, actions, prev_actions, achieved, hard = _toy_episode()
        out = self._relabeler().relabel(obs=obs, next_obs=next_obs, actions=actions,
                                        prev_actions=prev_actions, achieved_next=achieved,
                                        hard_terminal=hard)
        T = obs.shape[0]
        self.assertEqual(out["obs"].shape[0], T * 4)
        rows = np.repeat(np.arange(T), 4)
        goals = out["obs"][:, OBS_DIM:OBS_DIM + GOAL_DIM]
        # Every relabelled goal is the achieved position of a step >= t.
        for row, g in zip(rows, goals):
            self.assertGreaterEqual(g[0], row - 1e-6)
            self.assertTrue(np.any(np.isclose(achieved[:, 0], g[0])))
        # obs and next_obs carry the same relabelled goal; the rest is untouched.
        np.testing.assert_allclose(out["obs"][:, OBS_DIM:], out["next_obs"][:, OBS_DIM:])
        np.testing.assert_allclose(out["obs"][:, :OBS_DIM], obs[rows, :OBS_DIM])
        np.testing.assert_allclose(out["actions"], actions[rows])
        # Reward / done: +10 and terminal exactly when the goal is the step's own achieved state.
        hit = np.isclose(goals[:, 0], rows.astype(float))
        np.testing.assert_array_equal(out["rewards"] > 0, hit)
        np.testing.assert_array_equal(out["dones"] > 0.5, hit)
        self.assertTrue(hit.any())

    def test_invalid_achieved_states_are_never_proposed_as_goals(self):
        obs, next_obs, actions, prev_actions, achieved, hard = _toy_episode()
        achieved[:, 2] = 1.0        # falling puck everywhere...
        achieved[3, 2] = -1.0       # ...except step 3
        out = self._relabeler().relabel(obs=obs, next_obs=next_obs, actions=actions,
                                        prev_actions=prev_actions, achieved_next=achieved,
                                        hard_terminal=hard)
        goals = out["obs"][:, OBS_DIM:OBS_DIM + GOAL_DIM]
        np.testing.assert_allclose(goals[:, 0], 3.0)
        # Only steps t <= 3 have a valid future goal.
        self.assertEqual(out["obs"].shape[0], 4 * 4)
        self.assertEqual(out["n_valid"], 1)

    def test_goal_filter_restricts_candidates(self):
        obs, next_obs, actions, prev_actions, achieved, hard = _toy_episode()
        keep_x = 4.0
        out = self._relabeler(goal_filter=lambda g: np.isclose(g[:, 0], keep_x)).relabel(
            obs=obs, next_obs=next_obs, actions=actions, prev_actions=prev_actions,
            achieved_next=achieved, hard_terminal=hard)
        np.testing.assert_allclose(out["obs"][:, OBS_DIM], keep_x)
        self.assertEqual(out["n_valid"], 1)

    def test_no_valid_states_means_no_relabels(self):
        obs, next_obs, actions, prev_actions, achieved, hard = _toy_episode()
        achieved[:, 2] = 1.0
        self.assertIsNone(self._relabeler().relabel(obs=obs, next_obs=next_obs, actions=actions,
                                                    prev_actions=prev_actions, achieved_next=achieved,
                                                    hard_terminal=hard))
        self.assertIsNone(self._relabeler(k=0).relabel(obs=obs, next_obs=next_obs, actions=actions,
                                                       prev_actions=prev_actions, achieved_next=achieved,
                                                       hard_terminal=hard))

    def test_hard_terminal_stays_terminal_under_any_goal(self):
        obs, next_obs, actions, prev_actions, achieved, hard = _toy_episode()
        hard[-1] = True  # e.g. puck hit the bottom on the last step
        out = self._relabeler().relabel(obs=obs, next_obs=next_obs, actions=actions,
                                        prev_actions=prev_actions, achieved_next=achieved,
                                        hard_terminal=hard)
        rows = np.repeat(np.arange(obs.shape[0]), 4)
        self.assertTrue(np.all(out["dones"][rows == obs.shape[0] - 1] == 1.0))

    def test_final_strategy_uses_last_valid_state(self):
        obs, next_obs, actions, prev_actions, achieved, hard = _toy_episode()
        out = self._relabeler(strategy="final").relabel(obs=obs, next_obs=next_obs, actions=actions,
                                                        prev_actions=prev_actions, achieved_next=achieved,
                                                        hard_terminal=hard)
        self.assertEqual(out["obs"].shape[0], obs.shape[0])
        np.testing.assert_allclose(out["obs"][:, OBS_DIM], obs.shape[0] - 1)
        self.assertEqual(out["rewards"][-1], 10.0)


def _make_env(config_name, **overrides):
    config_path = REPO_ROOT / "configs" / "new_juggle" / "tasks" / config_name
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["air_hockey"]
    cfg["seed"] = 7
    cfg.update(overrides)
    return AirHockeyEnv(cfg)


class SparsePuckGoalTaskTests(unittest.TestCase):
    def test_position_goal_needs_contact(self):
        env = _make_env("sim_sysid_puck_goal.yaml")
        try:
            env.reset()
            ag = env.get_achieved_goal(env.current_state)
            self.assertEqual(ag.shape, (5,))
            self.assertEqual(ag[4], 0.0)
            self.assertEqual(env.get_desired_goal().shape, (2,))
            touched = ag.copy(); touched[4] = 1.0
            untouched = ag.copy(); untouched[4] = 0.0
            self.assertEqual(env.compute_reward(touched, ag[:2], {}), 10.0)
            self.assertEqual(env.compute_reward(untouched, ag[:2], {}), 0.0)
            far = touched.copy(); far[0] += 1.0
            self.assertEqual(env.compute_reward(far, ag[:2], {}), 0.0)
            # Goal on the puck but no contact yet: no reward, episode continues.
            env.set_goals(None, goal_pos=ag[:2])
            _, reward, done, truncated, info = env.step(np.zeros(2, dtype=np.float32))
            self.assertEqual(reward, 0.0)
            self.assertFalse(done)
            self.assertFalse(info["success"])
        finally:
            env.close()

    def test_contact_flag_latches_after_a_hit_and_clears_on_reset(self):
        env = _make_env("sim_sysid_puck_goal.yaml")
        try:
            hit = False
            for ep in range(30):
                env.reset()
                self.assertEqual(env.get_achieved_goal(env.current_state)[4], 0.0)
                env.set_goals(None, goal_pos=np.array([-0.9, 0.4]))  # out of the way
                for _ in range(120):
                    st = env.current_state
                    pad = np.array(st["paddles"]["paddle_ego"]["position"]); puck = np.array(st["pucks"][0]["position"])
                    a = np.array([np.clip((puck[0] - pad[0]) / 0.26, -1, 1), np.clip((puck[1] - pad[1]) / 0.12, -1, 1)], dtype=np.float32)
                    _, _, done, trunc, info = env.step(a)
                    if info["paddle_puck_collision_count"] > 0:
                        hit = True
                        self.assertEqual(env.get_achieved_goal(env.current_state)[4], 1.0)
                    if done or trunc:
                        break
                    if hit:
                        # Stays latched for the rest of the episode.
                        self.assertEqual(env.get_achieved_goal(env.current_state)[4], 1.0)
                if hit:
                    break
            self.assertTrue(hit, "scripted chase never touched the puck")
            env.reset()
            self.assertEqual(env.get_achieved_goal(env.current_state)[4], 0.0)
        finally:
            env.close()

    def test_speed_task_projects_achieved_to_speed(self):
        env = _make_env("sim_sysid_puck_goal_vel_hist2.yaml")
        try:
            env.reset()
            self.assertEqual(env.get_desired_goal().shape, (3,))
            ag = np.array([[-0.5, 0.1, -1.2, 0.9, 1.0]])
            dg = env.achieved_to_desired(ag)
            np.testing.assert_allclose(dg, [[-0.5, 0.1, 1.5]])
            self.assertEqual(env.compute_reward(ag[0], dg[0], {}), 10.0)
            self.assertEqual(env.compute_reward(ag[0], dg[0] + np.array([0, 0, 0.7]), {}), 0.0)
            # Same speed, different direction: still met.
            ag2 = np.array([[-0.5, 0.1, -1.5, 0.0, 1.0]])
            self.assertEqual(env.compute_reward(ag2[0], dg[0], {}), 10.0)
            self.assertTrue(env.goal_in_distribution(dg)[0])
            for _ in range(20):
                env.set_goals("fixed")
                self.assertTrue(env.goal_in_distribution(env.get_desired_goal()[None])[0])
                self.assertLess(env.goal_pos[0], 0.0)
                self.assertGreaterEqual(env.goal_speed, env.goal_shot_min_upward_speed)
            self.assertFalse(env.goal_in_distribution(np.array([[0.5, 0.0, 1.0]]))[0])
            self.assertFalse(env.goal_in_distribution(np.array([[-0.5, 0.0, 5.0]]))[0])
        finally:
            env.close()

    def test_intercept_shot_goal_conditions_on_the_spawned_puck(self):
        env = _make_env("sim_sysid_puck_goal_vel_hist2.yaml", goal_sampling="intercept_shot")
        try:
            for _ in range(10):
                obs, _ = env.reset()
                np.testing.assert_allclose(obs["desired_goal"], env.get_desired_goal())
                self.assertTrue(env.goal_in_distribution(env.get_desired_goal()[None])[0])
        finally:
            env.close()

    def test_goal_reached_terminates_with_plus_ten(self):
        env = _make_env("sim_sysid_puck_goal.yaml")
        try:
            env.reset()
            # Puck moving up through the goal, contact already made this episode.
            env.simulator.spawn_puck((0.0, 0.0), (-1.0, 0.0), "puck_0")
            env.simulator.instantiate_objects()
            env.current_state = env.simulator.get_current_state()
            env._puck_contacted = True
            env.set_goals(None, goal_pos=np.array([-0.05, 0.0]))
            _, reward, done, truncated, info = env.step(np.zeros(2, dtype=np.float32))
            self.assertEqual(reward, 10.0)
            self.assertTrue(done)
            self.assertIn("goal_reached", info["termination_reasons"])
            self.assertTrue(info["success"])
        finally:
            env.close()


class GoalEnvVectorTests(unittest.TestCase):
    def test_flat_obs_and_final_observation(self):
        envs = GoalEnvVector(lambda: _make_env("sim_sysid_puck_goal.yaml"))
        try:
            self.assertEqual(envs.single_observation_space.shape, (32,))
            obs, _ = envs.reset(seed=1)
            self.assertEqual(obs.shape, (1, 32))
            np.testing.assert_allclose(obs[0, 30:], envs.env.get_desired_goal(), atol=1e-6)
            done = False
            while not done:
                obs, r, term, trunc, infos = envs.step(np.zeros((1, 2), dtype=np.float32))
                self.assertEqual(infos["achieved_goal"].shape, (1, 5))
                done = bool(term[0] or trunc[0])
            self.assertIn("final_observation", infos)
            self.assertEqual(infos["final_observation"][0].shape, (32,))
            self.assertEqual(infos["final_achieved_goal"][0].shape, (5,))
            # Goal slice of the final observation is the episode's goal, not the new one.
            self.assertFalse(np.allclose(infos["final_observation"][0][30:], obs[0, 30:]))
        finally:
            envs.close()

    def test_trajectory_flush_writes_original_plus_relabeled(self):
        envs = GoalEnvVector(lambda: _make_env("sim_sysid_puck_goal.yaml"))
        try:
            compute_reward, goal_met, goal_filter, achieved_to_desired = make_goal_functions(envs.env)
            relabeler = HERRelabeler(observation_dim=30, goal_dim=2, compute_reward=compute_reward,
                                     goal_met=goal_met, k=4, seed=0)
            rb = TD3ReplayBuffer(buffer_size=10000, obs_shape=(32,), action_shape=(2,), device="cpu")
            traj = HEREpisodeTrajectory.empty()
            obs, _ = envs.reset(seed=3)
            steps = 0
            for _ in range(400):
                action = np.random.uniform(-1, 1, size=(1, 2)).astype(np.float32)
                next_obs, r, term, trunc, infos = envs.step(action)
                done = bool(term[0] or trunc[0])
                real_next = infos["final_observation"][0] if done else next_obs[0]
                ag = infos["final_achieved_goal"][0] if done else infos["achieved_goal"][0]
                traj.append_step(
                    obs=torch.as_tensor(obs[0]), next_obs=torch.as_tensor(real_next),
                    action=torch.as_tensor(action[0]), reward=torch.tensor(float(r[0])),
                    done=torch.tensor(float(term[0])), prev_action=torch.zeros(2),
                    achieved_goal=ag, hard_terminal=bool(infos["hard_terminal"][0]),
                )
                steps += 1
                obs = next_obs
                if done:
                    break
            stats = traj.flush_to_buffer(rb, relabeler)
            self.assertEqual(stats["original"], steps)
            self.assertEqual(len(rb), steps + stats["relabeled"])
            self.assertLessEqual(stats["relabeled"], 4 * steps)
            if stats["valid"] > 0:
                # Relabelled copies with reward 10 are exactly the terminal ones (no hard terminal here unless the puck died).
                r = rb.rewards[:len(rb)]
                self.assertGreater(float((r[steps:] > 0).sum()), 0)
            self.assertEqual(len(traj.observations), 0)
        finally:
            envs.close()


if __name__ == "__main__":
    unittest.main()

"""Continuing-task bootstrapping, online replay, and survival reporting."""

from collections import deque
import copy
from pathlib import Path
import unittest

import numpy as np
import torch
import yaml

from airhockey import AirHockeyEnv
from scripts.td3.helper.td3_episode_collection import EpisodeTrajectory, finalize_episode_if_done
from scripts.td3.helper.td3_args_validation import validate_args
from scripts.td3.helper.td3_graphed_update import GraphedTD3Update, h_transform
from scripts.td3.juggle_survival import diagnostic_body_positions, summarize_episodes
from scripts.td3.td3_training import Args, SingleEnvVector


class RecordingBuffer:
    def __init__(self):
        self.batches = []

    def add(self, **batch):
        self.batches.append(batch)


class ContinuingJuggleTests(unittest.TestCase):
    def test_diagnostic_positions_do_not_sample_sensors(self):
        import random
        from scripts.td3.juggle_original_sensing import original_sensing_config

        cfg = original_sensing_config()["air_hockey"]
        traces = []
        for record_truth in (False, True):
            random.seed(123)
            np.random.seed(123)
            env = AirHockeyEnv(copy.deepcopy(cfg))
            try:
                obs, _ = env.reset(seed=123)
                trace = [obs.copy()]
                for _ in range(20):
                    obs, _, term, trunc, _ = env.step(np.zeros(2))
                    if record_truth:
                        # Additional sensor calls change future noise/occlusions.
                        paddle, puck = diagnostic_body_positions(env.simulator)
                        np.testing.assert_allclose(paddle, env.current_state["paddles"]["paddle_ego"]["position"])
                        self.assertEqual(puck.shape, (2,))
                    trace.append(obs.copy())
                    if term or trunc:
                        break
                traces.append(np.asarray(trace))
            finally:
                env.close()
        np.testing.assert_array_equal(traces[0], traces[1])

    def test_true_state_counts_do_not_replace_historical_observed_counts(self):
        row = dict(steps=10000, failed=False, reached_horizon=True, contacts=10000,
                   juggles=100, true_state_juggles=0, end_reasons=["evaluation_horizon"],
                   peak_paddle_speed_m_s=0., peak_paddle_acceleration_m_s2=0.,
                   action_delta_rms=0., target_delta_rms_m=0., **{"return": 10000.})
        summary = summarize_episodes([row], 10000)
        self.assertEqual(summary["sustained_juggling_fraction"], 1.)
        self.assertEqual(summary["true_state_sustained_juggling_fraction"], 0.)

    def test_original_sensing_campaign_preserves_canonical_simulator_parameters(self):
        from scripts.td3.juggle_original_sensing import original_sensing_config, assert_original_sensing
        from scripts.td3.juggle_survival import evaluation_scenarios
        original = yaml.safe_load((Path(__file__).resolve().parents[3] /
                                   "configs/new_juggle/sysid_v2_hist2.yaml").read_text())["air_hockey"]
        restored = original_sensing_config()["air_hockey"]
        self.assertEqual(restored["simulator_params"], original["simulator_params"])
        self.assertEqual(restored["obs_type"], original["obs_type"])
        self.assertTrue(restored["domain_random"])
        self.assertEqual(set(restored["random_variables"]), {"paddle_density", "puck_damping", "gravity"})
        assert_original_sensing(restored)
        evaluation = evaluation_scenarios(restored, "training")["training"]
        self.assertEqual(evaluation, restored)
        evaluation["simulator_params"]["puck_noise"] = False
        self.assertTrue(restored["simulator_params"]["puck_noise"])
        with self.assertRaises(ValueError):
            assert_original_sensing(evaluation)

    def test_clean_benchmark_removes_sensor_corruption_without_changing_controller(self):
        from scripts.td3.juggle_survival import clean_sim_config
        cfg = yaml.safe_load((Path(__file__).resolve().parents[3] /
                              "configs/new_juggle/sysid_v2_hist2.yaml").read_text())["air_hockey"]
        clean = clean_sim_config(cfg)
        self.assertFalse(clean["domain_random"])
        for field in ("puck_noise", "enable_random_occlusions", "enable_observation_delay", "enable_action_delay"):
            self.assertFalse(clean["simulator_params"][field])
        for field in ("pid_kp", "pid_ki", "pid_kd", "hist_len", "max_force_timestep"):
            self.assertEqual(clean["simulator_params"][field], cfg["simulator_params"][field])
        self.assertTrue(cfg["simulator_params"]["puck_noise"])

    def test_loss_cost_only_applies_to_actual_puck_loss(self):
        cfg = yaml.safe_load((Path(__file__).resolve().parents[3] /
                              "configs/new_juggle/sysid_v2_hist2.yaml").read_text())["air_hockey"]
        cfg["puck_loss_penalty"] = 5.
        env = AirHockeyEnv(cfg)
        try:
            env.reset(seed=3)
            env._update_puck_pass_paddle_score = lambda puck, paddle: True
            env.get_base_reward = lambda state: (7., False)
            _, reward, term, trunc, info = env.step(np.zeros(2))
            self.assertTrue(term)
            self.assertFalse(trunc)
            self.assertEqual(float(reward), 2.)
            self.assertEqual(info['puck_loss_penalty'], 5.)
        finally:
            env.close()

    def test_sticky_contact_does_not_count_as_repeated_juggling(self):
        row = dict(steps=10000, failed=False, reached_horizon=True, contacts=10000,
                   juggles=0, end_reasons=["evaluation_horizon"], peak_paddle_speed_m_s=0.,
                   peak_paddle_acceleration_m_s2=0., action_delta_rms=0., target_delta_rms_m=0.,
                   **{"return": 10000.})
        summary = summarize_episodes([row], 10000)
        self.assertEqual(summary['sustained_juggling_fraction'], 0.)
        self.assertEqual(summary['mean_valid_juggling_steps'], 0.)

    def test_passed_puck_is_terminal_not_a_bootstrappable_time_limit(self):
        cfg = yaml.safe_load((Path(__file__).resolve().parents[3] /
                              "configs/new_juggle/sysid_v2_hist2.yaml").read_text())["air_hockey"]
        env = AirHockeyEnv(cfg)
        try:
            env.reset(seed=3)
            env._update_puck_pass_paddle_score = lambda puck, paddle: True
            _, _, term, trunc, info = env.step(np.zeros(2))
            self.assertTrue(term)
            self.assertFalse(trunc)
            self.assertIn("puck_passed_paddle", info["termination_reasons"])
        finally:
            env.close()

    def test_history_has_one_sample_per_control_step_when_opted_in(self):
        base = yaml.safe_load((Path(__file__).resolve().parents[3] /
                               "configs/new_juggle/sysid_v2_hist2.yaml").read_text())["air_hockey"]
        for once, samples in ((False, 2), (True, 1)):
            cfg = copy.deepcopy(base)
            cfg["simulator_params"]["history_sample_once_per_step"] = once
            env = AirHockeyEnv(cfg)
            try:
                env.reset(seed=3)
                before = len(env.simulator.puck_history)
                env.step(np.zeros(2))
                self.assertEqual(len(env.simulator.puck_history) - before, samples)
            finally:
                env.close()

    def test_time_limit_bootstraps_from_final_observation(self):
        cfg = yaml.safe_load((Path(__file__).resolve().parents[3] /
                              "configs/new_juggle/sysid_v2_hist2.yaml").read_text())["air_hockey"]
        cfg = copy.deepcopy(cfg)
        cfg.update(max_timesteps=0, terminate_on_puck_hit_bottom=False,
                   terminate_on_puck_pass_paddle=False, terminate_on_enemy_goal=False)
        env = SingleEnvVector(lambda: AirHockeyEnv(cfg))
        try:
            env.reset(seed=3)
            env.env.get_base_reward = lambda state: (7.0, False)
            for _ in range(3):
                _, rewards, terminated, truncated, info = env.step(np.zeros((1, 2)))
                if truncated[0]:
                    self.assertFalse(terminated[0])
                    self.assertEqual(rewards[0], 7.0)
                    self.assertIn("final_observation", info)
                    self.assertIn("max_timesteps_exceeded", info["final_info"][0]["truncation_reasons"])
                    break
            else:
                self.fail("Expected time-limit truncation")
        finally:
            env.close()

    def test_true_failure_remains_terminal(self):
        cfg = yaml.safe_load((Path(__file__).resolve().parents[3] /
                              "configs/new_juggle/sysid_v2_hist2.yaml").read_text())["air_hockey"]
        env = AirHockeyEnv(cfg)
        try:
            env.reset(seed=3)
            # Exercise the public step wrapper independently of physics.
            env.single_agent_step = lambda action: (np.zeros(30), 0.0, True, False, {})
            _, _, terminated, truncated, _ = env.step(np.zeros(2))
            self.assertTrue(terminated)
            self.assertFalse(truncated)
        finally:
            env.close()

    def test_online_chunks_do_not_duplicate_transitions_or_lose_return(self):
        trajectory = EpisodeTrajectory.empty()
        replay = RecordingBuffer()
        for step in range(7):
            trajectory.append_step(torch.tensor([float(step)]), torch.tensor([float(step + 1)]),
                                   torch.zeros(2), torch.tensor(1.0), torch.tensor(float(step == 6)),
                                   torch.zeros(2))
            if (step + 1) % 3 == 0:
                trajectory.flush_to_buffer(replay, keep_episode_return=True)
        returns = deque(maxlen=10)
        finalize_episode_if_done(True, trajectory, returns, 0.5, 0.0, replay, None, single_buffer=True)
        self.assertEqual(list(returns), [7.0])
        self.assertEqual(torch.cat([b["obs"] for b in replay.batches]).flatten().tolist(), list(range(7)))
        self.assertEqual(trajectory.episode_return, 0.0)
        self.assertEqual(torch.cat([b["dones"] for b in replay.batches]).tolist(), [0.] * 6 + [1.])

    def test_episode_ending_exactly_on_flush_keeps_return(self):
        trajectory = EpisodeTrajectory.empty()
        replay = RecordingBuffer()
        trajectory.append_step(torch.zeros(1), torch.zeros(1), torch.zeros(2), torch.tensor(3.),
                               torch.tensor(0.), torch.zeros(2))
        trajectory.flush_to_buffer(replay, keep_episode_return=True)
        restored = EpisodeTrajectory.from_state_dict(trajectory.state_dict(), "cpu")
        returns = deque(maxlen=10)
        finalize_episode_if_done(True, restored, returns, .5, 0., replay, None, single_buffer=True)
        self.assertEqual(list(returns), [3.])
        self.assertEqual(len(replay.batches), 1)
        self.assertEqual(restored.episode_return, 0.)

    def test_fixed_cadence_requires_flat_replay(self):
        with self.assertRaisesRegex(ValueError, "single_replay_buffer"):
            validate_args(Args(train_every_steps=32))
        validate_args(Args(train_every_steps=32, single_replay_buffer=True))
        with self.assertRaisesRegex(ValueError, "actor_learning_starts"):
            validate_args(Args(learning_starts=100, actor_learning_starts=50))

    def test_campaign_preserves_real_control_contract(self):
        root = Path(__file__).resolve().parents[3]
        base = yaml.safe_load((root / "configs/new_juggle/sysid_v2_hist2.yaml").read_text())["air_hockey"]
        cfg = yaml.safe_load((root / "configs/new_juggle/sustained/juggle.yaml").read_text())["air_hockey"]
        for key in ("center_offset_constant", "center_offset_constant_y", "obs_type"):
            self.assertEqual(cfg[key], base[key])
        for key in ("pid_kp", "pid_ki", "pid_kd", "hist_len", "max_force_timestep",
                    "length", "width", "x_min_lim", "x_max_lim", "y_min", "y_max"):
            self.assertEqual(cfg["simulator_params"][key], base["simulator_params"][key])
            self.assertNotIn(key, cfg["random_variables"])
        for key in ("action_x_ratio", "action_y_ratio"):
            self.assertEqual(cfg.get(key), base.get(key))

    def test_smoothing_loss_has_gradient_toward_previous_action(self):
        class Actor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.command = torch.nn.Parameter(torch.tensor([.5, -.5]))

            def get_action(self, obs):
                return self.command.expand(obs.shape[0], -1)

        class Critic(torch.nn.Module):
            def forward(self, obs, action):
                return h_transform(torch.ones((obs.shape[0], 1))) + action.sum(-1, keepdim=True) * 0

        updater = object.__new__(GraphedTD3Update)
        updater.actor = Actor()
        updater.qfs = [Critic()]
        updater.use_last_action = True
        updater.gamma, updater.h_eps = .99, .001
        updater.actor_action_delta_penalty = .1
        loss, _ = updater._actor_core(torch.zeros(4, 30), torch.zeros(4, 2))
        loss.backward()
        self.assertGreater(updater.actor.command.grad[0].item(), 0.)
        self.assertLess(updater.actor.command.grad[1].item(), 0.)

    def test_censored_survival_does_not_count_stationary_puck_as_juggling(self):
        row = dict(steps=1000, failed=False, reached_horizon=True, contacts=0,
                   end_reasons=["evaluation_horizon"], peak_paddle_speed_m_s=0.,
                   peak_paddle_acceleration_m_s2=0., action_delta_rms=0., target_delta_rms_m=0.,
                   **{"return": 1000.})
        summary = summarize_episodes([row], 1000)
        self.assertEqual(summary["failures"], 0)
        self.assertEqual(summary["horizon_survival_fraction"], 1.)
        self.assertEqual(summary["sustained_juggling_fraction"], 0.)
        self.assertAlmostEqual(summary["zero_failure_probability_upper_95"], .95)


if __name__ == "__main__":
    unittest.main()

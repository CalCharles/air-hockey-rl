"""Unit tests for the RMA baseline package (CPU only, seconds).

    .venv/bin/python -m pytest scripts/rma/tests -q
"""

import copy
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from scripts.rma.env_wrapper import EnvParamNormalizer, RMAEnvVector, raw_env_params
from scripts.rma.history import AdaptationDataset, StepHistoryBuffer, pad_episode
from scripts.rma.networks import (
    AdaptationModule,
    AdaptedActor,
    EnvFactorEncoder,
    RMAActor,
    step_feature_dim,
    step_state_features,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DR_CONFIG = REPO_ROOT / "configs" / "new_juggle" / "zeroshot_ablations" / "sim_paramrand_pm25.yaml"


def _load_dr_config():
    with open(DR_CONFIG) as f:
        return yaml.load(f, Loader=yaml.FullLoader)["air_hockey"]


class NormalizerTests(unittest.TestCase):
    def test_roundtrip_and_bounds(self):
        n = EnvParamNormalizer(["a", "b"], {"a": [2.0, 4.0], "b": [-1.0, 1.0]})
        self.assertEqual(n.dim, 2)
        np.testing.assert_allclose(n.normalize(np.array([3.0, 0.0])), [0.0, 0.0])
        np.testing.assert_allclose(n.normalize(np.array([2.0, 1.0])), [-1.0, 1.0])
        raw = np.array([2.5, -0.3], dtype=np.float32)
        np.testing.assert_allclose(n.denormalize(n.normalize(raw)), raw, atol=1e-6)
        again = EnvParamNormalizer.from_dict(n.to_dict())
        np.testing.assert_allclose(again.low, n.low)

    def test_missing_range_raises(self):
        with self.assertRaises(KeyError):
            EnvParamNormalizer(["a"], {})


class NetworkShapeTests(unittest.TestCase):
    def test_encoder_and_actor(self):
        actor = RMAActor(obs_dim=30, act_dim=2, env_param_dim=3, latent_dim=8, use_last_action=True)
        self.assertEqual(actor.policy_obs_dim, 35)
        x = torch.randn(5, 35)
        a = actor.get_action(x)
        self.assertEqual(a.shape, (5, 2))
        self.assertTrue(torch.all(a.abs() <= 1.0))
        # same result via the deployment path with the privileged latent
        obs, e, prev = actor.split_policy_obs(x)
        z = actor.encoder(e)
        torch.testing.assert_close(actor.get_action_from_latent(obs, prev, z), a)
        # encoder is part of the actor parameters (trained through the actor loss)
        names = [n for n, _ in actor.named_parameters()]
        self.assertTrue(any(n.startswith("encoder.") for n in names))
        self.assertTrue(any(n.startswith("actor.") for n in names))

    def test_actor_without_last_action(self):
        actor = RMAActor(obs_dim=30, act_dim=2, env_param_dim=3, latent_dim=4, use_last_action=False)
        self.assertEqual(actor.policy_obs_dim, 33)
        self.assertEqual(actor.get_action(torch.randn(2, 33)).shape, (2, 2))

    def test_adaptation_module_paper_dims(self):
        # H=50, kernels 8/5/5, strides 4/1/1 -> 11 -> 7 -> 3 time steps x 32 ch = 96
        phi = AdaptationModule(feature_dim=8, history_len=50, latent_dim=8)
        self.assertEqual(phi.conv_out_dim, 96)
        out = phi(torch.randn(4, 50, 8))
        self.assertEqual(out.shape, (4, 8))
        self.assertEqual(phi(torch.randn(50, 8)).shape, (1, 8))
        with self.assertRaises(ValueError):
            phi(torch.randn(4, 49, 8))
        with self.assertRaises(ValueError):
            AdaptationModule(feature_dim=8, history_len=5, latent_dim=8)

    def test_adapted_actor(self):
        actor = RMAActor(obs_dim=30, act_dim=2, env_param_dim=3, latent_dim=8)
        phi = AdaptationModule(feature_dim=8, history_len=50, latent_dim=8)
        deploy = AdaptedActor(actor, phi)
        a = deploy.get_action(torch.randn(3, 30), torch.zeros(3, 2), torch.randn(3, 50, 8))
        self.assertEqual(a.shape, (3, 2))
        with self.assertRaises(ValueError):
            AdaptedActor(actor, AdaptationModule(feature_dim=8, history_len=50, latent_dim=4))

    def test_step_features(self):
        obs = torch.arange(30, dtype=torch.float32)
        f = step_state_features(obs, "latest_frame")
        torch.testing.assert_close(f, torch.tensor([12.0, 13.0, 14.0, 27.0, 28.0, 29.0]))
        self.assertEqual(step_feature_dim(30, 2, "latest_frame"), 8)
        self.assertEqual(step_feature_dim(30, 2, "full_obs"), 32)
        # goal-conditioned tasks append the desired goal after the 30-dim history obs: still 8 features
        self.assertEqual(step_feature_dim(33, 2, "latest_frame"), 8)
        obs_goal = torch.cat([torch.arange(30, dtype=torch.float32), torch.tensor([100.0, 101.0, 102.0])])
        torch.testing.assert_close(step_state_features(obs_goal, "latest_frame"), torch.tensor([12.0, 13.0, 14.0, 27.0, 28.0, 29.0]))
        with self.assertRaises(ValueError):
            step_feature_dim(29, 2, "latest_frame")


class HistoryTests(unittest.TestCase):
    def test_buffer_padding_and_window_order(self):
        buf = StepHistoryBuffer(history_len=4, feature_dim=3, act_dim=1)
        buf.reset(np.array([1.0, 2.0]))
        w = buf.window()
        np.testing.assert_allclose(w, np.array([[1, 2, 0]] * 4))
        buf.push(np.array([1.0, 2.0]), np.array([0.5]))
        buf.push(np.array([3.0, 4.0]), np.array([-0.5]))
        w = buf.window()
        np.testing.assert_allclose(w[-1], [3, 4, -0.5])
        np.testing.assert_allclose(w[-2], [1, 2, 0.5])
        np.testing.assert_allclose(w[0], [1, 2, 0])

    def test_dataset_windows_match_online_buffer(self):
        H, Fs, A = 5, 2, 1
        T = 7
        rng = np.random.default_rng(0)
        states = rng.normal(size=(T, Fs)).astype(np.float32)
        actions = rng.normal(size=(T, A)).astype(np.float32)
        padded = pad_episode(states, actions, H)
        self.assertEqual(padded.shape, (H + T, Fs + A))
        ds = AdaptationDataset(H, Fs + A, latent_dim=2, env_param_dim=3)
        ds.add_episode(states, actions, np.array([0.1, 0.2]), np.array([0.0, 0.5, -0.5]))
        buf = StepHistoryBuffer(H, Fs + A, A)
        buf.reset(states[0])
        for t in range(T):
            np.testing.assert_allclose(ds.windows(np.array([t]))[0], buf.window(), atol=1e-6)
            buf.push(states[t], actions[t])
        np.testing.assert_array_equal(ds.steps_in_episode(ds.all_indices()), np.arange(T))
        self.assertEqual(ds.targets(ds.all_indices()).shape, (T, 2))

    def test_dataset_rolling_limit(self):
        ds = AdaptationDataset(3, 2, latent_dim=1, env_param_dim=1, max_steps=10)
        for i in range(5):
            ds.add_episode(np.ones((4, 1)), np.zeros((4, 1)), np.array([i]), np.array([0.0]))
        self.assertLessEqual(ds.num_steps, 10)  # oldest episodes dropped until <= max_steps
        self.assertEqual(ds.num_episodes, 2)
        np.testing.assert_array_equal(np.unique(ds.targets(ds.all_indices())), [3.0, 4.0])


class EnvWrapperTests(unittest.TestCase):
    def test_env_params_appended_and_change_per_reset(self):
        cfg = _load_dr_config()
        cfg["seed"] = 3
        cfg["max_timesteps"] = 5
        normalizer = EnvParamNormalizer.from_air_hockey_config(cfg)
        from airhockey import AirHockeyEnv

        envs = RMAEnvVector(lambda: AirHockeyEnv(copy.deepcopy(cfg)), normalizer)
        obs, _ = envs.reset(seed=0)
        self.assertEqual(obs.shape, (1, 33))
        e0 = obs[0, 30:]
        self.assertTrue(np.all(np.abs(e0) <= 1.0 + 1e-6))
        raw = raw_env_params(envs.env, normalizer.random_variables)
        np.testing.assert_allclose(normalizer.normalize(raw), e0, atol=1e-5)
        # step until the episode ends (5-step budget) and check the final obs keeps the OLD e
        seen_new = False
        for _ in range(40):
            obs, r, term, trunc, infos = envs.step(np.zeros((1, 2), dtype=np.float32))
            if "final_observation" in infos:
                np.testing.assert_allclose(infos["final_observation"][0][30:], e0, atol=1e-6)
                e1 = obs[0, 30:]
                if np.any(np.abs(e1 - e0) > 1e-6):
                    seen_new = True
                break
        self.assertTrue(seen_new, "domain randomization should re-sample e on reset")
        envs.close()


class GoalEnvWrapperTests(unittest.TestCase):
    """Goal-conditioned tasks: the wrappers build on GoalEnvVector (flat [obs, goal] + HER infos)."""

    GOAL_CONFIG = REPO_ROOT / "configs" / "new_juggle" / "tasks_v2" / "sim_dr3_puck_goal.yaml"

    def _cfg(self):
        with open(self.GOAL_CONFIG) as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)["air_hockey"]
        cfg["seed"], cfg["max_timesteps"], cfg["return_goal_obs"] = 3, 6, True
        return cfg

    def test_rma_env_vector_goal_layout_and_her_infos(self):
        from airhockey import AirHockeyEnv

        cfg = self._cfg()
        normalizer = EnvParamNormalizer.from_air_hockey_config(cfg)
        envs = RMAEnvVector(lambda: AirHockeyEnv(copy.deepcopy(cfg)), normalizer, goal=True)
        self.assertEqual((envs.observation_dim, envs.goal_dim, envs.raw_obs_dim), (30, 2, 32))
        obs, _ = envs.reset(seed=0)
        self.assertEqual(obs.shape, (1, 32 + 3))
        np.testing.assert_allclose(obs[0, 32:], normalizer.normalize(raw_env_params(envs.env, normalizer.random_variables)), atol=1e-5)
        saw_end = False
        for _ in range(30):
            e_before = obs[0, 32:].copy()
            obs, r, term, trunc, infos = envs.step(np.zeros((1, 2), dtype=np.float32))
            self.assertEqual(infos["achieved_goal"].shape[0], 1)
            self.assertEqual(infos["hard_terminal"].shape, (1,))
            if "final_observation" in infos:
                saw_end = True
                self.assertEqual(infos["final_observation"][0].shape, (35,))
                np.testing.assert_allclose(infos["final_observation"][0][32:], e_before, atol=1e-6)
                self.assertEqual(len(infos["final_achieved_goal"]), 1)
                break
        self.assertTrue(saw_end)
        envs.close()

    def test_history_env_vector_goal_layout(self):
        from airhockey import AirHockeyEnv
        from scripts.rma.env_wrapper import HistoryEnvVector

        cfg = self._cfg()
        envs = HistoryEnvVector(lambda: AirHockeyEnv(copy.deepcopy(cfg)), history_len=10, step_features="latest_frame", goal=True)
        self.assertEqual(envs.feature_dim, 8)
        obs, _ = envs.reset(seed=0)
        self.assertEqual(obs.shape, (1, 32 + 10 * 8))
        obs, *_ = envs.step(np.zeros((1, 2), dtype=np.float32))
        self.assertEqual(obs.shape, (1, 32 + 10 * 8))
        envs.close()


class GradientFlowTests(unittest.TestCase):
    def test_encoder_receives_actor_loss_gradient(self):
        actor = RMAActor(obs_dim=30, act_dim=2, env_param_dim=3, latent_dim=8)
        x = torch.randn(8, 35)
        loss = -actor.get_action(x).sum()
        loss.backward()
        grads = [p.grad for n, p in actor.named_parameters() if n.startswith("encoder.")]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
        self.assertTrue(any(g.abs().sum() > 0 for g in grads))


if __name__ == "__main__":
    unittest.main()


class HistoryBaselineTests(unittest.TestCase):
    def test_history_actor_layout(self):
        from scripts.rma.networks import HistoryActor

        actor = HistoryActor(obs_dim=30, act_dim=2, history_len=50, feature_dim=8, latent_dim=8)
        self.assertEqual(actor.policy_obs_dim, 30 + 400 + 2)
        x = torch.randn(4, 432)
        a = actor.get_action(x)
        self.assertEqual(a.shape, (4, 2))
        obs, window, prev = actor.split_policy_obs(x)
        self.assertEqual(window.shape, (4, 50, 8))
        torch.testing.assert_close(actor.get_action_from_latent(obs, prev, actor.encoder(window)), a)
        loss = -a.sum()
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in actor.named_parameters() if n.startswith("encoder.")))

    def test_history_env_vector_matches_online_buffer(self):
        from airhockey import AirHockeyEnv
        from scripts.rma.env_wrapper import HistoryEnvVector
        from scripts.rma.history import StepHistoryBuffer
        from scripts.rma.networks import step_state_features

        cfg = _load_dr_config()
        cfg["seed"] = 5
        cfg["max_timesteps"] = 6
        envs = HistoryEnvVector(lambda: AirHockeyEnv(copy.deepcopy(cfg)), history_len=4, step_features="latest_frame")
        obs, _ = envs.reset(seed=0)
        self.assertEqual(obs.shape, (1, 30 + 4 * 8))
        ref = StepHistoryBuffer(4, 8, 2)
        raw = obs[0, :30].astype(np.float32)
        ref.reset(step_state_features(torch.as_tensor(raw), "latest_frame").numpy())
        np.testing.assert_allclose(obs[0, 30:].reshape(4, 8), ref.window(), atol=1e-6)
        rng = np.random.default_rng(0)
        for _ in range(10):
            action = rng.uniform(-1, 1, size=(1, 2)).astype(np.float32)
            ref.push(step_state_features(torch.as_tensor(raw), "latest_frame").numpy(), action[0])
            obs, r, term, trunc, infos = envs.step(action)
            if "final_observation" in infos:
                np.testing.assert_allclose(infos["final_observation"][0][30:].reshape(4, 8), ref.window(), atol=1e-6)
                raw = obs[0, :30].astype(np.float32)
                ref.reset(step_state_features(torch.as_tensor(raw), "latest_frame").numpy())
            else:
                raw = obs[0, :30].astype(np.float32)
            np.testing.assert_allclose(obs[0, 30:].reshape(4, 8), ref.window(), atol=1e-6)
        envs.close()

    def test_bundle_roundtrip_history(self):
        import tempfile

        from scripts.rma.bundle import actor_from_meta, build_rma_meta, load_phase1, write_rma_meta
        from scripts.rma.networks import HistoryActor

        actor = HistoryActor(obs_dim=30, act_dim=2, history_len=50, feature_dim=8, latent_dim=8)
        actor.step_features = "latest_frame"
        n = EnvParamNormalizer(["a"], {"a": [0.0, 1.0]})
        meta = build_rma_meta(actor, n, [256, 128], 64, 2)
        self.assertEqual(meta["mode"], "history")
        with tempfile.TemporaryDirectory() as d:
            write_rma_meta(d, meta)
            torch.save(actor.state_dict(), f"{d}/model.pth")
            b = load_phase1(d)
            self.assertEqual(b["mode"], "history")
            x = torch.randn(2, 432)
            torch.testing.assert_close(b["actor"].get_action(x), actor.get_action(x))
        self.assertIsInstance(actor_from_meta(meta), HistoryActor)

"""Tests for the RMA / long-history eval agents on the deployment path.

These exercise the pieces that let `scripts.real.evaluate_policy` run an RMA
bundle through the frozen-policy eval pipeline: algorithm detection, bundle
loading, the tensor<->numpy adapter, and the per-episode history reset.

Skipped when the exported policy bundles are not present.
"""
from __future__ import annotations

import os
import unittest

import numpy as np
import torch

from scripts.real.evaluate_policy import detect_algo, resolve_model_path, resolve_train_args
from scripts.td3.helper.real_eval_agents import build_rma_eval_agent
from scripts.td3.helper.real_td3_runtime import Args

BUNDLE_ROOT = os.path.join("data", "policies_2026-09-19", "juggle")
RMA_BUNDLE = os.path.join(BUNDLE_ROOT, "rma_full")
HISTORY_BUNDLE = os.path.join(BUNDLE_ROOT, "drlong_full")
TD3_BUNDLE = os.path.join(BUNDLE_ROOT, "sysid")

OBS_DIM = 30
ACT_DIM = 2


def _build(bundle: str, obs_dim: int = OBS_DIM):
    train_args, _ = resolve_train_args("rma", bundle, None)
    return build_rma_eval_agent(
        args=Args(model_path=bundle),
        train_args=train_args,
        obs_dim=obs_dim,
        act_dim=ACT_DIM,
        action_low_np=np.full((ACT_DIM,), -1.0, dtype=np.float32),
        action_high_np=np.full((ACT_DIM,), 1.0, dtype=np.float32),
        device=torch.device("cpu"),
    )


def _policy_obs(seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    return torch.as_tensor(
        rng.normal(0.0, 0.3, size=(1, OBS_DIM + ACT_DIM)), dtype=torch.float32
    )


@unittest.skipUnless(os.path.isdir(RMA_BUNDLE), f"{RMA_BUNDLE} not present")
class RmaEvalAgentTests(unittest.TestCase):
    def test_detect_algo_distinguishes_bundle_kinds(self) -> None:
        self.assertEqual(detect_algo(RMA_BUNDLE), "rma")
        self.assertEqual(detect_algo(HISTORY_BUNDLE), "history")
        self.assertEqual(detect_algo(TD3_BUNDLE), "td3")
        # A weights file inside an RMA bundle still resolves to the bundle.
        self.assertEqual(detect_algo(os.path.join(RMA_BUNDLE, "model.pth")), "rma")

    def test_resolve_model_path_per_algo(self) -> None:
        # RMA needs the bundle dir (three files); TD3 needs a single weights file.
        self.assertEqual(resolve_model_path("rma", RMA_BUNDLE), RMA_BUNDLE)
        self.assertTrue(resolve_model_path("td3", TD3_BUNDLE).endswith("training_state.pth"))

    def test_adapted_agent_produces_valid_actions(self) -> None:
        bundle = _build(RMA_BUNDLE)
        self.assertEqual(bundle.metadata["rma_agent"], "AdaptedRMAAgent")
        self.assertIsNotNone(bundle.metadata["adaptation_module"])

        bundle.actor.eval()
        bundle.actor.on_episode_start(np.zeros(OBS_DIM, dtype=np.float32))
        action = bundle.actor.get_action(_policy_obs(0))
        self.assertEqual(tuple(action.shape), (1, ACT_DIM))
        self.assertTrue(torch.isfinite(action).all())
        self.assertTrue(bool((action.abs() <= 1.0).all()))

    def test_history_bundle_selects_history_agent(self) -> None:
        bundle = _build(HISTORY_BUNDLE)
        self.assertEqual(bundle.metadata["rma_agent"], "HistoryAgent")
        self.assertEqual(bundle.metadata["rma_mode"], "history")
        bundle.actor.on_episode_start(np.zeros(OBS_DIM, dtype=np.float32))
        self.assertEqual(tuple(bundle.actor.get_action(_policy_obs(1)).shape), (1, ACT_DIM))

    def test_get_action_before_episode_start_raises(self) -> None:
        bundle = _build(RMA_BUNDLE)
        with self.assertRaises(RuntimeError):
            bundle.actor.get_action(_policy_obs(2))

    def test_obs_dim_mismatch_is_caught_at_load(self) -> None:
        # A hist_len / task mismatch must fail loudly, not produce garbage actions.
        with self.assertRaises(ValueError):
            _build(RMA_BUNDLE, obs_dim=OBS_DIM + 3)

    def test_on_episode_start_clears_the_history_window(self) -> None:
        """Without the per-episode reset the window leaks the prior episode."""
        bundle = _build(RMA_BUNDLE)
        first_obs = np.zeros(OBS_DIM, dtype=np.float32)
        steps = [_policy_obs(i) for i in range(4)]

        bundle.actor.on_episode_start(first_obs)
        episode_a = [bundle.actor.get_action(s).clone() for s in steps]

        # Drive extra steps so the window is dirty, then start a fresh episode.
        for s in steps:
            bundle.actor.get_action(s)
        bundle.actor.on_episode_start(first_obs)
        episode_b = [bundle.actor.get_action(s).clone() for s in steps]

        for a, b in zip(episode_a, episode_b):
            torch.testing.assert_close(a, b)


if __name__ == "__main__":
    unittest.main()

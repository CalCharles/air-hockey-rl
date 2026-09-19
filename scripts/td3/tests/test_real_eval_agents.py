"""Tests for frozen-eval actor loading.

Sim ``training_state.pth`` files (and the exported policy bundles) often
omit replay buffers. Frozen eval must still load the actor.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.td3.deterministic_agent import DeterministicAgent
from scripts.td3.helper.real_eval_agents import build_td3_eval_agent
from scripts.td3.helper.real_td3_runtime import Args, TrainArgs, _load_training_state_checkpoint


class _DummySpaces:
    def __init__(self, obs_dim: int, act_dim: int) -> None:
        import gymnasium as gym

        self.single_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.single_action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
        )


def _make_actor(obs_dim: int = 4, act_dim: int = 2, fill: float = 0.42) -> DeterministicAgent:
    actor = DeterministicAgent(
        _DummySpaces(obs_dim, act_dim),
        action_scale=1.0,
        action_bias=0.0,
        hidden_layer_size=8,
        num_hidden_layers=1,
    )
    with torch.no_grad():
        for p in actor.parameters():
            p.fill_(fill)
    return actor


def _train_args() -> TrainArgs:
    return TrainArgs(
        agent_hidden_layer_size=8,
        agent_num_hidden_layers=1,
        q_hidden_layer_size=8,
        q_num_hidden_layers=1,
        use_last_action_in_policy_state=True,
    )


class BuildTd3EvalAgentCheckpointTests(unittest.TestCase):
    OBS_DIM = 4
    ACT_DIM = 2

    def _build(self, model_path: Path):
        args = Args(model_path=str(model_path), full_checkpoint_load="full_resume")
        return build_td3_eval_agent(
            args=args,
            train_args=_train_args(),
            obs_dim=self.OBS_DIM,
            act_dim=self.ACT_DIM,
            action_low_np=np.full((self.ACT_DIM,), -1.0, dtype=np.float32),
            action_high_np=np.full((self.ACT_DIM,), 1.0, dtype=np.float32),
            device=torch.device("cpu"),
        )

    def test_stripped_sim_training_state_loads_without_replay(self) -> None:
        actor = _make_actor(self.OBS_DIM + self.ACT_DIM, self.ACT_DIM)
        payload = {
            "actor": dict(actor.state_dict()),
            "actor_target": dict(actor.state_dict()),
            "learner_q_updates": 17,
            "learner_actor_updates": 3,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "training_state.pth"
            torch.save(payload, path)
            with self.assertRaises(KeyError):
                _load_training_state_checkpoint(str(path))
            bundle = self._build(path)

        loaded = next(bundle.actor.parameters()).flatten()[0].item()
        self.assertAlmostEqual(loaded, 0.42)
        self.assertEqual(bundle.metadata["q_updates"], 17)
        self.assertEqual(bundle.metadata["actor_updates"], 3)

    def test_raw_model_pth_loads(self) -> None:
        actor = _make_actor(self.OBS_DIM + self.ACT_DIM, self.ACT_DIM, fill=0.7)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pth"
            torch.save(actor.state_dict(), path)
            bundle = self._build(path)

        loaded = next(bundle.actor.parameters()).flatten()[0].item()
        self.assertAlmostEqual(loaded, 0.7)
        self.assertEqual(bundle.metadata["q_updates"], 0)


if __name__ == "__main__":
    unittest.main()

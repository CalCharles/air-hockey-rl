"""Tests for residual RL wiring in the real-world TD3 runtime library.

Covers:
- Relaxed `_load_training_state_checkpoint` defaults missing non-vital keys.
- `Args` exposes residual fields and they round-trip through `_build_args_file_defaults`.
- `_build_collector_actor` returns `ResidualActor` in residual mode and
  `DeterministicAgent` otherwise, and the residual head is zero-init.
- `_init_sync_learner_state` in residual mode wraps the actor as
  `ResidualActor`, optimizer covers only the residual head's parameters,
  and the critic is fresh.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.td3.helper.real_td3_runtime import (
    Args,
    TrainArgs,
    _build_args_file_defaults,
    _build_collector_actor,
    _init_sync_learner_state,
    _load_training_state_checkpoint,
    _NON_VITAL_TRAINING_STATE_DEFAULTS,
    _VITAL_TRAINING_STATE_KEYS,
)
from scripts.td3.helper.shared_replay import SharedTD3Replay
from scripts.td3.deterministic_agent import DeterministicAgent
from scripts.td3.residual_agent import ResidualActor


def _make_minimal_vital_state(obs_dim: int = 4, act_dim: int = 2) -> dict:
    """Build a training_state-shaped dict with vital keys only."""
    actor_dummy = DeterministicAgent(
        _DummySpaces(obs_dim, act_dim),
        action_scale=1.0,
        action_bias=0.0,
        hidden_layer_size=8,
        num_hidden_layers=1,
    )
    payload = {
        "actor": dict(actor_dummy.state_dict()),
        "actor_target": dict(actor_dummy.state_dict()),
        "qf1": {},
        "qf2": {},
        "qf1_target": {},
        "qf2_target": {},
        "success_replay_buffer": {},
        "failure_replay_buffer": {},
        "rng_states": {},
    }
    # Sanity: every vital key is present.
    for key in _VITAL_TRAINING_STATE_KEYS:
        assert key in payload
    return payload


class _DummySpaces:
    """Minimal spaces shim for DeterministicAgent."""

    def __init__(self, obs_dim: int, act_dim: int) -> None:
        import gymnasium as gym

        self.single_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.single_action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
        )


class LoadTrainingStateCheckpointDefaultsTests(unittest.TestCase):
    """`_load_training_state_checkpoint` should fill missing non-vital keys
    with documented defaults instead of raising."""

    def test_partial_non_vital_fields_get_defaults(self) -> None:
        payload = _make_minimal_vital_state()
        # Sim-style: some non-vital fields are present, others (real-only) are not.
        payload["q_optimizer"] = {"state": {}, "param_groups": []}
        payload["actor_optimizer"] = {"state": {}, "param_groups": []}
        payload["train_metrics"] = {"some_metric": 1.5}
        # Intentionally omit: learner_q_updates, learner_actor_updates,
        # collector_total_steps, run_elapsed_total_s, rolling50_*.

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "training_state.pth"
            torch.save(payload, path)
            loaded = _load_training_state_checkpoint(str(path))

        self.assertEqual(loaded["learner_q_updates"], 0)
        self.assertEqual(loaded["learner_actor_updates"], 0)
        self.assertEqual(loaded["collector_total_steps"], 0)
        self.assertEqual(loaded["run_elapsed_total_s"], 0.0)
        self.assertEqual(loaded["rolling50_task_reward_values"], [])
        self.assertEqual(loaded["rolling50_motion_reward_values"], [])
        self.assertEqual(loaded["rolling50_episode_length_values"], [])
        self.assertEqual(loaded["rolling50_estop_episode_flags"], [])
        # Existing values are preserved (not overwritten by defaults).
        self.assertEqual(loaded["train_metrics"], {"some_metric": 1.5})
        # Optimizers are NOT defaulted: their absence is the gate the learner
        # uses; fill them only when the source actually has them.
        self.assertIn("q_optimizer", loaded)
        self.assertIn("actor_optimizer", loaded)

    def test_missing_optimizers_are_not_defaulted(self) -> None:
        payload = _make_minimal_vital_state()
        # No optimizers, no counters — purely sim-source-style.

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "training_state.pth"
            torch.save(payload, path)
            loaded = _load_training_state_checkpoint(str(path))

        # q_optimizer/actor_optimizer remain absent so the learner's
        # `if "q_optimizer" in resume_checkpoint:` gate stays False.
        self.assertNotIn("q_optimizer", loaded)
        self.assertNotIn("actor_optimizer", loaded)
        # Counters/rolling50 are still defaulted for safe direct-access.
        self.assertEqual(loaded["learner_q_updates"], 0)
        self.assertEqual(loaded["rolling50_task_reward_values"], [])

    def test_missing_vital_keys_still_raise(self) -> None:
        payload = _make_minimal_vital_state()
        payload.pop("rng_states")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "training_state.pth"
            torch.save(payload, path)
            with self.assertRaises(KeyError):
                _load_training_state_checkpoint(str(path))

    def test_default_lists_are_independent_per_call(self) -> None:
        # Mutable defaults must be copied, not aliased to the module-level dict.
        payload = _make_minimal_vital_state()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "training_state.pth"
            torch.save(payload, path)
            loaded_a = _load_training_state_checkpoint(str(path))
            loaded_b = _load_training_state_checkpoint(str(path))

        loaded_a["rolling50_task_reward_values"].append(99.0)
        self.assertEqual(loaded_b["rolling50_task_reward_values"], [])
        self.assertEqual(_NON_VITAL_TRAINING_STATE_DEFAULTS["rolling50_task_reward_values"], [])


class ResidualArgsMappingTests(unittest.TestCase):
    """Residual fields exist on Args and round-trip through the YAML loader."""

    def test_residual_fields_are_default_off(self) -> None:
        args = Args()
        self.assertEqual(args.full_checkpoint_load, "full_resume")
        self.assertEqual(args.residual_scale, 0.15)
        self.assertEqual(args.residual_weight_decay, 0.0)
        self.assertIsNone(args.residual_ema_decay)
        self.assertEqual(args.residual_action_l2, 0.0)

    def test_residual_yaml_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "residual.yaml"
            path.write_text(
                "full_checkpoint_load: residual\n"
                "residual_scale: 0.1\n"
                "residual_weight_decay: 0.001\n"
                "residual_ema_decay: 0.9999\n"
                "residual_action_l2: 0.05\n",
                encoding="utf-8",
            )
            mapped, applied, ignored = _build_args_file_defaults(str(path))

        self.assertEqual(mapped["full_checkpoint_load"], "residual")
        self.assertEqual(mapped["residual_scale"], 0.1)
        self.assertEqual(mapped["residual_weight_decay"], 0.001)
        self.assertEqual(mapped["residual_ema_decay"], 0.9999)
        self.assertEqual(mapped["residual_action_l2"], 0.05)
        for key in (
            "full_checkpoint_load",
            "residual_scale",
            "residual_weight_decay",
            "residual_ema_decay",
            "residual_action_l2",
        ):
            self.assertIn(key, applied)
        self.assertEqual(ignored, [])


def _make_train_args() -> TrainArgs:
    return TrainArgs(
        action_scale=1.0,
        agent_hidden_layer_size=8,
        agent_num_hidden_layers=1,
        q_hidden_layer_size=8,
        q_num_hidden_layers=1,
        use_last_action_in_policy_state=True,
    )


class BuildCollectorActorTests(unittest.TestCase):
    OBS_DIM = 4
    ACT_DIM = 2

    def _bounds(self):
        action_low = np.full((self.ACT_DIM,), -1.0, dtype=np.float32)
        action_high = np.full((self.ACT_DIM,), 1.0, dtype=np.float32)
        return action_low, action_high

    def test_non_residual_returns_deterministic_agent(self) -> None:
        args = Args()
        action_low, action_high = self._bounds()
        actor = _build_collector_actor(
            args=args,
            train_args=_make_train_args(),
            obs_dim=self.OBS_DIM,
            act_dim=self.ACT_DIM,
            action_low_np=action_low,
            action_high_np=action_high,
            device=torch.device("cpu"),
        )
        self.assertIsInstance(actor, DeterministicAgent)
        self.assertNotIsInstance(actor, ResidualActor)

    def test_residual_returns_residual_actor_with_zero_head(self) -> None:
        args = Args(full_checkpoint_load="residual", residual_scale=0.1)
        action_low, action_high = self._bounds()
        actor = _build_collector_actor(
            args=args,
            train_args=_make_train_args(),
            obs_dim=self.OBS_DIM,
            act_dim=self.ACT_DIM,
            action_low_np=action_low,
            action_high_np=action_high,
            device=torch.device("cpu"),
        )
        self.assertIsInstance(actor, ResidualActor)
        head_weight = actor.residual.actor_mean_head.weight
        head_bias = actor.residual.actor_mean_head.bias
        self.assertEqual(float(head_weight.abs().sum().item()), 0.0)
        self.assertEqual(float(head_bias.abs().sum().item()), 0.0)
        # Residual scale propagated to the residual's action_scale buffer.
        self.assertAlmostEqual(float(actor.residual.action_scale.flatten()[0].item()), 0.1)


class InitSyncLearnerStateResidualTests(unittest.TestCase):
    OBS_DIM = 4
    ACT_DIM = 2

    def _save_source_actor(self, tmp_dir: Path) -> Path:
        """Save a DeterministicAgent state_dict at <tmp>/source/model.pth."""
        spaces = _DummySpaces(self.OBS_DIM + self.ACT_DIM, self.ACT_DIM)
        actor = DeterministicAgent(
            spaces,
            action_scale=1.0,
            action_bias=0.0,
            hidden_layer_size=8,
            num_hidden_layers=1,
        )
        # Set distinctive non-zero weights so we can verify they were loaded.
        with torch.no_grad():
            for p in actor.parameters():
                p.fill_(0.123)
        path = tmp_dir / "model.pth"
        torch.save(actor.state_dict(), path)
        return path

    def _make_args(self, source_path: Path, log_dir: Path, **overrides) -> Args:
        kwargs = dict(
            train_args="unused-in-this-test",
            args_file=None,
            config="unused-in-this-test",
            model_path=str(source_path),
            full_checkpoint_load="residual",
            residual_scale=0.1,
            collector_device="cpu",
            learner_device="cpu",
            log_parent_dir=str(log_dir),
            checkpoint_root_dir=str(log_dir),
        )
        kwargs.update(overrides)
        return Args(**kwargs)

    def _build_state(self, args: Args, log_dir: Path):
        replay = SharedTD3Replay(
            success_capacity=8,
            failure_capacity=8,
            obs_shape=(self.OBS_DIM,),
            action_shape=(self.ACT_DIM,),
        )
        action_low = np.full((self.ACT_DIM,), -1.0, dtype=np.float32)
        action_high = np.full((self.ACT_DIM,), 1.0, dtype=np.float32)
        return _init_sync_learner_state(
            args=args,
            train_args=_make_train_args(),
            replay=replay,
            stats={},
            obs_dim=self.OBS_DIM,
            act_dim=self.ACT_DIM,
            action_low_np=action_low,
            action_high_np=action_high,
            tb_log_dir=str(log_dir / "tb"),
            resume_checkpoint=None,
        )

    def test_residual_init_wraps_actor_and_freezes_base(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_path = self._save_source_actor(tmp_path)
            args = self._make_args(source_path, tmp_path)
            state = self._build_state(args, tmp_path)
            try:
                self.assertIsInstance(state.actor, ResidualActor)
                self.assertIsInstance(state.actor_target, ResidualActor)
                # Both wrappers share the same frozen base instance.
                self.assertIs(state.actor.base, state.actor_target.base)
                # Base parameters are frozen and equal the saved source weights.
                for p in state.actor.base.parameters():
                    self.assertFalse(p.requires_grad)
                    self.assertTrue(torch.allclose(p, torch.full_like(p, 0.123)))
                # Residual head is zero-init.
                head = state.actor.residual.actor_mean_head
                self.assertEqual(float(head.weight.abs().sum().item()), 0.0)
                self.assertEqual(float(head.bias.abs().sum().item()), 0.0)
                # actor_optimizer covers ONLY the residual head's params.
                opt_params = {id(p) for group in state.actor_optimizer.param_groups for p in group["params"]}
                expected = {id(p) for p in state.actor.residual.parameters()}
                self.assertEqual(opt_params, expected)
                # No EMA when residual_ema_decay is None.
                self.assertIsNone(state.actor_ema)
            finally:
                state.writer.close()

    def test_residual_init_with_ema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_path = self._save_source_actor(tmp_path)
            args = self._make_args(source_path, tmp_path, residual_ema_decay=0.99)
            state = self._build_state(args, tmp_path)
            try:
                self.assertIsInstance(state.actor_ema, ResidualActor)
                # EMA wraps the same base instance.
                self.assertIs(state.actor_ema.base, state.actor.base)
                # EMA residual params are frozen (no_grad copy).
                for p in state.actor_ema.residual.parameters():
                    self.assertFalse(p.requires_grad)
            finally:
                state.writer.close()


if __name__ == "__main__":
    unittest.main()

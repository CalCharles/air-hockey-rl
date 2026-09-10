"""Tests for the task plug-in points on ``ResetRunner`` / ``pick_reset_kind``.

Runs the reset runner against a fake env + fake FSM class so no robot or
Box2D is needed:

- SOFT path: ``post_soft_reset_hook`` runs after ``env.soft_reset()`` and
  before the paddle-history priming, so the primed obs already reflects
  the hook's side effect (goal appended via ``_append_goal_if_goal_env``);
- HARD path: ``force_fsm_after_hard_reset=True`` runs the FSM without stop
  flags and without the puck heuristic; the default keeps the historical
  ``HARD_SKIP_FSM`` downgrade;
- defaults are untouched → historical behaviour (hook is ``None``);
- ``pick_reset_kind`` honours ``periodic_every`` (0 disables).
"""
from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import torch

from scripts.td3.helper import real_reset_runner as rr
from scripts.td3.helper.real_reset_runner import (
    ResetKind,
    ResetRunner,
    StopFlags,
    pick_reset_kind,
)


class _FakeSimulator:
    def __init__(self):
        self.pos = np.array([-0.7, 0.0])
        self.paddle_history = [(-2.0, 0.0, 1)] * 5
        self.puck_history = [(-2.0, 0.0, 1)] * 5
        self.paddle_history_len = 5
        self.move_lims = (0.26, 0.12)

    def get_current_state(self):
        return {
            "paddles": {"paddle_ego": {"position": self.pos.copy(), "velocity": np.zeros(2)}},
            # Puck "visible" at mid-table so the juggle heuristic says
            # "no reset policy needed" on the hard path.
            "pucks": [{"position": np.array([-1.2, 0.0]), "occluded": np.array([0])}],
        }


class _FakeGoalEnv:
    """Goal env with ``return_goal_obs=False`` (goal appended to obs)."""

    return_goal_obs = False
    table_x_bot = -0.4

    def __init__(self):
        self.simulator = _FakeSimulator()
        self.obs_type = "history"
        self.goal = np.array([0.5, 0.5])
        self.calls: list[str] = []

    # --- reset surface used by ResetRunner ---
    def soft_reset(self):
        self.calls.append("soft_reset")

    def reset(self, seed=None, **kwargs):
        self.calls.append("hard_reset")
        return np.zeros(4), {}

    def step(self, action):
        self.calls.append("step")
        return np.zeros(4), 0.0, False, False, {}

    def get_observation(self, state_info, obs_type=None, puck_history=None, paddle_history=None):
        self.calls.append("get_observation")
        return np.asarray(paddle_history[-1][:2], dtype=np.float32)

    def get_desired_goal(self):
        return self.goal.copy()


class _FakeFSM:
    """Two-step FSM that succeeds immediately."""

    instances: list = []

    def __init__(self, env, rng):
        self.phase = "goto_start"
        self.total_steps = 0
        self.done = False
        self.done_reason = "in_progress"
        self.start_side = "fake"
        _FakeFSM.instances.append(self)

    def step(self, state):
        self.total_steps += 1
        if self.total_steps >= 2:
            self.done = True
            self.done_reason = "success"
        return np.zeros(2, dtype=np.float32)

    def close(self):
        pass


def _row(**kwargs):
    return {"pose": np.zeros(6)}


def _runner(env, **kwargs):
    runner = ResetRunner(
        env,
        device=torch.device("cpu"),
        reset_rng=np.random.default_rng(0),
        reset_policy_fsm_cls=_FakeFSM,
        build_split_episode_row=_row,
        latest_camera_frame=lambda env: None,
        **kwargs,
    )
    runner.MIN_RESET_DELAY_S = 0.0
    return runner


def _run(runner, kind, stop=StopFlags()):
    return runner.run(
        kind=kind,
        artifact_episode_id=1,
        episode_had_stop_flags=stop,
        episode_end_wall_time=0.0,
        pending_reset_artifact=None,
        next_reset_file_id=1,
    )


class SoftResetHookTests(unittest.TestCase):
    def setUp(self):
        _FakeFSM.instances.clear()

    def test_hook_runs_between_soft_reset_and_priming(self):
        env = _FakeGoalEnv()

        def hook(e):
            e.calls.append("hook")
            e.goal = np.array([0.9, -0.9])

        runner = _runner(env, post_soft_reset_hook=hook)
        with mock.patch.object(rr.time, "sleep"):
            result = _run(runner, ResetKind.SOFT)
        i_soft = env.calls.index("soft_reset")
        i_hook = env.calls.index("hook")
        i_obs = env.calls.index("get_observation")
        self.assertLess(i_soft, i_hook)
        self.assertLess(i_hook, i_obs)
        # Primed obs = [paddle xy] + [new goal].
        np.testing.assert_allclose(result.obs[-2:], [0.9, -0.9])
        self.assertEqual(result.kind_actual, ResetKind.SOFT)
        self.assertEqual(result.total_fsm_steps, 2)

    def test_default_has_no_hook(self):
        env = _FakeGoalEnv()
        runner = _runner(env)
        with mock.patch.object(rr.time, "sleep"):
            result = _run(runner, ResetKind.STARTUP)
        self.assertNotIn("hook", env.calls)
        np.testing.assert_allclose(result.obs[-2:], [0.5, 0.5])


class HardResetForceFSMTests(unittest.TestCase):
    def setUp(self):
        _FakeFSM.instances.clear()

    def test_default_downgrades_to_skip_fsm_without_stop(self):
        env = _FakeGoalEnv()
        runner = _runner(env)
        with mock.patch.object(rr.time, "sleep"):
            result = _run(runner, ResetKind.HARD_WITH_FSM)
        self.assertEqual(result.kind_actual, ResetKind.HARD_SKIP_FSM)
        self.assertEqual(len(_FakeFSM.instances), 0)
        self.assertIn("hard_reset", env.calls)

    def test_force_runs_fsm_after_hard_reset(self):
        env = _FakeGoalEnv()
        hook_calls = []
        runner = _runner(
            env,
            post_soft_reset_hook=lambda e: hook_calls.append(1),
            force_fsm_after_hard_reset=True,
        )
        with mock.patch.object(rr.time, "sleep"):
            result = _run(runner, ResetKind.HARD_WITH_FSM)
        self.assertEqual(result.kind_actual, ResetKind.HARD_WITH_FSM)
        self.assertEqual(len(_FakeFSM.instances), 1)
        self.assertEqual(result.total_fsm_steps, 2)
        self.assertEqual(hook_calls, [1])
        self.assertEqual(result.transition_reason, "hard_reset_reset_fsm_to_policy")
        # hard reset happened before the FSM stepped
        self.assertLess(env.calls.index("hard_reset"), env.calls.index("step"))

    def test_stop_still_forces_fsm_without_flag(self):
        env = _FakeGoalEnv()
        runner = _runner(env)
        with mock.patch.object(rr.time, "sleep"):
            result = _run(
                runner, ResetKind.HARD_WITH_FSM, stop=StopFlags(had_stop=True, had_protective_stop=True)
            )
        self.assertEqual(result.kind_actual, ResetKind.HARD_WITH_FSM)


class PickResetKindTests(unittest.TestCase):
    def test_default_periodic_three(self):
        self.assertEqual(pick_reset_kind(1, StopFlags()), ResetKind.SOFT)
        self.assertEqual(pick_reset_kind(3, StopFlags()), ResetKind.HARD_WITH_FSM)
        self.assertEqual(pick_reset_kind(6, StopFlags()), ResetKind.HARD_WITH_FSM)

    def test_custom_and_disabled_periodic(self):
        self.assertEqual(pick_reset_kind(3, StopFlags(), periodic_every=5), ResetKind.SOFT)
        self.assertEqual(pick_reset_kind(5, StopFlags(), periodic_every=5), ResetKind.HARD_WITH_FSM)
        self.assertEqual(pick_reset_kind(3, StopFlags(), periodic_every=0), ResetKind.SOFT)
        self.assertEqual(pick_reset_kind(300, StopFlags(), periodic_every=0), ResetKind.SOFT)

    def test_stop_always_hard(self):
        self.assertEqual(
            pick_reset_kind(1, StopFlags(had_stop=True), periodic_every=0), ResetKind.HARD_WITH_FSM
        )


if __name__ == "__main__":
    unittest.main()

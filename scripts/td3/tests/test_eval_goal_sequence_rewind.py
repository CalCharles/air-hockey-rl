"""Regression: eval goal grid must not advance on discarded episodes."""
from __future__ import annotations

import unittest

import numpy as np

from airhockey.airhockey_tasks.abstract_airhockey_goal_task import AirHockeyGoalEnv
from airhockey.airhockey_tasks.puck_goal_position import AirHockeyPuckGoalPositionEnv


def _goal_env_stub() -> AirHockeyGoalEnv:
    env = object.__new__(AirHockeyPuckGoalPositionEnv)
    env.goal_radius_type = "linear_decay"
    env.goal_pos = np.zeros(2, dtype=float)
    env._sync_goal_marker_to_simulator = lambda: None  # type: ignore[method-assign]

    def _set_goals(goal_radius_type, goal_pos=None, alt_goal_pos=None, goal_set=None):
        del goal_radius_type, alt_goal_pos, goal_set
        if goal_pos is not None:
            env.goal_pos = np.asarray(goal_pos, dtype=float)

    env.set_goals = _set_goals  # type: ignore[method-assign]
    env.set_goal_sequence([(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)])
    return env


class TestEvalGoalSequenceRewind(unittest.TestCase):
    def test_rewind_undoes_last_advance(self) -> None:
        env = _goal_env_stub()
        first = env._next_goal_pos_from_sequence()
        self.assertEqual(int(env._goal_sequence_idx), 1)
        env.rewind_goal_sequence()
        self.assertEqual(int(env._goal_sequence_idx), 0)
        again = env._next_goal_pos_from_sequence()
        np.testing.assert_allclose(first, again)

    def test_rewind_is_noop_without_sequence(self) -> None:
        env = _goal_env_stub()
        env.set_goal_sequence(None)
        env.rewind_goal_sequence()
        self.assertEqual(int(getattr(env, "_goal_sequence_idx", 0)), 0)

    def test_discard_then_reset_reuses_goal(self) -> None:
        env = _goal_env_stub()
        goal_at_start = env._next_goal_pos_from_sequence()
        self.assertEqual(int(env._goal_sequence_idx), 1)

        # Discarded attempt: rewind before the inter-episode reset consumes again.
        env.rewind_goal_sequence()
        goal_after_reset = env._next_goal_pos_from_sequence()
        np.testing.assert_allclose(goal_at_start, goal_after_reset)
        self.assertEqual(int(env._goal_sequence_idx), 1)

        # Kept attempt: no rewind; next reset should advance to the following goal.
        goal_after_kept_reset = env._next_goal_pos_from_sequence()
        np.testing.assert_allclose(goal_after_kept_reset, np.array([0.3, 0.4]))

    def test_prepare_goal_sequence_for_kept_index(self) -> None:
        env = _goal_env_stub()
        env.prepare_goal_sequence_for_kept_index(2)
        self.assertEqual(int(env._goal_sequence_idx), 3)
        np.testing.assert_allclose(env.goal_pos, np.array([0.5, 0.6]))


class TestRestartEvalArgs(unittest.TestCase):
    def test_restart_validation(self) -> None:
        from scripts.td3.extras.async_td3_real_eval import (
            EvalSpecificArgs,
            _validate_restart_eval_args,
        )

        _validate_restart_eval_args(
            EvalSpecificArgs(eval_episodes=20, restart_eval_from_episode=1)
        )
        _validate_restart_eval_args(
            EvalSpecificArgs(eval_episodes=20, restart_eval_from_episode=4)
        )
        with self.assertRaises(SystemExit):
            _validate_restart_eval_args(
                EvalSpecificArgs(eval_episodes=20, restart_eval_from_episode=0)
            )
        with self.assertRaises(SystemExit):
            _validate_restart_eval_args(
                EvalSpecificArgs(eval_episodes=20, restart_eval_from_episode=21)
            )

    def test_restart_episode_four_uses_fourth_goal(self) -> None:
        from scripts.td3.extras.async_td3_real_eval import _align_goal_sequence_for_restart

        env = _goal_env_stub()
        env.set_goal_sequence(
            [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6), (0.7, 0.8)]
        )
        _align_goal_sequence_for_restart(env, restart_from_episode=4)
        self.assertEqual(int(env._goal_sequence_idx), 4)
        np.testing.assert_allclose(env.goal_pos, np.array([0.7, 0.8]))


class TestEstopGoalRewind(unittest.TestCase):
    def test_should_rewind_and_eval_counting(self) -> None:
        from scripts.td3.extras.async_td3_real_eval import (
            _counts_toward_eval_set,
            _estop_triggers_goal_retry,
            _should_rewind_goal_after_episode,
        )

        self.assertTrue(
            _should_rewind_goal_after_episode(
                episode_kept=False,
                rewind_on_estop=False,
                had_estop=False,
                n_paddle_puck_hits=0,
            )
        )
        self.assertFalse(
            _should_rewind_goal_after_episode(
                episode_kept=True,
                rewind_on_estop=False,
                had_estop=True,
                n_paddle_puck_hits=0,
            )
        )
        self.assertTrue(
            _should_rewind_goal_after_episode(
                episode_kept=True,
                rewind_on_estop=True,
                had_estop=True,
                n_paddle_puck_hits=1,
            )
        )
        self.assertFalse(
            _should_rewind_goal_after_episode(
                episode_kept=True,
                rewind_on_estop=True,
                had_estop=True,
                n_paddle_puck_hits=2,
            )
        )
        self.assertTrue(
            _estop_triggers_goal_retry(
                rewind_on_estop=True, had_estop=True, n_paddle_puck_hits=1
            )
        )
        self.assertFalse(
            _estop_triggers_goal_retry(
                rewind_on_estop=True, had_estop=True, n_paddle_puck_hits=2
            )
        )
        self.assertFalse(
            _counts_toward_eval_set(
                episode_kept=True,
                rewind_on_estop=True,
                had_estop=True,
                n_paddle_puck_hits=1,
            )
        )
        self.assertTrue(
            _counts_toward_eval_set(
                episode_kept=True,
                rewind_on_estop=True,
                had_estop=True,
                n_paddle_puck_hits=2,
            )
        )
        self.assertTrue(
            _counts_toward_eval_set(
                episode_kept=True,
                rewind_on_estop=True,
                had_estop=False,
                n_paddle_puck_hits=0,
            )
        )

    def test_estop_rewind_reuses_goal_on_reset(self) -> None:
        from scripts.td3.extras.async_td3_real_eval import (
            _maybe_rewind_goal_sequence_after_episode,
        )

        env = _goal_env_stub()
        goal_at_start = env._next_goal_pos_from_sequence()
        self.assertEqual(int(env._goal_sequence_idx), 1)

        rewound = _maybe_rewind_goal_sequence_after_episode(
            env,
            episode_kept=True,
            rewind_on_estop=True,
            had_estop=True,
            n_paddle_puck_hits=1,
        )
        self.assertTrue(rewound)
        goal_after_reset = env._next_goal_pos_from_sequence()
        np.testing.assert_allclose(goal_at_start, goal_after_reset)
        self.assertEqual(int(env._goal_sequence_idx), 1)

    def test_multi_hit_estop_does_not_rewind(self) -> None:
        from scripts.td3.extras.async_td3_real_eval import (
            _maybe_rewind_goal_sequence_after_episode,
        )

        env = _goal_env_stub()
        env._next_goal_pos_from_sequence()
        self.assertEqual(int(env._goal_sequence_idx), 1)

        rewound = _maybe_rewind_goal_sequence_after_episode(
            env,
            episode_kept=True,
            rewind_on_estop=True,
            had_estop=True,
            n_paddle_puck_hits=2,
        )
        self.assertFalse(rewound)
        self.assertEqual(int(env._goal_sequence_idx), 1)
        goal_after_reset = env._next_goal_pos_from_sequence()
        np.testing.assert_allclose(goal_after_reset, np.array([0.3, 0.4]))

    def test_hard_then_soft_reset_needs_extra_rewind(self) -> None:
        from scripts.td3.helper.real_reset_runner import _rewind_goal_sequence_if_available

        env = _goal_env_stub()
        goal_at_start = env._next_goal_pos_from_sequence()
        self.assertEqual(int(env._goal_sequence_idx), 1)

        # End-of-episode rewind before inter-episode reset.
        _rewind_goal_sequence_if_available(env, reason="after estop")
        # Hard reset consumes the same goal again.
        np.testing.assert_allclose(env._next_goal_pos_from_sequence(), goal_at_start)
        self.assertEqual(int(env._goal_sequence_idx), 1)

        # Without compensation, soft reset would advance to the next goal.
        wrong_goal = env._next_goal_pos_from_sequence()
        np.testing.assert_allclose(wrong_goal, np.array([0.3, 0.4]))

        # Replay with compensation before soft reset.
        env.set_goal_sequence([(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)])
        env._goal_sequence_idx = 1
        _rewind_goal_sequence_if_available(env, reason="after estop")
        np.testing.assert_allclose(env._next_goal_pos_from_sequence(), goal_at_start)
        _rewind_goal_sequence_if_available(env, reason="before soft prime after hard reset")
        np.testing.assert_allclose(env._next_goal_pos_from_sequence(), goal_at_start)
        self.assertEqual(int(env._goal_sequence_idx), 1)


if __name__ == "__main__":
    unittest.main()

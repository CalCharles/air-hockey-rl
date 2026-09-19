"""Tests for the task-specific real-world eval hooks.

Covers:
- registry: the five canonical tasks resolve to their hooks class, unknown
  tasks fall through to ``GenericEvalHooks``;
- reset-side contract: puck tasks keep the puck FSM + juggle defaults,
  paddle-only tasks select ``PaddleRepositionFSM``, force the FSM after a
  hard reset, and disable the post-reset hold;
- ``on_soft_reset`` resamples the goal on goal envs (and is a no-op elsewhere);
- per-task metrics on synthetic split-schema rows, and that the resulting
  records aggregate cleanly through ``compute_eval_aggregate`` (``None``
  step counts are skipped rather than crashing).
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from scripts.td3.helper.juggle_counter import CONTACT_THRESH
from scripts.td3.helper.real_eval_stats import compute_eval_aggregate, format_eval_summary_console
from scripts.td3.helper.real_paddle_reposition_fsm import PaddleRepositionFSM
from scripts.td3.helper.real_task_eval_hooks import (
    RESET_STRATEGY_PADDLE_REPOSITION,
    RESET_STRATEGY_PUCK_FSM,
    TASK_EVAL_HOOKS,
    GenericEvalHooks,
    JuggleEvalHooks,
    PaddleReachEvalHooks,
    PaddleReachVelocityEvalHooks,
    PuckTouchEvalHooks,
    PuckVelocityEvalHooks,
    get_task_eval_hooks,
)


CANONICAL = {
    "puck_juggle_upper_half_reward": JuggleEvalHooks,
    "puck_touch": PuckTouchEvalHooks,
    "puck_velocity": PuckVelocityEvalHooks,
    "paddle_reach_position": PaddleReachEvalHooks,
    "paddle_reach_position_velocity": PaddleReachVelocityEvalHooks,
}


def _row(paddle_xy, puck_xy=(-2.0, 0.0), occluded=1.0, speed_xy=(0.0, 0.0)):
    return {
        "pose": np.array([paddle_xy[0], paddle_xy[1], 0, 0, 0, 0], dtype=np.float64),
        "speed": np.array([speed_xy[0], speed_xy[1], 0, 0, 0, 0], dtype=np.float64),
        "puck": np.array([puck_xy[0], puck_xy[1], occluded], dtype=np.float64),
    }


def _result(success=False, reasons=()):
    return SimpleNamespace(
        terminal=SimpleNamespace(episode_success=bool(success), episode_end_reasons=list(reasons))
    )


class _FakeGoalEnv:
    """Minimal ``AirHockeyGoalEnv``-shaped stand-in."""

    def __init__(self, goal=(-0.7, 0.1), goal_vel=None, goal_radius=0.05, goal_velocity_radius=0.5):
        self.goal_radius_type = "fixed"
        self.goal_pos = np.asarray(goal, dtype=np.float64)
        self.goal_vel = None if goal_vel is None else np.asarray(goal_vel, dtype=np.float64)
        self.goal_radius = goal_radius
        self.goal_velocity_radius = goal_velocity_radius
        self.set_goals_calls = []
        self.marker_synced = 0

    def set_goals(self, goal_radius_type, goal_pos=None, alt_goal_pos=None, goal_set=None):
        self.set_goals_calls.append(goal_radius_type)
        self.goal_pos = self.goal_pos + np.array([0.01, 0.0])

    def _sync_goal_marker_to_simulator(self):
        self.marker_synced += 1

    def get_desired_goal(self):
        if self.goal_vel is None:
            return self.goal_pos.copy()
        return np.concatenate([self.goal_pos, self.goal_vel])


class RegistryTests(unittest.TestCase):
    def test_five_canonical_tasks_registered(self):
        for task, cls in CANONICAL.items():
            with self.subTest(task=task):
                self.assertIs(TASK_EVAL_HOOKS[task], cls)
                self.assertIsInstance(get_task_eval_hooks(task), cls)

    def test_unknown_task_falls_back_to_generic(self):
        self.assertIsInstance(get_task_eval_hooks("puck_goal_position"), GenericEvalHooks)
        self.assertIsInstance(get_task_eval_hooks("not_a_task"), GenericEvalHooks)

    def test_series_and_rate_fields_cover_metric_keys(self):
        """Every task-specific metric key that should be summarised appears in
        the hooks' field lists (record-only keys are allowed to be absent)."""
        rows = [_row((-0.7, 0.0), puck_xy=(-0.7, 0.0), occluded=0.0)] * 3
        env = _FakeGoalEnv(goal_vel=(-1.0, 0.0))
        for task, cls in CANONICAL.items():
            hooks = cls()
            hooks.on_episode_start(env)
            metrics = hooks.compute_episode_metrics(result=_result(True), rows=rows, env=env)
            allowed = set(hooks.numeric_series_fields) | set(hooks.rate_fields) | {
                "goal_x", "goal_y", "goal_vx", "goal_vy",
            }
            with self.subTest(task=task):
                self.assertTrue(set(metrics) <= allowed, set(metrics) - allowed)


class ResetContractTests(unittest.TestCase):
    def test_puck_tasks_keep_juggle_reset_defaults(self):
        for cls in (JuggleEvalHooks, GenericEvalHooks, PuckTouchEvalHooks, PuckVelocityEvalHooks):
            hooks = cls()
            with self.subTest(cls=cls.__name__):
                self.assertEqual(hooks.reset_strategy, RESET_STRATEGY_PUCK_FSM)
                self.assertFalse(hooks.force_fsm_after_hard_reset)
                self.assertEqual(hooks.periodic_hard_reset_every, 3)
                self.assertIsNone(hooks.post_reset_transition_hold_steps)

    def test_juggle_hooks_bit_identical_surface(self):
        hooks = JuggleEvalHooks()
        self.assertEqual(hooks.min_timesteps, 50)
        self.assertEqual(
            hooks.numeric_series_fields,
            ("episode_return", "episode_juggles", "episode_contacts", "episode_reward", "episode_length"),
        )
        self.assertEqual(GenericEvalHooks().min_timesteps, 10)

    def test_paddle_only_tasks_use_reposition_reset(self):
        for cls in (PaddleReachEvalHooks, PaddleReachVelocityEvalHooks):
            hooks = cls()
            with self.subTest(cls=cls.__name__):
                self.assertEqual(hooks.reset_strategy, RESET_STRATEGY_PADDLE_REPOSITION)
                self.assertIs(hooks.make_reset_fsm_cls(), PaddleRepositionFSM)
                self.assertTrue(hooks.force_fsm_after_hard_reset)
                self.assertEqual(hooks.post_reset_transition_hold_steps, 0)
                self.assertEqual(hooks.min_timesteps, 1)

    def test_on_soft_reset_resamples_goal_and_syncs_marker(self):
        env = _FakeGoalEnv()
        before = env.goal_pos.copy()
        PaddleReachEvalHooks().on_soft_reset(env)
        self.assertEqual(env.set_goals_calls, ["fixed"])
        self.assertEqual(env.marker_synced, 1)
        self.assertFalse(np.allclose(before, env.goal_pos))

    def test_on_soft_reset_noop_without_goals(self):
        env = SimpleNamespace()  # no set_goals
        PaddleReachEvalHooks().on_soft_reset(env)
        JuggleEvalHooks().on_soft_reset(env)


class PaddleReachMetricsTests(unittest.TestCase):
    def test_reached_goal_metrics(self):
        env = _FakeGoalEnv(goal=(-0.7, 0.1), goal_radius=0.05)
        hooks = PaddleReachEvalHooks()
        hooks.on_episode_start(env)
        rows = [
            _row((-0.9, 0.1)),   # 0.20 m away
            _row((-0.8, 0.1)),   # 0.10
            _row((-0.74, 0.1)),  # 0.04 → inside radius at step 3
            _row((-0.71, 0.1)),  # 0.01
        ]
        m = hooks.compute_episode_metrics(
            result=_result(False, reasons=["goal_reached"]), rows=rows, env=env
        )
        self.assertTrue(m["episode_goal_reached"])
        self.assertAlmostEqual(m["episode_final_goal_dist_m"], 0.01, places=6)
        self.assertAlmostEqual(m["episode_min_goal_dist_m"], 0.01, places=6)
        self.assertEqual(m["episode_steps_to_goal"], 3)
        self.assertAlmostEqual(m["goal_x"], -0.7)
        self.assertAlmostEqual(m["goal_y"], 0.1)
        self.assertIn("goal_reached=1", hooks.format_kept_console_extras(m))

    def test_goal_snapshot_survives_goal_resample(self):
        """The goal recorded at episode start is used even if the env's goal
        changes before metrics are computed (the next reset resamples it)."""
        env = _FakeGoalEnv(goal=(-0.7, 0.0))
        hooks = PaddleReachEvalHooks()
        hooks.on_episode_start(env)
        env.set_goals("fixed")  # goal moves by +0.01 in x
        m = hooks.compute_episode_metrics(result=_result(), rows=[_row((-0.7, 0.0))], env=env)
        self.assertAlmostEqual(m["episode_final_goal_dist_m"], 0.0, places=9)

    def test_unreached_goal_leaves_steps_none(self):
        env = _FakeGoalEnv(goal=(-0.5, 0.0), goal_radius=0.05)
        hooks = PaddleReachEvalHooks()
        hooks.on_episode_start(env)
        m = hooks.compute_episode_metrics(result=_result(), rows=[_row((-0.9, 0.0))] * 5, env=env)
        self.assertFalse(m["episode_goal_reached"])
        self.assertIsNone(m["episode_steps_to_goal"])
        self.assertAlmostEqual(m["episode_min_goal_dist_m"], 0.4, places=6)
        self.assertIn("steps_to_goal=-", hooks.format_kept_console_extras(m))

    def test_empty_rows(self):
        env = _FakeGoalEnv()
        hooks = PaddleReachEvalHooks()
        hooks.on_episode_start(env)
        m = hooks.compute_episode_metrics(result=_result(), rows=[], env=env)
        self.assertIsNone(m["episode_final_goal_dist_m"])
        self.assertIsNone(m["episode_min_goal_dist_m"])


class PaddleReachVelocityMetricsTests(unittest.TestCase):
    def test_joint_criterion_for_steps_to_goal(self):
        env = _FakeGoalEnv(goal=(-0.7, 0.0), goal_vel=(-1.0, 0.0), goal_radius=0.05, goal_velocity_radius=0.5)
        hooks = PaddleReachVelocityEvalHooks()
        hooks.on_episode_start(env)
        rows = [
            _row((-0.72, 0.0), speed_xy=(0.0, 0.0)),    # in position, wrong velocity
            _row((-0.71, 0.0), speed_xy=(-0.8, 0.0)),   # both OK → step 2
            _row((-0.69, 0.0), speed_xy=(-1.0, 0.1)),
        ]
        m = hooks.compute_episode_metrics(result=_result(True), rows=rows, env=env)
        self.assertEqual(m["episode_steps_to_goal"], 2)
        self.assertAlmostEqual(m["episode_final_goal_vel_dist_mps"], 0.1, places=6)
        self.assertAlmostEqual(m["episode_min_goal_vel_dist_mps"], 0.1, places=6)
        self.assertAlmostEqual(m["goal_vx"], -1.0)
        self.assertTrue(m["episode_goal_reached"])
        self.assertIn("final_vel_dist=0.100m/s", hooks.format_kept_console_extras(m))


class PuckTaskMetricsTests(unittest.TestCase):
    def _approach_rows(self, n_far=10, n_contact=2, n_after=10):
        rows = []
        for _ in range(n_far):
            rows.append(_row((-0.7, 0.0), puck_xy=(-1.3, 0.0), occluded=0.0))
        for _ in range(n_contact):
            rows.append(_row((-0.7, 0.0), puck_xy=(-0.7 - CONTACT_THRESH / 2, 0.0), occluded=0.0))
        for i in range(n_after):
            rows.append(_row((-0.7, 0.0), puck_xy=(-0.8 - 0.05 * i, 0.0), occluded=0.0))
        return rows

    def test_touch_metrics(self):
        hooks = PuckTouchEvalHooks()
        m = hooks.compute_episode_metrics(result=_result(True), rows=self._approach_rows())
        self.assertTrue(m["episode_touched"])
        self.assertEqual(m["episode_contacts"], 1)
        self.assertEqual(m["episode_steps_to_touch"], 11)
        self.assertIn("steps_to_touch=11", hooks.format_kept_console_extras(m))

    def test_touch_miss(self):
        hooks = PuckTouchEvalHooks()
        rows = [_row((-0.7, 0.0), puck_xy=(-1.3, 0.0), occluded=0.0)] * 8
        m = hooks.compute_episode_metrics(result=_result(False), rows=rows)
        self.assertFalse(m["episode_touched"])
        self.assertEqual(m["episode_contacts"], 0)
        self.assertIsNone(m["episode_steps_to_touch"])

    def test_touch_ignores_occluded_contact_frames(self):
        hooks = PuckTouchEvalHooks()
        rows = [_row((-0.7, 0.0), puck_xy=(-0.7, 0.0), occluded=1.0)] * 8
        m = hooks.compute_episode_metrics(result=_result(False), rows=rows)
        self.assertFalse(m["episode_touched"])
        self.assertIsNone(m["episode_steps_to_touch"])

    def test_puck_velocity_upward_displacement(self):
        hooks = PuckVelocityEvalHooks()
        # Puck x: falls (+x) then, after contact, rises (-x). Only the -x
        # legs count; a step across an occluded frame is skipped.
        xs = [-1.0, -0.9, -0.8, -0.75, -0.8, -0.9, -1.05, -1.1]
        occ = [0, 0, 0, 0, 0, 0, 1, 0]
        rows = [_row((-0.7, 0.0), puck_xy=(x, 0.0), occluded=o) for x, o in zip(xs, occ)]
        m = hooks.compute_episode_metrics(result=_result(False), rows=rows)
        # -0.75→-0.8 (0.05), -0.8→-0.9 (0.10); the two steps touching the
        # occluded frame are skipped.
        self.assertAlmostEqual(m["episode_upward_displacement_m"], 0.15, places=9)
        self.assertAlmostEqual(m["episode_max_upward_step_m"], 0.10, places=9)
        self.assertIsInstance(m["episode_hit_puck"], bool)
        self.assertIn("upward=0.150m", hooks.format_kept_console_extras(m))


class AggregateIntegrationTests(unittest.TestCase):
    def _base_record(self, **extra):
        rec = {
            "episode_return": 10.0,
            "episode_reward": 10.0,
            "episode_length": 12.0,
            "episode_success": True,
            "had_protective_stop": False,
            "had_controller_disconnect": False,
            "readiness_fail_estop": False,
        }
        rec.update(extra)
        return rec

    def test_reach_records_aggregate_with_none_steps(self):
        hooks = PaddleReachEvalHooks()
        records = [
            self._base_record(
                episode_goal_reached=True, episode_final_goal_dist_m=0.01,
                episode_min_goal_dist_m=0.01, episode_steps_to_goal=7, goal_x=0.0, goal_y=0.0,
            ),
            self._base_record(
                episode_return=0.0, episode_reward=0.0, episode_success=False,
                episode_goal_reached=False, episode_final_goal_dist_m=0.3,
                episode_min_goal_dist_m=0.2, episode_steps_to_goal=None, goal_x=0.0, goal_y=0.0,
            ),
        ]
        agg = compute_eval_aggregate(
            records, numeric_fields=hooks.numeric_series_fields, rate_fields=hooks.rate_fields
        )
        self.assertEqual(agg["series"]["episode_steps_to_goal"]["count"], 1)
        self.assertAlmostEqual(agg["series"]["episode_steps_to_goal"]["mean"], 7.0)
        self.assertAlmostEqual(agg["rates"]["episode_goal_reached"]["rate"], 0.5)
        text = format_eval_summary_console(
            agg, n_target=2, n_attempts=2, n_discarded=0,
            numeric_fields=hooks.numeric_series_fields, rate_fields=hooks.rate_fields,
            field_format_overrides=hooks.field_format_overrides,
        )
        self.assertIn("episode_goal_reached", text)


if __name__ == "__main__":
    unittest.main()

"""Tests for ``PaddleRepositionFSM`` — the paddle-only between-episode reset.

Uses a kinematic fake env: each ``step`` moves the paddle by
``action * move_lims`` (with an optional first-order lag), so the FSM's
go-to / settle logic can be exercised without Box2D or the robot.
"""
from __future__ import annotations

import unittest

import numpy as np

from scripts.td3.helper.real_paddle_reposition_fsm import PaddleRepositionFSM


class _FakeSimulator:
    def __init__(self, start_xy, move_lims=(0.26, 0.12), lag=1.0):
        self.move_lims = tuple(move_lims)
        self.pos = np.asarray(start_xy, dtype=np.float64)
        self.vel = np.zeros(2, dtype=np.float64)
        self.lag = float(lag)

    def apply(self, action):
        delta = np.asarray(action, dtype=np.float64) * np.asarray(self.move_lims)
        step = delta * self.lag
        self.vel = step / 0.05
        self.pos = self.pos + step

    def get_current_state(self):
        return {
            "paddles": {
                "paddle_ego": {
                    "position": self.pos.copy(),
                    "velocity": self.vel.copy(),
                }
            },
            "pucks": [{"position": np.array([-2.0, 0.0]), "occluded": np.array([1])}],
        }


class _FakeEnv:
    """Workspace: x in [-1.0, -0.5], y in [-0.3, 0.3] (env frame)."""

    def __init__(self, start_xy=(-0.9, 0.2), random_paddle_spawn=True, **sim_kwargs):
        self.simulator = _FakeSimulator(start_xy, **sim_kwargs)
        self.random_paddle_spawn = random_paddle_spawn

    def get_paddle_workspace_bounds(self, margin=0.0, y=None):
        return -1.0 + margin, -0.5 - margin, -0.3 + margin, 0.3 - margin

    def get_paddle_configuration(self, name):
        return (-0.75, 0.0), (0.0, 0.0)

    def step(self, action):
        self.simulator.apply(action)


def _drive(fsm, env, max_steps=500):
    steps = 0
    while not fsm.done and steps < max_steps:
        action = fsm.step(env.simulator.get_current_state())
        env.step(action)
        steps += 1
    return steps


class PaddleRepositionFSMTests(unittest.TestCase):
    def test_reaches_random_target_and_settles(self):
        env = _FakeEnv()
        rng = np.random.default_rng(0)
        fsm = PaddleRepositionFSM(env, rng, settle_steps=3)
        # Target lies inside the (margin-inset) workspace.
        x, y = float(fsm.target_xy[0]), float(fsm.target_xy[1])
        self.assertGreaterEqual(x, -1.0 + fsm.spawn_margin_m)
        self.assertLessEqual(x, -0.5 - fsm.spawn_margin_m)
        self.assertGreaterEqual(y, -0.3 + fsm.spawn_margin_m)
        self.assertLessEqual(y, 0.3 - fsm.spawn_margin_m)

        _drive(fsm, env)
        self.assertTrue(fsm.done)
        self.assertEqual(fsm.done_reason, "success")
        self.assertEqual(fsm.phase, "settle")
        dist = float(np.linalg.norm(env.simulator.pos - fsm.target_xy))
        self.assertLess(dist, fsm.arrive_m)
        # The last few actions were zeros (settling) so the fake env's
        # velocity is zero.
        self.assertLess(float(np.linalg.norm(env.simulator.vel)), 1e-9)

    def test_per_step_displacement_is_capped(self):
        env = _FakeEnv(start_xy=(-1.0, -0.3))
        fsm = PaddleRepositionFSM(
            env, np.random.default_rng(1), target_xy=(-0.5, 0.3), max_step_m=0.05
        )
        state = env.simulator.get_current_state()
        action = fsm.step(state)
        delta = np.asarray(action) * np.asarray(env.simulator.move_lims)
        self.assertAlmostEqual(float(np.linalg.norm(delta)), 0.05, places=6)

    def test_fixed_spawn_uses_task_default_when_random_spawn_off(self):
        env = _FakeEnv(random_paddle_spawn=False)
        fsm = PaddleRepositionFSM(env, np.random.default_rng(2))
        np.testing.assert_allclose(fsm.target_xy, [-0.75, 0.0])

    def test_reset_rng_drives_target_sampling(self):
        env = _FakeEnv()
        a = PaddleRepositionFSM(env, np.random.default_rng(7)).target_xy
        b = PaddleRepositionFSM(env, np.random.default_rng(7)).target_xy
        c = PaddleRepositionFSM(env, np.random.default_rng(8)).target_xy
        np.testing.assert_allclose(a, b)
        self.assertFalse(np.allclose(a, c))

    def test_timeout_requests_hard_reset(self):
        # lag=0 → the paddle never moves.
        env = _FakeEnv(lag=0.0)
        fsm = PaddleRepositionFSM(
            env, np.random.default_rng(3), target_xy=(-0.6, 0.0), max_total_steps=20
        )
        steps = _drive(fsm, env)
        self.assertTrue(fsm.done)
        self.assertEqual(fsm.done_reason, "hard_reset_required")
        self.assertEqual(steps, 21)
        np.testing.assert_allclose(fsm.step(env.simulator.get_current_state()), [0.0, 0.0])

    def test_settle_waits_for_low_speed(self):
        env = _FakeEnv(start_xy=(-0.7, 0.0))
        fsm = PaddleRepositionFSM(
            env,
            np.random.default_rng(4),
            target_xy=(-0.7, 0.0),
            settle_steps=2,
            settle_speed_mps=0.01,
        )
        # Already at the target but reported moving fast: no finish yet.
        env.simulator.vel = np.array([1.0, 0.0])
        fsm.step(env.simulator.get_current_state())  # goto_start → settle
        fsm.step(env.simulator.get_current_state())  # settle 1
        fsm.step(env.simulator.get_current_state())  # settle 2, speed too high
        self.assertFalse(fsm.done)
        env.simulator.vel = np.zeros(2)
        fsm.step(env.simulator.get_current_state())
        self.assertTrue(fsm.done)
        self.assertEqual(fsm.done_reason, "success")

    def test_drift_during_settle_returns_to_goto(self):
        env = _FakeEnv(start_xy=(-0.7, 0.0))
        fsm = PaddleRepositionFSM(
            env, np.random.default_rng(5), target_xy=(-0.7, 0.0), arrive_m=0.02
        )
        fsm.step(env.simulator.get_current_state())
        self.assertEqual(fsm.phase, "settle")
        env.simulator.pos = np.array([-0.7 + 0.1, 0.0])
        action = fsm.step(env.simulator.get_current_state())
        self.assertEqual(fsm.phase, "goto_start")
        self.assertLess(float(action[0]), 0.0)  # heading back toward -x

    def test_duck_types_reset_policy_fsm_surface(self):
        env = _FakeEnv()
        fsm = PaddleRepositionFSM(env, np.random.default_rng(6))
        for attr in ("step", "done", "done_reason", "phase", "total_steps", "close", "start_side"):
            self.assertTrue(hasattr(fsm, attr), attr)
        fsm.close()


if __name__ == "__main__":
    unittest.main()

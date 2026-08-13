import math
import unittest
from pathlib import Path

import numpy as np
import yaml

from airhockey import AirHockeyEnv


class MultipuckStaggeredSpawnTests(unittest.TestCase):
    """Multi-puck resets place pucks on one juggle cycle, evenly spaced in arrival time."""

    def _make_env(self, num_pucks, **overrides):
        config_path = (
            Path(__file__).resolve().parents[3]
            / "configs"
            / "new_juggle"
            / "sysid_best_params_hist2.yaml"
        )
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)["air_hockey"]

        cfg["task"] = "puck_juggle_upper_half_reward"
        cfg["seed"] = 123
        cfg["num_pucks"] = num_pucks
        cfg["obs_type"] = "multipuck_history" if num_pucks > 1 else "history"
        # Keep the spawn/flight deterministic so arrival times are comparable.
        cfg.setdefault("simulator_params", {})["enable_random_occlusions"] = False
        cfg["simulator_params"]["puck_noise"] = False
        cfg.update(overrides)
        return AirHockeyEnv(cfg)

    @staticmethod
    def _true_states(env):
        """Spawned (position, velocity) in base coords, read from the Box2D bodies.

        ``env.current_state`` reports the *observed* puck position (delay
        interpolation / noise / occlusion), which is not the spawn state.
        """
        pucks = []
        for body in env.simulator.pucks.values():
            pucks.append(
                (
                    (-body.position[1], body.position[0]),
                    (-body.linearVelocity[1], body.linearVelocity[0]),
                )
            )
        paddle = env.simulator.paddles["paddle_ego"]
        return pucks, (-paddle.position[1], paddle.position[0])

    def _predicted_arrival_times(self, env):
        """Time for each spawned puck to fall back to the reach line, per the spawn model."""
        accel, damping = env._multipuck_fall_dynamics()
        reach_x = env._multipuck_reach_x()
        arrivals = []
        for (x0, _), (v0, _) in self._true_states(env)[0]:
            t = 0.0
            dt = 1e-4
            x, v = x0, v0
            while t < 60.0:
                v += (accel - damping * v) * dt
                x += v * dt
                t += dt
                if x >= reach_x and v > 0:
                    break
            arrivals.append(t)
        return arrivals

    def test_arrival_times_are_evenly_spaced(self):
        for num_pucks in (2, 3, 4, 5):
            env = self._make_env(num_pucks)
            try:
                for seed in range(5):
                    env.reset(seed=seed)
                    arrivals = self._predicted_arrival_times(env)
                    # puck_i arrives before puck_{i+1}
                    self.assertEqual(arrivals, sorted(arrivals))
                    gaps = np.diff(arrivals)
                    slot = arrivals[0]
                    self.assertGreater(slot, 0.0)
                    for gap in gaps:
                        self.assertAlmostEqual(gap, slot, delta=0.05 * slot)
            finally:
                env.close()

    def test_two_pucks_are_one_rising_one_falling(self):
        env = self._make_env(2)
        try:
            for seed in range(5):
                env.reset(seed=seed)
                pucks, _ = self._true_states(env)
                self.assertGreater(pucks[0][1][0], 0.0)  # soonest arrival is falling
                self.assertLess(pucks[1][1][0], 0.0)  # the other was just launched upward
        finally:
            env.close()

    def test_three_pucks_have_two_rising(self):
        env = self._make_env(3)
        try:
            for seed in range(5):
                env.reset(seed=seed)
                pucks, _ = self._true_states(env)
                rising = [p for p in pucks if p[1][0] < 0]
                self.assertEqual(len(rising), 2)
                # the one launched later is lower on the table (larger x)
                self.assertLess(rising[0][0][0], rising[1][0][0])
        finally:
            env.close()

    def test_spawns_stay_in_bounds_and_do_not_overlap(self):
        for num_pucks in (2, 4):
            env = self._make_env(num_pucks)
            try:
                for seed in range(10):
                    env.reset(seed=seed)
                    pucks, paddle = self._true_states(env)
                    bodies = [p[0] for p in pucks]
                    for x, y in bodies:
                        self.assertGreaterEqual(x, env.table_x_top + env.puck_radius - 1e-9)
                        self.assertLessEqual(x, env.table_x_bot - env.puck_radius + 1e-9)
                        self.assertGreaterEqual(y, env.table_y_left + env.puck_radius - 1e-9)
                        self.assertLessEqual(y, env.table_y_right - env.puck_radius + 1e-9)
                        self.assertGreaterEqual(
                            math.hypot(x - paddle[0], y - paddle[1]),
                            env.puck_radius + env.paddle_radius,
                        )
                    for i in range(len(bodies)):
                        for j in range(i + 1, len(bodies)):
                            distance = math.hypot(
                                bodies[i][0] - bodies[j][0], bodies[i][1] - bodies[j][1]
                            )
                            self.assertGreaterEqual(distance, 2.0 * env.puck_radius)
            finally:
                env.close()

    def test_stagger_can_be_disabled(self):
        env = self._make_env(2, multipuck_stagger=False)
        try:
            env.reset(seed=0)
            vx = [p[1][0] for p in self._true_states(env)[0]]
            # legacy path samples both pucks from the same independent distribution
            self.assertTrue(all(abs(v) <= 0.5 + 1e-9 for v in vx))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()

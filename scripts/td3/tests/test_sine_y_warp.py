"""Math + env-wiring tests for the puck-only sine y-warp.

Replaces the deleted ``test_observation_homography.py`` (the underlying
3x3 homography mechanism was removed 2026-05-07; see
``notes/scratch/experiments/2026-05-07_02-05_sim2sim-puck-obs-warp.md``).
"""

import unittest

import numpy as np

from airhockey.observation_homography import (
    apply_sine_y_warp_xy,
    make_sine_y_warp_fn,
)


class SineYWarpTests(unittest.TestCase):
    Y_LEFT = -0.4318
    Y_RIGHT = +0.4318
    A = 0.05

    def test_identity_at_left_edge(self):
        x_out, y_out = apply_sine_y_warp_xy(0.1, self.Y_LEFT, self.A, self.Y_LEFT, self.Y_RIGHT)
        self.assertEqual(x_out, 0.1)
        self.assertAlmostEqual(y_out, self.Y_LEFT, places=12)

    def test_identity_at_right_edge(self):
        x_out, y_out = apply_sine_y_warp_xy(0.1, self.Y_RIGHT, self.A, self.Y_LEFT, self.Y_RIGHT)
        self.assertEqual(x_out, 0.1)
        self.assertAlmostEqual(y_out, self.Y_RIGHT, places=12)

    def test_peak_at_midpoint(self):
        x_out, y_out = apply_sine_y_warp_xy(0.1, 0.0, self.A, self.Y_LEFT, self.Y_RIGHT)
        self.assertEqual(x_out, 0.1)
        self.assertAlmostEqual(y_out, self.A, places=12)

    def test_amplitude_zero_returns_input_unchanged(self):
        x_out, y_out = apply_sine_y_warp_xy(0.1, 0.0, 0.0, self.Y_LEFT, self.Y_RIGHT)
        self.assertEqual((x_out, y_out), (0.1, 0.0))

    def test_factory_returns_none_when_disabled(self):
        self.assertIsNone(make_sine_y_warp_fn(0.0, self.Y_LEFT, self.Y_RIGHT))

    def test_factory_returns_callable_when_enabled(self):
        fn = make_sine_y_warp_fn(self.A, self.Y_LEFT, self.Y_RIGHT)
        self.assertTrue(callable(fn))
        x, y = fn(0.1, 0.0)
        self.assertAlmostEqual(y, self.A, places=12)

    def test_monotonicity_guard_fires_at_bound(self):
        bound = (self.Y_RIGHT - self.Y_LEFT) / np.pi
        with self.assertRaises(ValueError):
            make_sine_y_warp_fn(bound, self.Y_LEFT, self.Y_RIGHT)

    def test_monotonicity_holds_below_bound(self):
        # Stretch factor stays positive for every y in [y_left, y_right].
        ys = np.linspace(self.Y_LEFT, self.Y_RIGHT, 1000)
        warped = np.array([
            apply_sine_y_warp_xy(0.0, float(y), self.A, self.Y_LEFT, self.Y_RIGHT)[1]
            for y in ys
        ])
        self.assertTrue(np.all(np.diff(warped) > 0))


if __name__ == "__main__":
    unittest.main()

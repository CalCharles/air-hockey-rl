"""Behavioral checks for the optional policy-side observer; no simulator edits."""
import unittest

import numpy as np

from scripts.analysis.puck_state_filter import PuckStateFilter


def observation(x, y=.1, occluded=False):
    obs = np.zeros(30, dtype=np.float64)
    obs[27:30] = [x, y, float(occluded)]
    return obs


class PuckFilterTests(unittest.TestCase):
    def test_latency_compensation_and_occlusion_prediction(self):
        observer = PuckStateFilter(gravity=0., damping=0., walls=False)
        observer.reset(observation(-.3 - .2 * .025))
        for step in range(1, 40):
            expected_x = -.3 + .2 * step * .05
            hidden = 25 <= step < 30
            obs = observation(999. if hidden else expected_x - .2 * .025,
                              -999. if hidden else .1, hidden)
            estimate = observer.update(obs)
            if step >= 15:
                np.testing.assert_allclose(estimate, [expected_x, .1, .2, 0.], atol=.001)

    def test_observer_preserves_observation_and_numpy_rng(self):
        np.random.seed(42)
        before = np.random.get_state()
        obs = observation(-.2)
        copy = obs.copy()
        observer = PuckStateFilter()
        observer.reset(obs)
        observer.update(obs)
        np.testing.assert_array_equal(obs, copy)
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_wall_prediction_remains_finite_during_occlusion(self):
        observer = PuckStateFilter(gravity=0., damping=0., latency=0.)
        observer.reset(observation(.8))
        for step in range(1, 5):
            observer.update(observation(.8 + .025 * step))
        for _ in range(100):
            estimate = observer.update(observation(999., -999., True))
            self.assertTrue(np.isfinite(estimate).all())
            self.assertLessEqual(abs(estimate[0]), 1.9304 / 2 - .03175 + 1e-6)
            self.assertGreaterEqual(np.linalg.eigvalsh(observer.covariance).min(), -1e-9)


if __name__ == "__main__":
    unittest.main()

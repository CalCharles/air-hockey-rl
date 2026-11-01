"""Sampled kinematic diagnostics: analytic trajectories and short recordings."""
import unittest
import numpy as np
from scripts.analysis.juggle_motion_metrics import sampled_motion


class MotionTests(unittest.TestCase):
    def test_constant_velocity_has_no_acceleration_or_jerk(self):
        time = np.arange(20)*.05
        metrics = sampled_motion(np.column_stack((2*time, -time)), .05)
        self.assertAlmostEqual(metrics['speed_m_s']['rms'], np.sqrt(5))
        self.assertLess(metrics['acceleration_m_s2']['maximum'], 1e-10)
        self.assertLess(metrics['jerk_m_s3']['maximum'], 1e-8)

    def test_constant_acceleration_is_measured_in_physical_units(self):
        time = np.arange(20)*.05
        metrics = sampled_motion(np.column_stack((1.5*time**2, 2*time**2)), .05)
        self.assertAlmostEqual(metrics['acceleration_m_s2']['p95'], 5.)
        self.assertLess(metrics['jerk_m_s3']['maximum'], 1e-8)

    def test_short_episode_has_missing_jerk_rather_than_zero(self):
        metrics = sampled_motion(np.zeros((2, 2)), .05)
        self.assertIsNone(metrics['jerk_m_s3']['rms'])
        self.assertEqual(metrics['jerk_m_s3']['samples'], 0)
        with self.assertRaises(ValueError):
            sampled_motion(np.zeros((4, 2)), 0)
        with self.assertRaises(ValueError):
            sampled_motion(np.full((4, 2), np.nan), .05)


if __name__ == '__main__':
    unittest.main()

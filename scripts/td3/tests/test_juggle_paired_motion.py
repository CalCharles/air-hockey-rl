"""Equal-duration comparisons must not hide late spikes or mismatched trials."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.analysis.juggle_paired_motion import compare


class PairedMotionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write(self, name, positions, seed=7, rate=20):
        root = self.root/name
        root.mkdir()
        np.savez(root/'episode.npz', true_paddle_xy=positions)
        data = dict(horizon=100, scenarios={'training': dict(
            config={'simulator_params': {'step_frequency': rate}},
            episodes=[dict(seed=seed, steps=len(positions), trajectory='episode.npz',
                           true_state_juggles=1, juggles=1)])})
        path = root/'results.json'
        path.write_text(json.dumps(data))
        return path

    def test_equal_prefix_excludes_unpaired_late_spike_and_preserves_analytic_units(self):
        time = np.arange(20)*.05
        # Cubic motion has constant jerk 6 m/s³; candidate scales it by half.
        baseline = np.column_stack((time**3, np.zeros(20)))
        candidate = baseline[:10]*.5
        baseline[10:, 0] += 100  # outside the shared ten-position prefix
        result = compare(self.write('baseline', baseline), self.write('candidate', candidate))
        self.assertEqual(result['episodes'][0]['paired_steps'], 10)
        metric = result['summary']['training']['metrics']['jerk_m_s3']
        self.assertAlmostEqual(metric['baseline_median_episode_p95'], 6.)
        self.assertAlmostEqual(metric['candidate_median_episode_p95'], 3.)
        self.assertAlmostEqual(metric['median_paired_ratio'], .5)

    def test_refuses_different_seeds_or_control_rates(self):
        xy = np.zeros((5, 2))
        baseline = self.write('baseline', xy)
        for candidate in [self.write('different_seed', xy, seed=8),
                          self.write('different_rate', xy, rate=25)]:
            with self.assertRaises(ValueError):
                compare(baseline, candidate)

    def test_missing_derivatives_and_zero_baseline_do_not_become_ratio_zero(self):
        result = compare(self.write('baseline', np.zeros((3, 2))),
                         self.write('candidate', np.zeros((3, 2))))
        metrics = result['summary']['training']['metrics']
        self.assertEqual(metrics['jerk_m_s3']['paired_episodes'], 0)
        self.assertIsNone(metrics['jerk_m_s3']['median_paired_ratio'])
        self.assertEqual(metrics['speed_m_s']['ratio_episodes'], 0)
        self.assertIsNone(metrics['speed_m_s']['median_paired_ratio'])


if __name__ == '__main__':
    unittest.main()

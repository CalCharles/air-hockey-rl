"""Only complete saved episodes may be reused after an interrupted evaluator."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scripts.analysis.juggle_long_reference_continue import trajectory_metrics


class SavedEpisodeTests(unittest.TestCase):
    def trajectory(self, folder, terminal=False):
        path=Path(folder)/'episode.npz'
        flags=np.zeros(12,dtype=bool);flags[-1]=terminal
        np.savez_compressed(path,action=np.zeros((12,2)),reward=np.ones(12),terminated=flags,
            truncated=np.zeros(12,dtype=bool),observation=np.zeros((12,30)),next_observation=np.zeros((12,30)),
            true_paddle_xy=np.zeros((12,2)),true_puck_xy=np.zeros((12,2)))
        return path

    def test_recording_limit_and_terminal_episodes_can_be_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            row=trajectory_metrics(self.trajectory(folder),7,12)
            self.assertEqual(row['steps'],12)
            self.assertEqual(row['reward'],12)
            self.assertEqual(row['end_reason'],'recording limit')
            row=trajectory_metrics(self.trajectory(folder,terminal=True),7,30)
            self.assertTrue(row['terminated'])
            self.assertNotEqual(row['end_reason'],'recording limit')

    def test_incomplete_saved_prefix_cannot_be_counted_as_completed(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError,'Incomplete episode'):
                trajectory_metrics(self.trajectory(folder),7,30)

    def test_corrupt_artifact_is_not_a_reusable_episode(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'episode.npz';path.write_bytes(b'incomplete')
            with self.assertRaises(ValueError):
                trajectory_metrics(path,7,30)


if __name__=='__main__':
    unittest.main()

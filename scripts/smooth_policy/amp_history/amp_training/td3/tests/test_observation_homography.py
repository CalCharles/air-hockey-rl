import unittest

import numpy as np

from airhockey.observation_homography import (
    apply_plane_homography_xy,
    sample_near_identity_homography,
)


class ObservationHomographyTests(unittest.TestCase):
    def test_identity_homography_preserves_point(self):
        homography = np.eye(3, dtype=np.float64)
        x_out, y_out = apply_plane_homography_xy(0.37, -0.11, homography)
        self.assertAlmostEqual(x_out, 0.37, places=7)
        self.assertAlmostEqual(y_out, -0.11, places=7)

    def test_sampled_homography_is_finite_and_normalized(self):
        rng = np.random.default_rng(123)
        homography = sample_near_identity_homography(rng)
        self.assertEqual(homography.shape, (3, 3))
        self.assertTrue(np.isfinite(homography).all())
        self.assertAlmostEqual(float(homography[2, 2]), 1.0, places=12)

    def test_single_point_round_trip_with_inverse(self):
        rng = np.random.default_rng(7)
        homography = sample_near_identity_homography(rng)
        inv_homography = np.linalg.inv(homography)

        x1, y1 = apply_plane_homography_xy(-0.42, 0.23, homography)
        x2, y2 = apply_plane_homography_xy(x1, y1, inv_homography)
        self.assertAlmostEqual(x2, -0.42, places=5)
        self.assertAlmostEqual(y2, 0.23, places=5)


if __name__ == "__main__":
    unittest.main()

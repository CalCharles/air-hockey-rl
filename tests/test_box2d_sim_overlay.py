"""Homography correspondence for the faint Box2D table overlay on the real camera."""
from __future__ import annotations

import unittest

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from airhockey.sims.real.overlay_utils import (
    Box2DEnvironmentOverlay,
    blend_masked_overlay,
    box2d_table_src_dst_points,
    robot_to_display_pixel,
    warp_box2d_environment_to_display,
)


def _require_cv2():
    if cv2 is None:
        raise unittest.SkipTest("opencv-python not installed")


class TestBox2DSimOverlay(unittest.TestCase):
    def test_src_corners_map_to_robot_display_pixels(self):
        table_length = 2.0
        table_width = 1.0
        center_offset = 1.0
        offset_constants = (2000.0, 500.0)
        downscale = 2.0
        src_hw = (40, 80)

        _, dst = box2d_table_src_dst_points(
            src_hw,
            table_length=table_length,
            table_width=table_width,
            center_offset=center_offset,
            offset_constants=offset_constants,
            visual_downscale_constant=downscale,
        )

        half_l = table_length / 2.0
        half_w = table_width / 2.0
        expected = np.array(
            [
                robot_to_display_pixel(
                    tx - center_offset,
                    ty,
                    offset_constants=offset_constants,
                    visual_downscale_constant=downscale,
                )
                for tx, ty in (
                    (half_l, -half_w),
                    (-half_l, -half_w),
                    (-half_l, half_w),
                    (half_l, half_w),
                )
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(dst, expected, atol=1e-5)

    def test_warp_lands_table_center_on_homography_display(self):
        _require_cv2()
        table_length = 2.0
        table_width = 1.0
        center_offset = 1.0
        offset_constants = (2000.0, 500.0)
        downscale = 2.0

        src_h, src_w = 40, 80
        table = np.zeros((src_h, src_w, 3), dtype=np.uint8)
        table[:, :] = (0, 0, 255)
        table[src_h // 2 - 2:src_h // 2 + 3, src_w // 2 - 2:src_w // 2 + 3] = (0, 255, 0)

        dst_hw = (600, 1200)
        warped, mask = warp_box2d_environment_to_display(
            table,
            dst_hw,
            table_length=table_length,
            table_width=table_width,
            center_offset=center_offset,
            offset_constants=offset_constants,
            visual_downscale_constant=downscale,
        )
        self.assertEqual(warped.shape[:2], dst_hw)
        self.assertEqual(mask.shape[:2], dst_hw)

        cx, cy = robot_to_display_pixel(
            0.0 - center_offset,
            0.0,
            offset_constants=offset_constants,
            visual_downscale_constant=downscale,
        )
        px, py = int(round(cx)), int(round(cy))
        self.assertGreater(int(mask[py, px]), 0)
        self.assertGreater(int(warped[py, px, 1]), 100)

        # Outside the table quad the warped mask should stay empty.
        self.assertEqual(int(mask[590, 50]), 0)

    def test_blend_is_faint_and_masked(self):
        _require_cv2()
        frame = np.full((20, 30, 3), 200, dtype=np.uint8)
        warped = np.zeros_like(frame)
        warped[5:15, 8:22] = (0, 0, 255)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        mask[5:15, 8:22] = 255

        blend_masked_overlay(frame, warped, mask, alpha=0.25)
        inside = frame[10, 15]
        outside = frame[1, 1]
        np.testing.assert_array_equal(outside, np.array([200, 200, 200]))
        self.assertLess(int(inside[0]), 200)
        self.assertGreater(int(inside[2]), 200)

    def test_overlay_class_from_config_and_apply(self):
        _require_cv2()
        table = np.full((20, 40, 3), (255, 0, 0), dtype=np.uint8)
        overlay = Box2DEnvironmentOverlay(
            alpha=0.5,
            table_length=2.0,
            table_width=1.0,
            center_offset=1.0,
            offset_constants=(2000.0, 500.0),
            visual_downscale_constant=2.0,
            table_image=table,
        )

        frame = np.zeros((600, 1200, 3), dtype=np.uint8)
        overlay.apply(frame)
        cx, cy = robot_to_display_pixel(
            -1.0,
            0.0,
            offset_constants=(2000.0, 500.0),
            visual_downscale_constant=2.0,
        )
        px, py = int(round(cx)), int(round(cy))
        self.assertGreater(int(frame[py, px, 0]), 50)
        self.assertEqual(int(frame[590, 50, 0]), 0)

    def test_disabled_config_returns_none(self):
        self.assertIsNone(Box2DEnvironmentOverlay.from_config(None))
        self.assertIsNone(Box2DEnvironmentOverlay.from_config({"alpha": 0.0}))
        self.assertIsNone(Box2DEnvironmentOverlay.from_config(0.0))


if __name__ == "__main__":
    unittest.main()

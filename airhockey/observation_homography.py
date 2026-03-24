import cv2
import numpy as np


def apply_plane_homography_xy(x, y, homography):
    """Apply a 3x3 homography to a single (x, y) point."""
    if homography is None:
        return float(x), float(y)
    points = np.array([[[float(x), float(y)]]], dtype=np.float32)
    warped = cv2.perspectiveTransform(points, np.asarray(homography, dtype=np.float64))
    return float(warped[0, 0, 0]), float(warped[0, 0, 1])


def sample_near_identity_homography(
    rng,
    affine_std=0.08,
    translation_std=0.01,
    perspective_std=0.04,
):
    """
    Sample a mild world-plane homography close to identity.

    Coordinates in this repo are meters with magnitudes around O(1), so
    these defaults produce small, systematic observation warps.
    """
    homography = np.eye(3, dtype=np.float64)
    homography[0, 0] += float(rng.normal(0.0, affine_std))
    homography[1, 1] += float(rng.normal(0.0, affine_std))
    homography[0, 1] += float(rng.normal(0.0, affine_std))
    homography[1, 0] += float(rng.normal(0.0, affine_std))
    homography[0, 2] += float(rng.normal(0.0, translation_std))
    homography[1, 2] += float(rng.normal(0.0, translation_std))
    homography[2, 0] += float(rng.normal(0.0, perspective_std))
    homography[2, 1] += float(rng.normal(0.0, perspective_std))
    if abs(homography[2, 2]) < 1e-8:
        homography[2, 2] = 1.0
    homography /= homography[2, 2]
    return homography


def pixel_homography_from_world_homography(homography_world, env, renderer):
    """Build output-frame pixel homography from world-frame homography."""
    world_corners = [
        (float(env.table_x_top), float(env.table_y_left)),
        (float(env.table_x_top), float(env.table_y_right)),
        (float(env.table_x_bot), float(env.table_y_right)),
        (float(env.table_x_bot), float(env.table_y_left)),
    ]
    warped_world_corners = [
        apply_plane_homography_xy(x, y, homography_world) for x, y in world_corners
    ]
    src_pixels = np.array(
        [renderer.world_xy_to_output_pixel(x, y) for x, y in world_corners],
        dtype=np.float32,
    )
    dst_pixels = np.array(
        [renderer.world_xy_to_output_pixel(x, y) for x, y in warped_world_corners],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(src_pixels, dst_pixels)

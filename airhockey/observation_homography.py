"""Observation warps applied to the puck position before it reaches the policy.

The active mechanism (2026-05-07 onward) is the edge-preserving sine y-warp
defined below. It models a partially-calibrated overhead puck tracker: the
table corners are anchored to ground truth, but interior reads bow off-true
along the lateral (y) axis. The warp is plumbed through
``airhockey/utils.py:get_observation_by_type`` via the ``puck_obs_warp_fn``
kwarg and is configured per-environment by three keys in
``simulator_params`` — see ``airhockey/sims/airhockey_box2d.py`` and
``notes/docs/environments/box2d/simulator-essentials.md``.

The older ``obs_position_homography`` mechanism (3x3 perspective matrix
applied to BOTH paddle and puck) was removed in favor of this targeted
puck-only path. See the experiment writeup at
``notes/scratch/experiments/2026-05-07_02-05_sim2sim-puck-obs-warp.md``.
"""

import numpy as np


def apply_sine_y_warp_xy(x, y, amplitude, y_left, y_right):
    """Edge-preserving sine warp on the y-coordinate.

        y_warp = y + amplitude * sin(pi * (y - y_left) / (y_right - y_left))

    Returns (x, y_warp); x is unchanged. The warp is the identity at the
    edges (y == y_left or y == y_right) and reaches +amplitude at the
    midpoint. Monotonic iff |amplitude| < (y_right - y_left) / pi.
    """
    if amplitude == 0.0:
        return float(x), float(y)
    width = y_right - y_left
    if width <= 0:
        return float(x), float(y)
    y_warp = y + amplitude * np.sin(np.pi * (y - y_left) / width)
    return float(x), float(y_warp)


def make_sine_y_warp_fn(amplitude, y_left, y_right):
    """Build a (x, y) -> (x, y_warped) callable for the sine y-warp.

    Returns None when amplitude == 0 so callers can branch on ``is None``
    to skip the warp entirely. Raises if the amplitude would break
    monotonicity, since a non-monotonic puck observation would be a
    silent footgun for any downstream policy.
    """
    if amplitude == 0.0:
        return None
    width = y_right - y_left
    if width <= 0:
        raise ValueError(
            f"sine_y_warp needs y_right > y_left, got [{y_left}, {y_right}]"
        )
    bound = width / np.pi
    if abs(amplitude) >= bound:
        raise ValueError(
            f"sine_y_warp amplitude={amplitude} breaks monotonicity "
            f"(must be < (y_right - y_left) / pi = {bound:.4f})"
        )

    def _warp(x, y):
        return apply_sine_y_warp_xy(x, y, amplitude, y_left, y_right)

    return _warp

"""
Velocity estimation from noisy position trajectories.

Fits a gravity-corrected linear model over a window of position observations
to estimate puck velocity at each timestep and at the collision moment.

Gravity correction matters even at g≈0.65: over a 0.5s window it introduces a
0.175 m/s systematic bias (= g*T/2) in the endpoint velocity if ignored.

Sign convention for `gravity` parameter:
  gravity=(gx, gy) encodes DECELERATION along each axis.
  Positive gx means vx DECREASES over time: pos(t) = pos0 + v0*t - 0.5*g*t², v(t) = v0 - g*t.
  To accelerate along an axis (velocity increasing), pass a NEGATIVE value.

IMPORTANT — Box2D in base coordinates:
  Box2D world gravity = (0, -0.65) in Box2D coords.
  Converting to base (x_base = -y_b2d, y_base = x_b2d):
    → puck ACCELERATES in +x_base direction (a_x = +0.65, so deceleration gx = -0.65).
  Pass gravity=(-0.65, 0.0) for this sim.
  For a vertically-oriented system where gravity decelerates upward motion, pass (0.0, g) with g > 0.

Typical usage:
    result = extract_pre_collision_velocities(
        positions, times, valid_mask, collision_idx,
        window_frames=10, gravity=(0.65, 0.0)
    )
    v_at_collision = result["v_at_end"]   # shape (2,) — [vx, vy] in base coords
"""

import numpy as np


def _weighted_lstsq(A, b, weights):
    """Weighted least squares: min sum_i w_i * (A[i] @ x - b[i])^2."""
    W = np.sqrt(weights)
    return np.linalg.lstsq(A * W[:, None], b * W, rcond=None)


def fit_velocity_from_positions(positions, times, valid_mask=None, gravity=(0.0, 0.0)):
    """
    Estimate puck velocity at each timestep and at the final timestep (collision moment)
    using gravity-corrected linear regression.

    Parameters
    ----------
    positions : (N, 2) array — observed (x, y) positions in metres
    times     : (N,)   array — timestamps in seconds (need not start at 0)
    valid_mask: (N,)   bool  — True for non-occluded frames; None means all valid
    gravity   : (gx, gy)    — deceleration in m/s² along each axis (positive = decelerates).
                               Model: pos(t) = pos0 + v0*t - 0.5*g*t², v(t) = v0 - g*t.
                               Examples:
                                 (-0.65, 0.0) — Box2D sim (puck accelerates in +x, so gx=-0.65)
                                 (0.0, -9.8)  — upward-positive y with real gravity below
                                 (0.0,  9.8)  — juggling where gravity decelerates upward motion
                                 (0.0,  0.0)  — flat real-world table (no in-plane gravity)

    Returns
    -------
    dict with keys:
        "v_at_end"   : (2,) array  — [vx, vy] at times[-1]
        "v_at_times" : (N, 2) array — [vx, vy] at each input timestep
        "snr"        : float        — |v_at_end| / residual_noise; higher = more trustworthy
        "n_valid"    : int          — number of valid frames used
    Returns None if fewer than 2 valid frames are available.
    """
    positions = np.asarray(positions, dtype=float)
    times = np.asarray(times, dtype=float)
    gx, gy = float(gravity[0]), float(gravity[1])
    N = len(times)

    if valid_mask is None:
        valid_mask = np.ones(N, dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)

    n_valid = int(valid_mask.sum())
    if n_valid < 2:
        return None

    weights = valid_mask.astype(float)
    t0 = times[0]
    dt = times - t0  # relative time, starts at 0

    # Design matrix for linear fit: [1, t]
    A = np.column_stack([np.ones(N), dt])

    v_at_times = np.zeros((N, 2))
    residuals = []

    for axis, (pos_col, g_axis) in enumerate([(positions[:, 0], gx), (positions[:, 1], gy)]):
        # pos(t) = pos0 + v0*(t-t0) - 0.5*g*(t-t0)^2
        # => pos(t) + 0.5*g*(t-t0)^2 = pos0 + v0*(t-t0)   [linear in [1, dt]]
        pos_corr = pos_col + 0.5 * g_axis * dt ** 2
        coeffs, res, _, _ = _weighted_lstsq(A, pos_corr, weights)
        # coeffs = [pos0, v_at_t0]
        v0 = coeffs[1]
        # v(t) = v0 - g*(t-t0)
        v_at_times[:, axis] = v0 - g_axis * dt
        residuals.append(np.sqrt(float(res[0]) / n_valid) if len(res) > 0 else 1e-6)

    v_at_end = v_at_times[-1]

    # SNR: ratio of signal magnitude to residual noise per observation
    noise_std = np.sqrt(residuals[0] ** 2 + residuals[1] ** 2)
    speed = float(np.linalg.norm(v_at_end))
    snr = speed / noise_std if noise_std > 1e-9 else float("inf")

    return {
        "v_at_end": v_at_end,
        "v_at_times": v_at_times,
        "snr": snr,
        "n_valid": n_valid,
    }


def extract_pre_collision_velocities(
    positions, times, valid_mask, collision_idx, window_frames=10, gravity=(0.0, 0.0)
):
    """
    Convenience wrapper: slices the window ending at collision_idx and calls
    fit_velocity_from_positions.

    Parameters
    ----------
    positions     : (N, 2) — full trajectory positions
    times         : (N,)   — full trajectory timestamps
    valid_mask    : (N,)   — occlusion flags (True = valid)
    collision_idx : int    — index of the last frame before collision
    window_frames : int    — number of frames to use (default 10 ≈ 0.5s at 20Hz)
    gravity       : (gx, gy) — effective gravity vector in m/s²

    Returns
    -------
    Same dict as fit_velocity_from_positions, or None if not enough valid data.
    """
    start = max(0, collision_idx - window_frames + 1)
    end = collision_idx + 1
    return fit_velocity_from_positions(
        positions[start:end],
        times[start:end],
        valid_mask[start:end],
        gravity=gravity,
    )

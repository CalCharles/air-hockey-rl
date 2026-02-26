"""Estimate puck velocities by fitting polynomial trajectories to free-flight segments.

 Detect contact events (paddle hits, wall hits)        ← puck_inflection.py
 Extract free-flight intervals between contacts
 Split each interval at x-peaks (direction reversals)  ← _split_at_peaks()
 Fit a separate parabola to each monotonic sub-segment
 Enforce vx = 0 exactly at peak timesteps
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from puck_inflection import find_contact_events, find_puck_x_peaks


def _merge_contact_windows(contact_events, gap_tolerance=2):
    """Merge contact events within gap_tolerance timesteps into windows.

    Returns list of (start, end) tuples for each merged contact window.
    """
    if not contact_events:
        return []

    timesteps = sorted(set(e["timestep"] for e in contact_events))
    windows = []
    win_start = timesteps[0]
    win_end = timesteps[0]

    for t in timesteps[1:]:
        if t <= win_end + gap_tolerance:
            win_end = t
        else:
            windows.append((win_start, win_end))
            win_start = t
            win_end = t
    windows.append((win_start, win_end))
    return windows


def segment_free_flights(contact_events, n_timesteps, gap_tolerance=2, min_length=3):
    """Segment a trajectory into free-flight intervals between contact windows.

    Args:
        contact_events: List of contact event dicts from find_contact_events().
        n_timesteps: Total length of the trajectory.
        gap_tolerance: Max gap to merge consecutive contacts into one window.
        min_length: Minimum number of timesteps for a valid segment.

    Returns:
        List of (start, end) tuples (inclusive) for free-flight intervals.
    """
    windows = _merge_contact_windows(contact_events, gap_tolerance)

    if not windows:
        if n_timesteps >= min_length:
            return [(0, n_timesteps - 1)]
        return []

    segments = []

    # Before first contact window
    first_contact_start = windows[0][0]
    if first_contact_start >= min_length:
        segments.append((0, first_contact_start - 1))

    # Between contact windows
    for i in range(len(windows) - 1):
        seg_start = windows[i][1] + 1
        seg_end = windows[i + 1][0] - 1
        if seg_end - seg_start + 1 >= min_length:
            segments.append((seg_start, seg_end))

    # After last contact window
    last_contact_end = windows[-1][1]
    if n_timesteps - 1 - last_contact_end >= min_length:
        segments.append((last_contact_end + 1, n_timesteps - 1))

    return segments


def _split_at_peaks(seg_start, seg_end, peaks, min_length=3):
    """Split a free-flight segment at x-peaks into monotonic sub-segments.

    The peak timestep is included in both adjacent sub-segments so each
    parabola naturally reaches vx~0 at the turning point.

    Args:
        seg_start: First timestep of the free-flight segment (inclusive).
        seg_end: Last timestep of the free-flight segment (inclusive).
        peaks: List of all peak timestep indices from find_puck_x_peaks().
        min_length: Minimum sub-segment length to keep.

    Returns:
        List of (sub_start, sub_end) tuples.
    """
    # Filter to peaks strictly inside the segment (not at boundaries)
    seg_peaks = sorted(p for p in peaks if seg_start < p < seg_end)

    if not seg_peaks:
        return [(seg_start, seg_end)]

    sub_segments = []
    prev = seg_start

    for peak in seg_peaks:
        # Ascending sub-segment: prev to peak (inclusive)
        if peak - prev + 1 >= min_length:
            sub_segments.append((prev, peak))
        prev = peak

    # Descending sub-segment: last peak to seg_end (inclusive)
    if seg_end - prev + 1 >= min_length:
        sub_segments.append((prev, seg_end))

    # If no sub-segment met min_length, fall back to whole segment
    if not sub_segments:
        sub_segments = [(seg_start, seg_end)]

    return sub_segments


def fit_segment(puck_pos, puck_occluded, start, end, dt=0.05,
                x_degree=2, y_degree=2, min_visible_points=None):
    """Fit polynomials to puck positions within a free-flight segment.

    Args:
        puck_pos: Full trajectory puck positions (N, 2) in base coordinates.
        puck_occluded: Boolean mask (N,), True where puck is occluded.
        start: First timestep index (inclusive).
        end: Last timestep index (inclusive).
        dt: Timestep duration in seconds.
        x_degree: Polynomial degree for x(t). Default 2 (parabolic).
        y_degree: Polynomial degree for y(t). Default 2 (quadratic).
        min_visible_points: Minimum visible points required. Defaults to max(degree)+1.

    Returns:
        dict with x_coeffs, y_coeffs, x_residual, y_residual, start, end
        or None if insufficient visible points.
    """
    if min_visible_points is None:
        min_visible_points = max(x_degree, y_degree) + 1

    indices = np.arange(start, end + 1)
    visible = ~puck_occluded[start:end + 1]

    if np.sum(visible) < min_visible_points:
        return None

    t_rel = (indices - start) * dt
    t_vis = t_rel[visible]
    x_vis = puck_pos[indices[visible], 0]
    y_vis = puck_pos[indices[visible], 1]

    x_coeffs = np.polyfit(t_vis, x_vis, x_degree)
    y_coeffs = np.polyfit(t_vis, y_vis, y_degree)

    x_fitted = np.polyval(x_coeffs, t_vis)
    y_fitted = np.polyval(y_coeffs, t_vis)
    x_residual = float(np.sqrt(np.mean((x_vis - x_fitted) ** 2)))
    y_residual = float(np.sqrt(np.mean((y_vis - y_fitted) ** 2)))

    return {
        "x_coeffs": x_coeffs,
        "y_coeffs": y_coeffs,
        "x_residual": x_residual,
        "y_residual": y_residual,
        "start": start,
        "end": end,
    }


def velocity_from_fit(coeffs, t_rel):
    """Compute velocity at relative time t_rel by differentiating the polynomial.

    Args:
        coeffs: Polynomial coefficients (highest degree first), from np.polyfit.
        t_rel: Relative time(s) in seconds (scalar or array).

    Returns:
        Velocity value(s) at t_rel.
    """
    deriv_coeffs = np.polyder(coeffs)
    return np.polyval(deriv_coeffs, t_rel)


def fit_trajectory_velocities(traj, dt=0.05, x_degree=2, y_degree=2,
                               gap_tolerance=2, min_segment_length=3):
    """Fit parabolic velocities for an entire trajectory.

    1. Detects contact events via find_contact_events()
    2. Segments into free-flight intervals
    3. Fits polynomials to each segment
    4. Assembles full (N, 2) velocity array
    5. Falls back to finite differences for unfitted timesteps

    Args:
        traj: Trajectory dict from load_trajectory() with 'puck' and 'paddle' keys.
        dt: Timestep duration.
        x_degree: Polynomial degree for x(t).
        y_degree: Polynomial degree for y(t).
        gap_tolerance: Max gap for merging contact events.
        min_segment_length: Minimum free-flight segment length.

    Returns:
        (N, 2) array of [vx, vy] velocity estimates for every timestep.
    """
    puck_pos = traj["puck"][:, :2]
    puck_occluded = traj["puck"][:, 2] > 0.5
    N = len(puck_pos)

    # Finite-difference fallback
    fd_vel = np.zeros((N, 2))
    if N > 1:
        fd_vel[1:] = (puck_pos[1:] - puck_pos[:-1]) / dt
        fd_vel[0] = fd_vel[1]
    fd_vel[puck_occluded] = 0.0

    # Start with fallback everywhere
    velocities = fd_vel.copy()

    # Detect contacts, peaks, and segment
    contacts = find_contact_events(traj)
    # print(f'Contact Segments: {contacts}')
    segments = segment_free_flights(contacts, N, gap_tolerance, min_segment_length)
    # print("="*60)
    # print(f'Segments: {segments}')
    # print("="*60)
    peaks = find_puck_x_peaks(puck_pos, puck_occluded)
    # print(f'Peaks: {peaks}')
    peak_set = set(peaks)

    # Fit each sub-segment (split at x-peaks) and fill in velocities
    for seg_start, seg_end in segments:
        sub_segs = _split_at_peaks(seg_start, seg_end, peaks, min_segment_length)
        # print(f'Sub-Segments: {sub_segs}')
        for sub_start, sub_end in sub_segs:
            fit = fit_segment(puck_pos, puck_occluded, sub_start, sub_end,
                              dt=dt, x_degree=x_degree, y_degree=y_degree)
            if fit is None:
                continue

            for t in range(sub_start, sub_end + 1):
                if puck_occluded[t]:
                    continue
                t_rel = (t - sub_start) * dt
                vx = velocity_from_fit(fit["x_coeffs"], t_rel)
                vy = velocity_from_fit(fit["y_coeffs"], t_rel)
                velocities[t] = [vx, vy]

        # Enforce vx=0 at peaks within this segment
        for peak in peaks:
            if seg_start <= peak <= seg_end and not puck_occluded[peak]:
                velocities[peak, 0] = 0.0

    return velocities


def get_velocity_at(traj, timestep, dt=0.05, x_degree=2, y_degree=2,
                    gap_tolerance=2, min_segment_length=3):
    """Get the fitted velocity at a specific timestep.

    Finds the free-flight segment containing the timestep, fits it,
    and returns the analytically derived velocity.

    For peak timesteps, the x-velocity will be near zero (parabola minimum).

    Args:
        traj: Trajectory dict from load_trajectory().
        timestep: Timestep index to evaluate velocity at.
        dt: Timestep duration.
        x_degree: Polynomial degree for x(t).
        y_degree: Polynomial degree for y(t).

    Returns:
        (vx, vy) tuple. Falls back to finite difference if timestep
        is not within a fittable segment.
    """
    puck_pos = traj["puck"][:, :2]
    puck_occluded = traj["puck"][:, 2] > 0.5
    N = len(puck_pos)

    contacts = find_contact_events(traj)
    segments = segment_free_flights(contacts, N, gap_tolerance, min_segment_length)
    peaks = find_puck_x_peaks(puck_pos, puck_occluded)
    peak_set = set(peaks)

    # If timestep is a peak, vx is exactly 0; still need vy from fit
    for seg_start, seg_end in segments:
        if seg_start <= timestep <= seg_end:
            sub_segs = _split_at_peaks(seg_start, seg_end, peaks, min_segment_length)
            for sub_start, sub_end in sub_segs:
                if sub_start <= timestep <= sub_end:
                    fit = fit_segment(puck_pos, puck_occluded, sub_start, sub_end,
                                      dt=dt, x_degree=x_degree, y_degree=y_degree)
                    if fit is not None:
                        t_rel = (timestep - sub_start) * dt
                        vx = float(velocity_from_fit(fit["x_coeffs"], t_rel))
                        vy = float(velocity_from_fit(fit["y_coeffs"], t_rel))
                        if timestep in peak_set:
                            vx = 0.0
                        return (vx, vy)

    # Fallback: finite difference
    if timestep > 0 and timestep < N:
        fd = (puck_pos[timestep] - puck_pos[timestep - 1]) / dt
        return (float(fd[0]), float(fd[1]))
    if timestep == 0 and N > 1:
        fd = (puck_pos[1] - puck_pos[0]) / dt
        return (float(fd[0]), float(fd[1]))
    return (0.0, 0.0)

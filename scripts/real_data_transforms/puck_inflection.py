"""Detect puck trajectory inflection points and relate them to contact events.

Terminology:
  Peaks   = local x-minima (puck furthest toward opponent, vx ≈ 0)
  Valleys = local x-maxima (puck closest to ego, typically at paddle contact)

Pipeline call stack (puck system identification):
  real_data_puck_pipeline.py  RealDataPuckPipeline.load_real_data()
    ├─ real_to_sim_observations.py  (loads trajectory data)
    │   └─ puck_velocity_fit.py  fit_trajectory_velocities()
    │       ├─ find_contact_events()         ◄── defined here
    │       └─ find_puck_x_peaks()           ◄── defined here
    └─ puck_inflection.py  load_peak_start_intervals()  ◄── YOU ARE HERE
        └─ reads inflection_<idx>.json logs → (peak, contact) interval tuples

  Batch pre-processing (run separately):
    puck_inflection.py --batch
      └─ analyze_and_log()  N → writes inflection_<idx>.json per trajectory

Key exports used by other modules:
  find_contact_events(traj)   → list of contact event dicts (paddle/wall hits)
  find_puck_x_peaks(...)      → list of timestep indices where puck x is at minimum
  load_peak_start_intervals() → {traj_idx: [(peak, contact), ...]} for CMA-ES sampling
  load_approach_intervals()   → {traj_idx: [(start, end), ...]} split at wall hits
"""

import os
import sys
import json
import argparse

import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, os.path.dirname(__file__))

from data_loading import (
    load_trajectory,
    load_all_trajectories,
    BOX2D_PADDLE_RADIUS,
    BOX2D_PUCK_RADIUS,
    BOX2D_TABLE_WIDTH,
    BOX2D_TABLE_LENGTH,
)

def _snap_to_visible(index, left_edge, right_edge, puck_occluded):
    """Snap an occluded detection to the nearest visible point in its plateau."""
    if not puck_occluded[index]:
        return index
    for offset in range(1, right_edge - left_edge + 2):
        for candidate in [index - offset, index + offset]:
            if left_edge <= candidate <= right_edge and not puck_occluded[candidate]:
                return candidate
    return None


def _find_extrema(signal, puck_occluded, min_prominence=0.02, min_distance=5):
    """Find peaks in `signal`, snapping occluded detections to visible points."""
    indices, props = find_peaks(signal, prominence=min_prominence,
                                plateau_size=1, distance=min_distance)
    result = []
    for i, idx in enumerate(indices):
        snapped = _snap_to_visible(
            int(idx), int(props["left_edges"][i]),
            int(props["right_edges"][i]), puck_occluded)
        if snapped is not None:
            result.append(snapped)
    return result


def find_puck_x_peaks(puck_pos, puck_occluded, **kwargs):
    return _find_extrema(-puck_pos[:, 0], puck_occluded, **kwargs)


def find_puck_x_valleys(puck_pos, puck_occluded, **kwargs):
    return _find_extrema(puck_pos[:, 0], puck_occluded, **kwargs)

def _find_occlusion_windows(puck_occluded):
    windows = []
    N = len(puck_occluded)
    t = 0
    while t < N:
        if puck_occluded[t]:
            start = t
            while t < N and puck_occluded[t]:
                t += 1
            windows.append((start, t - 1))
        else:
            t += 1
    return windows


def _pos(puck_pos, t):
    """Return {"x": ..., "y": ...} for a timestep."""
    return {"x": round(float(puck_pos[t, 0]), 5),
            "y": round(float(puck_pos[t, 1]), 5)}


def find_contact_events(traj):
    """Find contacts including estimated ones during occlusion."""
    paddle_pos = traj["paddle"][:, :2]
    puck_pos = traj["puck"][:, :2]
    puck_occluded = traj["puck"][:, 2] > 0.5
    N = len(puck_pos)

    hit_threshold = BOX2D_PADDLE_RADIUS + BOX2D_PUCK_RADIUS + 0.02
    occluded_hit_threshold = hit_threshold + 0.05

    side_boundary = (BOX2D_TABLE_WIDTH / 2) - BOX2D_PUCK_RADIUS - 0.05
    end_boundary = (BOX2D_TABLE_LENGTH / 2) - BOX2D_PUCK_RADIUS - 0.1

    def _classify_wall(px, py):
        if abs(py) >= side_boundary:
            return "wall_side"
        if px <= -end_boundary:
            return "wall_top"
        if px >= end_boundary:
            return "wall_bottom"
        return None

    events = []
    count = 0
    for t in range(N):
        if puck_occluded[t]:
            count +=1 
            continue
        dist = float(np.linalg.norm(puck_pos[t] - paddle_pos[t]))
        if dist < hit_threshold:
            events.append({
                "timestep": t,
                "type": "paddle",
                "estimated": False,
                "puck_pos": _pos(puck_pos, t),
                "paddle_dist": round(dist, 5),
            })
            continue
        wall = _classify_wall(puck_pos[t, 0], puck_pos[t, 1])
        if wall:
            events.append({
                "timestep": t,
                "type": wall,
                "estimated": False,
                "puck_pos": _pos(puck_pos, t),
            })

    visible_contact_set = {e["timestep"] for e in events}

    side_boundary_occ = side_boundary - 0.03
    end_boundary_occ = end_boundary - 0.03

    for occ_start, occ_end in _find_occlusion_windows(puck_occluded):
        pre = occ_start - 1
        if pre < 0:
            continue
        if any(t in visible_contact_set for t in range(max(0, pre - 2), min(N, occ_end + 3))):
            continue

        report_t = max(0, pre - 1)
        dist_pre = float(np.linalg.norm(puck_pos[pre] - paddle_pos[pre]))
        px, py = float(puck_pos[pre, 0]), float(puck_pos[pre, 1])

        base = {
            "timestep": report_t,
            "estimated": True,
            "occlusion_window": {"start": occ_start, "end": occ_end},
            "puck_pos": _pos(puck_pos, pre),
        }

        if dist_pre < occluded_hit_threshold:
            events.append({**base, "type": "paddle", "paddle_dist": round(dist_pre, 5)})
        elif abs(py) >= side_boundary_occ:
            events.append({**base, "type": "wall_side"})
        elif px <= -end_boundary_occ:
            events.append({**base, "type": "wall_top"})
        elif px >= end_boundary_occ:
            events.append({**base, "type": "wall_bottom"})

    events.sort(key=lambda e: e["timestep"])
    return events

def find_peaks_before_contacts(peaks, contact_events):
    """All peaks in each (prev_contact, contact) interval."""
    if not peaks or not contact_events:
        return []

    peak_arr = np.array(peaks)
    sorted_contacts = sorted(contact_events, key=lambda e: e["timestep"])
    results = []
    prev_t = -1

    for event in sorted_contacts:
        ct = event["timestep"]
        interval = [int(p) for p in peak_arr[(peak_arr > prev_t) & (peak_arr < ct)]]
        if interval:
            results.append({
                "peak_timesteps": interval,
                "contact_timestep": ct,
                "contact_type": event["type"],
                "num_peaks": len(interval),
                "last_peak_gap": ct - interval[-1],
            })
        prev_t = ct

    return results


def find_peaks_after_contacts(peaks, contact_events, n_timesteps):
    """All peaks in each (contact, next_contact) interval."""
    if not peaks or not contact_events:
        return []

    peak_arr = np.array(peaks)
    sorted_contacts = sorted(contact_events, key=lambda e: e["timestep"])
    results = []

    for i, event in enumerate(sorted_contacts):
        ct = event["timestep"]
        upper = sorted_contacts[i + 1]["timestep"] if i + 1 < len(sorted_contacts) else n_timesteps
        interval = [int(p) for p in peak_arr[(peak_arr > ct) & (peak_arr < upper)]]
        if interval:
            results.append({
                "peak_timesteps": interval,
                "contact_timestep": ct,
                "contact_type": event["type"],
                "num_peaks": len(interval),
                "first_peak_gap": interval[0] - ct,
            })

    return results

def find_approach_intervals(peaks, contact_events, puck_pos,
                            exclude_wall_hits=True):
    """Find intervals where the puck falls from a peak down to a paddle contact.

    Returns structured dicts with puck positions at the peak and contact,
    wall hits in between, and y-displacement for vertical-drop analysis.

    Args:
        peaks: List of peak timestep indices.
        contact_events: List of contact event dicts from find_contact_events().
        puck_pos: Puck positions (N, 2) in base coordinates.
        exclude_wall_hits: If True (default), discard intervals that contain
            any wall contacts (wall_top, wall_side, wall_bottom) between
            peak and paddle.
    """
    if not peaks or not contact_events:
        return []

    peak_arr = np.array(sorted(peaks))
    sorted_contacts = sorted(contact_events, key=lambda e: e["timestep"])
    paddle_contacts = [e for e in sorted_contacts if e["type"] == "paddle"]

    results = []
    used_peaks = set()

    for pc in paddle_contacts:
        ct = pc["timestep"]
        candidates = peak_arr[peak_arr < ct]
        if len(candidates) == 0:
            continue
        nearest_peak = int(candidates[-1])
        if nearest_peak in used_peaks:
            continue
        used_peaks.add(nearest_peak)

        wall_hits = [
            {"timestep": e["timestep"], "type": e["type"]}
            for e in sorted_contacts
            if e["type"].startswith("wall") and nearest_peak < e["timestep"] < ct
        ]

        if exclude_wall_hits and wall_hits:
            continue

        peak_pos = _pos(puck_pos, nearest_peak)
        contact_pos = _pos(puck_pos, ct)
        y_displacement = abs(peak_pos["y"] - contact_pos["y"])
        x_fall_height = contact_pos["x"] - peak_pos["x"]

        results.append({
            "interval": {"start": nearest_peak, "end": ct},
            "duration": ct - nearest_peak,
            "peak_puck_pos": peak_pos,
            "contact_puck_pos": contact_pos,
            "x_fall_height": round(x_fall_height, 5),
            "y_displacement": round(y_displacement, 5),
            "wall_hits": wall_hits,
            "num_wall_hits": len(wall_hits),
            "estimated_contact": pc.get("estimated", False),
        })

    return results


def find_vertical_approaches(approach_intervals, y_threshold=0.1):
    """Filter approach intervals where y barely changes (straight vertical drop).

    These are intervals where the puck falls nearly straight down from
    the apex to the paddle without significant lateral drift.
    """
    return [
        a for a in approach_intervals
        if a["y_displacement"] <= y_threshold
    ]


def find_paddle_to_paddle_intervals(peaks, contact_events, puck_pos):
    """Find full parabolic arcs: paddle hit → peak → paddle hit.

    Selects intervals where:
      1. A paddle contact occurs (start of arc)
      2. The puck rises to a peak (x-minimum, furthest from ego)
      3. Another paddle contact occurs (end of arc)
      4. No wall hits occur between the two paddle contacts

    This gives clean parabolic free-flight segments bounded by paddle
    interactions on both sides, ideal for puck physics identification.

    Args:
        peaks: List of peak timestep indices from find_puck_x_peaks().
        contact_events: List of contact event dicts from find_contact_events().
        puck_pos: Puck positions (N, 2) in base coordinates.

    Returns:
        List of dicts with interval, peak, paddle contact positions, etc.
    """
    if not peaks or not contact_events:
        return []

    peak_arr = np.array(sorted(peaks))
    sorted_contacts = sorted(contact_events, key=lambda e: e["timestep"])
    paddle_contacts = [e for e in sorted_contacts if e["type"] == "paddle"]

    if len(paddle_contacts) < 2:
        return []

    results = []
    used_peaks = set()

    # For each consecutive pair of paddle contacts, check if a peak exists between them
    for i in range(len(paddle_contacts) - 1):
        pc_start = paddle_contacts[i]
        pc_end = paddle_contacts[i + 1]
        t_start = pc_start["timestep"]
        t_end = pc_end["timestep"]

        # Find peaks between these two paddle contacts
        between_peaks = peak_arr[(peak_arr > t_start) & (peak_arr < t_end)]
        if len(between_peaks) == 0:
            continue

        # Check for wall hits between the two paddle contacts
        wall_hits = [
            e for e in sorted_contacts
            if e["type"].startswith("wall") and t_start < e["timestep"] < t_end
        ]
        if wall_hits:
            continue

        # Use the peak closest to the midpoint (typically there's exactly one)
        peak_t = int(between_peaks[len(between_peaks) // 2])
        if peak_t in used_peaks:
            continue
        used_peaks.add(peak_t)

        peak_pos = _pos(puck_pos, peak_t)
        start_pos = _pos(puck_pos, t_start)
        end_pos = _pos(puck_pos, t_end)

        results.append({
            "interval": {"start": t_start, "end": t_end},
            "duration": t_end - t_start,
            "peak_timestep": peak_t,
            "peak_puck_pos": peak_pos,
            "start_paddle_pos": start_pos,
            "end_paddle_pos": end_pos,
            "estimated_start": pc_start.get("estimated", False),
            "estimated_end": pc_end.get("estimated", False),
        })

    return results


def find_free_flight_segments(contacts, peaks, n_timesteps,
                              gap_tolerance=2, min_length=3):
    """Identify free-flight segments between merged contact windows.

    Contacts within gap_tolerance timesteps are merged into a single window.
    Segments shorter than min_length are discarded. Each segment is annotated
    with whether it contains an x-peak (where vx ~ 0).

    Args:
        contacts: List of contact event dicts from find_contact_events().
        peaks: List of peak timestep indices from find_puck_x_peaks().
        n_timesteps: Total trajectory length.
        gap_tolerance: Max gap to merge consecutive contacts into one window.
        min_length: Minimum segment length to include.

    Returns:
        List of dicts with keys:
            start: int, first timestep of free flight
            end: int, last timestep of free flight (inclusive)
            has_peak: bool, whether a peak is within this segment
            peak_idx: int or None, first peak timestep if has_peak
    """
    # Merge contacts into windows
    if contacts:
        timesteps = sorted(set(e["timestep"] for e in contacts))
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
    else:
        windows = []

    peak_set = set(peaks)

    # Build segments from gaps between windows
    segments = []

    def _make_segment(s, e):
        seg_peaks = [p for p in peaks if s <= p <= e]
        return {
            "start": s,
            "end": e,
            "has_peak": len(seg_peaks) > 0,
            "peak_idx": seg_peaks[0] if seg_peaks else None,
        }

    if not windows:
        if n_timesteps >= min_length:
            segments.append(_make_segment(0, n_timesteps - 1))
        return segments

    # Before first contact window
    if windows[0][0] >= min_length:
        segments.append(_make_segment(0, windows[0][0] - 1))

    # Between contact windows
    for i in range(len(windows) - 1):
        seg_start = windows[i][1] + 1
        seg_end = windows[i + 1][0] - 1
        if seg_end - seg_start + 1 >= min_length:
            segments.append(_make_segment(seg_start, seg_end))

    # After last contact window
    last_end = windows[-1][1]
    if n_timesteps - 1 - last_end >= min_length:
        segments.append(_make_segment(last_end + 1, n_timesteps - 1))

    return segments


def analyze_trajectory(traj, vertical_y_threshold=0.1):
    """Run full inflection + contact analysis on a trajectory."""
    from puck_velocity_fit import fit_trajectory_velocities

    puck_pos = traj["puck"][:, :2]
    puck_occluded = traj["puck"][:, 2] > 0.5
    n = len(puck_pos)

    peaks = find_puck_x_peaks(puck_pos, puck_occluded)
    valleys = find_puck_x_valleys(puck_pos, puck_occluded)
    contacts = find_contact_events(traj)
    all_inflections = sorted(peaks + valleys)
    approaches = find_approach_intervals(peaks, contacts, puck_pos,
                                         exclude_wall_hits=False)
    vertical = find_vertical_approaches(approaches, y_threshold=vertical_y_threshold)
    paddle_to_paddle = find_paddle_to_paddle_intervals(peaks, contacts, puck_pos)
    free_flight = find_free_flight_segments(contacts, peaks, n)
    occlusion_windows = [
        {"start": s, "end": e} for s, e in _find_occlusion_windows(puck_occluded)
    ]

    # Compute parabolic velocity estimates at peaks
    velocities = fit_trajectory_velocities(traj)
    peaks_with_velocity = []
    for p in peaks:
        vx, vy = float(velocities[p, 0]), float(velocities[p, 1])
        peaks_with_velocity.append({
            "timestep": p,
            "puck_pos": _pos(puck_pos, p),
            "velocity": {"vx": round(vx, 6), "vy": round(vy, 6)},
            "occluded": bool(puck_occluded[p]),
        })

    return {
        "peaks": peaks_with_velocity,
        "valleys": valleys,
        "contacts": contacts,
        "occlusion_intervals": occlusion_windows,
        "approach_intervals": approaches,
        "vertical_approaches": vertical,
        "paddle_to_paddle_intervals": paddle_to_paddle,
        "free_flight_segments": free_flight,
        "peaks_before_contacts": find_peaks_before_contacts(peaks, contacts),
        "valleys_before_contacts": find_peaks_before_contacts(valleys, contacts),
        "all_inflections_before_contacts": find_peaks_before_contacts(all_inflections, contacts),
        "peaks_after_contacts": find_peaks_after_contacts(peaks, contacts, n),
        "valleys_after_contacts": find_peaks_after_contacts(valleys, contacts, n),
        "all_inflections_after_contacts": find_peaks_after_contacts(all_inflections, contacts, n),
    }


_SECTIONS = [
    ("all_inflections_before_contacts", "All inflections before contacts", "before"),
    ("peaks_before_contacts",           "Peaks before contacts",           "before"),
    ("valleys_before_contacts",         "Valleys before contacts",         "before"),
    ("all_inflections_after_contacts",  "All inflections after contacts",  "after"),
    ("peaks_after_contacts",            "Peaks after contacts",            "after"),
    ("valleys_after_contacts",          "Valleys after contacts",          "after"),
]


def _print_section(result, key, label, direction):
    """Print inflection-before/after-contact intervals (disabled by default)."""
    pass


def analyze_and_log(data_dir, traj_idx, log_path=None):
    """Analyze a trajectory and optionally write a JSON log.

    Runs the full inflection + contact analysis, then serializes
    peaks, valleys, contacts, approach intervals, and free-flight
    segments to a JSON file for downstream consumption by the
    puck system identification pipeline.
    """
    traj = load_trajectory(data_dir, f"trajectory_data{traj_idx}.hdf5",
                           load_images=False)
    result = analyze_trajectory(traj)

    if log_path is not None:
        log_entry = {
            "traj_idx": traj_idx,
            "num_timesteps": len(traj["puck"]),
            "peaks": result["peaks"],
            "valleys": result["valleys"],
            "contacts": result["contacts"],
            "occlusion_intervals": result["occlusion_intervals"],
            "approach_intervals": result["approach_intervals"],
            "vertical_approaches": result["vertical_approaches"],
            "paddle_to_paddle_intervals": result["paddle_to_paddle_intervals"],
            "free_flight_segments": result["free_flight_segments"],
        }
        for key, _, _ in _SECTIONS:
            log_entry[key] = result[key]

        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(log_entry, f, indent=2)
        print(f"\nLogged to {log_path}")

    return result


def batch_analyze_and_log(data_dir, max_traj_idx=607, log_dir=None, force=False):
    """Process inflection analysis for all trajectories up to max_traj_idx.

    Skips trajectories that already have a log file or don't have an HDF5 file.
    Use force=True to regenerate all logs (e.g., after adding new analysis fields).
    """
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)

    processed = 0
    skipped_existing = 0
    skipped_missing = 0
    errors = 0

    for idx in range(max_traj_idx + 1):
        log_path = os.path.join(log_dir, f"inflection_{idx}.json")
        if os.path.exists(log_path) and not force:
            print('getting here')
            skipped_existing += 1
            continue

        traj_file = os.path.join(data_dir, f"trajectory_data{idx}.hdf5")
        if not os.path.exists(traj_file):
            skipped_missing += 1
            continue

        try:
            analyze_and_log(data_dir, idx, log_path=log_path)
            processed += 1
        except Exception as e:
            print(f"  Error processing trajectory {idx}: {e}")
            errors += 1

    print(f"\nBatch complete: {processed} processed, {skipped_existing} already existed, "
          f"{skipped_missing} no HDF5 file, {errors} errors")


def _build_peak_filters(data):
    """Build sets of fitted and occluded peak timesteps from a JSON log entry."""
    fitted_peaks = set()
    occluded_peaks = set()
    for p in data.get("peaks", []):
        if not isinstance(p, dict):
            continue
        t = p["timestep"]
        if p.get("occluded", False):
            occluded_peaks.add(t)
        vel = p.get("velocity", {})
        if vel.get("vx", None) == 0.0:
            fitted_peaks.add(t)
    return fitted_peaks, occluded_peaks


def load_approach_intervals(log_dir, traj_indices, require_fitted_peak=True):
    """Load inflection JSONs and build sampling intervals from approach intervals.

    Only returns intervals with no wall hits (wall_top, wall_side, wall_bottom).
    Intervals containing any wall contact are discarded entirely.

    Args:
        log_dir: Directory containing inflection_<idx>.json files.
        traj_indices: List of trajectory indices to load.
        require_fitted_peak: If True (default), only keep intervals whose
            peak has vx==0 from the parabolic fit.

    Returns:
        dict mapping traj_idx to list of (start, end) interval tuples.
    """
    intervals_by_traj = {}
    loaded = 0
    missing = 0
    skipped_wall = 0
    skipped_unfitted = 0

    for traj_idx in traj_indices:
        log_path = os.path.join(log_dir, f"inflection_{traj_idx}.json")
        if not os.path.exists(log_path):
            missing += 1
            continue

        with open(log_path, "r") as f:
            data = json.load(f)

        fitted_peaks, _ = _build_peak_filters(data)

        intervals = []
        for approach in data.get("approach_intervals", []):
            wall_hits = approach.get("wall_hits", [])
            if wall_hits:
                skipped_wall += 1
                continue

            iv_start = approach["interval"]["start"]
            iv_end = approach["interval"]["end"]

            if require_fitted_peak and iv_start not in fitted_peaks:
                skipped_unfitted += 1
                continue

            intervals.append((iv_start, iv_end))

        if intervals:
            intervals_by_traj[traj_idx] = intervals
            loaded += 1

    print(f"Loaded approach intervals for {loaded} trajectories "
          f"({missing} missing logs, {skipped_wall} wall-hit skipped, "
          f"{skipped_unfitted} unfitted-peak skipped)")
    return intervals_by_traj, traj_indices


def load_peak_start_intervals(log_dir, traj_indices, min_height=None,
                              max_y_displacement=None,
                              require_fitted_peak=True):
    """Load inflection JSONs and return intervals from peak to first paddle contact.

    Each interval starts at a peak (puck x-minimum, ~zero velocity) and ends
    at the first paddle contact after that peak. Wall contacts (wall_top,
    wall_side, wall_bottom) are skipped — if a wall hit occurs between the
    peak and the next paddle contact, the interval is discarded.

    Args:
        log_dir: Directory containing inflection_<idx>.json files.
        traj_indices: List of trajectory indices to load.
        min_height: Minimum x-displacement (fall height) from peak to paddle
            contact. Filters out shallow arcs.
        max_y_displacement: Maximum y-displacement between peak and paddle
            contact. Filters out arcs with large lateral drift.
        require_fitted_peak: If True (default), only keep intervals whose
            peak has vx==0 from the parabolic fit (properly fitted segment).

    Returns:
        dict mapping traj_idx to list of (peak_start, paddle_contact_end) tuples.
    """
    intervals_by_traj = {}
    loaded = 0
    missing = 0
    total_intervals = 0
    skipped_wall = 0
    skipped_unfitted = 0

    for traj_idx in traj_indices:
        log_path = os.path.join(log_dir, f"inflection_{traj_idx}.json")
        if not os.path.exists(log_path):
            missing += 1
            continue

        with open(log_path, "r") as f:
            data = json.load(f)

        fitted_peaks, _ = _build_peak_filters(data)

        intervals = []
        for approach in data.get("approach_intervals", []):
            wall_hits = approach.get("wall_hits", [])
            if wall_hits:
                skipped_wall += 1
                continue

            iv_start = approach["interval"]["start"]
            iv_end = approach["interval"]["end"]

            if require_fitted_peak and iv_start not in fitted_peaks:
                skipped_unfitted += 1
                continue

            if min_height is not None:
                x_fall = approach.get("x_fall_height", 0)
                if x_fall < min_height:
                    continue

            if max_y_displacement is not None:
                y_disp = approach.get("y_displacement", 0)
                if y_disp > max_y_displacement:
                    continue

            intervals.append((iv_start, iv_end))

        if intervals:
            intervals_by_traj[traj_idx] = intervals
            total_intervals += len(intervals)
            loaded += 1

    print(f"Loaded {total_intervals} peak-start intervals from {loaded} trajectories "
          f"({missing} missing logs, {skipped_wall} wall-hit skipped, "
          f"{skipped_unfitted} unfitted-peak skipped)")
    return intervals_by_traj


def load_paddle_to_paddle_intervals(log_dir, traj_indices,
                                    require_fitted_peak=True):
    """Load inflection JSONs and return paddle→peak→paddle parabolic arc intervals.

    Each interval spans a full parabolic arc: starting at a paddle contact,
    rising to a peak (x-minimum), and ending at the next paddle contact,
    with no wall hits in between.

    Args:
        log_dir: Directory containing inflection_<idx>.json files.
        traj_indices: List of trajectory indices to load.
        require_fitted_peak: If True (default), only keep intervals whose
            peak has vx==0 from the parabolic fit (properly fitted segment).

    Returns:
        dict mapping traj_idx to list of (start, end) interval tuples.
    """
    intervals_by_traj = {}
    loaded = 0
    missing = 0
    total_intervals = 0
    skipped_unfitted = 0

    for traj_idx in traj_indices:
        log_path = os.path.join(log_dir, f"inflection_{traj_idx}.json")
        if not os.path.exists(log_path):
            missing += 1
            continue

        with open(log_path, "r") as f:
            data = json.load(f)

        fitted_peaks, _ = _build_peak_filters(data)

        intervals = []
        for p2p in data.get("paddle_to_paddle_intervals", []):
            iv_start = p2p["interval"]["start"]
            iv_end = p2p["interval"]["end"]
            peak_t = p2p.get("peak_timestep")

            if require_fitted_peak and peak_t not in fitted_peaks:
                skipped_unfitted += 1
                continue

            intervals.append((iv_start, iv_end))

        if intervals:
            intervals_by_traj[traj_idx] = intervals
            total_intervals += len(intervals)
            loaded += 1

    print(f"Loaded {total_intervals} paddle-to-paddle intervals from {loaded} trajectories "
          f"({missing} missing logs, {skipped_unfitted} unfitted-peak skipped)")
    return intervals_by_traj


if __name__ == "__main__":
    DEFAULT_DATA_DIR = "/data2/air_hockey/air_hockey_state_data/datastor1/calebc/public/data/mouse/state_data_all_new/"

    parser = argparse.ArgumentParser(description="Detect puck x-direction inflection points")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--traj-idx", type=int, default=None,
                        help="Single trajectory index to process")
    parser.add_argument("--batch", action="store_true",
                        help="Process all trajectories up to --max-traj-idx")
    parser.add_argument("--max-traj-idx", type=int, default=864,
                        help="Max trajectory index for batch mode")
    parser.add_argument("--log-path", type=str, default=None,
                        help="Path to save JSON log (default: logs/inflection_<idx>.json)")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate logs even if they already exist")
    args = parser.parse_args()

    if args.batch:
        batch_analyze_and_log(args.data_dir, max_traj_idx=args.max_traj_idx,
                              force=args.force)
    else:
        traj_idx = args.traj_idx if args.traj_idx is not None else 100
        if args.log_path is None:
            log_dir = os.path.join(os.path.dirname(__file__), "logs")
            args.log_path = os.path.join(log_dir, f"inflection_{traj_idx}.json")
        analyze_and_log(args.data_dir, traj_idx, log_path=args.log_path)

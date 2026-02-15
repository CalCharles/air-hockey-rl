"""Detect puck trajectory inflection points and relate them to contact events.

Peaks = local x-minima (puck furthest toward opponent).
Valleys = local x-maxima (puck closest to ego).
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
    end_boundary = (BOX2D_TABLE_LENGTH / 2) - BOX2D_PUCK_RADIUS - 0.05

    def _classify_wall(px, py):
        if abs(py) >= side_boundary:
            return "wall_side"
        if px <= -end_boundary:
            return "wall_top"
        if px >= end_boundary:
            return "wall_bottom"
        return None

    events = []
    for t in range(N):
        if puck_occluded[t]:
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

def find_approach_intervals(peaks, contact_events, puck_pos):
    """Find intervals where the puck falls from a peak down to a paddle contact.

    Returns structured dicts with puck positions at the peak and contact,
    wall hits in between, and y-displacement for vertical-drop analysis.
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

        peak_pos = _pos(puck_pos, nearest_peak)
        contact_pos = _pos(puck_pos, ct)
        y_displacement = abs(peak_pos["y"] - contact_pos["y"])

        results.append({
            "interval": {"start": nearest_peak, "end": ct},
            "duration": ct - nearest_peak,
            "peak_puck_pos": peak_pos,
            "contact_puck_pos": contact_pos,
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

def analyze_trajectory(traj, vertical_y_threshold=0.1):
    """Run full inflection + contact analysis on a trajectory."""
    puck_pos = traj["puck"][:, :2]
    puck_occluded = traj["puck"][:, 2] > 0.5
    n = len(puck_pos)

    peaks = find_puck_x_peaks(puck_pos, puck_occluded)
    valleys = find_puck_x_valleys(puck_pos, puck_occluded)
    contacts = find_contact_events(traj)
    all_inflections = sorted(peaks + valleys)
    approaches = find_approach_intervals(peaks, contacts, puck_pos)
    vertical = find_vertical_approaches(approaches, y_threshold=vertical_y_threshold)

    return {
        "peaks": peaks,
        "valleys": valleys,
        "contacts": contacts,
        "approach_intervals": approaches,
        "vertical_approaches": vertical,
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
    pass
    # entries = result[key]
    # print(f"\n  {label} ({len(entries)} intervals):")
    # for e in entries:
    #     pts = e["peak_timesteps"]
    #     if direction == "before":
    #         print(f"    {e['num_peaks']}x at t={pts} -> {e['contact_type']} "
    #               f"t={e['contact_timestep']}  (last gap={e['last_peak_gap']})")
    #     else:
    #         print(f"    {e['contact_type']} t={e['contact_timestep']} -> "
    #               f"{e['num_peaks']}x at t={pts}  (first gap={e['first_peak_gap']})")


def analyze_and_log(data_dir, traj_idx, log_path=None):
    """Analyze a trajectory, print summary, optionally write JSON log."""
    traj = load_trajectory(data_dir, f"trajectory_data{traj_idx}.hdf5",
                           load_images=False)
    result = analyze_trajectory(traj)

    # print(f"\n{'='*60}")
    # print(f"Trajectory {traj_idx}  ({len(traj['puck'])} timesteps)")
    # print(f"{'='*60}")
    # print(f"  Peaks (toward opponent): {len(result['peaks'])} at {result['peaks']}")
    # print(f"  Valleys (toward ego):    {len(result['valleys'])} at {result['valleys']}")
    # print(f"  Contacts: {len(result['contacts'])}")
    # for c in result["contacts"]:
    #     est = " [estimated]" if c.get("estimated") else ""
    #     print(f"    t={c['timestep']:4d}  {c['type']:12s}  "
    #           f"pos=({c['puck_pos']['x']:+.4f}, {c['puck_pos']['y']:+.4f}){est}")

    # print(f"\n  Approach intervals (peak -> paddle): {len(result['approach_intervals'])}")
    # for a in result["approach_intervals"]:
    #     iv = a["interval"]
    #     est = " [estimated]" if a["estimated_contact"] else ""
    #     print(f"    t={iv['start']} -> t={iv['end']}  ({a['duration']} steps, "
    #           f"dy={a['y_displacement']:.4f}, {a['num_wall_hits']} wall hits){est}")

    # n_vert = len(result["vertical_approaches"])
    # print(f"\n  Vertical approaches (dy <= 0.1): {n_vert}")
    # for a in result["vertical_approaches"]:
    #     iv = a["interval"]
    #     print(f"    t={iv['start']} -> t={iv['end']}  "
    #           f"peak_y={a['peak_puck_pos']['y']:+.4f}  "
    #           f"contact_y={a['contact_puck_pos']['y']:+.4f}  "
    #           f"dy={a['y_displacement']:.4f}")

    # for key, label, direction in _SECTIONS:
    #     _print_section(result, key, label, direction)

    if log_path is not None:
        log_entry = {
            "traj_idx": traj_idx,
            "num_timesteps": len(traj["puck"]),
            "peaks": result["peaks"],
            "valleys": result["valleys"],
            "contacts": result["contacts"],
            "approach_intervals": result["approach_intervals"],
            "vertical_approaches": result["vertical_approaches"],
        }
        for key, _, _ in _SECTIONS:
            log_entry[key] = result[key]

        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(log_entry, f, indent=2)
        print(f"\nLogged to {log_path}")

    return result


if __name__ == "__main__":
    DEFAULT_DATA_DIR = "/data2/air_hockey/air_hockey_state_data/datastor1/calebc/public/data/mouse/state_data_all_new/"

    parser = argparse.ArgumentParser(description="Detect puck x-direction inflection points")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--traj-idx", type=int, default=100)
    parser.add_argument("--log-path", type=str, default=None,
                        help="Path to save JSON log (default: logs/inflection_<idx>.json)")
    args = parser.parse_args()

    if args.log_path is None:

        log_dir = os.path.join(os.path.dirname(__file__), "logs")
        args.log_path = os.path.join(log_dir, f"inflection_{args.traj_idx}.json")

    analyze_and_log(args.data_dir, args.traj_idx, log_path=args.log_path)

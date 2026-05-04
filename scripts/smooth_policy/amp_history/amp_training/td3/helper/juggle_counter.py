"""Per-episode paddle-puck contact and juggle counter.

A *contact* is a frame where ``||paddle_xy - puck_xy|| < paddle_radius +
puck_radius + slop``. A *juggle* is a contact whose long-term puck-x
direction flips: comparing puck Δx over the window 25→5 frames *before*
the contact against the window 5→25 frames *after*. Wall bounces
(side/top walls) never trigger because the paddle stays near the robot
side and is far from those walls.

Validated against staged real-world episode hdf5s — see
``notes/scratch/juggle_count_validation/count_juggles.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


PADDLE_RADIUS = 0.0508
PUCK_RADIUS = 0.03175
CONTACT_SLOP = 0.02
CONTACT_THRESH = PADDLE_RADIUS + PUCK_RADIUS + CONTACT_SLOP  # ~0.0826 m

WINDOW = 25            # long-term direction window: 25 frames before/after
INNER = 5              # ignore the 5 frames immediately around contact
MIN_DISPLACEMENT = 0.02  # m; reject windows with < 2 cm motion (noise)
COOLDOWN_FRAMES = 30   # min frame gap between accepted juggles

JUGGLE_SUCCESS_THRESHOLD = 2  # juggles >= this -> juggle-success episode


@dataclass(frozen=True)
class JuggleCounts:
    """Per-episode summary returned by :func:`count_episode_juggles`.

    ``n_contacts`` counts every paddle-puck contact event (one per
    cluster of contiguous frames where distance < threshold), even those
    that did not produce a long-term direction flip. ``n_juggles`` is
    the subset of those that passed the direction-flip + cooldown
    filter.
    """

    n_contacts: int
    n_juggles: int

    @property
    def juggle_success(self) -> bool:
        return self.n_juggles >= JUGGLE_SUCCESS_THRESHOLD


def _find_contact_event_centers(dist: np.ndarray) -> list[int]:
    """Group contiguous frames where ``dist < CONTACT_THRESH`` into events,
    returning the index of the local-minimum frame in each cluster."""
    in_contact = dist < CONTACT_THRESH
    events: list[int] = []
    i, n = 0, len(dist)
    while i < n:
        if in_contact[i]:
            j = i
            while j < n and in_contact[j]:
                j += 1
            events.append(i + int(np.argmin(dist[i:j])))
            i = j
        else:
            i += 1
    return events


def _classify_event(t: int, puck_x: np.ndarray, T: int) -> str:
    pre_a, pre_b = max(0, t - WINDOW), max(0, t - INNER)
    post_a, post_b = min(T - 1, t + INNER), min(T - 1, t + WINDOW)
    if pre_b - pre_a < 5 or post_b - post_a < 5:
        return "edge"
    dx_pre = puck_x[pre_b] - puck_x[pre_a]
    dx_post = puck_x[post_b] - puck_x[post_a]
    if abs(dx_pre) < MIN_DISPLACEMENT or abs(dx_post) < MIN_DISPLACEMENT:
        return "tiny"
    if np.sign(dx_pre) != np.sign(dx_post):
        return "juggle"
    return "no_flip"


def count_juggles_from_arrays(
    paddle_xy: np.ndarray,
    puck_xy: np.ndarray,
    puck_occluded: np.ndarray,
) -> JuggleCounts:
    """Compute (n_contacts, n_juggles) from per-step (T, 2) position arrays.

    Inputs share a single table-frame coordinate system. ``puck_occluded``
    must be a length-T array of 0/1 floats — frames where the puck was
    occluded are ignored when scanning for contact (we do not trust the
    sentinel position).
    """
    paddle_xy = np.asarray(paddle_xy, dtype=np.float64).reshape(-1, 2)
    puck_xy = np.asarray(puck_xy, dtype=np.float64).reshape(-1, 2)
    occluded = np.asarray(puck_occluded, dtype=np.float64).reshape(-1) > 0.5
    T = paddle_xy.shape[0]
    if T == 0 or puck_xy.shape[0] != T or occluded.shape[0] != T:
        return JuggleCounts(n_contacts=0, n_juggles=0)

    dist = np.linalg.norm(paddle_xy - puck_xy, axis=1)
    dist_for_contact = dist.copy()
    dist_for_contact[occluded] = np.inf

    events = _find_contact_event_centers(dist_for_contact)
    if not events:
        return JuggleCounts(n_contacts=0, n_juggles=0)

    n_juggles = 0
    last_juggle = -10**9
    puck_x = puck_xy[:, 0]
    for t in events:
        if _classify_event(t, puck_x, T) != "juggle":
            continue
        if (t - last_juggle) < COOLDOWN_FRAMES:
            continue
        n_juggles += 1
        last_juggle = t

    return JuggleCounts(n_contacts=len(events), n_juggles=n_juggles)


def count_juggles_from_rows(rows: Sequence[Mapping[str, np.ndarray]]) -> JuggleCounts:
    """Extract paddle xy / puck xy / occlusion from split-schema HDF5 rows.

    The orchestrator builds these rows via ``_build_split_episode_row``;
    each row carries ``pose`` (paddle xy in cols 0:2) and ``puck`` (puck
    xy in cols 0:2, occlusion flag in col 2). Empty trajectories return
    a zero count.
    """
    if not rows:
        return JuggleCounts(n_contacts=0, n_juggles=0)
    paddle_xy = np.asarray([np.asarray(r["pose"], dtype=np.float64)[:2] for r in rows])
    puck = np.asarray([np.asarray(r["puck"], dtype=np.float64)[:3] for r in rows])
    return count_juggles_from_arrays(
        paddle_xy=paddle_xy,
        puck_xy=puck[:, :2],
        puck_occluded=puck[:, 2],
    )

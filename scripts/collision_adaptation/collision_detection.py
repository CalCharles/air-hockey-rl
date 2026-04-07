"""
Collision detection helpers for air-hockey collision adaptation scripts.

Background
----------
The Box2D simulator appends one entry to `collision_listener.collision_forces`
for *every contact point* in *every PostSolve callback*.  A single physical
paddle-puck collision can produce multiple entries in one step (two contact
points → two entries, or the contact persists across sub-steps).

Detection pattern used throughout the adaptation scripts:

    prev = len(sim.get_collision_forces())
    env.step(action)
    new_entries = sim.get_collision_forces()[prev:]
    hit = any(is_paddle_puck_collision(cf) for cf in new_entries)

This module provides:
  - is_paddle_puck_collision(cf)  — predicate on a single collision_forces entry
  - StepCollisionDetector         — stateful helper that wraps the pattern above

Notes on the collision_forces entry format
------------------------------------------
Each entry is a dict produced in CollisionForceListener.PostSolve
(airhockey/sims/airhockey_box2d.py):

    {
        'bodyA':          <body.userData string, e.g. "paddle_ego" or "puck_0">
        'bodyB':          <body.userData string>
        'normal_force':   float   (impulse / 60.0)
        'contact_normal': (float, float)  — Box2D coords
    }

bodyA/B values come from Box2D body.userData which is set to string labels
like "paddle_ego", "puck_0", "table_wall" when the bodies are created.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from airhockey.sims.real.velocity_estimator import fit_velocity_from_positions


# ---------------------------------------------------------------------------
# Shared constants — single source of truth for tier/speed definitions
# ---------------------------------------------------------------------------

TIERS = ("low", "mid", "high")
SPEED_BREAKPOINTS = (0.25, 0.75)  # m/s — must match CollisionForceListener defaults
PUCK_HISTORY_PAD = 5              # spawn_puck() prepends this many padding entries


def speed_tier(speed: float) -> str:
    """Return 'low', 'mid', or 'high' based on SPEED_BREAKPOINTS."""
    lo, hi = SPEED_BREAKPOINTS
    if speed < lo:
        return "low"
    if speed < hi:
        return "mid"
    return "high"


# ---------------------------------------------------------------------------
# Predicate
# ---------------------------------------------------------------------------

def is_paddle_puck_collision(cf: dict) -> bool:
    """Return True if collision_forces entry is a paddle ↔ puck contact.

    Checks whether one body contains "paddle" and the other contains "puck"
    in its userData string.  Works regardless of which body is A vs B.
    """
    a = str(cf.get("bodyA", ""))
    b = str(cf.get("bodyB", ""))
    has_paddle = "paddle" in a or "paddle" in b
    has_puck = "puck" in a or "puck" in b
    return has_paddle and has_puck


# ---------------------------------------------------------------------------
# Stateful per-step detector
# ---------------------------------------------------------------------------

class StepCollisionDetector:
    """Tracks new paddle-puck collision entries across env steps.

    Usage
    -----
        detector = StepCollisionDetector(sim)

        env.reset()
        detector.reset()           # sync baseline with post-reset state

        for _ in range(n_steps):
            env.step(action)
            new_entries = detector.step()  # entries added during this step
            if new_entries:
                # at least one paddle-puck collision happened this step
                ...

    Parameters
    ----------
    sim : AirHockeyBox2D
        The simulator object; must expose get_collision_forces().
    """

    def __init__(self, sim):
        self._sim = sim
        self._prev_count: int = 0

    def reset(self) -> None:
        """Sync baseline to current collision_forces length (call after env.reset())."""
        self._prev_count = len(self._sim.get_collision_forces())

    def step(self) -> list[dict]:
        """Return all paddle-puck collision entries added since the last call.

        Call once per env step *after* env.step().  Returns an empty list if
        no paddle-puck collision occurred during that step.
        """
        forces = self._sim.get_collision_forces()
        new_entries = [cf for cf in forces[self._prev_count:] if is_paddle_puck_collision(cf)]
        self._prev_count = len(forces)
        return new_entries

    def any_collision_this_step(self, new_entries: list[dict]) -> bool:
        """Convenience: True if new_entries (from step()) is non-empty."""
        return len(new_entries) > 0


# ---------------------------------------------------------------------------
# Wall-puck collision predicate
# ---------------------------------------------------------------------------

def is_wall_puck_collision(cf: dict) -> bool:
    """Return True if collision_forces entry is a wall ↔ puck contact."""
    a = str(cf.get("bodyA", ""))
    b = str(cf.get("bodyB", ""))
    has_wall = "wall" in a or "wall" in b
    has_puck = "puck" in a or "puck" in b
    return has_wall and has_puck


# ---------------------------------------------------------------------------
# Position-only collision detection
# ---------------------------------------------------------------------------

@dataclass
class CollisionEvent:
    """A collision detected from position trajectories alone."""
    frame_idx: int
    collision_type: str         # "paddle" or "wall"
    speed_before: float         # m/s
    speed_after: float          # m/s
    velocity_before: np.ndarray # (2,) m/s
    velocity_after: np.ndarray  # (2,) m/s
    tier: str                   # "low" / "mid" / "high"


# Default wall bounds for the heavy config table.
# Table half-length = 0.9652, half-width = 0.4318 (base coords).
# Collision happens when puck center reaches wall - puck_radius (0.03175).
_DEFAULT_WALL_BOUNDS = {
    "x_min": -0.9652 + 0.03175,
    "x_max":  0.9652 - 0.03175,
    "y_min": -0.4318 + 0.03175,
    "y_max":  0.4318 - 0.03175,
}


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle between two 2-D vectors in degrees. Returns 0 if either is near-zero."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def detect_collisions_from_positions(
    puck_positions: np.ndarray,
    paddle_positions: np.ndarray,
    times: np.ndarray,
    valid_mask: np.ndarray | None = None,
    gravity: tuple[float, float] = (-0.65, 0.0),
    window_frames: int = 10,
    min_snr: float = 8.0,
    speed_change_threshold: float = 0.1,
    angle_change_threshold: float = 30.0,
    paddle_proximity_radius: float = 0.10,
    wall_bounds: dict | None = None,
    wall_proximity_margin: float = 0.03,
) -> list[CollisionEvent]:
    """Detect collisions from raw position trajectories (no Box2D callbacks).

    Scans for velocity discontinuities using sliding-window velocity fits,
    then classifies each candidate as paddle or wall collision based on
    spatial proximity.

    Parameters
    ----------
    puck_positions   : (N, 2) puck x,y positions in base coords
    paddle_positions : (N, 2) paddle x,y positions in base coords
    times            : (N,) timestamps in seconds
    valid_mask       : (N,) bool, True = non-occluded. None means all valid.
    gravity          : (gx, gy) deceleration for velocity fitting
    window_frames    : frames per velocity-fit window (pre and post)
    min_snr          : minimum SNR to accept a velocity fit
    speed_change_threshold : m/s, min ||v_after - v_before|| to flag
    angle_change_threshold : degrees, min angle change to flag
    paddle_proximity_radius : metres, max puck-paddle distance for paddle collision
    wall_bounds      : dict with x_min, x_max, y_min, y_max (puck center at collision)
    wall_proximity_margin : metres, extra inward margin for wall detection (puck may
                     have bounced away by one frame at the detection point)

    Returns
    -------
    List of CollisionEvent sorted by frame_idx.
    """
    puck_positions = np.asarray(puck_positions, dtype=float)
    paddle_positions = np.asarray(paddle_positions, dtype=float)
    times = np.asarray(times, dtype=float)
    N = len(puck_positions)

    if valid_mask is None:
        valid_mask = np.ones(N, dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)

    if wall_bounds is None:
        wall_bounds = dict(_DEFAULT_WALL_BOUNDS)

    dt = float(times[1] - times[0]) if N > 1 else 0.05
    g_arr = np.array(gravity, dtype=float)

    # ------------------------------------------------------------------
    # Step 1: scan for velocity discontinuities
    # ------------------------------------------------------------------
    candidates = []  # (frame_idx, v_before, v_after, delta_v_norm)

    for i in range(window_frames, N - window_frames):
        # Pre-collision window: [i - window_frames, i)
        pre_slice = slice(i - window_frames, i)
        pre_result = fit_velocity_from_positions(
            puck_positions[pre_slice], times[pre_slice],
            valid_mask[pre_slice], gravity=gravity,
        )
        if pre_result is None or pre_result["snr"] < min_snr:
            continue

        # Post-collision window: [i, i + window_frames)
        post_slice = slice(i, i + window_frames)
        post_result = fit_velocity_from_positions(
            puck_positions[post_slice], times[post_slice],
            valid_mask[post_slice], gravity=gravity,
        )
        if post_result is None or post_result["snr"] < min_snr:
            continue

        # v_before: extrapolate pre-window endpoint by one dt with gravity
        v_before = pre_result["v_at_end"] - g_arr * dt
        # v_after: velocity at first frame of post-window
        v_after = post_result["v_at_times"][0]

        delta_v = v_after - v_before
        delta_v_norm = float(np.linalg.norm(delta_v))
        angle_change = _angle_between(v_before, v_after)

        if delta_v_norm > speed_change_threshold or angle_change > angle_change_threshold:
            candidates.append((i, v_before.copy(), v_after.copy(), delta_v_norm))

    # ------------------------------------------------------------------
    # Step 2: deduplicate — merge candidates within window_frames of each other
    # ------------------------------------------------------------------
    deduped = []
    for cand in candidates:
        if deduped and cand[0] - deduped[-1][0] < window_frames:
            # Keep the one with larger discontinuity
            if cand[3] > deduped[-1][3]:
                deduped[-1] = cand
        else:
            deduped.append(cand)

    # ------------------------------------------------------------------
    # Step 3: classify and build events
    # ------------------------------------------------------------------
    events = []
    for frame_idx, v_before, v_after, _ in deduped:
        puck_pos = puck_positions[frame_idx]
        paddle_pos = paddle_positions[frame_idx]

        # Paddle proximity check
        dist_to_paddle = float(np.linalg.norm(puck_pos - paddle_pos))
        if dist_to_paddle < paddle_proximity_radius:
            collision_type = "paddle"
        # Wall proximity check (with margin for post-bounce displacement)
        elif (puck_pos[0] <= wall_bounds["x_min"] + wall_proximity_margin or
              puck_pos[0] >= wall_bounds["x_max"] - wall_proximity_margin or
              puck_pos[1] <= wall_bounds["y_min"] + wall_proximity_margin or
              puck_pos[1] >= wall_bounds["y_max"] - wall_proximity_margin):
            collision_type = "wall"
        else:
            continue  # noise or occlusion artifact

        speed_before = float(np.linalg.norm(v_before))
        speed_after = float(np.linalg.norm(v_after))
        tier = speed_tier(speed_before)

        events.append(CollisionEvent(
            frame_idx=frame_idx,
            collision_type=collision_type,
            speed_before=speed_before,
            speed_after=speed_after,
            velocity_before=v_before,
            velocity_after=v_after,
            tier=tier,
        ))

    return events

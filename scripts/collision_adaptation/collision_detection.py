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

from typing import Sequence


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

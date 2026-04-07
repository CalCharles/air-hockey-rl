# Collision Detection — Bugs & Plan for Position-Only Detection

Status: working notes, 2026-04-06

Goal: detect collisions from **raw position trajectories only** (no Box2D callbacks), so the same code works on real-world camera data.

---

## Current architecture

| Layer | File | What it does | Uses privileged data? |
|-------|------|-------------|----------------------|
| Box2D listener | `airhockey/sims/airhockey_box2d.py:106-420` (`CollisionForceListener`) | Hooks `PreSolve`/`PostSolve`, records contacts, applies deterministic restitution impulses | Yes — IS the privileged source |
| Step detector | `scripts/collision_adaptation/collision_detection.py` (`StepCollisionDetector`) | Tracks growth of `collision_forces` list, filters paddle-puck entries | Yes — reads `get_collision_forces()` |
| Position-based rollout | `scripts/collision_adaptation/rollout_position_based.py` | Estimates pre/post speeds from `puck_history` positions | **Hybrid** — detection still uses `get_collision_forces()`, only speed estimation is position-based |
| Replay rollout | `scripts/collision_adaptation/rollout_replay.py` | Collects oracle scenarios, replays in learner | Yes — detection via `get_collision_forces()`, paddle vel from Box2D body |

---

## Bugs

### ~~1. `collision_detection.py` is unused by position-based code~~ — FIXED

All callers now import `StepCollisionDetector` and `is_paddle_puck_collision` from `collision_detection.py`.

### 2. No actual position-based collision detection exists

Despite the name, `rollout_position_based.py` still calls `sim.get_collision_forces()` to know *when* collisions happen. It only uses positions for *speed estimation*. For real-world data there are no Box2D callbacks, so collision detection itself must come from positions.

### ~~3. Velocity extrapolation bias at collision boundary~~ — FIXED

`_estimate_collision_speeds` now extrapolates `v_at_end` forward by one `dt` using gravity before computing `speed_before`.

### ~~4. Approach speed definition mismatch~~ — DOCUMENTED

Center-to-center and contact-normal are equivalent for circular bodies (which is all we have). Added a note to `_estimated_approach_speed` flagging that non-circular shapes would require an update.

### ~~5. Duplicated constants~~ — FIXED

`TIERS`, `SPEED_BREAKPOINTS`, `PUCK_HISTORY_PAD`, and `speed_tier()` now live in `collision_detection.py` as the single source of truth. All consumers import from there.

---

## Plan: position-only collision detection

The goal is a detector that takes only `(N, 2)` puck positions + `(N, 2)` paddle positions (+ timestamps) and returns collision events with estimated pre/post speeds.

### Collision signature in position data

A collision manifests as a **velocity discontinuity**: the puck's velocity vector changes abruptly between adjacent frames. Two ingredients:

1. **Velocity jump**: fit velocity in a sliding window before and after each candidate frame. Flag frames where speed or direction changes beyond a threshold.
2. **Proximity**: the puck must be near a wall boundary or the paddle at the discontinuity frame. This disambiguates collisions from noise/occlusion.

### Proposed interface

```python
@dataclass
class CollisionEvent:
    frame_idx: int              # index in the trajectory
    collision_type: str         # "paddle" or "wall"
    speed_before: float         # m/s, estimated
    speed_after: float          # m/s, estimated
    velocity_before: np.ndarray # (2,) m/s
    velocity_after: np.ndarray  # (2,) m/s
    tier: str                   # "low" / "mid" / "high"

def detect_collisions_from_positions(
    puck_positions: np.ndarray,     # (N, 2)
    paddle_positions: np.ndarray,   # (N, 2)
    times: np.ndarray,              # (N,)
    valid_mask: np.ndarray | None,  # (N,) bool, None = all valid
    gravity: tuple[float, float] = (-0.65, 0.0),
    window_frames: int = 10,
    min_snr: float = 8.0,
    speed_change_threshold: float = 0.1,   # m/s min |delta_v| to flag
    angle_change_threshold: float = 30.0,  # degrees
    paddle_proximity_radius: float = 0.05, # metres
    wall_bounds: tuple = None,             # table boundaries for wall detection
) -> list[CollisionEvent]:
    ...
```

### Detection algorithm sketch

1. For each frame `i` in `[window_frames, N - window_frames)`:
   - Fit velocity in `[i - window_frames, i)` → `v_before`
   - Fit velocity in `[i, i + window_frames)` → `v_after`
   - Compute `delta_speed = |v_after| - |v_before|` (or use vector difference)
   - Compute `angle_change = angle(v_before, v_after)`
2. Flag frame `i` as a candidate if `|delta_v| > threshold` OR `angle_change > threshold`
3. Classify candidate:
   - If `||puck_pos[i] - paddle_pos[i]|| < paddle_proximity_radius` → paddle collision
   - If puck is near a wall boundary → wall collision
   - Otherwise → discard (noise, occlusion artifact)
4. Deduplicate: merge candidates within `window_frames` of each other (keep the one with largest discontinuity)
5. Return `CollisionEvent` list

### Where this should live

`scripts/collision_adaptation/collision_detection.py` — extend the existing module. Keep `StepCollisionDetector` for sim-only use, add `detect_collisions_from_positions` alongside it. The rollout scripts should then use the position-based detector instead of `get_collision_forces()`.

### Open questions

- What paddle proximity radius works? Depends on paddle + puck radii in sim config.
- Should we use a residual-based detector (fit one window across the candidate, check if residual spikes) instead of two-window velocity comparison?
- How to handle occlusion runs? If `valid_mask` has gaps near a collision, the windows shrink. Need minimum valid frames.
- Wall boundary coordinates — these are in the sim config but need to be passed in or looked up for real-world data.

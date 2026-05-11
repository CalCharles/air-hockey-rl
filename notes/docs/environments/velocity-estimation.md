# Puck Velocity Estimation from Noisy Positions

Part of the sim-real alignment pipeline. Before restitution parameters can be fitted,
reliable velocity estimates are needed from position trajectories that are noisy,
occasionally occluded, and observation-delayed — without privileged access to
the Box2D body's true velocity.

---

## Problem statement

The real-world tracking pipeline (and Box2D's simulated observation pipeline) provides:
- Puck positions at each timestep: Gaussian noise σ = 0.01 m, plus occlusion runs
- Observation delay: 0.025 s ± 25% jitter (`enable_observation_delay: true`)
- No direct velocity measurement

Naive finite differences amplify noise badly. Over a 10-step (0.5 s) window,
regression across all frames reduces velocity uncertainty by ~4× vs single-frame
differencing.

---

## Approach: gravity-corrected linear regression

The puck follows a kinematic model with constant acceleration (gravity) per axis:

```
pos(t) = pos₀ + v₀·(t−t₀) + 0.5·a·(t−t₀)²
```

Rearranging removes the gravity term and leaves a linear system:

```
pos_corr(t) = pos(t) − 0.5·a·(t−t₀)²  =  pos₀ + v₀·(t−t₀)
```

Fitting `pos_corr = pos₀ + v₀·dt` via weighted least squares (weight=0 for occluded
frames) gives `v₀` at the window start, from which velocity at any time follows:

```
v(t) = v₀ + a·(t−t₀)
```

### Gravity axis — Box2D coordinate system

> **Critical:** gravity is **not** in the y-axis of base coordinates.

Box2D world gravity = `(0, −0.65)` in Box2D frame.  
Coordinate conversion: `x_base = −y_b2d`, `y_base = x_b2d`.

Result:
- **`x_base`** = length direction (±0.965 m) — gravity **accelerates** puck in +x
- **`y_base`** = width direction  (±0.432 m) — no gravity

The estimator's `gravity` parameter encodes **deceleration** (positive = slows down).
Acceleration in +x means deceleration = −0.65, so pass:

```python
gravity = (-0.65, 0.0)   # (gx, gy) deceleration in base coords
```

For a vertically-oriented system where gravity decelerates upward motion, pass
`gravity = (0.0, 9.8)`. For a flat real-world table, pass `gravity = (0.0, 0.0)`.

---

## Implementation

**Module:** `airhockey/sims/real/velocity_estimator.py`

### `fit_velocity_from_positions`

```python
result = fit_velocity_from_positions(
    positions,        # (N, 2) — observed (x, y) in base coords
    times,            # (N,)   — timestamps in seconds
    valid_mask=None,  # (N,)   — True for non-occluded frames
    gravity=(0.0, 0.0),  # (gx, gy) deceleration in m/s²
)
```

Returns a dict (or `None` if fewer than 2 valid frames):

| Key | Shape | Description |
|-----|-------|-------------|
| `v_at_end` | `(2,)` | `[vx, vy]` at `times[-1]` — the collision moment |
| `v_at_times` | `(N, 2)` | Velocity evaluated at each input timestep |
| `snr` | float | `|v_at_end| / residual_noise` — higher = more trustworthy |
| `n_valid` | int | Number of non-occluded frames used in the fit |

### `extract_pre_collision_velocities`

Convenience wrapper for the restitution fitting loop:

```python
result = extract_pre_collision_velocities(
    positions, times, valid_mask,
    collision_idx,       # last frame before collision
    window_frames=10,    # frames to use (default 10 ≈ 0.5 s at 20 Hz)
    gravity=(-0.65, 0.0),
)
```

### Typical usage

```python
from airhockey.sims.real.velocity_estimator import (
    fit_velocity_from_positions,
    extract_pre_collision_velocities,
)

# Before collision (10 frames, 20 Hz, Box2D sim)
result = extract_pre_collision_velocities(
    positions, times, valid_mask,
    collision_idx=collision_idx,
    window_frames=10,
    gravity=(-0.65, 0.0),
)

v_before = result["v_at_end"]      # velocity at collision moment
snr      = result["snr"]           # use to weight/skip low-quality estimates
```

---

## Validation results (Box2D sim)

Evaluated against ground-truth Box2D body velocities across 7 collision-free
trajectories varying speed and direction. 10-step windows (0.5 s) at 20 Hz,
noise std = 0.01 m, occlusion rate ≈ 2.5%, observation delay ≈ 0.025 s.

| Scenario | GT speed | Abs err | **Rel err** | SNR |
|---|---|---|---|---|
| Slow along length (0.2 m/s) | 0.45 m/s | 0.031 m/s | **6.8%** | 48 |
| Medium along length (0.6 m/s) | 0.81 m/s | 0.075 m/s | **9.3%** | 82 |
| Fast along length (1.2 m/s) | 1.35 m/s | 0.114 m/s | **8.5%** | 114 |
| Medium against gravity (−0.6 m/s) | 0.26 m/s | 0.022 m/s | **8.5%** | 24 |
| Diagonal length+width | 0.76 m/s | 0.062 m/s | **8.2%** | 89 |
| Diagonal against gravity+width | 0.28 m/s | 0.018 m/s | **6.3%** | 27 |
| Pure width direction (0.3 m/s) | 0.38 m/s | 0.053 m/s | **13.8%** | 30 |

**Typical relative error: ~8%.** Error is roughly speed-independent, indicating it is
dominated by the observation delay (~0.025 s timing offset) rather than position noise.

The pure-width-direction outlier (13.8%) has no gravity correction and lower SNR;
at slow speeds in that axis the noise-to-signal ratio is less favourable.

### Implications for restitution fitting

- 8% velocity error → similar relative error in the restitution ratio |v_after|/|v_before|
- Gradient descent over many collisions averages out random errors
- **Downweight or skip** estimates with SNR < ~20 (slow or heavily occluded collisions)
- The delay causes a consistent ~`delay × |v|` positional offset; since both pre- and
  post-collision windows are equally affected, the ratio is largely preserved

---

## Visualization scripts

Two scripts in `notes/scratch/` for development/verification:

| Script | Purpose |
|--------|---------|
| `viz_velocity_estimator.py` | Synthetic ground-truth examples (6 panels) |
| `collect_and_evaluate_velocity_estimator.py` | Live Box2D trajectories vs GT body velocity |

Run from repo root:
```bash
source .venv/bin/activate
python3 notes/scratch/viz_velocity_estimator.py
python3 notes/scratch/collect_and_evaluate_velocity_estimator.py
# Output: notes/scratch/velocity_estimator_viz/
```

Each plot shows: raw observed positions (blue), gravity-corrected positions (green +),
fitted trajectory (red), estimated velocity arrows (blue), and GT velocity arrows (orange).

---

## Pipeline context

```
real HDF5 trajectory  /  Box2D episode
        │
  detect collision event (collision_idx)   ← not yet implemented
        │
  extract_pre_collision_velocities(...)    ← this module
        │
  { v_at_end, v_at_times, snr }
        │
  compare v_before / v_after              ← restitution = |v_after_n| / |v_before_n|
        │
  nudge params via external optimizer       ← bridge module no longer in tree
```

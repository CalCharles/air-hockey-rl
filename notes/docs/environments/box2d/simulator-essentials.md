# Box2D simulator essentials



Primary simulator file: [`airhockey/sims/airhockey_box2d.py`](../../../../airhockey/sims/airhockey_box2d.py).

## Bound-related configs (important)

- **PID workspace clipping** (`use_pid=True` path), **inside Box2D only**:
  - `x_min_lim`, `x_max_lim`, `y_min`, `y_max`
  - `top_abs`, `bot_abs`, `max_bias_p`, `max_bias_m`
  - These become `lims` / `edge_lims` in [`airhockey_box2d.py`](../../../../airhockey/sims/airhockey_box2d.py); applied by `_clip_limits()` and `_clip_pid_target_to_workspace()`.
  - Keys `paddle_bounds` / `paddle_edge_bounds` may appear under `simulator_params` (the env copies them for API parity) but **this simulator does not read them**.
- **Per-step movement limits** (not global workspace):
  - `rmax_x`, `rmax_y` (stored as `move_lims`)
  - Applied in `_compute_pid_target_pos()` via `_get_edge()`.
- **Physical table limits**:
  - From `length`, `width` → `table_x_min/max`, `table_y_min/max`.
  - These are hard world walls in Box2D.

## Two layers: sim vs env (paddle limits)

1. **Box2D** (`air_hockey.simulator_params`): `x_min_lim` … `y_max` and `top_abs` … `max_bias_m` constrain targets inside the sim (PID / transition). See class docstring on `AirHockeyBox2D`.

2. **Base env** ([`airhockey_base.py`](../../../../airhockey/airhockey_base.py)): top-level `air_hockey.paddle_bounds` (rectangle `x_min`, `x_max`, `y_min`, `y_max`) and `air_hockey.paddle_edge_bounds` (`top_abs`, `bot_abs`, `max_bias_p`, `max_bias_m`) feed `get_clip_limits()` in `single_agent_step()` and **clip the policy action before** `simulator.get_transition()`.

Keep YAML values consistent across both places; otherwise pre-step action gating and in-sim clipping can disagree. If `paddle_bounds` / `paddle_edge_bounds` are omitted, the env falls back to table-wide defaults and loose edge defaults, while Box2D still uses its own `lims`.

## Practical maximum reach (default table)

Given default Box2D dimensions in [`airhockey/sims/airhockey_box2d.py`](../../../../airhockey/sims/airhockey_box2d.py):

- `length = 1.9304`, `width = 0.8636`, `paddle_radius = 0.0508`
- Table center-coordinate extents:
  - `x in [-0.9652, 0.9652]`
  - `y in [-0.4318, 0.4318]`
- Practical paddle-center extents (subtract radius from walls):
  - `x in [-0.9144, 0.9144]`
  - `y in [-0.3810, 0.3810]`

If configured bounds exceed these, reachable motion is still capped by table walls.

## Coordinate reminder

- Internal Box2D and base coordinates are converted through:
  - `base_coord_to_box2d()`
  - `convert_from_box2d_coords()`
- Keep frame conventions consistent when adding or diagnosing bounds logic.

## Delay toggles and shared delay value

- Box2D supports independent toggles for:
  - `enable_action_delay`
  - `enable_observation_delay`
- Both features use the same `delay_seconds` value (clamped per step to `[0, time_per_step]`).
- The delay value is fixed — the older `randomize_delay` / `delay_relative_range` per-step jitter mechanism was removed in the 2026-05-11 randomization cleanup.
- Observation-delay snapshots can show stale paddle `acceleration`/`jerk` fields because those derivatives are refreshed after the full step; positions/velocities remain mid-step consistent.

### ⚠ Subtle side effect: `enable_observation_delay` changes `puck_history` sampling rate

`get_singleagent_transition` (`airhockey_box2d.py:1681`) splits each 20 Hz
env step into one or more sub-steps based on the breakpoints
`{0, t_obs, t_action, time_per_step}`. The paddle/puck history lists
are appended **inside** that sub-step loop (line 1830 / 1837), so the
number of history entries per env step depends on which delay toggles
are on:

| Config | Sub-steps / env step | Appends / env step | Effective sampling rate |
|---|---:|---:|---:|
| `enable_observation_delay: true`  (canonical baseline) | 2 | 2 | ~40 Hz |
| `enable_observation_delay: false` (e.g. zero-shot ablation) | 1 | 1 | 20 Hz |

The 20 Hz **env / control step** is unchanged by either setting (every
`step()` advances sim time by exactly `time_per_step = 1/20 s`). What
changes is the temporal *density* of the puck/paddle history that the
policy reads via `puck_history[-5:]` / `paddle_history[-5:]` (see
[`obs construction`](../observation-action-spaces.md#temporal-density-caveat)).
The real-world simulator (`air_hockey_real.py:1549`) does not have a
sub-step loop and appends exactly once per 20 Hz env step — so it
matches the `enable_observation_delay: false` density, **not** the
canonical training density.

## Paddle-puck contact caveat (important)

Empirically in this Box2D setup, paddle-puck outcomes are often less stable than expected from ideal rigid-body intuition:

- Contact behavior is highly sensitive to **relative velocity at impact** (especially along the collision normal).
- Changing `paddle_density` / `puck_density` can matter, but in many practical runs it has a weaker effect than impact timing and approach speed.
- Small differences in pre-contact motion (controller force limits, damping, jitter cadence, action lag, and step timing) can dominate the post-contact result.

Practical guidance:

- Treat density sweeps as a secondary knob for tuning contact behavior.
- First tune and compare pre-contact velocity profiles and impact timing, then use density for finer adjustment.

## Collision physics

All collision restitution is handled by a custom `CollisionForceListener` that **disables Box2D's built-in restitution** and re-applies it deterministically in `PostSolve`. This avoids jitter from Box2D's global `b2_velocityThreshold`.

### Bodies and restitution values

| Body | Shape | Restitution (fixture) |
|---|---|---|
| Puck | Circle (`puck_radius`) | `puck_restitution` (e.g. 1.09145) |
| Paddle | Circle (`paddle_radius`) | `paddle_restitution` (default 1.0) |
| Side walls (left/right) | `b2EdgeShape` | `side_wall_restitution` (e.g. 0.99) |
| End walls (top/bottom) | `b2EdgeShape` | `end_wall_restitution` (e.g. 0.70) |

Friction is 0.0 everywhere. Gravity is a downward `gravity` value (e.g. `-0.65 m/s²`) simulating table tilt.

### Puck ↔ wall

**PreSolve:** computes `incoming_speed` (puck's normal-component speed toward the wall), disables Box2D's restitution (`contact.restitution = 0.0`), and stores `{incoming_speed, normal_inward, restitution}` keyed by puck name.

**PostSolve:** reads the stored entry and applies a corrective impulse:
- If `incoming_speed >= puck_wall_restitution_threshold_speed` (default 0.25 m/s): `v_out = incoming_speed * restitution`
- If `incoming_speed < threshold`: enforces a minimum rebound of `puck_wall_min_rebound_speed_below_threshold` (default 0.1 m/s)

The impulse is `J = mass * (target_outgoing - current_outgoing)` applied along the inward normal.

### Puck ↔ paddle

**PreSolve:** computes relative approach speed of puck w.r.t. paddle along the contact normal. Uses `combined_e = max(puck_restitution, paddle_restitution)`. Disables Box2D restitution and stores state in `_pending_paddle_puck`.

**PostSolve:** enforces `v_rel_desired = e * approach_speed` via a momentum-conserving reduced-mass impulse:

```
j = (v_rel_desired - v_rel_post) * m_paddle * m_puck / (m_paddle + m_puck)
```

Applied as `+j` to puck and `-j` to paddle along the contact normal.

A `puck_restitution > 1.0` (e.g. 1.09145) means the puck leaves slightly faster than it arrived — simulating a springy puck.

### Paddle force model

The paddle is a **dynamic body** driven by a PID controller (`pid_kp`, `pid_kd`, `pid_ki`). Each step the policy outputs a target position; the PID computes a force that is applied to the paddle body. The paddle's velocity at contact time is the result of this accumulated force, so the collision outcome depends on PID tuning and the step timing. `paddle_damping` provides heavy deceleration between steps.

### Quick-reference with `sysid_best_params_hist2.yaml`

```
Puck-side wall:   e = 0.99,   threshold = 0.25 m/s,  min rebound = 0.1 m/s
Puck-end wall:    e = 0.70,   same threshold logic
Puck-paddle:      e = 1.09145  (superelastic — puck gains energy from springiness)
Puck damping:     0.25
Paddle damping:   17
Gravity:         -0.65 m/s²
PID gains:        Kp=5000, Kd=200, Ki=0
```

## Per-tier collision scaling and external parameter interface

The simulator supports per-speed-tier restitution multipliers that an external optimizer (or another RL agent) can update during training without restarting.

### Speed tiers

| Tier | Incoming speed range |
|---|---|
| `low` | < 0.25 m/s |
| `mid` | 0.25 – 0.75 m/s |
| `high` | ≥ 0.75 m/s |

Breakpoints are configurable. The effective restitution for each collision is `base_restitution * scale[tier]`.

### Simulator API

```python
# Push new scales at an episode boundary.
sim.set_collision_scales(
    wall_scales=[1.0, 1.0, 1.0],    # [low, mid, high]
    paddle_scales=[1.0, 1.0, 1.0],
    speed_breakpoints=(0.25, 0.75), # optional, m/s
)

# Read per-tier stats accumulated during the episode (resets counters).
stats = sim.get_episode_collision_stats()
# stats = {
#   "wall":   {"low": {"count", "mean_speed_in", "mean_speed_out"}, ...},
#   "paddle": {"low": ..., "mid": ..., "high": ...},
# }
```

### Status file schema (legacy / unused)

The simulator can still emit per-tier collision stats via `get_episode_collision_stats()`, but the external `CollisionParamManager` bridge that originally consumed them was removed in the May 2026 cleanup. The schema below documents the intended status JSON in case the bridge is revived.



```json
{
  "episode": 1042,
  "current_params": {
    "wall_scales": [1.0, 1.0, 1.0],
    "paddle_scales": [1.0, 1.0, 1.0],
    "speed_breakpoints": [0.25, 0.75]
  },
  "collision_stats": {
    "wall": {
      "low":  {"count": 12, "mean_speed_in": 0.14, "mean_speed_out": 0.10},
      "mid":  {"count": 34, "mean_speed_in": 0.48, "mean_speed_out": 0.47},
      "high": {"count":  8, "mean_speed_in": 0.91, "mean_speed_out": 0.95}
    },
    "paddle": {
      "low":  {"count":  3, "mean_speed_in": 0.11, "mean_speed_out": 0.12},
      "mid":  {"count": 21, "mean_speed_in": 0.41, "mean_speed_out": 0.44},
      "high": {"count":  9, "mean_speed_in": 0.88, "mean_speed_out": 0.98}
    }
  },
  "episode_outcome": {
    "total_reward": 12.4,
    "juggle_count": 7,
    "termination_reason": "puck_hit_bottom"
  }
}
```

## Puck observation sine y-warp (sim2sim perception error)

Edge-preserving sine warp on the puck's `y` (sideways) observation only. Models a partially-calibrated overhead tracker — corners anchored to the table side walls, interior reads bow off-true. Lives in `airhockey/observation_homography.py:apply_sine_y_warp_xy` and is plumbed through `airhockey/utils.py` via the `puck_obs_warp_fn` kwarg.

```
y_obs = y_true + A · sin(π · (y_true − y_left) / (y_right − y_left))
x_obs = x_true                                    # x is unchanged
```

- Edges preserved: `y_obs == y_true` at both side walls.
- Peak deviation `+A` at the midline (`y = 0`).
- Monotonic iff `|A| < (y_right − y_left) / π ≈ 0.275 m` at full table width. Enforced by `make_sine_y_warp_fn`.
- Paddle observations untouched. Physics untouched (collisions still resolve at the *true* puck position; only what gets written into the puck-history slots of the obs vector is warped).

**Config keys** (in `air_hockey.simulator_params`):

| Key | Default | Meaning |
|---|---:|---|
| `puck_obs_sine_warp_amplitude` | `0.0` | `A` in meters. `0.0` = warp disabled (no-op, `puck_obs_warp_fn` is `None`). |
| `puck_obs_sine_warp_y_left`  | `null` | Left edge of the warp domain. `null` defaults to `−width/2`. |
| `puck_obs_sine_warp_y_right` | `null` | Right edge. `null` defaults to `+width/2`. |

**Canonical sim2sim target using this warp**: `configs/new_juggle/sim2sim_warp075_p30.yaml` (paddle −30% + warp 0.075). See [`notes/scratch/experiments/2026-05-07_02-05_sim2sim-puck-obs-warp.md`](../../../scratch/experiments/2026-05-07_02-05_sim2sim-puck-obs-warp.md) for the rationale and the visualization at `/tmp/sine_warp_viz.png`.

The older `obs_position_homography` (3×3 perspective matrix applied to both paddle and puck) was removed in favor of this puck-only mechanism. See the experiment writeup for what was removed.


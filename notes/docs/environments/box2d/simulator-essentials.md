# Box2D simulator essentials

Mirror of [`.cursor/rules/project-box2d-simulator.mdc`](../../../../.cursor/rules/project-box2d-simulator.mdc).

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

- Box2D now supports independent toggles for:
  - `enable_action_delay`
  - `enable_observation_delay`
- Both features use the same `delay_seconds` value (clamped per step to `[0, time_per_step]`).
- Optional fluctuation:
  - `randomize_delay` enables per-step randomization of the realized delay.
  - `delay_relative_range` controls multiplicative half-width (for example `0.25` means `delay_seconds * [0.75, 1.25]` before clamping).
- Observation-delay snapshots can show stale paddle `acceleration`/`jerk` fields because those derivatives are refreshed after the full step; positions/velocities remain mid-step consistent.

## Paddle-puck contact caveat (important)

Empirically in this Box2D setup, paddle-puck outcomes are often less stable than expected from ideal rigid-body intuition:

- Contact behavior is highly sensitive to **relative velocity at impact** (especially along the collision normal).
- Changing `paddle_density` / `puck_density` can matter, but in many practical runs it has a weaker effect than impact timing and approach speed.
- Small differences in pre-contact motion (controller force limits, damping, jitter cadence, action lag, and step timing) can dominate the post-contact result.

Practical guidance:

- Treat density sweeps as a secondary knob for tuning contact behavior.
- First tune and compare pre-contact velocity profiles and impact timing, then use density for finer adjustment.

Evidence reference:

- Contact-scenario implementation and metrics collection: [`scripts/box2d_paddle_puck_contact_scenario.py`](../../../../scripts/box2d_paddle_puck_contact_scenario.py) (see parsing of `paddle:puck` sweep pairs, per-run `paddle_density` / `puck_density` recording, and pre/post-contact speed logging).

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

### Quick-reference with `pid_noise_constant_upper_half_custom_sim_params.yaml`

```
Puck-side wall:   e = 0.99,   threshold = 0.25 m/s,  min rebound = 0.1 m/s
Puck-end wall:    e = 0.70,   same threshold logic
Puck-paddle:      e = 1.09145  (superelastic — puck gains energy from springiness)
Puck damping:     0.25
Paddle damping:   17
Gravity:         -0.65 m/s²
PID gains:        Kp=5000, Kd=200, Ki=0
```

## Observation homography (sim-to-real)

When `obs_position_homography` is enabled in the simulator config, observations are warped through a perspective homography matrix before reaching the policy. This simulates camera-like positional distortion for sim-to-real transfer training.

- The homography matrix can be overridden via training args.
- [`validate_obs_homography_gif.py`](../../../../scripts/smooth_policy/validate_obs_homography_gif.py) renders world-space frames with the homography warp applied, for visual verification.

This is a separate feature from the real camera homography documented in [`../real-world/homography.md`](../real-world/homography.md).

## Paddle boundary visualization utility

For Box2D boundary debugging and validation, use:

- [`scripts/box2d_boundary_validation/validate_paddle_bounds_gif.py`](../../../../scripts/box2d_boundary_validation/validate_paddle_bounds_gif.py)

This script renders a looped GIF of the paddle tracing the effective boundary and overlays the actual clipped boundary polygon (including corner cuts from edge-shaping).

### Parameters (brief)

- **Workspace bounds**
  - `--x-min`, `--x-max`, `--y-min`, `--y-max`: rectangle-style workspace limits to test.
- **Corner / edge shaping**
  - `--top-abs`: slope factor for right-side corner tapering versus `y`.
  - `--max-bias-m`, `--max-bias-p`: right-edge bias terms used in `x_max(y) = min(x_max, max_bias_m - top_abs*y, max_bias_p + top_abs*y)`.
  - `--bot-abs`: accepted for parity with config, but current clip math keeps `x_min` fixed and does not currently apply this term.
- **Coordinate-frame interpretation**
  - `--limits-frame raw_robot|centered`: whether input x-limits are raw-robot style (converted by `center_offset_constant`) or already centered.
- **Traversal controls**
  - `--loops`: number of perimeter loops.
  - `--steps-per-edge`: sampling density along each perimeter segment.
  - `--control-substeps`: inner control updates per waypoint.
  - `--action-scale`: per-update movement magnitude scaling.
  - `--position-tol`: tolerance used for boundary-touch and violation checks.
- **Rendering / outputs**
  - `--fps`: GIF framerate.
  - `--name`, `--output-dir`: output naming/path.
  - `--renderer-orientation`: renderer orientation (project rule defaults this to `vertical`).
  - `--config-path`: YAML source of `air_hockey.simulator_params`.

### Artifacts

- GIF: `runs/paddle_boundary_validation/<name>.gif`
- Summary JSON: `runs/paddle_boundary_validation/<name>.json`

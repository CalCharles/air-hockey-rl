# Box2D simulator essentials

Mirror of [`.cursor/rules/project-box2d-simulator.mdc`](../../../../.cursor/rules/project-box2d-simulator.mdc).

Primary simulator file: [`airhockey/sims/airhockey_box2d.py`](../../../../airhockey/sims/airhockey_box2d.py).

## Bound-related configs (important)

- **PID workspace clipping** (`use_pid=True` path):
  - `x_min_lim`, `x_max_lim`, `y_min`, `y_max`
  - `top_abs`, `bot_abs`, `max_bias_p`, `max_bias_m`
  - Applied by `_clip_limits()` and `_clip_pid_target_to_workspace()`.
- **Per-step movement limits** (not global workspace):
  - `rmax_x`, `rmax_y` (stored as `move_lims`)
  - Applied in `_compute_pid_target_pos()` via `_get_edge()`.
- **Physical table limits**:
  - From `length`, `width` → `table_x_min/max`, `table_y_min/max`.
  - These are hard world walls in Box2D.

## Env-level bounds interaction

- Environment-level action clipping is handled in [`airhockey/airhockey_base.py`](../../../../airhockey/airhockey_base.py), using:
  - `paddle_bounds` (x/y min/max)
  - `paddle_edge_bounds` (edge shaping)
- This clipping occurs in `single_agent_step()` before calling simulator transition.

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

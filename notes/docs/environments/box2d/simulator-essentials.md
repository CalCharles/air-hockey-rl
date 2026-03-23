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

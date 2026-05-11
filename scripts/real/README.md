# Real-robot helpers

Scripts and utilities that talk to the UR5 + camera stack, called by the async TD3 real-world entrypoints under [`scripts/td3/extras/`](../td3/extras/).

| File | Role |
|------|------|
| `rollout_reset_policy_real.py` | Reset-policy FSM. Imported by all three active real-world entrypoints (`async_td3_real.py`, `async_td3_real_eval.py`, `async_td3_real_teleop_eval.py`). |
| `agent.py` | Lightweight environment wrapper consumed by `rollout_reset_policy_real.py`. |
| `teleoperate.py` | Mouse-paddle teleop control on the physical robot. |
| `calibrate_robo_camera.py` | Camera calibration helper. |
| `aruco_detection.py` | ArUco puck-tracking utilities. |
| `generate_homography.py` | Build `Mimg.npy` / `Mrob.npy` from hand-chosen point pairs (see [`notes/docs/environments/real-world/homography.md`](../../notes/docs/environments/real-world/homography.md)). |
| `visualize_saved_trajectory.py` | Replay a saved HDF5 trajectory. |

## Sourced ROS env: avoiding the `scripts` package collision

Real-world entry points often run in a shell where `/opt/ros/iron/setup.bash` has been sourced. ROS prepends `/opt/ros/iron/lib/python3.10/site-packages` to `PYTHONPATH`, and that directory contains its own `scripts/` package. When you launch a script directly (`python scripts/real/foo.py`), Python's `sys.path` becomes:

1. `…/scripts/real` (the script's own directory)
2. ROS `PYTHONPATH` entries (including the conflicting `scripts/`)
3. The repo root, *but only because* `easy-install.pth` added it from `pip install -e .` (so it lands later, behind ROS)

A naive guard like `if str(REPO_ROOT) not in sys.path: sys.path.insert(0, ...)` is a no-op here — `REPO_ROOT` is already on `sys.path`, just in the wrong position — and `import scripts.td3.agent` resolves to the ROS package, which fails with `ModuleNotFoundError: No module named 'catkin_pkg'`.

Two ways to avoid this:

- Run with `python -m scripts.real.foo` from the repo root; `-m` puts the cwd at `sys.path[0]`, so the local `scripts` package wins.
- Or, at the very top of the script — **before any other imports** — force the repo root to position 0 unconditionally:

  ```python
  import sys
  from pathlib import Path

  REPO_ROOT = Path(__file__).resolve().parents[2]
  _REPO_ROOT_STR = str(REPO_ROOT)
  while _REPO_ROOT_STR in sys.path:
      sys.path.remove(_REPO_ROOT_STR)
  sys.path.insert(0, _REPO_ROOT_STR)
  ```

Use this pattern for any new entry point that imports `scripts.*`.

## Transition holds

There are two different "slowdown / smoothing" mechanisms in the real stack:

- **Async TD3 transition holds** — shared runtime library `scripts/td3/helper/real_td3_runtime.py` defines the `Args` knobs; the collector entrypoint `scripts/td3/extras/async_td3_real.py` drives the holds via `helper/real_transition_hold.py`. Used for reset-to-policy handoff, actor sync, and genuine safety recovery.
- **Rollout startup / cooldown logic** — `rollout_reset_policy_real.py`:
  - `--startup-hold-steps` forces zero action for the first few normal-mode steps.
  - `reset_cooldown` prevents immediate re-entry into reset mode after leaving it.
- **Simulator-side holds** — command-path recovery smoothing in `airhockey/sims/air_hockey_real.py`. YAML keys under `air_hockey.simulator_params`:
  - `transition_hold_steps_on_estop_enter`
  - `transition_hold_steps_on_estop_clear`
  - `transition_hold_steps_on_safety_rearm`
  - `transition_hold_debug`

## Keypresses during rollout

The async real-world rollout loop reads keypresses:

- `y` — save current trajectory, then reset
- `q` — reset without saving
- `x` — exit script

### What happens on `y`

The entrypoint calls `eval_env.reset(seed=None, write_traj=True)`. In `airhockey/sims/air_hockey_real.py`, `reset(..., write_traj=True)`:
- merges buffered image/value frames with `merge_trajectory(...)`,
- writes one HDF5 trajectory file with `write_trajectory(...)`,
- increments the trajectory index for the next save,
- clears the temporary image buffer (`./temp/images/`).

### Where files are saved

Controlled by `air_hockey.simulator_params.save_path` in the rollout config; the async entrypoints set this via `--data-root-dir`. File naming is `trajectory_data{N}.hdf5`, where `N` starts from the next available index at startup.

### HDF5 contents

Each saved file contains:

- `train_img`: compressed image sequence (`gzip`, level 9), shape `[T, H, W, C]`.
- `train_vals`: compressed proprioceptive/state sequence (`gzip`, level 9), shape `[T, D]`.

`train_vals` columns (see `airhockey/sims/real/proprioceptive_state.py`): `cur_time(1) tidx(1) i(1) estop(1) safety(1) pose(6) speed(6) force(6) acc(3) desired_pose(6) puck(3)` — total width 35 per timestep.

## Real-robot workspace bounds (meters, robot frame)

From `airhockey/sims/air_hockey_real.py`:

- `x_min_lim = -0.8`, `x_max_lim = -0.33`
- `y_min = -0.3582`, `y_max = 0.350`

### Conversion to table-centered coordinates

The simulator uses `x_offset = 1.2` to convert robot-frame X to table-frame X:

- `table_x = robot_x + x_offset`
- `table_y = robot_y`

With that offset, the workspace maps to `table_x ∈ [0.40, 0.87]`, `table_y ∈ [-0.3582, 0.350]`. Table dimensions are `length = 1.9304` → `table_x ∈ [-0.9652, 0.9652]`, `width = 0.8636` → `table_y ∈ [-0.4318, 0.4318]`.

### Edge restriction on `x_max`

`clip_limits(...)` applies a y-dependent near-edge cap:

- `x_max = min(x_max_lim, max_bias_m - top_abs * y, max_bias_p + top_abs * y)`

So the allowed near-edge X shrinks at high `|y|` (corner cut-in). `x_min` is a fixed hard line.

## Action scaling: real vs Box2D

How the same 2D policy action `[ax, ay]` is handled differs between simulators.

### Real (`airhockey/sims/air_hockey_real.py`)

- In RL mode, `take_action(...)` interprets the action as a normalized delta and computes `move_vector = [ax, ay] * [rmax_x, rmax_y]`, target = `current_pose[:2] + move_vector`.
- Per-step scaling is controlled by `rmax_x = 0.26`, `rmax_y = 0.12`.
- `action_x_scaling` / `action_y_scaling` defaults exist but are not used in the real transition logic.
- `compute_rect(...)` / `compute_pol(...)` / `clip_limits(...)` enforce workspace limits, so the realized motion can be smaller than `[ax*rmax_x, ay*rmax_y]`.

### Box2D (`airhockey/sims/airhockey_box2d.py`)

- Before dynamics, action is converted with `convert_to_box2d_coords(...)`: `[ax, ay] → [ay, -ax]`.
- During transition, force is scaled by `force = force * [action_x_scaling, action_y_scaling]`.
- Scaling happens after coordinate conversion, so the scalings act on Box2D axes (post-transform), not on raw policy-frame `[ax, ay]`.

### Practical implication

To tune real rollouts, prioritize `rmax_x` / `rmax_y` plus workspace/safety clipping. To tune Box2D action responsiveness, use `action_x_scaling` / `action_y_scaling`. Matching behavior across real and Box2D requires calibrating both sets of parameters, not copying only one pair.

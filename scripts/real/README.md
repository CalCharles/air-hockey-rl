### Trajectory Saving for Real Rollouts

### Transition Hold Notes

There are two different "slowdown/smoothing" mechanisms in the real stack:

- Async TD3 transition holds
  - configured in `scripts/smooth_policy/amp_history/amp_training/td3/async_td3_real.py`
  - used for reset-to-policy handoff, actor sync, and genuine safety recovery
- Rollout startup/cooldown logic in `rollout_reset_policy_real.py`
  - `--startup-hold-steps` forces zero action for the first few normal-mode steps
  - `reset_cooldown` prevents immediate re-entry into reset mode after leaving it

These are not the same thing:

- `--startup-hold-steps` is a simple zero-action buffer in normal mode
- `reset_cooldown` is a reset-trigger debounce
- simulator `safety_rearm` / `estop_clear` holds are command-path recovery smoothing in `airhockey/sims/air_hockey_real.py`

If you want to tune the real simulator holds directly in YAML, the relevant keys live under `air_hockey.simulator_params`:

- `transition_hold_steps_on_estop_enter`
- `transition_hold_steps_on_estop_clear`
- `transition_hold_steps_on_safety_rearm`
- `transition_hold_debug`

When running `rollout_new.py`:

```bash
python scripts/real/rollout_new.py --config-path configs/real_configs/rollout_config.yaml --model <path_to_model>
```

or with an explicit save path override:

```bash
python scripts/real/rollout_new.py --config-path configs/real_configs/rollout_config.yaml --model <path_to_model> --save-path ./data/rollout/my_run
```

`rollout_constant.py` supports the same override:

```bash
python scripts/real/rollout_constant.py --config-path configs/real_configs/rollout_config.yaml --timesteps 200 --action 0.0 0.0 --save-path ./data/rollout/my_constant_run
```

keypress handling in the rollout loop is:

- `y`: save current trajectory, then reset
- `q`: reset without saving
- `x`: exit script

#### What happens when you press `y`
- `rollout_new.py` calls `eval_env.reset(seed=None, write_traj=True)`.
- In the real simulator (`airhockey/sims/air_hockey_real.py`), `reset(..., write_traj=True)`:
  - merges buffered image/value frames with `merge_trajectory(...)`
  - writes one HDF5 trajectory file with `write_trajectory(...)`
  - increments trajectory index for the next save
  - clears the temporary image buffer (`./temp/images/`)

#### Where files are saved
- Saved path is controlled by `air_hockey.simulator_params.save_path` in your config.
- You can override that path per run with `--save-path` in `rollout_new.py` or `rollout_constant.py`.
- Path precedence is:
  1. CLI `--save-path` (if provided)
  2. YAML `air_hockey.simulator_params.save_path`
- File naming format:
  - `trajectory_data{N}.hdf5` (for example `trajectory_data0.hdf5`, `trajectory_data1.hdf5`)
- `N` starts from the next available index in `save_path` (computed at startup).

#### HDF5 formatting
Each saved file contains:

- `train_img`: compressed image sequence (`gzip`, level 9), typically shaped like `[T, H, W, C]`
- `train_vals`: compressed proprioceptive/state sequence (`gzip`, level 9), shaped like `[T, D]`

`train_vals` columns follow this order (see `airhockey/sims/real/proprioceptive_state.py`):

- `cur_time` (1)
- `tidx` (1)
- `i` (1)
- `estop` (1)
- `safety` (1)
- `pose` (6)
- `speed` (6)
- `force` (6)
- `acc` (3)
- `desired_pose` (6)
- `puck` (3)

Total `train_vals` width is 35 per timestep.



Rolling out a constant policy:
python scripts/real/rollout_constant.py --config-path configs/real_configs/rollout_config.yaml --timesteps 150 --action 0.05 -0.02 --clip --save-path data/constant/action --auto-gif

### Demonstration imitation rollout

`rollout_imitation.py` replays a saved demonstration without any learned policy.
Control behavior is:

1. Move from current paddle position to the first demo paddle pose.
2. During rollout, find the closest demo paddle pose and apply the action used there.
3. Stop immediately when the final demo state is reached.

Example:

```bash
python scripts/real/rollout_imitation.py \
  --demo-hdf5 new_data/reset/demo1/trajectory_data0_timesteps_90_200.hdf5 \
  --config-path configs/real_configs/rollout_config.yaml \
  --verbose
```

Quick non-robot preview (no environment stepping):

```bash
python scripts/real/rollout_imitation.py \
  --demo-hdf5 new_data/reset/demo1/trajectory_data0_timesteps_90_200.hdf5 \
  --config-path configs/real_configs/rollout_config.yaml \
  --dry-run --dry-run-steps 12
```

### Real workspace bounds in meters

In `airhockey/sims/air_hockey_real.py`, the active real-controller limits are:

- `x_min_lim = -0.8`
- `x_max_lim = -0.33`
- `y_min = -0.3582`
- `y_max = 0.350`

These values are in the robot frame (same frame as `pose` and `desired_pose` in `train_vals`).

#### Conversion to table-centered coordinates

The simulator uses `x_offset = 1.2` to convert robot-frame X to table-frame X:

- `table_x = robot_x + x_offset`
- `table_y = robot_y`

With that offset, the workspace above maps to:

- `table_x in [0.40, 0.87]`
- `table_y in [-0.3582, 0.350]`

For reference, table dimensions in this config are:

- `length = 1.9304` -> `table_x in [-0.9652, 0.9652]`
- `width = 0.8636` -> `table_y in [-0.4318, 0.4318]`

#### Additional edge restriction on `x_max`

`clip_limits(...)` applies a y-dependent near-edge cap:

- `x_max = min(x_max_lim, max_bias_m - top_abs * y, max_bias_p + top_abs * y)`

So the allowed near-edge X can shrink at high `|y|` (corner cut-in), even if `x_max_lim` is unchanged.
Also note `x_min` is currently a fixed hard line (`x_min = x_min_lim`).

### Action scaling: real vs Box2D

How the same 2D policy action `[ax, ay]` is handled differs between simulators.

#### Real environment (`airhockey/sims/air_hockey_real.py`)

- In RL mode, `take_action(...)` interprets the action as a normalized delta and computes:
  - `move_vector = [ax, ay] * [rmax_x, rmax_y]`
  - target before constraints: `[x, y] = current_pose[:2] + move_vector`
- Effective per-step scaling in real is therefore controlled by `rmax_x` and `rmax_y`.
- `action_x_scaling` and `action_y_scaling` exist in defaults but are not used in the real transition logic.
- After target construction, `compute_rect(...)` / `compute_pol(...)` and `clip_limits(...)` enforce workspace limits, so the realized motion can be smaller than `[ax*rmax_x, ay*rmax_y]`.

- rmax_x=0.26, rmax_y=0.12

#### Box2D simulator (`airhockey/sims/airhockey_box2d.py`)

- Before dynamics, action is converted with `convert_to_box2d_coords(...)`:
  - `[ax, ay] -> [ay, -ax]`
- During transition, force is scaled by:
  - `force = force * [action_x_scaling, action_y_scaling]`
- So Box2D uses `action_x_scaling/action_y_scaling` directly in control, unlike real.
- Since scaling happens after coordinate conversion, these scalings act on Box2D axes (post-transform), not on raw `[ax, ay]` in the original policy frame.

#### Practical implication

- To tune real rollouts, prioritize `rmax_x/rmax_y` (plus workspace and safety clipping behavior).
- To tune Box2D action responsiveness, use `action_x_scaling/action_y_scaling`.
- Matching behavior across real and Box2D requires calibrating both sets of parameters, not copying only one pair.
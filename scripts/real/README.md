
### Trajectory Saving for Real Rollouts

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
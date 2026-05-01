# Robosuite simulator (MuJoCo)

Status: **legacy backend, currently broken, not used for active TD3 training.**
Kept around for the PPO/SAC baseline configs in
[`configs/baseline_configs/robosuite/`](../../../../configs/baseline_configs/robosuite).
This page documents what is wired up today and what would need to change to run
[`td3_training.py`](../../../../scripts/smooth_policy/amp_history/amp_training/td3/td3_training.py)
on it without altering training-side args or the TD3 output structure.

> **"Currently broken"** means: even before considering TD3 compatibility, the
> existing test-rendering script (see "Test rendering" below) does not run as-is
> against the current sim — its config path is wrong and the camera image keys
> it reads (`current_state["sideview_image"]`) are not populated by
> `AirHockeyRobosuite.get_current_state()`. Treat any call site that touches
> robosuite as needing verification, not just the trainer.

Primary file: [`airhockey/sims/airhockey_robosuite.py`](../../../../airhockey/sims/airhockey_robosuite.py)
(class `AirHockeyRobosuite`, ~1069 lines).

---

## How robosuite plugs into the env stack

The `AirHockeyEnv(cfg)` factory and the abstract `AirHockeyBaseEnv` are
simulator-agnostic. The sim backend is selected by `air_hockey.simulator`:

```yaml
air_hockey:
  simulator: robosuite          # box2d | robosuite | real
  simulator_params: { ... }     # forwarded to the chosen sim's __init__
```

Dispatch lives in [`airhockey/airhockey_base.py:97-104`](../../../../airhockey/airhockey_base.py)
(`get_robosuite_simulator_fn` → `from airhockey.sims import AirHockeyRobosuite`).
The package `__init__` ([`airhockey/__init__.py`](../../../../airhockey/__init__.py))
registers the class with robosuite's env registry on import:

```python
from .airhockey_robosuite import AirHockeyRobosuite
from robosuite.environments.base import register_env
register_env(AirHockeyRobosuite)
```

Imports are wrapped in `try/except` because robosuite/MuJoCo doesn't install on
Apple Silicon — both `airhockey/__init__.py` and `airhockey/sims/__init__.py`
print a warning and continue if the import fails.

### Asset side-effects at import time

`airhockey/__init__.py` performs file copies into the robosuite package on
import (these run regardless of which simulator the run uses):

- Copies `assets/arenas/air_hockey_table.xml` into robosuite's `assets_root`.
- Copies `assets/robots/ur5e/robot.xml` into `<robosuite>/robots/custom_ur5e/`,
  creating the folder by `shutil.copytree` from the stock `ur5e` assets the
  first time.

These exist so that the table XML and a custom UR5e variant are findable via
`robosuite_xml_path_completion` at sim construction time.

### Custom robosuite components

Loaded from [`airhockey/sims/`](../../../../airhockey/sims/):

- [`controllers/`](../../../../airhockey/sims/controllers) — custom controller registration
- [`robots/`](../../../../airhockey/sims/robots) — custom UR5e robot
- [`grippers/`](../../../../airhockey/sims/grippers) — round paddle gripper
- [`utils/RobosuiteTransforms`](../../../../airhockey/sims/utils) — coordinate utility

These are imported eagerly (also under `try/except`) in `airhockey/__init__.py`
so they register with robosuite's factories before any env is constructed.

---

## What `AirHockeyRobosuite` does

### Construction

`__init__(**kwargs)` (lines 157-321) takes a flat dict (a `simulator_params`
section from YAML, plus a few env-level keys forwarded by the base env: `seed`,
`paddle_bounds`, `paddle_edge_bounds`, `center_offset_constant`).

Notable defaults baked into the sim:

| Key | Default | Notes |
|---|---|---|
| `robots` | `["UR5e"]` | Stock UR5e (not the custom one) |
| `controller_configs` | `{'arm': 'OSC_POSE'}` | OSC end-effector control |
| `gripper_types` | `None` | No gripper attached |
| `control_freq`, `step_frequency` | 20, 20 | Policy frequency |
| `horizon` | 400 | Robosuite-internal episode cap |
| `has_renderer` | `False` | On-screen GUI |
| `has_offscreen_renderer` | `True` | For camera obs |
| `use_camera_obs` | `True` | Cameras: `["birdview", "sideview"]` |
| `table_tilt` | 0.0 | Radians; juggle YAML uses 0.09 |
| `table_elevation`, `depth`, `rim_width` | 0.0, 0.0505, 0.05 | Geometry |
| `max_paddle_vel`, `max_puck_vel` | 2.0, 10.0 | Used by env |
| `osc_kp`, `osc_damping_ratio` | 150, 1 | OSC gains |
| `osc_output_max_pos` | `[0.05, 0.05, 0.05]` | Per-axis delta clamp (m) |
| `puck_radius`, `puck_density`, `puck_damping` | 0.03165, 30, 0.01 | Puck params |

The OSC controller config is assembled by `_build_controller_config` (lines
322-362) and passed through `robosuite_env_cfg`.

### Lifecycle (mirroring the abstract `AirHockeySim` API)

- `reset(seed=None, **kwargs)` (line 381) — calls `robosuite_env.reset()` if
  one already exists, resets `timestep`, primes `puck_history` to a length-5
  filler list `[(-2 + center_offset_constant, 0, 1)] * 5`. **On first call
  only**, parses `arenas/air_hockey_table.xml` via `xmltodict` into
  `self.xml_config` and updates table geometry.
- `update_table(top_solref, bot_solref, left_solref, right_solref)` (line 713)
  — overwrites wall solrefs in the parsed XML. Called by the base env from
  `reset` (`AirHockeyBaseEnv.reset`) only when `simulator_name == "robosuite"`
  using `self.solrefs` (read from `simulator_params.{top,bot,left,right}_solref`).
- `spawn_puck(pos, vel, name)` / `spawn_block(pos, vel, name)` (lines 618, 530)
  — append the body to `self.xml_config` with damping wired from
  `self.puck_damping`, store initial `(pos, vel)` for later state injection.
- `spawn_paddle(pos, vel, name)` (line 736) — stores the desired EEF
  position/velocity in `initial_obj_configurations`. The actual paddle is the
  UR5e EEF, not a separately spawned body.
- `instantiate_objects()` (line 457) — first call only: writes the assembled
  XML to a temp file, builds `RobosuiteEnv(xml_fp=...)` with table size /
  friction / offset and the puck/block names, then calls `set_obj_configs()`
  to inject qpos/qvel for each object. Subsequent calls re-inject only.
- `set_object_links()` (line 445) — populates `paddle_name_list = ["gripper0_right_eef"]`
  plus sorted puck/block name lists.
- `get_transition(action)` (line 757) — see "Action interpretation" below.
- `get_current_state()` (line 810) — pulls EEF pos/vel and per-puck pos/vel
  out of `robosuite_env._get_observations()`, runs them through
  `robosuite_to_high_level_coords` / `robosuite_to_high_level_vel`, and returns
  the same `state_info` dict shape Box2D produces (`{'paddles': {'paddle_ego':
  {position, velocity, acceleration, force}}, 'pucks': [...], 'blocks': [...]}`).
- `get_contacts()` — passthrough to `robosuite_env.get_contacts()`.
- `start_callbacks(**kwargs)` — no-op (Box2D and real both override).

### Action interpretation

`get_transition(action)`:

1. `translate_action` (line 740) maps the policy's 2-D `[ax, ay] ∈ [-1, 1]`
   into a 6-D OSC delta:
   ```
   dx  = -ax * x_to_x_prime_ratio * action_x_scaling   # cos(table_tilt) factor
   dy  = -ay * action_y_scaling
   dz  =  transform_z(-ax * action_x_scaling)          # sin(table_tilt) → keep on tilted plane
   d{roll, pitch, yaw} = 0
   ```
   `action_x_scaling`, `action_y_scaling` default to 1.0 and are applied **before** OSC's own
   `output_max_pos` clamp of `±0.05 m` per axis.
2. Inner loop runs `int(control_timestep / model_timestep)` mjsim sub-steps,
   calling `_pre_action(action, policy_step=True)` only on the first sub-step
   (matches robosuite's policy/internal-control split).
3. After stepping, populates `paddle_ego.force` from `cfrc_ext` of
   `gripper0_right_eef` and `paddle_ego.acceleration ≈ Δv`.
4. Appends the puck position to `self.puck_history` (or filler if no puck).

### Coordinate frame

The sim has two frames. The "high-level" frame matches Box2D's centered
convention (`x ∈ [-length/2, length/2]`, `y ∈ [-width/2, width/2]`). Robosuite
internal frame has `x ∈ [rim_width, length-rim_width]` plus a vertical tilt.
Conversion is handled by:

- `high_level_to_robosuite_coords / _vel`
- `robosuite_to_high_level_coords / _vel` (note: `_vel` flips both x and y signs)

### Sysid params and which ones the sim *actually* uses

`AirHockeyRobosuite.__init__` accepts `puck_density`, `puck_damping`,
`puck_radius`, `paddle_radius`, `block_density`, `gravity`, `force_scaling`,
`paddle_damping`, `paddle_density`, `max_force_timestep` — but most are
defaults-only placeholders. Read the source for what is consumed:

| Param | Used? | Where |
|---|---|---|
| `puck_density` | ✅ | `spawn_puck` puck mass (line 624) |
| `puck_damping` | ✅ | injected as joint damping in puck XML (lines 644, 653) |
| `puck_radius` | ✅ | XML geom size + collision check |
| `block_width`, `block_density` | ✅ | `spawn_block` |
| `paddle_radius` | ✅ | env-level only; sim stores it |
| `gravity` | ❌ | accepted, never read; MuJoCo gravity is in the table XML |
| `paddle_damping` | ❌ | accepted, never read |
| `paddle_density` | ❌ | accepted, never read |
| `force_scaling` | ❌ | accepted, never read |
| `max_force_timestep` | ❌ | accepted, never read |
| `pid_kp`, `pid_kd`, `pid_ki` | ❌ | no PID controller; OSC instead |
| `puck_noise*`, `enable_random_occlusions*` | ❌ | not implemented |
| `enable_observation_delay`, `enable_action_delay`, `delay_seconds` | ❌ | not implemented (a TODO at line 768 mentions action_lag) |
| `enable_action_force_attenuation*`, `enable_puck_delay_interpolation*` | ❌ | not implemented |
| `enable_paddle_puck_strength_randomization` etc. | ❌ | not implemented |
| `top/bot/left/right_solref` | ✅ | wired through env → `update_table` |

Bottom line: the system-ID parameters from `sysid_best_params.yaml` (gravity,
puck_damping, paddle_density, pid_kp/kd) **do not transfer**. Walls (via
solrefs) and puck mass/damping are the only sysid-style knobs that work.

---

## Test rendering

Primary test-rendering entrypoint:
[`scripts/test_controller.py`](../../../../scripts/test_controller.py).

What it does:

1. Loads
   `configs/baseline_configs/puck_height_robosuite.yaml` via
   `os.path.join(dir_path, '../configs', 'baseline_configs/puck_height_robosuite.yaml')`.
2. Builds an `AirHockeyEnv` and an
   [`AirHockeyRenderer`](../../../../airhockey/renderers/render.py).
3. Steps a hardcoded scripted action (`[0.01, 0.0165]` for the first 32 steps,
   then `[-1, 0.0165]`) until the episode terminates.
4. At each step, calls `renderer.get_frame()` (Box2D-style top-down rendering)
   and reads `eval_env.current_state["sideview_image"]` (the robosuite
   off-screen camera frame), concatenates the two side-by-side, displays via
   `cv2.imshow("AirHockey", ...)`, and accumulates frames into a GIF written to
   `../eval_gifs/<task>.gif`.

> **Quick-fix history (2026-04-30):** the script was repaired enough to
> render one episode end-to-end. What was patched:
>
> 1. **Config path.** Was `configs/baseline_configs/puck_height_robosuite.yaml`,
>    fixed to `configs/baseline_configs/robosuite/puck_height_robosuite.yaml`
>    (the `baseline_configs/` layout had been reshuffled into per-simulator
>    subdirectories without updating this script).
> 2. **`paddle_history`.** `AirHockeyRobosuite` did not maintain it at all,
>    so `AirHockeyBaseEnv.reset()` crashed before the first observation
>    even when `obs_type: vel` (the env passes `paddle_history` as a kwarg
>    unconditionally). Initialized in `reset()` with the same length-5 filler
>    Box2D uses, appended in `get_transition()` with a `0` valid-flag.
> 3. **Headless mode.** `puck_height_robosuite.yaml` has `has_renderer: true`,
>    which opens a GLFW window. The script now overrides
>    `simulator_params.has_renderer = False`,
>    `has_offscreen_renderer = True` and gates `cv2.imshow` /
>    `cv2.waitKey` behind a `AIRHOCKEY_HEADLESS=1` env var (default 1).
> 4. **Missing `seed` field.** Neither the YAML nor the defaults provided
>    `simulator_params.seed`; the script now injects `seed: 43` at both
>    `air_hockey.seed` and `air_hockey.simulator_params.seed`.
>
> Camera-image plumbing (`birdview_image` / `sideview_image` keys in
> `current_state`) was already in `get_current_state()` — line 857-859,
> `for key in obs.keys(): if 'image' in key: state_info[key] = obs[key]`.
> What broke this earlier was just that the env never reached `step()`.
>
> **Run command:**
> ```bash
> MUJOCO_GL=egl PYOPENGL_PLATFORM=egl AIRHOCKEY_HEADLESS=1 \
>   .venv/bin/python scripts/test_controller.py
> ```
> Output: `eval_gifs/puck_height_robosuite.gif` (15 frames at 30 fps; the
> hardcoded scripted action drives the paddle out of bounds in ~16 steps,
> which truncates the episode — that's a script-level issue, not a sim-level
> one).

### Other places that consume robosuite frames

- [`scripts/teleop.py`](../../../../scripts/teleop.py) — interactive teleop
  loop that reads `current_state['sideview_image']` and `birdview_image` keys
  (same expectation, same brittleness).
- [`scripts/evaluate_model.py`](../../../../scripts/evaluate_model.py),
  [`scripts/utils.py`](../../../../scripts/utils.py) (`Evaluator._eval`),
  [`scripts/domain_adaptation/utils.py`](../../../../scripts/domain_adaptation/utils.py) —
  PPO/SAC eval helpers that branch on `simulator_name == 'robosuite'` to
  collect per-camera GIFs (`feval_robosuite_<view>.gif`). Used during the old
  baseline training, not by `td3_training.py`.
- [`scripts/utils/recreate_scene.py`](../../../../scripts/utils/recreate_scene.py),
  [`scripts/scripts/visualize_model.py`](../../../../scripts/scripts/visualize_model.py) —
  call `robosuite.suite.make` directly (bypassing `AirHockeyEnv`); legacy.
- [`scripts/real/generate_homography.py:14`](../../../../scripts/real/generate_homography.py)
  — produces `assets/robosuite/Mimg.npy` and `Mrob.npy` used by the teleop
  view warp.

### `AirHockeyRenderer` and `robosuite_view`

The Box2D-style top-down renderer ([`airhockey/renderers/render.py`](../../../../airhockey/renderers/render.py))
takes an optional `robosuite_view` kwarg (defaults to `""`). When non-empty,
`render` calls `merge_robosuite_frame` (line 434) which reads
`self.airhockey_env.current_state[self.robosuite_view]` and concatenates that
camera frame with the synthetic top-down image. This is the rendering surface
that needs camera-image plumbing in `get_current_state` to work.

---

## Configs that exist for robosuite

15 task YAMLs in
[`configs/baseline_configs/robosuite/`](../../../../configs/baseline_configs/robosuite),
all written for the old PPO/SAC baseline pipeline:

```
puck_juggle_robosuite.yaml         puck_strike_robosuite.yaml
puck_catch_robosuite.yaml          puck_touch_robosuite.yaml
puck_height_robosuite.yaml         puck_reach_robosuite.yaml
puck_vel_robosuite.yaml            paddle_position_robosuite.yaml
paddle_vel_robosuite.yaml          paddle_pos_neg_regions_robosuite.yaml
move_block_robosuite.yaml          strike_crowd_robosuite.yaml
multipuck_juggle_robosuite.yaml    test_robosuite.yaml
gcrl_pos_robosuite.yaml            gcrl_pos_vel_robosuite.yaml
gc_pos_dense_robosuite.yaml        gcrl_pos_curriculum_robosuite.yaml
```

These have `algorithm: ppo`, `obs_type: vel` (not `history`), and a much
larger `paddle_bounds` rectangle than the Box2D juggle setup. They have not
been touched since the TD3 work began.

---

## Gaps for running TD3 against robosuite (same args, same output structure)

The TD3 training script ([`td3_training.py`](../../../../scripts/smooth_policy/amp_history/amp_training/td3/td3_training.py))
is simulator-agnostic at the seam: `make_env` (line 643) calls
`AirHockeyEnv(config["air_hockey"])` and the trainer never branches on the
simulator name. So the **TD3 `Args` dataclass and the run-output layout
(`runs/default_training/{task_name}/{run_name}_{timestamp}/` with `args.yaml`,
`config.yaml`, TensorBoard scalars, checkpoints, sample GIFs) require no
changes.** All gaps live in env/sim code or the YAML you point `--config` at.

### Hard blockers (must fix to start a run)

1. ~~**`paddle_history` is never set on the robosuite sim.**~~ **Fixed
   2026-04-30** as part of the test-render quick fix:
   - Symptom (pre-fix): `AttributeError: 'AirHockeyRobosuite' object has no
     attribute 'paddle_history'` on first reset, even with `obs_type: vel`.
   - Path: `AirHockeyBaseEnv.reset()` →
     `get_observation(..., paddle_history=self.simulator.paddle_history, ...)`
     ([`airhockey_base.py:394`](../../../../airhockey/airhockey_base.py)).
     `obs_type=history` (the default for TD3, used by `sysid_best_params*.yaml`)
     reads it at [`airhockey/utils.py:76`](../../../../airhockey/utils.py).
   - Fix: `reset` now initializes
     `self.paddle_history = [(-2 + center_offset_constant, 0, 1)] * 5`
     (mirroring Box2D's filler), and `get_transition` appends
     `[paddle_x, paddle_y, 0]` per step. **Sufficient for `obs_type: vel`
     to run; not yet validated end-to-end for `obs_type: history`.**

2. **No `observation_state_info` / `observation_puck_history` /
   `observation_paddle_history` (observation-delay snapshots).**
   - The base env at [`airhockey_base.py:881-893`](../../../../airhockey/airhockey_base.py)
     uses `getattr(simulator, "observation_state_info", None)` and falls back to
     the raw `puck_history` / `paddle_history`, so this only blocks if (1) is
     fixed. Listed here so the same structure as Box2D can be added if you ever
     want observation-delay parity in robosuite (currently no delay support at
     all).

3. **`has_renderer: true` in the existing juggle YAMLs.**
   - Will try to open an on-screen GLFW window and crash a headless training
     box. Set `has_renderer: false`, `has_offscreen_renderer: true`.

### Behavioral mismatches (won't crash, but mean "TD3 trained on robosuite" is not comparable to "TD3 trained on Box2D")

4. **Action semantics differ.** Box2D's `td3_recommended` uses `use_pid: true`
   with `pid_kp=9000, pid_kd=50` over `paddle_radius=0.0508` and an action
   scaled to `[0.26 m, 0.12 m]` (env-level `action_x_ratio`,
   `action_y_ratio`). Robosuite uses OSC_POSE with `osc_kp=150` and a per-axis
   delta clamp of `±0.05 m`, then a separate `action_x_scaling`,
   `action_y_scaling` (default 1.0). Same `[-1, 1]^2` action goes through
   different physics.
   - `td3_training.py:792-795` reads `config["air_hockey"]["use_pid"]` to set
     `action_scale`. If absent (the case in the robosuite YAMLs),
     `action_scale = args.action_scale` is used — semantically wrong if you
     copy a Box2D-trained run's args.
   - To be apples-to-apples you'd need OSC gains and `osc_output_max_pos` tuned
     so a `[1, 1]` action moves the paddle a similar distance per step as the
     PID setup.

5. **Sysid parameters from `sysid_best_params.yaml` don't transfer** (see
   table above). `gravity`, `puck_damping`, `paddle_density`, `pid_*` either
   don't exist or aren't read on the robosuite path. The realism story used
   for sim2real on Box2D does not apply to a robosuite run.

6. **No domain-randomization features.** None of `puck_noise`,
   `enable_random_occlusions`, `enable_observation_delay`, `enable_action_delay`,
   `enable_action_force_attenuation`, `enable_puck_delay_interpolation`,
   per-collision strength/direction randomization are implemented in the
   robosuite sim. The robosuite path is effectively a clean sim.

7. **Reward shaping fields** (`jerk`) are not populated. `get_current_state`
   sets `acceleration` (Δv proxy) and `force`, but no `jerk`. This is gated
   behind `jerk_penalty_coeff != 0.0` in `airhockey_base.py:735`, so it's
   silent at the default coeff of 0.0 — but turning the coeff on with a
   robosuite sim would silently stop firing the penalty.

8. **`obs_position_homography`** is not exposed (`getattr(..., None)` → no-op).
   Acceptable; just means real-world homography augmentation has no robosuite
   counterpart.

9. **No `soft_reset`.** The real-world adapter has it; Box2D and robosuite
   don't. Only used by some real-world flows, not by `td3_training.py`.

10. **`num_envs > 1` and `AsyncVectorEnv`.** `td3_training.py:676-680`
    enforces `num_envs == 1` regardless of sim, so robosuite's heavier MuJoCo
    fork-cost is not an extra issue here. But note that the trainer still
    wraps in `AsyncVectorEnv` — robosuite's MuJoCo state has historically been
    tricky over fork. Worth verifying empirically.

### What the minimum-viable TD3-on-robosuite YAML pair would look like

- **Sim YAML** (call it `sysid_robosuite.yaml`): mirror
  `sysid_best_params.yaml` but with `simulator: robosuite`, drop the keys
  the robosuite sim doesn't read, set sane `osc_*` and `action_{x,y}_scaling`,
  set `has_renderer: false`, and change `obs_type` to a value that doesn't
  require history (`vel`) **or** add `paddle_history` support to
  `AirHockeyRobosuite` (preferred — keeps obs-shape parity with Box2D).
- **TD3 args YAML**: identical to `td3_recommended.yaml` aside from `config:
  pointing at the new sim YAML. No `Args` field needs to change.
- **Output structure**: identical. `td3_training.py` writes `runs/default_training/
  {task_name}/{run_name}_{timestamp}/` with `args.yaml` + `config.yaml` +
  TensorBoard + checkpoints + GIFs regardless of simulator.

### Suggested order of work

1. Add `paddle_history` to `AirHockeyRobosuite` (cheapest unblocker).
2. Fix `has_renderer` defaults / build a headless-friendly base sim YAML.
3. Tune OSC `kp` / `output_max_pos` / `action_*_scaling` so a `[1, 1]` action
   has comparable per-step displacement to PID Box2D.
4. (Optional) Port domain-randomization knobs (puck noise, occlusions, action
   delay) if you want sim2real story parity, not just "sanity-check on a
   different physics engine."

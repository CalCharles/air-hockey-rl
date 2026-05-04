# Robosuite simulator (MuJoCo)

Status: **legacy backend, basic functionality restored 2026-05-01, not used for
active TD3 training.** Kept around for the PPO/SAC baseline configs in
[`configs/baseline_configs/robosuite/`](../../../../configs/baseline_configs/robosuite).
This page documents what is wired up today and what would need to change to run
[`td3_training.py`](../../../../scripts/smooth_policy/amp_history/amp_training/td3/td3_training.py)
on it without altering training-side args or the TD3 output structure.

> **What "basic functionality restored" means.** As of 2026-05-01 the sim
> can: (a) place the UR5e EEF on the table at the env-intended `paddle_ego`
> start position, (b) hold position with `action=zeros` (no controller
> blow-up), (c) translate `[ax, ay] ∈ [-1, 1]` policy actions into actual
> paddle motion via OSC, (d) render the puck visibly on the table from any
> camera, (e) attach the **yellow round paddle (`RoundGripper`)** to the
> EEF — visible as `gripper0_right_wiping_surface2br`, (f) initialize the
> puck-juggle task with the puck spawning at the +x end of the tilted table
> and sliding toward the agent under gravity. Ten compounding bugs had to be
> fixed across two debugging passes to get here — see the [Bug-fix log
> (2026-04-30 → 2026-05-01)](#bug-fix-log-2026-04-30--2026-05-01) section
> below.
>
> The sim is **not yet** production-grade for TD3: action scaling, sysid
> parity, IK pose tuning for the gripper-on length offset, and domain-rand
> features are still missing — see the [TD3 compatibility
> gaps](#gaps-for-running-td3-against-robosuite-same-args-same-output-structure)
> section.

Primary file: [`airhockey/sims/airhockey_robosuite.py`](../../../../airhockey/sims/airhockey_robosuite.py)
(class `AirHockeyRobosuite`, ~1100 lines).

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

- [`controllers/`](../../../../airhockey/sims/controllers) — custom controller registration. **Broken on robosuite 1.5+** (uses old `robosuite.controllers.base_controller` import). Not currently needed since we use the stock `OSC_POSE` controller.
- [`robots/`](../../../../airhockey/sims/robots) — custom UR5e robot. **Broken on robosuite 1.5+** (uses old `load_controller_config` import). Not currently used; we use the stock `UR5e` robot from robosuite.
- [`grippers/`](../../../../airhockey/sims/grippers) — round paddle gripper (`RoundGripper`, registered as `"RoundGripper"` in `GRIPPER_MAPPING`). The actual paddle: a yellow cylinder of radius 0.0508 m, named `gripper0_right_wiping_surface2br` in the loaded model. **Works** on robosuite 1.5+. Set `gripper_types: 'RoundGripper'` in `simulator_params` to attach it to the EEF.
- [`utils/RobosuiteTransforms`](../../../../airhockey/sims/utils) — coordinate utility.

> **Bug fixed 2026-05-01.** These four imports used to be in a single
> `try/except` block. When `controllers` failed (it does on robosuite 1.5+),
> the whole try block aborted, so `grippers` and `utils.RobosuiteTransforms`
> never ran. Result: `'RoundGripper' not in GRIPPER_MAPPING` even though the
> file was present. Now each module is imported in its own try/except so
> failures are isolated.

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
| `gripper_types` | `None` | Defaults to UR5e's Robotiq85. Set to `'RoundGripper'` to attach the yellow paddle (see [Gripper attachment](#gripper--paddle-attachment)) |
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
  See "Puck spawn details" below for two non-obvious fixes (rgba + z-tilt
  compensation) landed 2026-05-01.
- `spawn_paddle(pos, vel, name)` (line 736) — stores the desired EEF
  position/velocity in `initial_obj_configurations`. The actual paddle is the
  UR5e EEF, not a separately spawned body. **The desired pose is NOT applied
  here** — `set_obj_configs` overrides robot qpos to a hand-tuned
  tabletop-reach pose on every reset (see "Robot pose override" below).
- `instantiate_objects()` (line 457) — first call only: writes the assembled
  XML to a temp file, builds `RobosuiteEnv(xml_fp=...)` with table size /
  friction / offset and the puck/block names, then calls `set_obj_configs()`
  to inject qpos/qvel for each object. Subsequent calls re-inject only.
- `set_obj_configs()` — runs every reset. Now (post-2026-05-01) does three
  things in this order:
  1. Disable problematic collisions (pedestal, default-gripper finger/knuckle
     self-contacts) by zeroing `geom_contype`/`geom_conaffinity`. Re-applied
     every reset because `hard_reset=True` rebuilds the MuJoCo model from XML
     defaults each time.
  2. Override robot joints to `tabletop_init_qpos`
     (`[-0.3388, -1.553, 2.1471, -2.4853, -1.3923, -1.991]`) so the EEF lands
     at world `(0.333, 0.0, 0.795)` (env's intended `paddle_ego` start
     position, 5 mm above the table). Then resync the OSC controller via
     `update(force=True)` + manual `goal_pos = world_to_origin_frame(ref_pos)`.
  3. Inject puck and block slide-joint qpos/qvel as before.
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

1. `translate_action` (line ~810) maps the policy's 2-D `[ax, ay] ∈ [-1, 1]`
   into a 6-D OSC delta:
   ```
   dx  = -ax * x_to_x_prime_ratio * action_x_scaling   # cos(table_tilt) factor
   dy  = -ay * action_y_scaling
   dz  = -x_to_z_ratio * ax * action_x_scaling         # sin(table_tilt) — keeps EEF on tilted plane as it moves in x
   d{roll, pitch, yaw} = 0
   ```
   `action_x_scaling`, `action_y_scaling` default to 1.0 and are applied **before** OSC's own
   `output_max_pos` clamp of `±0.05 m` per axis.

   > **Bug fixed 2026-05-01.** The previous `dz` was
   > `self.transform_z(-ax * action_x_scaling)`, where `transform_z` returns
   > an ABSOLUTE z position (≈ `table_elevation - depth/2 ≈ 0.69 m`), not a
   > delta. OSC saw a constant ~0.69 z-input every step regardless of the
   > policy action, clipped it to `+0.05 m` per step, and dragged the EEF
   > upward — main visible symptom: paddle drifted off the table and out of
   > bounds in 1-2 steps under any action. Replaced with the tilt-compensated
   > delta above.
2. Inner loop runs `int(control_timestep / model_timestep)` mjsim sub-steps,
   calling `_pre_action(action, policy_step=True)` only on the first sub-step
   (matches robosuite's policy/internal-control split).
3. After stepping, populates `paddle_ego.force` from `cfrc_ext` of
   `gripper0_right_eef` and `paddle_ego.acceleration ≈ Δv`.
4. Appends the puck position to `self.puck_history` and the EEF position to
   `self.paddle_history` (or filler entries if no puck/paddle yet).

### Gripper / paddle attachment

The yellow round paddle that the agent uses to hit the puck is implemented as
a `RoundGripper` (subclass of robosuite's `GripperModel`) defined in
[`airhockey/sims/grippers/round_gripper.py`](../../../../airhockey/sims/grippers/round_gripper.py),
backed by [`assets/grippers/round_gripper.xml`](../../../../assets/grippers/round_gripper.xml).
The XML places a **yellow cylinder** (radius 0.0508 m, half-height 0.015 m,
named `wiping_surface2br`) on the EEF site, and the cylinder also doubles as
the contact geom that makes paddle-puck contact.

To attach it to the robot:

```yaml
air_hockey:
  simulator_params:
    gripper_types: RoundGripper      # <-- without this, you get the UR5e's
                                     #     default Robotiq85Gripper
```

> **Bugs fixed 2026-05-01 (multiple).** Three things were preventing the
> RoundGripper from actually being attached even when set in the YAML:
>
> 1. **Registration was silently failing.** See "Custom robosuite components"
>    above — the try/except in `airhockey/__init__.py` was aborting on the
>    first broken import, so the gripper module never registered.
> 2. **Wrong key in `load_robots_configs`.** The function passed
>    `controller_config`, `mount_type` to `Robot.__init__`, but robosuite's
>    Robot expects `composite_controller_config`, `base_type`. Robot
>    silently fell back to defaults — including the default Robotiq85.
> 3. **`gripper_type` was never threaded through.** `load_robots_configs`
>    didn't include the gripper type at all, and the function returned the
>    empty input list (`return robot_configs`) instead of the populated
>    `self.robot_configs`. So `get_robots` instantiated `Robot(robot_type=...,
>    idn=...)` with no per-robot config and `gripper_type` defaulted to
>    `"default"` → loaded `UR5e.default_gripper` = `Robotiq85Gripper`.
>
> All three are fixed: each module imports independently;
> `load_robots_configs` accepts `gripper_types` and uses the right keys;
> the function returns `self.robot_configs`. After the fix, the loaded
> MuJoCo model contains `gripper0_right_round_gripper` body and
> `gripper0_right_wiping_surface2br` geom (the yellow paddle disc).

### Robot pose override

`spawn_paddle(pos, vel, name)` only **records** the desired EEF pose
(`initial_obj_configurations['paddles'][name]`) — it never moves the UR5e.
Robosuite's stock `init_qpos` for the UR5e is
`[-0.47, -1.735, 2.48, -2.275, -1.59, -1.991]`, which puts the EEF at world
z=0.685 — that's **10 cm BELOW the table surface** (top at z=0.787 at the
table center). Letting OSC try to recover from there explodes qvel on the
first `sim.step` (`NaN/Inf in QACC at DOF 6`).

To work around this, `set_obj_configs` writes a hand-tuned **tabletop-reach
pose** (`tabletop_init_qpos`) every reset. The pose was computed via
damped-least-squares IK on the EEF position Jacobian, with target = world
`(0.333, 0.0, 0.795)` (the env-coord paddle_ego start of `(0.79, 0)` mapped
through `high_level_to_robosuite_coords`, plus 5 mm clearance above the
table-center top). Override sequence:

1. Set `sim.data.qpos[joint_idx] = tabletop_qpos`, `sim.data.qvel[...] = 0`
2. `sim.forward()` (refreshes derived kinematics like body_xpos)
3. `cc.update_state()` — refreshes per-controller `origin_pos` / `origin_ori`
4. For each part_controller: `pc.update_initial_joints(tabletop_qpos)` then
   `pc.update(force=True)` — refreshes `ref_pos` / `ref_ori_mat` from sim
5. `pc.goal_pos = pc.world_to_origin_frame(pc.ref_pos)` and
   `pc.goal_ori = ref_ori_mat` — manually written in BASE frame, because
   robosuite's stock `reset_goal` writes world-frame `ref_pos` directly into
   `goal_pos` (which OSC stores in BASE frame). That mismatch is what was
   making the EEF fly off by ~0.4 m on the first step.
6. `sim.data.ctrl[:] = 0` then `sim.step()` — settle once with zero ctrl so
   leftover torques from any previous controller cycle don't perturb the
   freshly-set pose.

If you ever need a different start pose, override the instance attribute
`tabletop_init_qpos` BEFORE the first reset, e.g.:

```python
env.simulator.tabletop_init_qpos = my_qpos
env.reset()
```

### Collision overrides applied at every reset

The MuJoCo model rebuilt by `hard_reset=True` includes two sets of
collisions that should never fire and which break dynamics:

- **Pedestal vs. robot.** The robot is mounted on `fixed_mount0_pedestal`
  (world `(-0.2, 0, 0.74)`). The robot's own shoulder body sits at world
  `(-0.196, 0.011, 0.785)` — INSIDE the pedestal collision volume. With
  `solref=-100000 -250` style stiffness on contact, this generates massive
  constraint forces that cause `NaN/Inf in QACC` on the first step.
- **Default gripper self-collision.** A gripper is added even with
  `gripper_types: None` in the YAML, and its `inner_finger_collision` and
  `inner_knuckle_collision` geoms intersect at the rest pose, generating
  spurious internal contact forces.

`set_obj_configs` zeroes `contype` / `conaffinity` on every geom whose name
contains `pedestal`, `mount`, or matches `gripper *(finger|knuckle) *collision`.
The visual geoms (group=1) are unaffected. Reapplied each reset because
`hard_reset` rebuilds the model from XML.

### Puck spawn details

`spawn_puck(pos, vel, name, affected_by_gravity=True)` builds an XML `<body>`
for the puck with several non-obvious choices, fixed across 2026-04-30 and
2026-05-01:

1. **The puck body is named `"base"` (the OUTER body) and contains a child
   body named e.g. `"puck_0"` (the INNER body).** A single `<freejoint>`
   (was `slide_x` + `slide_y` + `yaw_hinge` until 2026-05-01) is attached at
   the OUTER level, so the joint moves the whole base body; the geom lives on
   the inner body. `body_xpos[puck_0]` and `body_xpos[base]` track each other
   (puck_0 has body_pos = (0,0,0) inside base).

   > **Bug fixed 2026-05-01.** The original three-slide-joint setup locked
   > the puck at its spawn z. That made the air-hockey puck-juggle task
   > impossible: the puck couldn't fall under gravity or bounce up off the
   > paddle. Replaced with a single free joint so MuJoCo gravity acts
   > naturally and paddle contact propels the puck correctly. The
   > `set_obj_configs` injector was rewritten accordingly: free joint qpos is
   > 7 elements `[x, y, z, qw, qx, qy, qz]`, qvel is 6 elements `[vx, vy, vz,
   > wx, wy, wz]`. Initial pose is written world-frame; angular velocity is
   > zeroed; linear velocity uses the spawn `vel` for x and y.

   > **Bug fixed 2026-05-01.** The `affected_by_gravity` parameter was
   > accepted but never used. Default was `False` (vs. Box2D's `True`),
   > which would have prevented juggle even if the joint allowed z motion.
   > Default is now `True`, and the per-puck flag is honored via
   > `body_gravcomp` (1.0 = full antigravity / no gravity, 0.0 = normal
   > gravity).

2. **Color is set via `@rgba`, not `@material`.** The previous code used
   `"@material": "green"` referencing the `<material name="green" .../>`
   asset in the table XML. That material lookup silently fails when the
   puck body is appended via `xmltodict` to the parsed XML (the assets
   namespace doesn't propagate the way you'd expect), so MuJoCo fell back
   to `rgba=[0.5, 0.5, 0.5, 1]` (gray). On a white table from a 4 m birdview
   camera, a 3 cm gray puck is essentially invisible — that's the original
   "I don't see the puck" complaint. Now hard-coded to bright green:
   `"@rgba": "0.05 0.85 0.15 1"`.

3. **The spawn z compensates for table tilt.** The table is tilted
   `axisangle="0 1 0 -table_tilt"` (with `table_tilt=0.09` rad in
   `puck_height_robosuite.yaml`). That means the table top z varies linearly
   with x: `table_top_z(x) = table_elevation + sin(table_tilt) * (x - table_full_size[0])`.
   At the puck spawn x (≈ world x=1.97, the +x end of the table), the table
   top is at z ≈ 0.870, but the previous spawn used a flat
   `z = table_elevation + puck_height/2 = 0.792` — placing the puck
   **8 cm UNDERNEATH the table surface** at that x. The puck body_xpos
   reported the right value (1.97, -0.21, 0.79), so it was NOT obvious from
   logs; from any camera looking down, the white table occluded the puck.
   Now `z_pos = table_top_z(x) + puck_height/2 + 0.001` (1 mm clearance
   above the tilted top).

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

Two scripts:

### Multi-camera diagnostic render (preferred)

[`scripts/render_robosuite_views.py`](../../../../scripts/render_robosuite_views.py)
(added 2026-05-01). Builds the env, runs a 60-step rollout with a gentle
`action = [0, 0.3 * sin(step * 0.15)]` paddle wobble, and saves per-camera
GIFs plus a 2-row stacked grid GIF.

Cameras rendered (defined in [`assets/arenas/air_hockey_table.xml`](../../../../assets/arenas/air_hockey_table.xml)):

| Name | Pos | Use |
|---|---|---|
| `birdview` | `(0.8, 0, 4.0)` top-down | Full table from above, robot at image bottom, puck at top |
| `agentview` | `(0.5, 0, 3.0)` top-down | Closer top-down, robot fills bottom third |
| `frontview` | `(1.0, 0, 1.45)` angled from front | Mostly the robot's body — too close to be useful |
| `sideview` | `(1.0, 2.85, 1.6)` from the side | Best angle for seeing the puck on the (tilted) table with the robot reaching across |
| `backview` | `(-1, 0, 1.45)` opposite side from `frontview` | New 2026-04-30. Looks at the agent end of the table from the puck-spawn side |
| `puckview` | `(1.5, 0, 2.5)` top-down close-up over the puck-spawn area | New 2026-05-01. Specifically positioned so the 3 cm puck is unambiguously visible. Note: positions below z≈2 from this point fail to render anything (probably below near clip + lighting); z=2.5 is the lowest stable height |

The script flips `sideview`, `frontview`, `backview` with `np.flipud` (those
camera quaternions output upside-down framebuffers); `birdview`, `agentview`,
`puckview` come out right-side-up natively. See `NEEDS_VFLIP` in the script.

Run command:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python scripts/render_robosuite_views.py
```

Outputs go to `eval_gifs/views/{birdview,agentview,frontview,sideview,backview,puckview,grid}.gif`
plus first/last-frame stills. `grid.gif` is the most compact way to confirm
the sim is working: top row top-down views (puck visible), bottom row side
views (robot visibly on the table reaching toward the puck).

### Puck-juggle initialization render

[`scripts/render_puck_juggle.py`](../../../../scripts/render_puck_juggle.py)
(added 2026-05-01). Loads `puck_juggle_robosuite.yaml`, attaches the round
paddle (`gripper_types: 'RoundGripper'`), and runs a 120-step rollout from
the puck-juggle initial state. First half holds the paddle still so you can
see the puck slide toward it under gravity; second half pulses the paddle
forward to attempt strikes. Outputs to `eval_gifs/juggle/`.

What you should see:

- `puckview_f000.png` — bright green puck at top of the table (spawn).
- `sideview` GIF — puck slides from the puck-spawn end toward the agent end;
  yellow paddle disc visible on the gripper.
- `backview` GIF — clearest view of the round paddle attached to the wrist.

Diagnostic dump from the script:

```
After env init/reset:
  puck world pos = [1.98, 0.20, 0.876]   # at +x end of tilted table, on surface
  EEF world pos  = [0.15, -0.04, 0.838]  # paddle held above agent end
  puck z gravity affected? gravcomp=0.0  # 0.0 = normal gravity
Rolled out 120 steps
Puck z range: [0.701, 0.873]              # 17 cm vertical drop as it slides
Final puck pos: [0.08, 0.20, 0.703]       # rests against rim_home wall
```

### Original concatenated-render (Box2D + robosuite side-by-side)

[`scripts/test_controller.py`](../../../../scripts/test_controller.py).

What it does:

1. Loads
   `configs/baseline_configs/robosuite/puck_height_robosuite.yaml`.
2. Builds an `AirHockeyEnv` and an
   [`AirHockeyRenderer`](../../../../airhockey/renderers/render.py).
3. Steps `action = np.zeros(2)` (post-fix; was a hardcoded scripted action
   that drove the paddle out of bounds).
4. At each step, calls `renderer.get_frame()` (Box2D-style top-down rendering)
   and reads `eval_env.current_state["birdview_image"]` (post-fix; was
   `sideview_image` with a wrong-axis flip), concatenates the two
   side-by-side, optionally displays via `cv2.imshow` (gated behind
   `AIRHOCKEY_HEADLESS=0`), and accumulates frames into a GIF written to
   `eval_gifs/<task>_robosuite.gif`.

Run command:
```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl AIRHOCKEY_HEADLESS=1 \
  .venv/bin/python scripts/test_controller.py
```
Output: `eval_gifs/puck_height_robosuite.gif`.

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

## Bug-fix log (2026-04-30 → 2026-05-01)

Two debugging passes fixed ten compounding bugs total. Each one masked the
next, so they had to be debugged in order.

### Pass 1 (2026-05-01 morning): get robot on the table, controllable, puck visible

| # | Bug | Symptom | Fix location |
|---|---|---|---|
| 1 | `paddle_history` never maintained | `AttributeError` on first `env.reset()` (env unconditionally passes `paddle_history` kwarg into `get_observation`) | `reset` (init filler), `get_transition` (per-step append) |
| 2 | UR5e default `init_qpos` puts EEF 10 cm BELOW table | OSC tries to recover from inside-the-table pose, blows up: `WARNING: Nan, Inf or huge value in QACC at DOF 6` | `set_obj_configs` writes IK-solved `tabletop_init_qpos` |
| 3 | OSC controller goal/ref out of sync after qpos override | EEF jumps ~0.5 m on first env.step toward old home pose. Root cause: stock `reset_goal` writes world-frame `ref_pos` into base-frame `goal_pos` | `set_obj_configs` calls `pc.update(force=True)` then writes `goal_pos = world_to_origin_frame(ref_pos)` directly |
| 4 | Pedestal & default-gripper collisions | Robot shoulder body sits inside pedestal collision volume; gripper inner-finger self-collides with inner-knuckle. With stiff `solref`, generates massive constraint forces → qvel explodes to 200+ rad/s in one step | `set_obj_configs` zeros `contype`/`conaffinity` on `pedestal*`/`mount*`/`gripper *(finger|knuckle) *collision` geoms (re-applied each reset because `hard_reset` rebuilds model) |
| 5 | `translate_action` z-component returned ABSOLUTE position | `dz = transform_z(...) ≈ 0.69 m` regardless of action — OSC sees a constant +0.69 z-input, drags EEF up off the table over ~10 steps | `translate_action` now uses `dz = -x_to_z_ratio * ax * action_x_scaling` (tilt-compensated delta) |
| 6 | Puck spawn z ignores table tilt → puck spawns UNDER table | Puck `body_xpos` reports correct value but rendered cameras see only white table — table occludes the under-table puck. Also: `@material:"green"` reference silently failed → puck rendered as 0.5,0.5,0.5 gray (also invisible against white) | `spawn_puck` now: (a) uses `@rgba="0.05 0.85 0.15 1"` directly, (b) computes `z = table_elevation + sin(table_tilt)*(x_pos - table_full_size[0]) + puck_height/2 + 0.001` |

After pass 1: `env.step(np.zeros(2))` holds EEF at world `(0.333, 0, 0.795)`
for 60+ steps with sub-mm drift. `env.step([0, 0.5])` produces smooth +y
paddle motion. The puck is clearly visible from `birdview`, `sideview`,
`backview`, and the new `puckview` cameras.

### Pass 2 (2026-05-01 afternoon): paddle on the gripper + puck juggle dynamics

| # | Bug | Symptom | Fix location |
|---|---|---|---|
| 7 | `RoundGripper` registration silently failed | The yellow paddle was implemented as a custom `GripperModel` but `'RoundGripper' not in robosuite.models.grippers.GRIPPER_MAPPING`. Root cause: the four optional-import lines in `airhockey/__init__.py` were in a single `try/except`; the first one (`controllers`) fails on robosuite 1.5+ due to old API references, aborting the whole try block before reaching `grippers` | Each module now imported in its own `try/except` so failures are isolated |
| 8 | `gripper_types: 'RoundGripper'` was ignored even after #7 | Even with RoundGripper registered, the loaded MuJoCo model contained Robotiq85 bodies. Three layered bugs: (a) `load_robots_configs` passed `controller_config` (wrong key — robosuite expects `composite_controller_config`); (b) it passed `mount_type` (wrong — expects `base_type`); (c) it never threaded `gripper_type` at all, AND returned the empty input list instead of the populated `self.robot_configs` | `RobosuiteEnv.__init__` now forwards `gripper_types` from `robosuite_env_params`; `load_robots_configs` accepts `gripper_types`, uses the right keys, and returns `self.robot_configs` |
| 9 | Puck couldn't move in z (no juggle possible) | The puck was attached via three constrained slide joints (`x`, `y`, `yaw`) — no z motion. Even with gravity enabled, the puck couldn't fall onto the paddle or bounce off it. This silently broke the puck-juggle task whenever robosuite was the simulator | Replaced the three slides with a single MuJoCo `<freejoint>`; updated `set_obj_configs` to write 7-element pose qpos and 6-element velocity qvel for the free joint |
| 10 | `affected_by_gravity` was accepted but ignored | Default was `False` in robosuite spawn (vs. `True` in Box2D), so even after #9 the puck wouldn't fall under gravity unless the task explicitly enabled it. Most tasks (including PuckJuggle) call `spawn_puck(pos, vel, name)` with no gravity arg | Default is now `True`. Per-puck flag stored in `self.puck_gravity_flags` and applied via `body_gravcomp` (1.0 = full antigravity, 0.0 = normal gravity) every reset |

After pass 2: the round paddle (yellow disc, geom name
`gripper0_right_wiping_surface2br`, radius 0.0508 m) is attached to the EEF
and visible in all camera views. The puck spawns at the +x end of the
tilted table and slides toward the agent end under gravity, demonstrating
that the juggle task initialization is functional. Verified via
[`scripts/render_puck_juggle.py`](../../../../scripts/render_puck_juggle.py).
**Caveat:** the puck currently stops against `rim_home_top` before reaching
the paddle because the IK-solved EEF pose was tuned for the no-gripper EEF
body, so with the round paddle attached the paddle face hangs ~10 cm above
the table at the agent end. Re-tuning the IK target to put the paddle
surface (not the EEF body) on the tilted plane is a follow-up.

The new collision-disabling and IK-pose machinery is **harmless to the old
PPO/SAC training pipeline** — those configs would have hit the same OSC
crash, so any prior runs either avoided this code path or worked around it
with different YAMLs.

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
   2026-04-30.** See bug-fix log #1.

2. ~~**Robot starts inside the table; OSC crashes on first step.**~~ **Fixed
   2026-05-01.** See bug-fix log #2-#5 (four interlocking bugs). Sim now
   holds the EEF on the table with `action=zeros` and tracks delta inputs
   correctly. Verified with `scripts/render_robosuite_views.py`.

3. ~~**Puck not visible (spawns under tilted table; rendered gray).**~~
   **Fixed 2026-05-01.** See bug-fix log #6.

4. **No `observation_state_info` / `observation_puck_history` /
   `observation_paddle_history` (observation-delay snapshots).**
   - The base env at [`airhockey_base.py:881-893`](../../../../airhockey/airhockey_base.py)
     uses `getattr(simulator, "observation_state_info", None)` and falls back to
     the raw `puck_history` / `paddle_history`, so this only blocks if (1) is
     fixed. Listed here so the same structure as Box2D can be added if you ever
     want observation-delay parity in robosuite (currently no delay support at
     all).

5. **`has_renderer: true` in the existing juggle YAMLs.**
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

1. ~~Add `paddle_history` to `AirHockeyRobosuite` (cheapest unblocker).~~
   **Done 2026-04-30.**
2. ~~Get robot on the table + controllable + puck visible.~~
   **Done 2026-05-01** (six-bug debugging session, see [bug-fix log](#bug-fix-log-2026-05-01)).
3. Fix `has_renderer` defaults in the YAMLs / build a headless-friendly base
   sim YAML for the TD3 path.
4. Tune OSC `kp` / `output_max_pos` / `action_*_scaling` so a `[1, 1]` action
   has comparable per-step displacement to PID Box2D. Currently a `[0, 0.5]`
   action moves the paddle ~5 mm/step in y, which is ~10× slower than the
   Box2D PID setup.
5. (Optional) Port domain-randomization knobs (puck noise, occlusions, action
   delay) if you want sim2real story parity, not just "sanity-check on a
   different physics engine."

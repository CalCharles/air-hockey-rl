# Robot data collection

Scripted, open-loop data collection — no policy. Two collectors run the **same** trial
battery, one on the real UR5 and one in Box2D, so the two sets of trajectories can be
compared directly. Neither touches a puck.

`collect_paddle_motion.py` also carries **two further experiments**, each behind its own
flag and its own early exit in `main()`, sharing the env, the geometry and the file format
with the battery above but nothing else:

- `--puck-collision` — an interactive camera-triggered puck/paddle collision battery.
  See [Experiment 2](#experiment-2-puckpaddle-collisions---puck-collision).
- `--jerk` — an interactive out-and-back reversal battery that measures how hard the arm
  shakes when a trajectory turns around, and how much a linear deceleration ramp softens
  it. Paddle only, no puck. See [Experiment 3](#experiment-3-reversal-jerk---jerk).

For policy rollouts / training see [`scripts/td3/extras/`](../td3/extras/); for teleop and
calibration helpers see [`scripts/real/`](../real/).

| File | Role |
|------|------|
| `trial_plan.py` | Shared trial plan + start-pose geometry. Both collectors build their conditions here, so the same flags give the same trials in the same order. |
| `collect_paddle_motion.py` | Real UR5 battery. One HDF5 per trial. Also hosts the `--puck-collision` and `--jerk` experiments. |
| `collect_paddle_motion_sim.py` | Box2D battery. One HDF5 + one GIF per trial, plus a summary GIF per direction. |
| `diagnose_reset_descent.py` | Measures when the paddle actually reaches the table after a reset. Use it to check the clamp fix below, or to chase a "the arm starts before it's down" report. |
| `test_camera_puck_detection.py` | Live camera + puck-detection viewer. No robot, no env — opens the camera directly, runs the same homography + detector the real env uses, and draws the detected puck circle. Use it to check the camera index / lighting / detector thresholds before a session. |
| `replay_paddle_motion.py` | Replay a collected session's start poses + actions through Box2D and render both tracks overlaid. |

```bash
# print the trial plan; touches nothing
python -m scripts.robot_data_collection.collect_paddle_motion --dry-run

# real robot (36 trajectories in one session)
python -m scripts.robot_data_collection.collect_paddle_motion \
    --out-dir data/robot_data_collection/paddle_motion_$(date +%Y%m%d_%H%M)

# sim
python -m scripts.robot_data_collection.collect_paddle_motion_sim \
    --out-dir data/robot_data_collection/paddle_motion_sim_$(date +%Y%m%d_%H%M)

# experiment 2: interactive puck/paddle collisions on the real robot
python -m scripts.robot_data_collection.collect_paddle_motion --puck-collision

# experiment 3: interactive reversal-jerk battery (no puck)
python -m scripts.robot_data_collection.collect_paddle_motion --jerk
```

Run with `python -m` from the repo root (both modules also force-fix `sys.path`, see the
[ROS `scripts` package collision](../real/README.md#sourced-ros-env-avoiding-the-scripts-package-collision)).

**Note:** a `--directions` value starting with `-` must use the equals form
(`--directions=-x,+y`), or argparse reads it as a flag.

## The plan

| Knob | Default | Flag |
|------|---------|------|
| Directions (lines) | `+x, -x, +y, -y, diagpos, diagneg` | `--directions` |
| Curves (half arcs) | `arc_wide, arc_medium, arc_tight` | `--curves` |
| Curve speeds | 3 per shape, m/s along the arc | `--curve-speeds` |
| Curve tracking | `closed-loop` (action computed from the measured pose) | `--curve-tracking open-loop` |
| Curve steps | same as `--action-steps` (20) | `--curve-steps` |
| Action → velocity gain | 3.2 1/s real, 10 1/s sim | `--curve-velocity-gain` |
| Magnitudes | `0.33, 0.66, 1.0` | `--deltas` |
| Magnitude meaning | `action` (delta **is** the action) | `--delta-mode workspace` |
| Repeats per condition | 3 | `--repeats` |
| Constant-action steps | 20 | `--action-steps` |
| Zero-action settle steps | 3 | `--settle-steps` |
| Start pose | `max-room` | `--start-mode fixed` |
| Post-reset hold (real only) | 3.0 s | `--reset-settle-s` |
| Ordering | `grouped` (repeats back-to-back) | `--order interleaved` |

6 directions × 3 deltas × 3 repeats = 54 line trials, plus 3 arc shapes × 3 speeds × 3
repeats = 27 curve trials — **81 trajectories** total. Run one family alone with
`--curves=none` or `--directions=none`.

| Direction | Travel | Action at delta d |
|---|---|---|
| `+x` / `-x` | vertical (robot x, along table length); `-x` is the strike direction, toward table centre | `(±d, 0)` |
| `+y` / `-y` | horizontal (robot y, lateral) | `(0, ±d)` |
| `diagpos` | positive diagonal, **bottom-left → top-right** | `(-d, +d)` |
| `diagneg` | negative diagonal, **bottom-right → top-left** | `(-d, -d)` |

Frame conventions: `+x` is toward the robot base (the *bottom* of the vertical render, `x_max`),
`-x` toward table centre (*top*, `x_min`); `+y` is *right* (`y_max`), `-y` *left* (`y_min`).
Diagonals also answer to `-x+y` / `-x-y` and `diag+` / `diag-`.

### What `delta` means (`--delta-mode action`, the default)

`delta` **is** the action. Every axis a direction travels gets `±delta`; axes it doesn't travel
are zero. At `delta = 0.33`:

| Direction | Action |
|---|---|
| `+x` / `-x` | `(±0.33, 0.00)` |
| `+y` / `-y` | `(0.00, ±0.33)` |
| `diagpos` | `(-0.33, +0.33)` |
| `diagneg` | `(-0.33, -0.33)` |

Both diagonals carry a **negative x** component: they travel toward table centre (the strike
direction, "up" the table) and differ only in the y sign.

Because `rmax_x` (0.26 m/step at `|action|=1`) and `rmax_y` (0.12) differ, and the room differs
too (~0.33 m in x vs ~0.73 m in y from the diagonal corners), the axes run out at different
times. x saturates within a couple of steps and the paddle then slides along the far edge for
the remaining timesteps — accepted behaviour in this mode.

`--delta-mode workspace` is the alternative: `delta` becomes the fraction of the available room
to cover over the trial and the per-axis action is derived as
`action_i = delta * room_i / (action_steps * rmax_i)`, so both axes finish together — a straight
corner-to-corner path that never clips. It costs speed (top diagonal ~0.80 m/s at 20 steps vs
5.73 m/s commanded in `action` mode) and decouples `delta` from the action actually sent.

### Curved trajectories (`--curves`)

Three half-arc shapes, each swept at three speeds. The path is a semi-ellipse whose chord lies
along the bottom edge: the paddle starts at one end, rises up the table, and comes back down at
the other end, sweeping left to right without ever reversing in y.

| Shape | Start → end (y) | Half-width `a` | Height `b` | `b/a` | Character |
|---|---|---|---|---|---|
| `arc_wide` | `-0.340` → `+0.380` | 0.360 m | 0.160 m | 0.44 | mild bow, **bottom corner to bottom corner** |
| `arc_medium` | `-0.310` → `+0.350` | 0.330 m | 0.260 m | 0.79 | rounded, starts nearer the centre |
| `arc_tight` | `-0.280` → `+0.320` | 0.300 m | 0.360 m | 1.20 | tall arc, starts nearest the centre |

`arc_wide`'s half-width is the **workspace maximum** — 0.36 m is half the 0.74 m y span less the
boundary margin, so it cannot be widened further.

### The action is a velocity command, not a displacement

This is the single fact the arc code turns on, and getting it wrong is what made the arcs come out
a third of the table wide on the robot.

Both simulators rebuild the command target from the **current** pose every step —

```
target = pose_now + action * move_lims          # AirHockeyReal.get_transition
```

— and hand it to a tracking controller (`servoL(lookahead=0.2, gain=700)` on the UR5, a PID in
Box2D). The command never integrates: the paddle sits a fixed lead behind its target, and a fixed
lead settles at a fixed **speed**:

```
v_realised  ≈  G · (action · move_lims)          [m/s]
```

Measured off the straight-line battery in `data/robot_data_collection/paddle_motion_20260909_1808`:

| | G | action 1.0 sustains | realised / commanded step |
|---|---|---|---|
| **real UR5** (dt ≈ 0.048 s) | **3.2 1/s** | 0.83 m/s in x, 0.39 m/s in y | **15 %** |
| **Box2D** (dt = 0.05 s) | **10 1/s** | 2.6 m/s in x, 1.2 m/s in y | **50 %** |

So the robot's paddle is ~3× less responsive than the sim's, and neither one realises the per-step
displacement the action nominally names.

### Closed-loop arc tracking (default)

`--curve-tracking closed-loop` computes each action **online, from the measured paddle pose**.
Every step the tracker projects the paddle onto the target ellipse, places a carrot
`--curve-lookahead-steps` (default 2) steps further along, and asks for the action that produces
the requested path speed toward it:

```
action = clip( speed · unit(carrot − p) / (G · move_lims),  −1, +1 )
```

Two properties follow, and neither held for the old open-loop schedule:

* **The shape is right whatever `G` is.** The carrot is re-derived from the measured pose every
  step, so a mis-estimated gain changes how *fast* the sweep goes, not what it traces. Measured in
  Box2D, the realised path stays within **3 mm** of the ideal ellipse at every speed.
* **Running out of time is graceful.** When the requested speed is past what the axis can deliver
  the action just saturates at 1.0 and the paddle sweeps as far along the **full-width** arc as it
  can. That is the intended trade at 20 steps (below).

The feedback signal on the real robot is `getTargetTCPPose()` — the same quantity
`get_transition` anchors its command on, not the actual TCP pose, which trails it.

`--curve-velocity-gain` overrides `G`; each collector defaults to its own backend's value. To
recalibrate from a session: fit `speed_y / action_y` over the steady part of the `ypos`/`yneg` line
trials and divide by `rmax_y`.

### 20 steps does not fit a full-width arc — on purpose

`--curve-steps` (default: whatever `--action-steps` is, so 20) fixes every curve trial at the same
length as the line trials. At 20 steps × 0.048 s the paddle has **0.96 s**, and the arm tops out
near **0.39 m/s laterally**, so it covers roughly the first half of a corner-to-corner arc:

| Condition | speed | arc reached | realised y span | steps to complete |
|---|---|---|---|---|
| `arc_wide` | 0.22 / 0.32 / 0.45 m/s | 26 / 38 / 49 % | 0.16 / 0.26 / 0.35 m | 77 / 53 / 41 |
| `arc_medium` | 0.24 / 0.35 / 0.50 m/s | 26 / 38 / 49 % | 0.12 / 0.22 / 0.32 m | 78 / 53 / 42 |
| `arc_tight` | 0.28 / 0.42 / 0.60 m/s | 27 / 41 / 50 % | 0.09 / 0.21 / 0.30 m | 75 / 50 / 41 |

(Predictions against the identified first-order arm, printed by `--dry-run`. They run slightly
conservative: Box2D over-delivered them by 10–13 %.)

The alternative — shrinking the arc until it finishes in 20 steps — is what the old code
effectively did, and it is the thing being fixed: **a narrow arc traced fully is not what the
battery is for.** Every trial now sweeps a corner-to-corner arc and stops partway; at the top speed
that is 0.35 m of lateral travel against **0.22–0.25 m for every condition before**, and it is the
widest the arm can produce in the time.

Pass `--action-steps 60` (or `--curve-steps 60`, which leaves the line trials at 20) for a battery
where the arcs complete.

The three speeds span 2× and are anchored at the top by what the arm can hold — the fastest
saturates the y axis around the apex. They are deliberately *not* spread down to a crawl: with the
step count fixed, a slower path speed simply means less arc covered.

### The old open-loop schedule (`--curve-tracking open-loop`)

Kept so pre-fix runs stay reproducible. It bakes a fixed per-step displacement schedule:

```
theta: -90° → +90° over N steps,  N = round(arc_length / (speed · dt))
dx = b·sin(theta)·dtheta   →  ax = dx / rmax_x     (negative, then positive: up then down)
dy = a·cos(theta)·dtheta   →  ay = dy / rmax_y     (always ≥ 0: never reverses)
```

which is only the truth if the paddle reaches its target within one step. `--curve-tracking-gain`
(default 2.0) scaled the whole schedule to compensate, and 2.0 is a **Box2D** number: at
`G·dt = 0.5` it landed `arc_wide` at 0.72 m of realised sweep, so the sim looked correct. On the
robot `G·dt = 0.15` the same schedule delivered **0.24 m**, and the gain could not be raised to fix
it — it is clamped per trial so no action reaches 1.0, and a fast sweep has no headroom left.
Both flags are ignored under closed-loop tracking, as is `--min-curve-steps`.

In this mode a curve carries its own length (`N` above, 17–52 steps) rather than `--curve-steps`.

### Start poses (`--start-mode max-room`)

Each direction starts at the **far end of every axis it travels**, so the run is spent in free
space rather than pinned against a limit — which is exactly what makes the diagonals run corner
to corner. Axes a direction does *not* travel keep their `--base-robot-xy` coordinate (default
`-0.68 0.0`, the config's `hitting` reset pose), so every trial on an axis shares one starting
line. Poses are inset `--start-margin` (default 0.01 m) from the boundary, and on the x axis the
far end is the chamfered `effective_x_max`, not raw `x_max_lim`.

| Direction | Start (robot frame) | Room |
|-----------|--------------------|------|
| `+x` | `(-0.820, 0.000)` | x: 0.400 m |
| `-x` | `(-0.430, 0.000)` | x: 0.400 m |
| `+y` | `(-0.680, -0.340)` | y: 0.730 m |
| `-y` | `(-0.680, +0.380)` | y: 0.730 m |
| `diagpos` | `(-0.502, -0.340)` — bottom-left | x: 0.328 m, y: 0.730 m |
| `diagneg` | `(-0.502, +0.380)` — bottom-right | x: 0.328 m, y: 0.730 m |

The diagonal starts sit at x = -0.502 rather than -0.430 because the far edge is chamfered
(`effective_x_max`) near the corners — the arm cannot reach the true corner, so the start pose is
clamped to the reachable one at its own y.

`--start-mode fixed` restores the older behaviour: every trial starts at `--base-robot-xy`.

On the real robot the collector rewrites `reset_pose[0][:2]` before each `env.reset()`, using
the same mutable field the env's own random/preset reset paths write to.

### Saturation is still expected at the large deltas

Commanded displacement per step is `action * (rmax_x, rmax_y)` = `action * (0.26 m, 0.12 m)`.
In the default `action` mode the trials do saturate, by design: `20 × 0.26 = 5.2 m` on x and
`20 × 0.12 = 2.4 m` on y both exceed the 0.40 m / 0.73 m available, so the paddle ends every
trial against a limit, and a diagonal saturates x first and slides along the far edge into the
corner for the remaining steps. What `max-room` buys is the **transient plus a clean
steady-state velocity plateau** before that — the part worth comparing. Both scripts print the
predicted saturation step per direction before starting.

`--delta-mode workspace` avoids saturation entirely — verified across a full 54-trial sim
battery: no commanded target ever reached a workspace limit, worst deviation from a straight
start→end line 0.6 mm.

### No timestep smoothing

Both sims run every commanded target through a `filter_update` that averages the last
`hist_len` `(target - pose)` deltas. Both bundled configs set `hist_len: 1`, where that
average degenerates to the raw clipped target. The real config additionally zeroes the three
`transition_hold_steps_*` knobs so a recovery hold can never overwrite a commanded action
mid-trial. Pass your own `--config` and each script warns if a guarantee is missing.

### The paddle must be seated before the first step

Nothing in the stack used to detect table contact — `apply_negative_z_force` is open-loop,
`getActualTCPForce` was logged but never acted on, and `reset()` carried a TODO for exactly
this. `reset()` applied the clamp once after its first `moveL`, then let `forceMode` expire
(~2 s controller-side timeout) across its sleeps and the space-bar wait; its **final** `moveL`
parked the tool back at the reset pose's z with nothing re-clamping afterwards, because the
legacy post-stage clamp is gated on `high_reset`, which is hardcoded `False`. The paddle
therefore finished its descent during the first steps of the episode, while the policy was
already commanding x/y.

`reset_verify_seated: true` closes the loop. Before reset returns, `_verify_paddle_seated`
holds the clamp on and polls actual TCP z at 50 Hz until one of:

- **`z_settled`** — z stayed inside `reset_verify_range_m` for a full `reset_verify_window_s`
  (tuned: **1 cm over 1 s**);
- **`contact_force`** — `|TCP force z|` crossed `reset_verify_force_n`, if that is set > 0;
- **`timeout`** — `reset_verify_timeout_s` elapsed, which tears the clamp down, re-establishes
  it and restarts the window, up to `reset_verify_max_retries` times.

A final failure warns loudly but does not raise — a stuck reset shouldn't kill a collection run.
The verdict is stashed on `simulator.last_reset_verify_info`.

Two properties of this design are easy to miss:

- **The clamp must be live for the test to mean anything.** An unclamped paddle hanging
  motionless in mid-air has a perfectly constant z, so a settle check without a downward push
  would pass on exactly the failure it exists to catch. The loop refreshes the clamp every
  0.25 s while polling, and if not one clamp call lands it reports `clamp_unavailable` rather
  than "settled".
- **The band is a velocity bound.** `range / window` = 1 cm / 1 s means any residual creep
  slower than **1 cm/s** counts as settled. Tighten `reset_verify_range_m` if a slow drag shows
  up on this table.

**The knob defaults to `true`**, so every real-robot config gets the seat check — a reset that
returns with the paddle airborne corrupts the first steps of every episode, which is not
something to opt into. [`paddle_motion_config.yaml`](../../configs/robot_data_collection/paddle_motion_config.yaml)
restates it explicitly along with the tuned thresholds. Set `reset_verify_seated: false` to
restore the old unverified reset. Budget ~1.2 s per reset for the settle window (worst case
`timeout x (retries + 1)` = 10 s before it gives up and warns).

A third property to know: if the settled z lands within 2 mm of the reset pose's own z, the
run is flagged `settled_at_parked_height` and warns — once the paddle is airborne a `forceMode`
call only re-establishes contact if the commanded z is at or below the table, so settling
exactly where the `moveL` parked it means the clamp likely found nothing to press into.

To check it on hardware:

```bash
# legacy — expect a drop AFTER reset returns
python -m scripts.robot_data_collection.diagnose_reset_descent \
    --no-verify --start-preset bottom --out /tmp/descent_off.json

# with the seat check — expect "already seated at reset return"
python -m scripts.robot_data_collection.diagnose_reset_descent \
    --verify --start-preset bottom --out /tmp/descent_on.json
```

The diagnostic logs actual TCP z off RTDE at three points — the instant `reset()` returns,
during an idle watch where nothing is commanded, and once per `env.step` — then prints a
per-step table, what `reset()` itself concluded, and a verdict. `--action -1 0` drives it with
real motion instead of holding still. `--verify-window-s` / `--verify-range-m` /
`--verify-force-n` override the criterion for tuning; `--start-preset` covers the workspace
extremes (`bottom` = the paddle's own end, high robot x) and `--start-robot-xy` takes an
explicit pose.

### The 3-second post-reset hold (real only)

`--reset-settle-s` (default 3.0) holds at the reset pose before each trial so the arm is
fully at rest and every trial starts from the same state. The wait is **sliced and re-clamped**
rather than a plain sleep: UR's `forceMode` has a ~2 s controller-side timeout and nothing
refreshes it while we idle, so an unguarded 3 s pause would drop compliance and let the paddle
lift off the table. See
[`paddle-clamping-coverage-gap.md`](../../notes/docs/environments/real-world/paddle-clamping-coverage-gap.md).

## Output

```
<out-dir>/
├── manifest.json                                  # session metadata + one entry per trial
├── traj_000_xpos_delta0.33_trial1.hdf5
├── ...
└── gifs/                                          # sim only
    ├── traj_000_xpos_delta0.33_trial1.gif
    └── summary_xpos.gif                           # one per direction, whole delta sweep
```

### Real (`collect_paddle_motion.py`)

| Dataset | Shape | Contents |
|---------|-------|----------|
| `train_vals` | `[T, 35]` | Standard proprioceptive row — see the `vals_column_names` attr and [`proprioceptive_state.py`](../../airhockey/sims/real/proprioceptive_state.py). `pose_*` is the actual TCP pose in **robot frame**; `desired_pose_*` is the servoL target actually sent. |
| `train_img` | `[T, 240, 320, 3]` | Camera frames (gzip-9). Omit with `--no-save-images`. |
| `actions` | `[T, 2]` | The action passed to `env.step` each timestep. |
| `observations` | `[T+1, 30]` | Env observations, including the post-reset one at index 0. |
| `is_settle_step` | `[T]` | 1 during the zero-action settle phase, 0 during the constant-action phase. |
| `step_start_time`, `step_end_time` | `[T]` | Wall-clock bounds of each `env.step`. |
| `command_block_reason` | `[T]` | Anything other than `none` means the servoL was suppressed. |

If the robot reports a protective stop the trial is cut short (`aborted_at_step` attr,
`protective_stop` in the manifest) and the session stops after saving. Clear the stop, then
re-run only the missing conditions with `--directions` / `--deltas`.

### Sim (`collect_paddle_motion_sim.py`)

The puck is parked at the far end of the table and re-frozen every step, matching a real
session on a cleared table.

| Dataset | Shape | Contents |
|---------|-------|----------|
| `paddle_pos_robot` / `paddle_pos_table` | `[T, 2]` | Ground-truth paddle position. `_robot` = table frame minus `center_offset_constant`. |
| `paddle_vel` | `[T, 2]` | Ground-truth paddle velocity. |
| `target_pos_robot` / `target_pos_table` | `[T, 2]` | PID target after clipping (`last_target_position`). |
| `actions`, `observations`, `is_settle_step` | | Same meaning as the real files. |

## Experiment 2: puck/paddle collisions (`--puck-collision`)

A different experiment out of the same file. Everything above this section describes the
scripted battery, which runs on a cleared table; `--puck-collision` takes an early exit in
`main()` and none of the plan code runs, so the two cannot interfere.

**The question it answers:** what does a puck do when a paddle driving up the table at a
known constant action hits it, as a function of that action and of how fast the puck is
coming down?

**One trial:**

1. You type the puck **release height** — free text, `top` / `3/4` / `1/2`. It is a *label*:
   it is written into the file and the manifest and commands nothing. It records where you
   physically placed the puck.
2. You type the **delta** — the vertical action magnitude, `0` to `1`. The action sent is
   `(-delta, 0)`: straight up the table, y held at zero. **`0` is allowed** and means the
   paddle holds still: it drives to its start pose and stays there while the puck arrives,
   which is the stationary-paddle control condition.
3. You type the **y offset** — signed metres to shift the paddle sideways off the start
   pose. `0` parks it where every trial used to start; `+0.05` / `-0.05` move it across the
   table. This is the knob that sets how square or glancing the impact is.
4. You type the **trigger x** — the observation-frame x of the black line nearest the robot,
   the one whose crossing fires the strike. Normally a bare ENTER to reuse the last trial's
   value; type a number on the trials where you want to move the line.
5. That fourth ENTER starts the trial. Nothing moves before it. The arm resets to the bottom
   of the table (the `-x` vertical trials' start pose, robot `x = -0.430`) **with the y
   offset already applied**, and settles.
6. The script **arms**: it polls the overhead camera and runs the env's own puck detector,
   printing where it sees the puck. Release the puck now. The paddle does not move.
7. The instant a fresh detection **crosses the trigger line** moving toward the robot, the
   constant action is commanded for `--collision-action-steps` steps — the paddle drives
   straight up the table into the incoming puck from the offset y it has been holding.
8. A zero-action tail (`--collision-post-steps`) records the rebound, and the trial is
   saved as `collision_<idx>_h<height>_delta<d>_y<offset>.hdf5`.

Then you are prompted again. A bare ENTER at any prompt reuses the previous trial's value,
so a block of repeats at one condition is four ENTERs. Bad input re-asks only that one
question. `q` at any prompt finishes the session.

### The trigger line

The strike fires when the puck crosses the **black line nearest the robot** — the one just
in front of the paddle, *not* the bold line across the middle of the table.

`--trigger-x` is that line's position in **observation frame**, where the table centre
is `0.0`:

```
x_obs = +0.9652   bottom rail, behind the robot        (robot x = -0.235)
x_obs = +0.770    the paddle's start pose              (robot x = -0.430)
   ^^^ the near black line is somewhere around here — MEASURE IT
x_obs = +0.370    the paddle's furthest reach up-table (robot x = -0.830)
x_obs =  0.0      the bold black centre line — NOT the trigger  (robot x = -1.200)
x_obs = -0.9652   top rail, far end                    (robot x = -2.165)
```

**It is asked once per trial**, so you can move the line between trials without restarting
the session; a bare ENTER reuses the previous answer and `--trigger-x` pre-fills the first
prompt. There is no built-in default: where that line lands in observation coordinates
depends on your table, your camera and the homography — nothing in this repo knows it, and
quietly falling back to the centre line would run a different experiment from the one you
asked for.

Two ways to measure it once:

```bash
# with the robot: adjust the number until the drawn line sits on the black one
python -m scripts.robot_data_collection.collect_paddle_motion --puck-collision \
    --show-arm-view --trigger-x 0.6

# without the robot: hover the cursor over the line, read the robot-frame x,
# add center_offset_constant (1.2)
python -m scripts.robot_data_collection.test_camera_puck_detection
```

The puck rolling down toward the robot has an **increasing** `x_obs`, and the paddle never
reaches the centre line — contact happens somewhere in `x_obs ∈ [0.37, 0.77]` while the
puck is still travelling.

The trigger is a genuine crossing, not a half-plane test: `--trigger-min-far-frames` fresh
detections must land beyond the line before a detection at or past it fires. That is what
stops it arming on a puck already sitting on the robot's half, or on one stray detection.
Stale frames (the detector's last-known-position fallback) are recorded but never
trigger — the detector goes stale exactly when your hand reaches in to release the puck,
and firing on a held-over position would launch the paddle at nothing.

Every trial prints and stores `trigger_to_action_s`, the measured detection-to-first-command
latency, alongside the `trigger_x_obs` it used. If the paddle is consistently arriving late,
type a **more negative** trigger x on the next trial (fire earlier) rather than changing
anything else.

### Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--puck-collision` | off | Run this experiment instead of the scripted battery. |
| `--trigger-x` | — | **Seed** for the per-trial trigger-line prompt. The line itself is typed per trial; this just pre-fills the first one. Omit it and the first trial has to type a value. |
| `--trigger-min-far-frames` | `2` | Fresh detections needed beyond the line before a crossing can fire. |
| `--arm-timeout-s` | `25` | No crossing in this long → the trial is discarded (not saved) and you are re-prompted. |
| `--arm-print-every` | `15` | Print the detected puck position every N camera frames while armed. |
| `--arm-clamp-every-s` | `0.5` | Re-apply the paddle clamp this often while armed (forceMode times out after ~2 s). |
| `--show-arm-view` | off | Live rectified view with the trigger line drawn. |
| `--camera-buffersize` | `1` | `CAP_PROP_BUFFERSIZE` for this session. The trigger is latency-critical; `0` leaves the backend default. |
| `--collision-action-steps` | `20` | Constant-action steps after the trigger. |
| `--collision-post-steps` | `20` | Zero-action steps after the strike, recording the rebound. |
| `--collision-settle-steps` | `0` | Zero-action steps *before* arming. Off by default — see below. |

`--config`, `--out-dir`, `--reset-settle-s`, `--no-save-images`, `--no-wait`, `--dry-run`,
`--start-mode` / `--base-robot-xy` / `--start-margin` all mean what they mean above. The
plan flags (`--directions`, `--deltas`, `--repeats`, `--curves`, …) are ignored: the
direction is always `-x` and the delta always comes from the prompt.

`--collision-settle-steps` is 0 on purpose. Those steps would sit in `train_vals` separated
from the strike by however long you took to release the puck, so they read as a gap in the
trajectory rather than as an at-rest baseline. The arm-phase puck track (below) is the
better baseline.

### Output

Same layout as the scripted battery — one HDF5 per trial plus `manifest.json` — with the
trials named `collision_<idx>_h<height>_delta<d>_y<offset>.hdf5` and these additions:

| Dataset / attr | Shape | Contents |
|----------------|-------|----------|
| `arm_puck_track` | `[K, 4]` | The puck's whole approach at **camera rate** (~30 Hz), not env-step rate: `frame_time_s, puck_x_obs, puck_y_obs, puck_occluded`. This is where the incoming puck speed comes from. |
| `step_phase` | `[T]` | `settle` / `action` / `post` per step. `is_settle_step` stays 1 only for `settle`. |
| `y_offset` (attr) | | The offset you typed, in metres. Also in the filename. |
| `trigger_x_obs` (attr) | | The trigger line this trial used, as typed. |
| `start_robot_xy_commanded` (attr) | `[2]` | The offset start pose this trial actually parked at, after workspace clipping. |
| `session_start_robot_xy` (attr) | `[2]` | The unshifted start pose every trial in the session shares. |
| `terminated`, `truncated` | `[T]` | The task's own flags, recorded and **ignored** — the juggle task terminates the moment the puck reaches the bottom rail, which here is a normal outcome, so the schedule runs to the end regardless. |
| `puck_height_label` (attr) | | Exactly what you typed. |
| `trigger_puck_obs`, `trigger_time` (attrs) | | Where and when the crossing fired. |
| `trigger_to_action_s` (attr) | | Detection → first commanded step, in seconds. |
| `trigger_approach_speed_obs_m_s` (attr) | | Puck speed along x at the crossing, from the last two fresh detections. |
| `arm_wait_s`, `arm_poll_rate_hz`, `arm_polls`, `arm_fresh_frames` (attrs) | | Arm-phase diagnostics. |

`train_vals`, `train_img`, `actions`, `observations`, `step_start_time`, `step_end_time`
and `command_block_reason` are unchanged from the real battery above; `train_vals`'
`puck_x` / `puck_y` columns are in observation frame, the same frame as `arm_puck_track`.

#### Restarting into an existing directory

Point `--out-dir` at a directory that already holds trials and the session **continues the
numbering**: it reads the highest `collision_<n>_…` index on disk and starts at `n + 1`, so
nothing is overwritten. Files that don't parse as `collision_<n>_…` are ignored, so a
hand-renamed or half-written one can't push the counter somewhere odd. The session prints
where it is starting from.

`manifest.json` is rewritten by each session, so the earlier one is carried into a
`previous_sessions` list (oldest first, flattened) rather than lost — the current session
stays at the top level, and `first_trial_index` says where it began. Per-trial metadata is
duplicated in each HDF5's attrs regardless, so the manifest is a convenience index, not the
only copy.

Trials that time out without a crossing are **not** written — they appear in the manifest's
`skipped_trials` with a reason, and you re-enter the condition to retry.

### Why the arm phase re-runs the detector itself

It mirrors `AirHockeyReal.poll_puck_detection` (the detection-only tick the rollout startup
gate uses: no timestep advances, no motion is commanded) with two differences. The
rectified frame is handed back so `--show-arm-view` can draw on it, and nothing is appended
to `sim.images` — `merge_trajectory` pairs `sim.images` with `sim.vals` one-to-one and only
`env.step` appends to `vals`, so an arm-phase frame left in `images` would shift every image
of the saved trial by one. `sim.puck_history` *is* appended to, deliberately: the detector
gates candidates on the predicted next position, so it needs the same warm history while
armed that it has mid-rollout.

The clamp refresh during the arm phase is not optional. The wait is operator-paced and
routinely longer than UR's ~2 s forceMode timeout, so without it the paddle goes compliant-
free and lifts off the table before the puck ever arrives — the same failure documented in
[paddle-clamping-coverage-gap.md](../../notes/docs/environments/real-world/paddle-clamping-coverage-gap.md).

## Experiment 3: reversal jerk (`--jerk`)

A third experiment out of the same file, and like `--puck-collision` it takes an early exit
in `main()`, so none of the plan code runs and the three cannot interfere. Paddle only —
no puck, no camera.

**The question it answers:** how violently does the arm shake when a trajectory reverses,
and how much of that goes away if you ramp into the turnaround instead of reversing at full
speed?

**One trial** holds a constant action, ramps it linearly to zero over `t` steps and
immediately reverses, in `--jerk-steps` (20) commanded steps:

| phase | steps | action on the travel axis |
|-------|-------|---------------------------|
| `out` | `n_out` | `+delta`, constant |
| `slow` | `t` | `+delta * (t − k)/t`, `k = 1..t` — ends at exactly `0` |
| `back` | `N − n_out − t` | `−delta`, constant |

`n_out` and `t` are **both typed per trial** and used exactly as typed: 10 steps out, `t`
slowing, `20 − 10 − t` back. The ramp is straight-line interpolation from `delta` to zero —
consecutive scales differ by exactly `1/t` and the `t`-th step commands precisely `0`, so
the paddle is at rest at the turnaround before `−delta` is sent. At `delta = 1`, `n_out = 10`:

```
t=0   1 1 1 1 1 1 1 1 1 1                          -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
t=2   1 1 1 1 1 1 1 1 1 1  .5   0                  -1 -1 -1 -1 -1 -1 -1 -1
t=4   1 1 1 1 1 1 1 1 1 1  .75 .5  .25  0          -1 -1 -1 -1 -1 -1
t=6   1 1 1 1 1 1 1 1 1 1  .83 .67 .5  .33 .17  0  -1 -1 -1 -1
```

**Four conditions**, the two halves of the experiment:

| condition | axis | starts at | travels |
|-----------|------|-----------|---------|
| `up_down` | x (vertical, table length) | bottom of the table | up, then back down |
| `down_up` | x | top of the table | down, then back up |
| `right_left` | y (horizontal, lateral) | left edge | right, then back left |
| `left_right` | y | right edge | left, then back right |

Each starts at the far end of the axis it is about to travel — `start_pose_for` on the
**outbound** direction, the same "opposite end of the table" rule the scripted battery uses
— so the whole out-and-back runs in free space rather than against a limit.

**`t` is the experiment.** At `t = 0` the commanded action jumps from `+delta` to `−delta`
between two consecutive timesteps: the hardest reversal the action space can express, and
the one that shakes the arm. Raising `t` spreads that same reversal over a linear ramp. The
ramp's last step is exactly zero, so `t = 1` degenerates to a one-timestep pause. Sweep `t`
at fixed `delta` and fixed `n_out`, then read `acc_x` / `acc_y` / `acc_z` out of `train_vals`
(see [`vals_column_names`](#real-collect_paddle_motionpy)).

**The prompts** work exactly like the collision battery's: condition, delta, `n_out`, then
`t`, each taking a bare ENTER to reuse the previous trial's answer and `q` to finish. The
last one doubles as the "ready?" gate — nothing moves until it is answered. The condition
accepts `1`–`4`, the full key (`up_down`), or a short form (`up` / `down` / `right` / `left`).
`n_out` is asked before `t` so `t` can be capped at what `n_out` leaves behind, which is what
keeps the three phases summing to `N`.

```
[jerk] trial 0: condition (1=up_down, 2=down_up, 3=right_left, 4=left_right), or q to finish: 3
[jerk] trial 0: delta -- action magnitude on the travel axis, 0 <= d <= 1 (0.33 / 0.66 / 1.0): 1.0
[jerk] trial 0: n_out -- timesteps held at full delta in the initial direction, out of 20: 10
[jerk] trial 0: t -- timesteps ramping linearly to a stop before the reversal, 0 to 10 (0 = reverse at full delta): 4

[0] jerk_000_right_left_delta1.00_out10_slow04  [horizontal: toward +y (right) first, then back left]
  10 out + 4 slow + 6 back = 20 steps at |action|=1 (0.120 m/step); travel 1.380 m of 0.730 m room.
```

### Unattended batches (`--jerk-batch`)

`--jerk-batch` replaces the prompts with a scripted plan and runs it start to finish. Bare
`--jerk-batch` runs the default battery — 174 trials:

| block | delta | `t` | repeats | trials |
|-------|-------|-----|---------|--------|
| `up_down` | 0.66, 1.00 | 5, 4, 3, 2, 1, 0 | ×1 | 12 |
| `down_up` | 0.33, 0.66, 1.00 | 5, 4, 3, 2, 1, 0 | ×3 | 54 |
| `right_left` | 0.33, 0.66, 1.00 | 5, 4, 3, 2, 1, 0 | ×3 | 54 |
| `left_right` | 0.33, 0.66, 1.00 | 5, 4, 3, 2, 1, 0 | ×3 | 54 |

Blocks run in the order written; inside a block the order is delta-major, then `t`
descending, then repeat — so each delta's `t` sweep finishes before the next delta starts.
The repeat number lands in the file and the manifest as `repeat` (the filename is already
unique via the trial index).

Pass a SPEC to run something else: semicolon-separated
`condition:deltas:ts:repeats[:n_out]` blocks. The optional 5th field sets that block's
`n_out`, which is how you give the short x axis a shorter cruise than y:

```bash
# the default 174-trial battery
python -m scripts.robot_data_collection.collect_paddle_motion --jerk --jerk-batch \
    --out-dir data/robot_data_collection/reversal_jerk_$(date +%Y%m%d_%H%M)

# same sweep, but n_out chosen per axis so the outbound leg stays on the table
python -m scripts.robot_data_collection.collect_paddle_motion --jerk --jerk-batch \
    'up_down:0.33:5,4,3,2,1,0:3:2;down_up:0.33:5,4,3,2,1,0:3:2;right_left:0.33,0.66,1.0:5,4,3,2,1,0:3:4;left_right:0.33,0.66,1.0:5,4,3,2,1,0:3:4'

# see the plan, the per-block room check and the trial count without touching the robot
python -m scripts.robot_data_collection.collect_paddle_motion --jerk --jerk-batch --dry-run
```

The whole spec is parsed and validated **before** the env is built, so a typo fails
immediately rather than partway through a long session. The single ENTER safety gate still
applies (add `--no-wait` to skip it), a protective stop still stops the session, and Ctrl-C
still writes the manifest for whatever completed. Re-running a batch into a directory that
already holds trials continues the numbering, so an interrupted battery can be resumed by
running the remaining blocks into the same `--out-dir`.

### Watch the room — nothing clips `n_out` for you

The action is a **velocity command** of `rmax_axis * delta` metres per step, so `n_out` steps
ask for `n_out * rmax * delta` metres. The workspace is 0.41 m on x and 0.74 m on y, so
`n_out = 10` at `delta = 1.0` asks for 2.6 m on x: the paddle pins against the far edge after
two steps and spends the rest of the outbound leg stalled there, which makes the ramp and the
reversal start from a standstill and measures nothing.

Trials that overrun still run and still record — `saturates_at_step` is in the file — but a
clean reversal wants `n_out * step_m` inside the room. The plan report prints the predicted
pin step and the largest `n_out` that fits for every condition × delta before the arm moves,
and each trial prints it again as it starts:

```
[jerk] up_down     start robot xy=(-0.430,+0.000), room 0.400 m on x.  [vertical: up the table (away from the robot) first, then back down]
[jerk]   delta=0.33  step 0.086 m -> out 0.858 m + ramp 0.129 m = 0.987 m  !! asks 0.987 m of 0.400 m -- pins against the far edge at step 5; n_out <= 3 would stay inside at this delta and t
[jerk] right_left  start robot xy=(-0.680,-0.340), room 0.730 m on y.  [horizontal: toward +y (right) first, then back left]
[jerk]   delta=0.33  step 0.040 m -> out 0.396 m + ramp 0.059 m = 0.455 m
[jerk]   delta=1     step 0.120 m -> out 1.200 m + ramp 0.180 m = 1.380 m  !! asks 1.380 m of 0.730 m -- pins against the far edge at step 7; n_out <= 4 would stay inside at this delta and t
```

y is the axis with room to work in. On x, a 20-step out-and-back only stays inside the
workspace at the small deltas — `--jerk-deltas` previews all three so you can see which
`n_out` each one can take before you start typing trials.

### Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--jerk` | off | Run this experiment instead of the scripted battery. |
| `--jerk-steps` | `20` | Commanded steps per trial, split `out + t + back`. |
| `--jerk-batch` | off | Run a scripted battery with no prompts. Bare flag = the 174-trial default plan; a SPEC runs your own. See [above](#unattended-batches---jerk-batch). |
| `--jerk-out-steps` | `10` | **Seed** for the per-trial `n_out` prompt. |
| `--jerk-slowdown-steps` | `0` | **Seed** for the per-trial `t` prompt. `0` is the bang-bang reversal — the baseline the ramps are compared against. |
| `--jerk-delta` | — | **Seed** for the per-trial delta prompt. Omit it and the first trial has to type one. |
| `--jerk-condition` | — | **Seed** for the per-trial condition prompt. |
| `--jerk-deltas` | `0.33 0.66 1.0` | Deltas the plan report previews per condition. It does not *run* them — the trials are the ones you type. |
| `--jerk-settle-steps` | `0` | Zero-action steps before the outbound leg, on top of `--reset-settle-s`. An at-rest acceleration baseline in the same file as the reversal. |

`--config`, `--out-dir`, `--reset-settle-s`, `--no-save-images`, `--no-wait`, `--dry-run`,
`--start-mode` / `--base-robot-xy` / `--start-margin` all mean what they mean above. The
plan flags (`--directions`, `--deltas`, `--repeats`, `--curves`, …) are ignored: the
direction pair and the delta come from the prompt.

```bash
# geometry only, no robot
python -m scripts.robot_data_collection.collect_paddle_motion --jerk --dry-run

# first prompts pre-filled: 10 out + 4 slowing + 6 back
python -m scripts.robot_data_collection.collect_paddle_motion --jerk \
    --jerk-condition up_down --jerk-delta 0.66 \
    --jerk-out-steps 10 --jerk-slowdown-steps 4 \
    --out-dir data/robot_data_collection/reversal_jerk_$(date +%Y%m%d_%H%M)
```

### Output

Same layout as the other two — one HDF5 per trial plus `manifest.json` — with the trials
named `jerk_<idx>_<condition>_delta<d>_slow<t>.hdf5` and these additions:

| Key | What |
|-----|------|
| `step_phase` (dataset) | `settle` / `out` / `slow` / `back`, one per commanded step. This is what segments the acceleration trace against the commanded reversal. |
| `slowdown_scales` (dataset) | The ramp multipliers this trial used, so the schedule is reconstructable from the file without re-deriving it from `t`. |
| `condition_key`, `axis`, `motion` | `up_down` …; `x`/`y`; `vertical`/`horizontal`. |
| `out_direction_key` / `back_direction_key` | The two `DIRECTIONS` keys the legs travel, with their `*_vec` pairs. |
| `out_steps`, `slowdown_steps`, `back_steps`, `total_steps` | The split, as typed. |
| `repeat` | Which repeat of an identical condition this is (always 1 for a typed trial; 1..N under `--jerk-batch`). |
| `step_m`, `room_m`, `out_m`, `ramp_m`, `travel_m`, `back_m` | What the split asks for, in metres on the travel axis. |
| `saturates_at_step` | `-1` if the outbound leg fits; otherwise the 1-indexed commanded step at which the paddle starts pinning against the far edge. Use it to drop stalled trials from an analysis. |
| `max_out_steps_fitting` | The largest `n_out` that would have stayed inside the workspace at this delta and `t`. |

Restarting into a directory that already holds `jerk_<n>_…` trials continues the numbering
and carries the old manifest forward under `previous_sessions`, exactly as the collision
battery does.

## Replaying a session through Box2D

`replay_paddle_motion.py` closes the loop: it reads each trial's **start pose** and
**action sequence** out of the HDF5, resets Box2D to that start pose (puck parked and
frozen), steps it through exactly those actions, records the trajectory the sim went
through, and renders a GIF with both tracks overlaid.

```bash
python -m scripts.robot_data_collection.replay_paddle_motion \
    --session data/robot_data_collection/paddle_motion_sim_20260909_1608

# a subset, no GIFs
python -m scripts.robot_data_collection.replay_paddle_motion \
    --session <dir> --trials traj_018 traj_024 --no-gifs
```

It takes either kind of session and reads which from the file layout:

- a **real** session — the overlay is a genuine sim-vs-real comparison and the error curve
  is the transfer gap;
- a **sim** session — the replay should land on top of itself, which is how you check the
  pipeline is faithful before trusting it on real data.

#### One-step alignment (real sources only)

`AirHockeyReal.get_transition` reads the TCP pose (`_resolve_state_pose_speed`) and logs the
`train_vals` row (`get_state_array` → `self.vals.append`) **before** it sends that step's
`ctrl.servoL` — grep those three call sites in `airhockey/sims/air_hockey_real.py` to confirm
the order, which is what matters here. So
`train_vals[i]` is the state observed *before* `actions[i]` took effect, while the sim
collector records post-step state. The replay shifts the real trajectory forward one step so
`positions[i]` is the state `actions[i]` produced, and drops the final action whose result is
therefore unobserved (23 steps in → 22 compared). Without this, every real-vs-sim number is
biased by one 50 ms step. `--no-real-step-shift` disables it for inspecting the raw alignment.

Output lands in `<session>/replay/` (override with `--out-dir`): one
`<trial>_replay.hdf5` per trial holding `source_pos_robot` / `source_vel` /
`source_target_robot` alongside `replay_pos_robot` / `replay_vel` /
`replay_target_robot` and the per-step `position_error`, plus
`gifs/<trial>_overlay.gif`, a `summary_<direction>_overlay.gif`, and a `manifest.json`
carrying per-trial and session-wide mean/max error.

In the GIF the **green** track is the source trajectory from the file and the **magenta**
track is what Box2D just produced; the header shows the live position error in mm. Source
is drawn wider and underneath, so a perfect match reads as a magenta core inside a green
halo rather than one track silently hiding the other.

`--config` selects the Box2D config used for the replay (default: the collection config).
Pointing it at a perturbed config is a quick way to see how a given parameter change moves
the tracks apart.

### Metrics

Every replay prints a table and writes `metrics.csv` (one row per trial) and `metrics.txt`
(the rendered table) next to the trajectories. Per-step arrays `position_error` and
`delta_error` are in each trial's HDF5.

| Metric | Definition |
|--------|-----------|
| `sum_pos_err_mm` | `Σ_t ‖p_source(t) − p_replay(t)‖` — per-timestep position differences, summed over the trial |
| `sum_delta_err_mm` | `Σ_t ‖Δp_source(t) − Δp_replay(t)‖` where `Δp(t) = p(t) − p(t−1)` — per-timestep motion differences, summed |
| `max_pos_err_mm` | worst-case `‖p_source − p_replay‖` over the trial |
| `max_delta_err_mm` | worst-case per-step motion difference |

The table rolls up per-trial rows into per-condition subtotals and one overall aggregate:
sums add, maxima take the worst case, and a per-trial mean row keeps the aggregate
comparable to an individual trial.

The two families answer different questions. **Position error accumulates** — once the tracks
separate they stay apart, so a large `sum_pos_err` mostly reflects how early the divergence
started. **Delta error is local**: it is the per-step motion residual, so it localises *where*
the dynamics disagree and is the one to regress against when tuning sim parameters.

### Self-check on the 2026-09-09 sim session

Replaying `paddle_motion_sim_20260909_1608` through its own collection config reproduces
all 36 trials **exactly** — mean and max position error 0.000 mm. Start pose plus action
sequence is therefore a complete description of a trial: nothing else about the run needs
to be carried for a replay to be faithful. Replaying the same session with `pid_kp` halved
and `paddle_density` halved instead gives mean 84 mm / max 247 mm, so the overlay does
separate when dynamics differ.

## Comparing the two later

Per timestep, the directly comparable quantities (both in **robot frame**, both from the same
start pose under the same action schedule with smoothing off):

| Sim | Real |
|-----|------|
| `paddle_pos_robot` | `train_vals[:, 5:7]` (`pose_x`, `pose_y`) |
| `paddle_vel` | `train_vals[:, 11:13]` (`speed_x`, `speed_y`) |
| `target_pos_robot` | `train_vals[:, 26:28]` (`desired_pose_x/y`) |

[`scripts/visualization/replay_real_in_sim.py`](../visualization/replay_real_in_sim.py) is the
existing side-by-side renderer if you want frames rather than curves.

Two things to know before reading too much into a sim run:

- **Sim repeats are bit-identical.** Box2D is deterministic and the collection config turns off
  puck noise and occlusions, so the 3 repeats of a condition produce the same trajectory. The
  repeats exist for the real side; use `--repeats 1` if you only want sim data.
- **`+y` stops at the rail, not the workspace limit.** The sim's side wall pins the paddle
  centre at `width/2 - paddle_radius = 0.381 m`, inside the configured `y_max = 0.39`. `-y`
  reaches its `-0.35` workspace limit first. This asymmetry is a real sim/real difference, not
  a collection bug.

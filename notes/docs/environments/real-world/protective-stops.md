# UR5 protective stops: causes and diagnostics

Reference for interpreting UR5 protective-stop errors seen during real-robot
rollouts and teleop. Focuses on the failure modes we've actually hit in this
codebase; cross-links to the relevant fixes.

## What a protective stop is

The UR5 controller runs an internal safety loop (~500 Hz) that compares each
joint's **actual** encoder position against the **commanded** position the
motion planner is streaming. If the gap exceeds a per-joint tolerance, the
controller assumes something has physically interfered with the motion and
trips a protective stop — stopping all motion and requiring manual clear on
the teach pendant.

Our Python side sees this as:

- `self.rcv.isProtectiveStopped()` → `True`
- The reward-side teardown path reports
  `terminated (protective_stop)` from
  `scripts/real/rollout_new.py` and the reset FSM
  (`airhockey/sims/air_hockey_real.py:robot_command_readiness`).

## "Position deviates from path (SHOULDER)" — what it means

The pendant text `Position deviates from path (SHOULDER)` is the specific
message when the **shoulder** joint (joint 1, base rotation) deviates from
the planner's time-parametrized joint trajectory. Other joint names
(`ELBOW`, `WRIST_1/2/3`, `BASE`) mean the same error on a different joint.

"Path" here is *not* a geometric line — it is the sequence of joint angles
per millisecond the planner generated for the current `moveL` / `moveJ` /
`servoL`. Tripping the check means actual-vs-commanded drifted mid-move.

## Common causes, with examples

1. **Contact without compliance** *(the failure on 2026-04-17 — see below)*
   - Tool rests on or presses into a surface while a plain `moveL` continues
     to advance the commanded pose *through* that surface. Without
     `forceMode` compliance, actual z is pinned at the surface while
     commanded z keeps descending — shoulder torque saturates, position
     gap opens, trip.
   - *Mitigation in this repo*: `apply_negative_z_force(self.ctrl, self.rcv)`
     is called at three sync sites
     (`airhockey/sims/air_hockey_real.py:1448`,
     `airhockey/sims/air_hockey_real.py:1490`,
     `airhockey/sims/air_hockey_real.py:1595`) and/or by the async z-force
     worker (`_async_z_force_worker`, currently broken — see below).

2. **Physical obstruction / collision.** External object pushes on the tool
   or an arm link.
   - *Example*: paddle clips the table rail during an aggressive `moveL`;
     rail refuses to move; encoders stall; trip.

3. **Stale compliance from a prior command.** Force mode from an earlier
   call is still active when a plain `moveL` commands exact tracking —
   compliance drags the actual pose off the commanded path.
   - *Mitigation*: reset sequence calls `self.ctrl.forceModeStop()` at
     `airhockey/sims/air_hockey_real.py:1349` before any `moveL`.

4. **Aggressive velocity / acceleration for the arm configuration.**
   Commanded speed exceeds what feed-forward torque can sustain; inertia
   lag grows into a position gap. Most common on the shoulder with a
   near-fully-extended arm.

5. **Near a singularity.** `moveL` is Cartesian; controller inverts to
   joint-space. Near singularities a small Cartesian step demands near-
   infinite joint velocity — controller can't follow.

6. **Joint limit clipping.** Trajectory would require a joint past its
   soft limit; commanded is clipped; actual diverges from what the planner
   expected.

7. **Wrong payload / CoG.** `setPayload()` mass or centre-of-gravity
   wrong → gravity compensation torque wrong → joint droop → trip under
   load.

## The 2026-04-17 incident — root cause and fix

### Symptom

Running

```
python scripts/real/rollout_new.py \
    --config-path configs/real_configs/rollout_td3_config.yaml \
    --model <ckpt>/training_state.pth \
    --train-args <train_run_dir>/args.yaml \
    ...
```

triggered `Episode ended due to: terminated (protective_stop)`. The
controller pendant reported **"Position deviates from path (SHOULDER)"**.
The failure occurred during the reset's up-then-down `moveL` sequence,
before any sideways policy motion.

### Root cause

Interaction of two unrelated facts:

1. `async_z_force_enabled` defaults to `True` in
   `airhockey/sims/air_hockey_real.py:336`.
2. The async z-force worker **always fails to spawn** with
   `One of the RTDE input registers are already in use!` — see
   [`known-issues.md`](../../known-issues.md) entry #2 and
   [`async-z-force-future-steps.md`](async-z-force-future-steps.md) for
   the full diagnosis.
3. A mutual-exclusion gate added 2026-04-17 skips every sync
   `apply_negative_z_force(...)` call site when
   `async_z_force_enabled=True` (see diff:
   `airhockey/sims/air_hockey_real.py:1448`,
   `airhockey/sims/air_hockey_real.py:1486`,
   `airhockey/sims/air_hockey_real.py:1592`).

Net result with the default config: **no force-mode compliance was being
engaged at all** — neither async (worker dead) nor sync (gated off). The
reset's downward `moveL` targeted the table surface expecting compliance
to absorb the overshoot; without it, rigid contact → shoulder saturation →
"position deviates from path (SHOULDER)".

This is cause #1 in the list above.

### Fix applied

Set `async_z_force_enabled: false` under `simulator_params:` in every
real-sim config. This re-enables the three sync clamp sites and restores
force-mode compliance during reset and each env step. Files updated
2026-04-17:

- `configs/real_configs/rollout_td3_config.yaml`
- `configs/real_configs/rollout_config.yaml`
- `configs/real_configs/mouse_config.yaml`
- `configs/real_configs/primitive_exploration_config.yaml`
- `configs/baseline_configs/random_configs/puck_vel_real.yaml`
- `configs/baseline_configs/random_configs/paddle_pos_neg_regions_real_preset.yaml`
- `configs/baseline_configs/random_configs/puck_height_real.yaml`

Each entry is commented pointing to
[`async-z-force-future-steps.md`](async-z-force-future-steps.md) so the
override can be removed cleanly once the threads+lock redesign lands.

## Diagnostic checklist

When a "position deviates from path" trip fires, walk this list in order
before assuming it's a mechanical collision:

1. **Is the tool on/near the table surface at trip time?** If yes, suspect
   cause #1. Confirm by checking whether any `apply_negative_z_force` call
   ran in the rollout log — grep stdout for `[control_gate] ... forceMode`.
   If none fired, check `async_z_force_enabled` in the loaded sim config.
2. **Was `forceModeStop` called before the offending `moveL`?** Grep log
   for `[control_gate] reset:start forceModeStop done`. If absent, suspect
   cause #3.
3. **What was the commanded speed/accel?** Check `self.reset_pose[1]`
   (speed) and `self.reset_pose[2]` (accel) in
   `airhockey/sims/air_hockey_real.py`. Defaults are conservative; look
   for recent overrides.
4. **Is the arm near full extension?** Causes #4 and #5 become likely.
5. **Was `setPayload` called with the correct paddle mass/CoG?** Check
   `airhockey/sims/real/robot_control.py` or the env `__init__` path.

## References

- `airhockey/sims/air_hockey_real.py:336` — `async_z_force_enabled`
  default.
- `airhockey/sims/air_hockey_real.py:1349` — `forceModeStop` at reset
  start.
- `airhockey/sims/air_hockey_real.py:1448,1486,1592` — sync z-force call
  sites (gated on `not self.async_z_force_enabled`).
- `airhockey/sims/real/robot_control.py` — `apply_negative_z_force` helper.
- `notes/docs/known-issues.md` #2 — async z-force worker RTDE register
  conflict.
- `notes/docs/environments/real-world/async-z-force-future-steps.md` —
  threads+lock redesign plan.

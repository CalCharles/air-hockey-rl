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
  the async real-world entrypoint (`scripts/td3/extras/async_td3_real.py`) and the reset FSM
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
     is called synchronously at three sites in `AirHockeyReal` —
     `reset()` main_stage, `reset()` post_stage (high-reset only), and once
     per `get_transition()`. There is **no** background loop reapplying force
     between env steps; UR's `forceMode` has a ~2 s controller-side timeout,
     so any phase where the main loop pauses for longer (long blocking
     `moveL`s inside reset, inter-episode waits, transition holds) drops
     clamping. See
     [`paddle-clamping-coverage-gap.md`](paddle-clamping-coverage-gap.md).

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
python -m scripts.td3.extras.async_td3_real_eval \
    --config configs/real_configs/rollout_config.yaml \
    --args-file configs/td3_real_world/td3_residual.yaml \
    --model-path <ckpt>/training_state.pth \
    --train-args <train_run_dir>/args.yaml \
    ...
```

triggered `Episode ended due to: terminated (protective_stop)`. The
controller pendant reported **"Position deviates from path (SHOULDER)"**.
The failure occurred during the reset's up-then-down `moveL` sequence,
before any sideways policy motion.

### Root cause

Interaction of two unrelated facts (state at the time of incident):

1. `async_z_force_enabled` defaulted to `True` in `AirHockeyReal`.
2. The async z-force worker **always failed to spawn** with `One of the RTDE
   input registers are already in use!` — only one
   `RTDEControl(FLAG_USE_EXT_UR_CAP)` connection is allowed per robot, and the
   main process already held it.
3. A mutual-exclusion gate skipped every sync `apply_negative_z_force(...)`
   call site when `async_z_force_enabled=True`.

Net result with the default config: **no force-mode compliance was being
engaged at all** — neither async (worker dead) nor sync (gated off). The
reset's downward `moveL` targeted the table surface expecting compliance
to absorb the overshoot; without it, rigid contact → shoulder saturation →
"position deviates from path (SHOULDER)".

This is cause #1 in the list above.

### Fix applied

The immediate 2026-04-17 fix was to set `async_z_force_enabled: false` under
`simulator_params:` in every real-sim config so the three sync clamp sites
fired again. On 2026-05-05 the async path was removed entirely — the worker
function, multiprocessing flags, the `_sync_async_z_force_flags` helper, and
the `not self.async_z_force_enabled` gates on the three sync sites are all
gone, and the per-config `async_z_force_enabled: false` overrides were
stripped along with them. Sync clamping is now the only path; its limits
(>2 s gaps during reset / inter-episode phases break clamping) are documented
in [`paddle-clamping-coverage-gap.md`](paddle-clamping-coverage-gap.md).

## Diagnostic checklist

When a "position deviates from path" trip fires, walk this list in order
before assuming it's a mechanical collision:

1. **Is the tool on/near the table surface at trip time?** If yes, suspect
   cause #1. Confirm by checking whether any `apply_negative_z_force` call
   ran in the rollout log — grep stdout for `[control_gate] ... forceMode`.
   If none fired in the seconds leading up to the trip, the wrench had
   probably timed out (UR `forceMode` ~2 s timeout); read
   [`paddle-clamping-coverage-gap.md`](paddle-clamping-coverage-gap.md).
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

- `airhockey/sims/air_hockey_real.py` — `forceModeStop` at the start of
  `reset()`, plus the three `apply_negative_z_force` sync call sites
  (grep for `apply_negative_z_force`).
- `airhockey/sims/real/robot_control.py` — `apply_negative_z_force` helper.
- [`paddle-clamping-coverage-gap.md`](paddle-clamping-coverage-gap.md) — the
  >2 s coverage gap left by the sync-only path, stopgaps, and the
  threaded-worker fix sketch.

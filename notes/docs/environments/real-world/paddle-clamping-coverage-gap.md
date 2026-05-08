# Paddle clamping: sync-only coverage gap

The paddle is held against the table by repeated downward `forceMode` calls
(`apply_negative_z_force` in `airhockey/sims/real/robot_control.py`). All
clamping is synchronous and tied to the main control loop. There used to be an
out-of-process async worker that re-applied force at 150 Hz; it never worked
(RTDE register conflict with the main process) and was removed 2026-05-05. This
doc captures the gap that single-path leaves and what to do about it.

## How clamping fires today

Three call sites in `airhockey/sims/air_hockey_real.py`, all on the main
process's `self.ctrl`:

1. **`reset()` main_stage** — once after the main reset `moveL` completes.
2. **`reset()` post_stage** — once more, only when `high_reset and not above_table`. `high_reset` defaults `False`, so this site is dead in the standard real-world configs.
3. **`get_transition()`** — once per env step (~20 Hz cadence) while the
   policy / reset FSM is stepping the env.

Grep `apply_negative_z_force` to find the live call sites; the call sites move
when `air_hockey_real.py` is reorganised, so trust the grep over hard-coded
line numbers in any doc.

## The controller-side timeout

UR's `forceMode` has a **~2 second timeout on the controller side**: if no fresh `forceMode`
call lands within that window, compliance is dropped and the wrench dies. The
robot then goes rigid in z and any residual command pose is followed exactly.

At the per-step `get_transition` cadence (~20 Hz, ~50 ms per step), the timeout
is invisible — every step refreshes it. The gap appears whenever the main loop
is *not* stepping the env.

## Where the gap opens

Phases where `get_transition` does not run for >2 s:

- **Inside `reset()` itself**, between the main_stage `apply_negative_z_force` call and the next `get_transition` of the new episode. The intervening sequence — `time.sleep(0.7)`, optional space-bar wait, blocking final-stage `moveL`, `time.sleep(0.2)`, `time.sleep(0.7)` — usually exceeds 2 s. In particular, the explicit `forceModeStop()` at the very start of the next reset cancels any force still in flight.
- **Between episodes**, while the orchestrator runs reset-FSM stop checks, episode-artifact handling, transition holds after a protective-stop clear, and any "wait for N consecutive puck detections" gates before kicking off the next policy episode.
- **Long blocking `moveL` calls inside reset** (the high-reset pre-stage, main-stage, and final-stage moves are all `asynchronous=False`).
- **Any training-side stall** that pauses the collector — learner backpressure, slow checkpoint write, slow camera frame fetch.

In any of these cases the wrench expires, the paddle goes compliant-free in z,
and gravity / residual contact force lifts the paddle slightly off the table.
Once it's airborne, the *next* `forceMode` call only re-establishes contact if
the commanded z target is at or below the table; if a `moveL` between the
expiry and the next force call moved the commanded z away from the table, the
next force call has no surface to press into and the paddle keeps drifting.

This is the failure mode behind reports like "the arm keeps trying to reset
but no force is applied down" — every reset cycle re-issues `forceModeStop` →
`moveL` → one `forceMode` and then leaves the paddle alone for several
seconds, and once contact is lost neither the reset FSM nor subsequent resets
re-establish it.

## Stopgap mitigations (small, no architecture change)

In rough increasing-effort order:

1. **Pump `apply_negative_z_force` after every blocking `moveL` inside `reset()`** — there's currently only one such call (after the main_stage `moveL`); add one after the final-stage `moveL` and one immediately before `reset()` returns. Cheap, makes the post-reset gap shorter than 2 s in the common case.
2. **Pump it from any orchestrator-side wait loop that can run >1.5 s** — the reset-FSM stop-clear poll, the puck-detection gate, transition-hold loops. Either expose a public `AirHockeyReal.refresh_z_clamp()` and call it from the wait loop, or have the orchestrator call `env.simulator.air_hockey_env.ctrl.forceMode(...)` directly with the same args (the latter ties the orchestrator to the RTDE handle, so the public method is preferable).
3. **Sleep cadence**: any deliberate `time.sleep(s)` with `s >= 1.5` near the control path should be replaced with a poll loop that pumps the force every ~500 ms.

These keep the single-process, single-`RTDEControl` design and just plug holes.

## Real fix (when there's appetite for it)

Run a background **thread** (not process — the UR controller only allows one
`RTDEControl(FLAG_USE_EXT_UR_CAP)` connection per robot, total) that calls
`self.ctrl.forceMode(...)` at ~150 Hz for the lifetime of the program, sharing
the main process's `self.ctrl`. This is what the (now-deleted) async worker
was trying to do via `multiprocessing.Process`, which is fundamentally
incompatible with the one-`RTDEControl`-per-robot constraint.

Sketch of the work:

- `threading.Thread(target=worker, daemon=True)` started once in `AirHockeyReal.__init__` right after `self.ctrl` is constructed; stopped on `close()`.
- A single `threading.Lock` (`self._ctrl_lock`) wrapped around **every** `self.ctrl.*` call site in the module — ~20-30 sites, mechanical but must be complete. ur_rtde's C++ implementation is not documented thread-safe; one unlocked call is enough to interleave protocol frames and corrupt the connection.
- Worker honours pause flags for: protective stop, controller disconnect, transition holds, `control_off`, `above_table`. All of these already exist in `AirHockeyReal`; the thread just reads them.
- During main-loop `moveL`s, the thread should be paused via the existing transition-hold flag; resets already engage these holds.
- Tick-period drift logging — if any tick takes >1.5 s the wrench is at risk; alert the operator instead of silently dropping clamp.

## Pointers

- `airhockey/sims/air_hockey_real.py` — `reset()`, `get_transition()`, the call sites listed above.
- `airhockey/sims/real/robot_control.py` — `apply_negative_z_force` helper (single point of `forceMode` invocation).
- `notes/docs/environments/real-world/protective-stops.md` — what happens when clamping is *completely* absent (rigid contact during reset → "position deviates from path (SHOULDER)" trip).

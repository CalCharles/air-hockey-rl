# Async z-force: future steps

Plan for turning the currently-broken background z-force worker into a thread
that actually runs for the lifetime of the program. Capturing this so we can
revisit later without re-deriving the reasoning.

## Current state (2026-04-17)

- `AirHockeyReal` applies a downward wrench (`apply_negative_z_force`) to keep
  the paddle pinned to the table. There are two code paths today:
  - **Async worker**: `_async_z_force_worker` spawned as a
    `multiprocessing.Process` from `_start_async_z_force_worker_if_needed`
    (`airhockey/sims/air_hockey_real.py:711`). Target rate 150 Hz.
  - **Sync calls on the main loop**: `apply_negative_z_force(self.ctrl, …)`
    inline at three sites — `reset()` main_stage, `reset()` post_stage, and
    every `get_transition()` call (per env step).
- The async worker **always fails to start** with
  `worker startup failed: One of the RTDE input registers are already in use!`.
  Cause: the main process already holds `self.ctrl =
  RTDEControl(host, freq, FLAG_USE_EXT_UR_CAP)`, which claims the UR
  controller's External-Control URCap input registers. The child process tries
  to open its own `RTDEControl` with the same flag and is denied. This is
  inherent to multiprocessing — separate processes = separate TCP connections =
  second claim fails.
- What's actually keeping the paddle pinned today: the **sync call sites** on
  the main process using `self.ctrl`. The async worker silently does nothing.
- As of 2026-04-17 the two paths are **mutually exclusive** under the
  `async_z_force_enabled` flag — sync sites are gated on
  `not self.async_z_force_enabled`, so the flag cleanly selects one path. To
  keep clamping while the worker is broken, set
  `async_z_force_enabled: false` in the sim config. This override has been
  applied in all seven real-sim configs (see
  [`known-issues.md`](../../known-issues.md) entry #2 for the file list)
  after the default `True` + broken worker combination caused a "position
  deviates from path (SHOULDER)" protective stop on 2026-04-17 — see
  [`protective-stops.md`](protective-stops.md).

## Why we want the async path working

The sync calls only fire when the main loop is stepping (`env.step()`) or
during reset. During other phases — our new "wait for 5 consecutive puck
detections" gate, long user-input pauses, between episodes, training idle
stretches — nothing presses the paddle down and it drifts up off the table
surface. `forceMode` also has a ~2 s controller-side timeout, so bursts of
sync calls don't persist beyond the main-loop cadence. A true background
loop at 150 Hz holds contact continuously.

## Hard constraint

**One `RTDEControl(FLAG_USE_EXT_UR_CAP)` per robot, total.** The URCap
allocates a fixed register block; the controller denies a second claim. No
flag, no retry, no workaround. Any background force loop must share the one
`RTDEControl` instance that already lives in the main process.

## Design: threads, not processes

Drop `multiprocessing.Process`, use `threading.Thread`. Threads share memory,
so the worker calls `self.ctrl.forceMode(...)` on the exact instance the main
loop uses. No second RTDE connection. This is the core fix.

## What the port requires

1. **Swap the spawn machinery**
   - `multiprocessing.Process(target=_async_z_force_worker, …)` →
     `threading.Thread(target=…, daemon=True)`.
   - `multiprocessing.Event()` → `threading.Event()`.
   - `multiprocessing.Value("i", 0)` for `hold_active_flag`,
     `control_off_flag`, `above_table_flag` → plain `bool`/`int` attributes (or
     `threading.Event` each). `_sync_async_z_force_flags` still works; it just
     writes Python attributes.
   - Drop the `robot_host` arg from the worker signature — it uses
     `self.ctrl` and `self.rcv` directly.

2. **Mutex discipline on `self.ctrl` — the non-trivial part**

   ur_rtde's C++ implementation is not documented thread-safe for concurrent
   calls on one `RTDEControl` instance. Concurrent writes interleave protocol
   frames on the RTDE socket (multi-byte messages with sequence numbers); a
   corrupted frame drops the connection or mis-executes a command.

   Required: a single `threading.Lock` (`self._ctrl_lock`) around **every**
   `self.ctrl.*` call site in the module. ~20–30 call sites — grep
   `self\.ctrl\.` — mechanical but must be complete. One unlocked call is
   enough to corrupt state under load.

   Cheaper structural alternative (deferred): make the async thread the only
   writer to `self.ctrl`. Main loop posts requests onto a `queue.Queue`; the
   thread drains + runs the 150 Hz tick between drains. True single-writer,
   clean architecture, but every main-loop `self.ctrl` call becomes
   "enqueue + await ack" — significant rewrite. Not worth it unless we find
   the lock version unacceptable.

3. **Cooperation via existing flags**

   Force mode and motion commands interact on the robot side even with a lock:
   - `moveL` during an active force mode: they interleave via the compliance
     axes, but explicit `forceModeStop` → `moveL` → re-enable is the clean
     pattern we use today in resets.
   - `servoStop`/`forceModeStop`: if the async tick runs right after, it
     re-engages force mode, which may break the caller's intent.

   The existing multiprocessing worker already handles this correctly via
   `hold_active_flag`, `control_off_flag`, `above_table_flag` — the main loop
   sets them to pause the worker during resets / above-table moves / estop
   holds. Port this design unchanged; the thread honors the same flags.

   During long `moveL`s the main loop should still explicitly pause the thread
   (set `hold_active=True`), run the move, then unpause. This already happens
   in the reset sequence — carry it over.

4. **Lifecycle: start at `__init__`, not at `reset()`**

   Today `_start_async_z_force_worker_if_needed` runs from inside `reset()`.
   For "running at all times while the program is happening," start the thread
   once in `AirHockeyReal.__init__` right after `self.ctrl` is constructed
   (around `air_hockey_real.py:410`), and stop it on shutdown (`__del__`,
   atexit, or explicit `close()`). `reset()` can pause via flags but shouldn't
   (re)start the thread.

## Subtleties to check during implementation

- **`forceMode` timeout (~2 s).** If the thread ever goes >2 s between ticks
  (heavy GIL contention, long-held lock), compliance fades and the paddle
  floats until the next tick. 150 Hz → ~6.7 ms ticks, so only pathological
  stalls break this. Main-loop `moveL` under the lock taking 300 ms is fine;
  3 s is not. Log tick period and alert on drift.
- **Protective stop handling stays as-is.** The worker already reads
  `rcv.isProtectiveStopped()` and skips the tick. Ports over unchanged — just
  reads `self.rcv.isProtectiveStopped()` directly from shared state.
- **Camera / numpy / policy inference don't touch `self.ctrl`.** They run
  concurrently with the force-mode thread, no lock needed there.
- **Reconnection path.** If `self.ctrl` disconnects (controller restart), both
  the main loop and the async thread must coordinate a single reconnect.
  Today each process would try independently. With threads it's one
  `self.ctrl`, one reconnect path, behind the lock — simpler.
- **Revert the mutual-exclusion gating** added 2026-04-17 at the three sync
  call sites (`apply_negative_z_force` in `reset()` main_stage, `reset()`
  post_stage, `get_transition()`) once the thread is reliably running —
  though leaving the gating in place is also fine; the sync sites are dead
  code when async is enabled.

## Scope estimate

- Swap spawn + flags: ~30 lines.
- Lock-everywhere discipline: ~20–30 call sites, mechanical.
- Lifecycle move to `__init__`/shutdown: small.
- Testing: verify 150 Hz tick rate holds under policy rollout, force-mode
  never times out, no RTDE protocol errors under `moveL` + concurrent ticks,
  reset sequence still clean.

## Files to touch

- `airhockey/sims/air_hockey_real.py` — worker, start/stop, flag types, lock
  around `self.ctrl.*`, lifecycle.
- `notes/docs/known-issues.md` — update entry 1 once fixed.

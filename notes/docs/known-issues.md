# Known Issues

## 1. Paddle clamping coverage gap during reset / idle phases

The paddle is clamped to the table by synchronous `apply_negative_z_force`
calls (see `airhockey/sims/real/robot_control.py`) at three sites in
`AirHockeyReal`:

- once after the main-stage `moveL` in `reset()`
- once after the post-stage `moveL` in `reset()` (only when `high_reset and not above_table`, both default `False` in standard real configs — dead in practice)
- once per env step inside `get_transition()`

UR's `forceMode` has a **~2 second controller-side timeout**. Anywhere the
main loop is *not* stepping the env for >2 s — long blocking `moveL`s inside
reset, post-reset sleeps, "wait for puck" gates, transition holds after
protective-stop clears, training-side stalls — the wrench expires and the
paddle goes compliant-free in z. Once contact is lost, the next force call
only re-establishes clamping if the commanded z target is still at the table;
a `moveL` in the gap window can lift the commanded z and prevent recovery,
producing "the arm keeps trying to reset but no force is applied down" loops.

A previous out-of-process async worker (`_async_z_force_worker` /
`multiprocessing.Process`) was meant to plug the gap by re-applying force at
150 Hz. It never worked: the UR controller only allows one
`RTDEControl(FLAG_USE_EXT_UR_CAP)` connection per robot total, and the worker
process always failed startup with `One of the RTDE input registers are
already in use!`. The worker, multiprocessing flags, `_sync_async_z_force_flags`
helper, and the three `not self.async_z_force_enabled` gates on the sync call
sites were all removed 2026-05-05.

Stopgaps and the threaded-worker fix are written up in
[`environments/real-world/paddle-clamping-coverage-gap.md`](environments/real-world/paddle-clamping-coverage-gap.md).
The full incident write-up for the 2026-04-17 protective-stop trip caused by
having *zero* clamping (default `True` flag + broken async worker + sync sites
gated off) is in
[`environments/real-world/protective-stops.md`](environments/real-world/protective-stops.md).

# Final history comparisons and a lower actor learning rate are queued

- **Date**: 2026-10-29 23:21 UTC recording/writeup time, not new training start
- **Status**: in-flight (queues waiting; follow-up training planned)
- **Run dirs**: existing three history runs; planned `runs/td3/noisy_history_low_actor_lr_20261029_seed0`
- **Configs**: `configs/td3/sustained/noisy_history_low_actor_lr_scratch.yaml`

## Question
Does a smaller actor learning rate improve reliable survival under unchanged canonical noisy simulation?

## Setup
Fresh random actor/critic/encoder, original history recipe and6M budget; only actor LR changes3e-4→1e-4. Launch after original full6M and both frozen final comparisons; defer if either reaches32/32 selection caps. See [E042](../../../docs/experiment_logbook.md#e042) for exact sources, metadata and guard checks.

## Results
None yet. Existing three learners are still running; all six final evaluators and the follow-up queue are waiting. Target/online gaps motivate a hypothesis without proving instability is the cause.

## Next
Evaluate final models, inspect queued launch status, and assess any promising winner using unused independent starts. Never warm-start new training.

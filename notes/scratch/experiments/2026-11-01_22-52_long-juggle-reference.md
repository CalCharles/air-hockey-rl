# A fixed historical target policy enters a longer independent reference test

- **Date**: 2026-11-01 22:52 UTC launch/writeup.
- **Status**: in-flight
- **Run dirs**: `runs/td3/juggle_long_reference_20261101`
- **Configs**: frozen original history arguments/config, copied from `runs/td3/noisy_history_20261029_seed0/target_actor_3m6_bundle`; canonical noisy sensing and established full DR retained.

## Purpose and setup

While [six weaker smoothness learners](2026-11-01_22-48_weaker-smoothness-sweep.md) run, evaluate the unchanged retained 3.6M target actor over 32 new starts 9000000–9000031 with 100,000-step recordings. This measures longer-duration juggling and sampled physical motion under the same simulator. The policy was selected before these starts; any later policy comparison reusing them must label them as reused, not independent evidence for that later selection. This is not new policy training or a checkpoint warm start.

The copied frozen model SHA256 is `c8a1df78ade41cfbe6845a84f4b21480f512973be7a89f0baf66daf5d452b31b`, matching the earlier independent-test actor. Canonical configuration guards and previously reviewed simulator/base-env/config hashes passed before submission. No observations, dynamics, noise, occlusions, delays, interpolation, history sampling, PID, smoothing, geometry, contacts/workspace or randomization distribution changed. Only the evaluation recording horizon and new reset seeds differ.

## Execution

CPU-only Slurm job **115864** requests six CPUs, 16G memory and a four-hour limit with `--gres=none` and empty `CUDA_VISIBLE_DEVICES`. Confirmed running on slurm-node-003. Worker recorded evaluation start **2026-11-01 22:52:36.988738 UTC / 16:52:36.988738 -0600**. The six GPU learners remain live, and total user GPU allocations remain nine including three unrelated existing jobs. No performance result yet.

The worker invokes the existing evaluator with `--scenario-set training --episodes 32 --horizon 100000 --seed-base 9000000 --gif-steps 200`, then checks frozen weights and runs read-only physical speed/acceleration/jerk analysis. Raw episode steps, physical/sensed juggles, reward and motion distributions are primary; no binary acceptance gate. Maximum recording length is censoring, not proof of infinite juggling. Sampled Cartesian motion is not a UR5 protective-stop prediction.

## Artifacts and reproduction

- [Launch record, exact command and worker/batch sources](../../../docs/juggling-long-reference-launch.json).
- Local frozen inference bundle: `runs/td3/juggle_long_reference_20261101/frozen_target_3m6/`.
- Local progress: `status.json`, `slurm.log`; results planned under `independent_100k_eval/` with JSON, per-episode NPZ/GIFs and `motion_metrics.json`.
- Reference metrics from the earlier 20k independent test: [historical note](2026-10-31_19-00_history-training-plots.md).
- Reproduce using the exact command in the launch record and a fresh output directory; no need to retrain the historical model.

Results and figures are not yet available. Inspected actual Slurm process state and frozen model hash; source compiles. Corrected an escaped trailing-newline formatting defect in the worker status file without restarting or changing the live evaluation; the exact originally executed worker source remains in the launch record. The evaluator's result JSON is unaffected. Completion status may need the same read/write normalization when inspected after the worker exits.

## Concurrent initial checkpoint evidence

At documentation time, four live weaker-penalty learners had saved their first 100k-step eight-start evaluations: mean physical juggles 0–0.125 and mean episode steps 37.75–41.875. These are early training diagnostics, not evidence of full-budget policy quality. Published the [moving raw progress report](../../../docs/juggling-smooth-sweep.md) and its first checkpoint figure; all learners continue unchanged.

# Fresh history training tests whether penalizing action changes improves smoothness

- **Date**: 2026-10-31 22:43 UTC launch, from the runner status timestamp.
- **Status**: in-flight
- **Run dirs**: `runs/td3/noisy_history_smooth_20261031_seed0`
- **Configs**: [smooth scratch](../../../configs/td3/sustained/noisy_history_smooth_scratch.yaml), [unchanged simulator](../../../configs/new_juggle/sustained/original_sensing.yaml)

## Question and setup

Does penalizing abrupt commands improve motion while retaining useful juggling? The fresh actor, critics and 50-step history encoder initialize randomly. Relative to history seed 0, only run name and `actor_action_delta_penalty: 0.05` change. The actor loss adds `0.05 * mean((action - previous_action)^2)` to its normalized-Q objective. This is command smoothness regularization, not measured motor energy. The 6M-step budget, seed 0, gamma 0.995, sensing, dynamics, PID, action mapping/smoothing, workspace and full established ±25% DR remain fixed.

Before launch, compared effective YAMLs, passed the canonical environment/scratch initialization guards and checked simulator/base-env/config hashes against the previously reviewed originals in the actor-LR queue provenance. The simulator diff against merge `94acd81932c59f3cd3b827d6f39a0283747a1beb` was inspected during preparation; optional single-sample history remains disabled. Existing smoothness-gradient test passed during preparation ([A045](../../../docs/agent_logbook.md#a045)).

## Execution and tests

Slurm job **115589**, `juggle-smooth`, one GPU, six CPUs, 32G memory, 12-hour limit, running on `slurm-node-003` at first inspection. Runner recorded training start at **2026-10-31T22:43:21.203811+00:00**, which is **2026-10-31 17:43:21 -0500** America/Chicago. Initial status means process launch, not evidence of learning quality.

Built-in checkpoint evaluations use eight reused starts and 5k-step recordings every 100k steps. A durable orchestrator also queues online and target tests at fixed 1M, 3M and 6M steps: 32 reused comparison starts 700000–700031, 20k-step recordings. Each test copies a frozen inference bundle, checks input and weight hashes, and preserves per-episode JSON/NPZ plus the evaluator's GIF outputs. Final testing waits for successful full-budget completion. Raw metrics include physical/sensed juggles, episode steps, undiscounted reward, action/target delta RMS and physical peak paddle speed/acceleration. These repeated comparison starts are selection data, not independent generalization evidence. No binary acceptance criterion or early stopping is used. If an auxiliary test fails, its error is recorded and the learner is preserved to finish its authorized budget.

The wrapper additionally performs its established 32-start final 10k-step evaluation from seed 200000. Compare that with matching baseline protocol rather than mixing it with the 20k comparison. Recording limits truncate episode durations and juggle counts.

## Artifacts and reproduction

- [Launch provenance, exact batch source and reviewed hashes](../../../docs/juggling-smooth-training-launch.json)
- [Exact orchestrator source snapshot](../../../docs/juggling-smooth-training-watch-source.txt)
- Local logs: `runs/td3/noisy_history_smooth_20261031_seed0_slurm.log` and the run's `training.log`, `status.json`, `smooth_test_status.json`.
- Planned checkpoints: run's `train/checkpoint_*`; final weights under `train/`.
- Planned comparison outputs: `paired_{1000000,3000000,6000000}_{online,target}_20k_eval/` with source hashes and frozen bundles.
- Reproduce training with a fresh, nonexistent output path: `.venv/bin/python -m scripts.td3.juggle_noisy_scratch --root runs/td3/CHOOSE_NEW_SMOOTH_RUN --args-file configs/td3/sustained/noisy_history_smooth_scratch.yaml`.

No performance figures are available yet for this new trial. [Previous training curves](../../../docs/assets/juggling/history-checkpoint-curves.png) and [analysis](2026-10-31_19-00_history-training-plots.md) provide historical context only. Future reports should compare raw juggling and smoothness metrics together, without claiming measured improvements before results exist.

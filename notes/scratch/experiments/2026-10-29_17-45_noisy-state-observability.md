# Longer deployable history modestly improves saved-state estimation

- **Date**: 2026-10-29 17:45 UTC
- **Status**: done
- **Run dirs**: `runs/td3/noisy_scratch_20261028_seed0/observability/`
- **Configs**: CLI arguments in the [raw diagnostic results](../../../docs/juggling-observability.json)

## Question
Is there useful state information in control-step history beyond the canonical noisy 30-dimensional observation? This tests a possible policy-memory change without altering observations, physics or DR, and without training a policy.

## Setup
Saved trajectories from E008's selected and final scratch actors (64 episodes, seeds 300000–300031 under each actor). Split by seed across both actors: 24 training seeds (48 episodes, 31,817 usable transitions) and eight held-out seeds (16 episodes). Inputs contain only deployable observations/actions. Baseline receives canonical observation plus previous action; memory model also receives 12 prior latest-frame/action pairs, padded at reset. The current action and future frames are excluded.

Each fresh supervised MLP has two ELU hidden layers and predicts physical puck position immediately before the current action plus average physical velocity over the preceding 50 ms control step. Targets derive from saved actual body positions, never policy input. Train-only feature/target normalization; Adam 0.001, batch 256, 3,000 updates, initializer/sampling seed 0. No actor, critic, residual base, environment rollout or policy checkpoint is trained/loaded/modified in this diagnostic. Regression weights are discarded and will not initialize any future actor. Long episodes contribute more training transitions; test metrics equally weight episodes and aggregate paired actors within each held-out seed.

## Results

| Model | Parameters | Mean episode 2D position RMSE (cm) | Mean episode prior-step velocity RMSE (m/s) |
|---|---:|---:|---:|
| Canonical input, width 64 | 6,532 | 4.4436 | 0.7694 |
| Canonical input, width 128 | 21,252 | 4.2508 | 0.7438 |
| 12-step memory, width 64 | 12,676 | 4.1023 | 0.6587 |

![Held-out state-estimation errors](../../../docs/juggling-observability.png)

Figure generated 2026-10-29 from [raw results](../../../docs/juggling-observability.json). Each dot is one held-out seed, averaging its two actor episodes; connecting lines preserve pairing. Bars show those means. Memory reduces velocity error about 11.4% and position error about 3.5% relative to the larger short-input control. Mean paired per-seed velocity-error improvement 0.08506 m/s, seed-bootstrap 95% interval [0.07588, 0.09455] (10,000 resamples, default_rng seed 42). This reflects eight held-out environment seeds and one regression training seed, not policy training-seed variability. Physical velocity target is a finite-step mean, not post-contact instantaneous velocity.

## Reproduce

```sh
.venv/bin/python -u -m scripts.analysis.juggle_observability --roots runs/td3/noisy_scratch_20261028_seed0/selected_paired_eval runs/td3/noisy_scratch_20261028_seed0/final_paired_eval --out <new-results.json> --dt 0.05
```

Control-period argument was made explicit after this run; its default equals the 0.05 s used for every reported target. A targeted synthetic-data test verifies pre-action alignment, absence of current-action/future-state leakage and seed-group assignment.

## Conclusion
Control-step history contains useful additional predictive information in the saved noisy trajectories. The benefit is modest, absolute errors remain substantial, and this is supervised estimation on frozen-policy data rather than learned control. It does not establish improved juggling, sufficiency of memory, contact-time estimation quality or performance under a new policy's distribution.

## Next
Continue the three running scratch RL comparisons. A from-scratch history policy is a justified next architecture candidate if those do not meet sustained-survival requirements; do not reuse these regressor weights. At the latest 600k checkpoints baseline/reduced-exploration survival means are 311.5/308.375 on eight reused starts, both 0/8 caps, so the early 500k exploration ranking is not yet robust.

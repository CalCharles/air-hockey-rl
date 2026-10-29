# Read-only randomized-physics audit

Recorded 2026-10-29 15:27:48 -0500. Reconstructed all 32 initial resets of the completed history2.1M paired20k evaluation. Each initial30-dimensional observation matches its stored float32 array exactly. No policy actions or training launches; simulator and distribution unchanged. The driver refuses a mismatched observation or out-of-range physics parameter.

[Driver](../../../scripts/analysis/juggle_physics_audit.py), [per-start recovered parameters](../../../docs/juggling-history-2m1-physics.csv).

```sh
.venv/bin/python -m scripts.analysis.juggle_physics_audit --results runs/td3/noisy_history_20261029_seed0/paired_2m1_20k_eval/results.json --out docs/juggling-history-2m1-physics.csv
```

Each parameter's range fraction0–1 denotes position within its established ±25% interval, not a new distribution. Exploratory splits below are post hoc and involve only four capped episodes; they do not establish causality or independent generalization.

| Parameter | Lower-half starts / mean steps / caps | Upper-half starts / mean steps / caps |
|---|---|---|
| Paddle density | 12 / 5411.17 / 0 | 20 / 8059.35 / 4 |
| Puck damping | 16 / 6812.94 / 2 | 16 / 7319.63 / 2 |
| Gravity (upper means weaker negative gravity) | 16 / 5628.25 / 1 | 16 / 8504.31 / 3 |

All four caps have paddle-density fractions above0.5, but survival/density Pearson correlation over all starts is only0.056; do not infer a monotone density effect. Gravity correlation is0.361, damping0.115. Gravity fractions cover0.0046–0.9984 and density0.0072–0.9968 across the32starts, confirming a broad sampled range rather than nominal-only evaluation. These data motivate keeping the entire distribution and checking additional seeds, not narrowing physics. No training change follows from these small exploratory subgroups.

Validation: actual32reset replay assertions, range assertions, documentation links/anchors/TOCs and whitespace. No new rollout GIF for reset-only analysis; [E019](../../../docs/experiment_logbook.md#e019) contains the latest published clips.

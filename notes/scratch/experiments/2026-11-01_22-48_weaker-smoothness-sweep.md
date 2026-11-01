# Six fresh trials test weaker smoothness penalties while preserving juggling physics

- **Date**: 2026-11-01 22:48 UTC writeup; actual runner starts below.
- **Status**: in-flight
- **Related**: [failed 0.05 trial](2026-11-01_17-06_smooth-history-results.md)
- **Run dirs**: `runs/td3/noisy_history_smooth_{0005,002,008}_20261101_seed{0,1}`
- **Configs**: [0.0005 seed 0](../../../configs/td3/sustained/noisy_history_smooth_0005_seed0.yaml), [seed 1](../../../configs/td3/sustained/noisy_history_smooth_0005_seed1.yaml); [0.002 seed 0](../../../configs/td3/sustained/noisy_history_smooth_002_seed0.yaml), [seed 1](../../../configs/td3/sustained/noisy_history_smooth_002_seed1.yaml); [0.008 seed 0](../../../configs/td3/sustained/noisy_history_smooth_008_seed0.yaml), [seed 1](../../../configs/td3/sustained/noisy_history_smooth_008_seed1.yaml)

## Objective and evidence

The user authorized continuing simulation research toward high-performance juggling with smooth enough motion for eventual real-robot transfer, under a maximum of 16 total concurrent user GPUs. Infinite juggling and low UR5 stop risk cannot be established by finite Box2D episodes. This campaign aims to improve raw physical/sensed juggles and episode steps jointly with command and sampled physical motion distributions. It does not operate the real robot.

The completed coefficient-0.05 trial greatly reduced command changes but averaged only about one physical juggle and 95–97 steps. This motivates a weaker coefficient sweep with replication, not changing the simulator to make it easier. Each new actor/critic/history encoder starts randomly. Relative to the original history seed 0 recipe, only coefficient, run name and replication seed change; training remains 6M steps, gamma 0.995. Prior historical baselines provide comparison; repeated selection starts are not independent generalization evidence.

## Launch and simulator contract

| Coefficient | Seed | Slurm job | Runner start UTC | America/Chicago |
|---|---:|---|---|---|
| 0.0005 | 0 | 115858 | 2026-11-01 22:46:53.312558 | 2026-11-01 16:46:53 -0600 |
| 0.0005 | 1 | 115859 | 2026-11-01 22:46:53.320543 | 2026-11-01 16:46:53 -0600 |
| 0.002 | 0 | 115860 | 2026-11-01 22:46:53.280769 | 2026-11-01 16:46:53 -0600 |
| 0.002 | 1 | 115861 | 2026-11-01 22:46:53.283638 | 2026-11-01 16:46:53 -0600 |
| 0.008 | 0 | 115862 | 2026-11-01 22:46:53.284248 | 2026-11-01 16:46:53 -0600 |
| 0.008 | 1 | 115863 | 2026-11-01 22:46:53.283936 | 2026-11-01 16:46:53 -0600 |

Each job requests one GPU, six CPUs, 32G RAM and an 18-hour limit. Checked all pending/running user GPU allocations before each submission: three preexisting GPUs plus six new jobs gives nine total, below 16. Existing jobs remain untouched. Disk headroom at submission was 59G; a disk-headroom check preceded each job. No artifact cleanup was performed.

Passed exact effective-YAML comparisons, canonical environment and scratch guards, and saved-trainer-argument checks (coefficient, seed, `model_path: null`, history warm start false). Reviewed simulator/base-env diffs against merge `94acd81932c59f3cd3b827d6f39a0283747a1beb` and matched hashes to the previous reviewed launch. Preexisting changes are terminal/reward bookkeeping and optional history sampling that remains disabled. The canonical sysid-v2 sensing/control recipe, noise, occlusions, delays/interpolation, history sampling, PID, action smoothing, geometry, contacts/workspace and full established DR distribution remain unchanged.

[Launch provenance](../../../docs/juggling-smooth-sweep-launch.json) records exact batch commands, per-job manifests/hashes, initial queue, GPU count before every submission and runner timestamps. Jobs were confirmed running; TensorBoard then confirmed actor updates beyond the 20k-step random-exploration phase in all six runs (latest observed training steps approximately 22k–24k). This verifies learner activity, not policy quality.

## Testing and motion measurements

[Durable orchestration](../../../scripts/td3/extras/run_juggle_smooth_trial.py) runs each full-budget learner while evaluating frozen online and target copies at 1M, 3M and 6M. Each matched comparison uses 32 reused starts 700000–700031 and a 20k-step recording limit. All candidate comparisons report raw physical/sensed juggles, episode steps, reward and command changes. There is no binary acceptance gate or performance-driven early stopping. Each checkpoint also has existing eight-start 5k validation; the wrapper retains its existing final 32-start 10k protocol from seed 200000. Keep the protocols separate.

The new [read-only motion analysis](../../../scripts/analysis/juggle_motion_metrics.py) calculates distributions of speed, acceleration and jerk by finite differences of saved true paddle positions at the canonical control rate. It records per-episode RMS, median, p95, p99 and maximum for whole episodes and the first 250 steps. It does not affect policy inputs, training reward or environment state. Shorter episodes still contribute fewer observations; matched prefixes do not eliminate differing visited states or survivor bias. These sampled quantities exclude joint torques, UR5 dynamics and substep motion/contact spikes, so they do not predict protective stops.

Analytic tests cover constant velocity, constant acceleration, short recordings and invalid inputs. All three tests passed. Applying the diagnostic to the historical final online policies gave median episode-p95 speed/acceleration/jerk of 1.773 m/s, 17.546 m/s², 331.707 m/s³ for baseline versus 0.934, 5.577, 110.872 for coefficient 0.05; shorter episodes and different behavior prevent a causal physical-safety interpretation. This is new analysis of old saved trajectories, not new rollouts.

[Progress report](../../../docs/juggling-smooth-sweep.md) and [raw progress CSV](../../../docs/juggling-smooth-sweep.csv) are generated from available artifacts. New performance figures are unavailable until checkpoint evaluations exist. Planned longer tests on untouched starts follow comparison of the juggling/smoothness tradeoff; no candidate is yet selected for them, and no future result is claimed. Long tests remain censored observations, not proof of infinite juggling.

## Reproduction and continuation

- GPU launch commands and full source hashes: [launch record](../../../docs/juggling-smooth-sweep-launch.json).
- Fresh one-job orchestration: `.venv/bin/python -m scripts.td3.extras.run_juggle_smooth_trial --root NEW_ROOT --args-file CONFIG --manifest MANIFEST` inside an allocated GPU job. The manifest must contain reviewed source/config hashes; existing output roots are rejected.
- Report: `.venv/bin/python -m scripts.analysis.juggle_smooth_sweep`.
- Motion metrics: `.venv/bin/python -m scripts.analysis.juggle_motion_metrics --results RUN/paired_STEP_ROLE_20k_eval/results.json`.
- Tests: `.venv/bin/python -m unittest scripts.td3.tests.test_juggle_motion_metrics`.
- Local checkpoints/NPZ/GIFs/statuses remain in ignored run roots; source/configs, launch record and report are tracked. Frozen weights are hash checked after evaluation. Auxiliary test failures are recorded without interrupting the authorized full-budget learner.

Continue from the measured tradeoff: compare both training seeds, inspect failures and GIFs, then run independent longer trajectories for retained candidates. Any next policy still trains from scratch with the simulator contract intact. The goal remains active and unsolved.

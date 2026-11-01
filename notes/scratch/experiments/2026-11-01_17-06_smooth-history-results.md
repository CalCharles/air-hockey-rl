# Action-change penalty 0.05 reduced command changes but did not learn useful juggling

- **Date**: 2026-11-01 17:06 UTC retrospective analysis/writeup; no new rollouts.
- **Status**: done
- **Updates**: [launch note](2026-10-31_22-43_smooth-history-scratch.md); this dated completion record preserves its original in-flight status.
- **Run dirs**: `runs/td3/noisy_history_smooth_20261031_seed0`, baseline `runs/td3/noisy_history_20261029_seed0`
- **Configs**: [smooth scratch](../../../configs/td3/sustained/noisy_history_smooth_scratch.yaml), [baseline scratch](../../../configs/td3/sustained/noisy_history_scratch.yaml), [unchanged canonical noisy simulator](../../../configs/new_juggle/sustained/original_sensing.yaml)

## Setup and completion

The fresh actor, critics and history encoder trained for all 6M steps with `actor_action_delta_penalty: 0.05`; baseline effective differences remain only that coefficient and run name. The actor objective adds `0.05 * mean((action - previous_action)^2)` against normalized `(1-gamma)*Q`. This is a command-change penalty, not measured physical motor energy. Sensing, dynamics, PID, action smoothing, geometry, terminations and the established randomization distribution were unchanged. No warm start was used.

Slurm job 115589 completed with exit code 0 in 7:05:18, including evaluations. The runner recorded start **2026-10-31 17:43:21.203811 -0500** and completion of training plus its 10k final evaluation **2026-11-01 00:47:49.469522 -0500**; the auxiliary evaluation orchestrator completed **2026-11-01 00:48:37.544200 -0500**. These timestamps record process/status milestones, not exact rollout start/end times. This writeup was recorded after the daylight-saving change, at offset -0600. See [completion evidence](../../../docs/juggling-smooth-training-completion.json) and [original launch provenance](../../../docs/juggling-smooth-training-launch.json).

## Raw results

Each row averages 32 episodes. The six smooth milestone rows and two baseline final rows use the identical reused starts 700000–700031 and 20k-step recording limit. The `smooth final 10k` row instead uses seed base 200000 and a 10k limit; it is a separate protocol. Physical and sensed juggles use their existing trajectory/direction-reversal counters; contact timesteps are not juggle counts. Reward is the undiscounted episode sum. The actor regularizer is not subtracted from the reported environment reward.

| Policy | Physical juggles | Sensed juggles | Episode steps | Median steps | Reward | Action-change RMS | Target-change RMS (m) |
|---|---:|---:|---:|---:|---:|---:|---:|
| smooth 1000000 online | 0.34 | 0.34 | 57.66 | 50.50 | 29.47 | 0.03381 | 0.00741 |
| smooth 1000000 target | 0.34 | 0.25 | 60.38 | 54.00 | 31.31 | 0.03466 | 0.00775 |
| smooth 3000000 online | 0.97 | 0.97 | 101.75 | 97.50 | 56.81 | 0.04197 | 0.00811 |
| smooth 3000000 target | 0.94 | 0.81 | 97.88 | 93.50 | 57.53 | 0.03934 | 0.00795 |
| smooth 6000000 online | 0.84 | 0.88 | 94.84 | 96.00 | 59.09 | 0.04960 | 0.00987 |
| smooth 6000000 target | 1.00 | 0.91 | 97.31 | 91.00 | 59.62 | 0.04763 | 0.00977 |
| baseline 6000000 online | 77.53 | 68.03 | 5,447.00 | 2,419.50 | 3,991.75 | 0.40835 | 0.07393 |
| baseline 6000000 target | 138.41 | 121.72 | 9,859.81 | 7,798.50 | 7,363.72 | 0.39745 | 0.07267 |
| smooth final 10k | 1.19 | 1.03 | 103.72 | 100.00 | 61.75 | 0.04556 | 0.00929 |

![Final raw juggling and smoothness comparison](../../../docs/assets/juggling/smooth-final-comparison.png)

Figure generated 2026-11-01 from the historical saved results. Blue bars are the original history seed 0; orange bars are the action-change penalty trial. Counts and lengths are episode means, action-change RMS is averaged across episodes, and acceleration is the largest recorded value across each set of 32 episodes. This is one training seed for each condition, without a population uncertainty estimate.

The final online command-change RMS fell from 0.40835 to 0.04960 (87.9% lower), and target RMS from 0.39745 to 0.04763 (88.0% lower). Physical juggle means fell from 77.53 to 0.84 online and from 138.41 to 1.00 target. Mean lengths fell from 5,447 to 94.84 steps online and 9,859.81 to 97.31 steps target. The earlier 1M and 3M measurements also show fewer than one physical juggle per episode. This setting did not recover useful juggling by the full budget.

Final online largest recorded acceleration fell from 58.00 to 28.62 m/s²; target from 57.04 to 32.51 m/s². Episode lengths and visited states differ strongly, so maxima have different observation opportunities and these descriptive changes do not isolate physical smoothness at matched behavior. Likewise lower command changes may reflect little useful motion. Recording limits truncate durations/counts in baseline episodes; the smooth final episodes all ended well before 20k. No binary acceptance gate is used.

## Interpretation and next work

The 0.05 setting trades away useful juggling in this single-seed comparison. A weaker coefficient is a reasonable hypothesis, not an established fix or explanation of the full learning dynamics. No follow-up training was launched for this documentation task. Evaluate future changes jointly on physical/sensed juggles, episode steps, reward, command changes and motion metrics; keep canonical physics, sensing, control and DR unchanged, and initialize new policies from scratch.

## Sources and reproduction

- [Summary CSV with all nine protocols and source paths](../../../docs/assets/juggling/smooth-results-summary.csv)
- [Raw 288 episode rows](../../../docs/assets/juggling/smooth-results-episodes.csv)
- [Source hashes and figure-generation timestamp](../../../docs/assets/juggling/smooth-results-provenance.json)
- [Read-only export/plot script](../../../scripts/analysis/juggle_smoothness_results.py)
- [Exact launch/test orchestrator](../../../docs/juggling-smooth-training-watch-source.txt)
- Original local ignored JSON, NPZ trajectories, frozen bundles and checkpoints remain under the run directories above. Completed smooth outputs are `paired_{1000000,3000000,6000000}_{online,target}_20k_eval/` and `final_eval/`; per-test `frozen_provenance.json` records model hashes and commands. Final checkpoints are in `train/`; no checkpoint binary is added to Git.

Reproduce from the repository root with `.venv/bin/python -m scripts.analysis.juggle_smoothness_results` using the saved local ignored run artifacts. The script checks episode counts and recomputes summary means before exporting; the tracked CSVs and figure remain independently reviewable without the original run directory. Validation for this documentation task: raw-summary consistency, figure readability, relative links/images, unique anchors, complete ordered TOCs and `git diff --check`. No new policy rollouts or learning tests were performed.

# Two fresh trials combine a weak action-change penalty with longer discounting

- **Date**: 2026-11-01 23:09 UTC launch/writeup.
- **Status**: in-flight
- **Run dirs**: `runs/td3/noisy_history_smooth_0005_g0999_20261101_seed{0,1}`
- **Configs**: [seed 0](../../../configs/td3/sustained/noisy_history_smooth_0005_g0999_seed0.yaml), [seed 1](../../../configs/td3/sustained/noisy_history_smooth_0005_g0999_seed1.yaml)
- **Related**: [six weaker-penalty trials](2026-11-01_22-48_weaker-smoothness-sweep.md), [previous longer-discount scratch trial](2026-10-31_19-00_history-training-plots.md), [actual target-motion audit](2026-11-01_22-59_pid-target-replay-audit.md).

## Hypothesis and controls

Test whether gamma 0.999 with the weak action-change coefficient 0.0005 offers a better long-duration juggling/motion tradeoff than the current gamma-0.995 sweep. The earlier unregularized longer-discount final target averaged 198.63 physical juggles and 14,141.53 episode steps on the reused 32-start 20k comparison. That does not prove this combination will improve performance; it motivates the comparison while the original six learners continue unchanged.

Both new policies initialize actor, critics and history encoder randomly and train for 6M steps. Relative to the previous longer-discount scratch recipe, only run name, coefficient, replication seed and checkpoint cadence differ. Gamma is 0.999, action-change coefficient 0.0005, and checkpoint interval 500k rather than 100k to reduce storage use. Relative to the gamma-0.995 weak-penalty pair, discount and checkpoint cadence differ. The cadence affects available validation observations, so do not compare mismatched latest checkpoints as fixed-budget results. Paired 1M/3M/6M tests remain at identical training steps.

The actor objective is still normalized `(1-gamma)*Q` plus the same squared action-change cost. Raw environment reward is unchanged; actor-loss magnitudes across discounts are not a performance metric. Physical/sensed juggles, episode steps, reward and motion distributions remain primary, without binary acceptance gates.

## Launch and invariants

Slurm jobs **115866** (seed 0) and **115867** (seed 1) were submitted and confirmed running on slurm-node-008 and slurm-node-009 respectively. Each requests one GPU, six CPUs, 32G memory and an 18-hour limit. The [launch record](../../../docs/juggling-smooth-longdiscount-launch.json) contains exact submission timestamps and batch sources; runner starts were 2026-11-01 23:09:03.312581 UTC (seed 0) and 23:09:04.068727 UTC (seed 1), or 17:09:03/04 -0600 America/Chicago. These are process-start recording milestones, distinct from this writeup's recording time. Saved trainer arguments independently confirm gamma 0.999, coefficient 0.0005, checkpoint interval 500k, null model path and no history warm start.

GPU requests were counted before each submission, including pending and running jobs: counts were nine then ten, yielding **11 total user GPUs** after both submissions, including the three preexisting allocations. The 16-GPU ceiling is preserved. The separate long-reference job uses no GPU. Disk headroom was approximately 59.6 GiB; saving every 500k reduces new checkpoint storage without deleting historical artifacts.

Canonical environment/scratch guards and exact effective-YAML differences passed before launch. Simulator/base-env/config hashes match the reviewed six-trial manifest and previous strong-penalty launch. The simulator remains at the same reviewed branch implementation with default-false optional single-sample history disabled. Noise, occlusions, delay/interpolation, observation sampling, PID, action smoothing, geometry, contacts, limits and the full established DR distribution are unchanged. No pretrained weights or residual base policy is used. No shared trainer/updater/evaluator source was edited while the original six jobs run.

## Testing and artifacts

The same [durable trial runner](../../../scripts/td3/extras/run_juggle_smooth_trial.py) attaches frozen online and target tests at 1M/3M/6M: 32 reused starts 700000–700031, 20k-step recording limit. Built-in eight-start 5k validation now occurs every 500k. The wrapper also retains its final 32-start 10k protocol from seed 200000. Physical motion distributions are calculated read-only from saved trajectories. Legacy `target_delta_rms_m` is a displacement-change proxy; direct PID-target measurements require the separately verified replay diagnostic.

[Progress report](../../../docs/juggling-smooth-sweep.md), [raw summary CSV](../../../docs/juggling-smooth-sweep.csv) and its generator now include both launch records, showing gamma and seed explicitly in curve labels. Existing six-trial results remain separate curves. No result or figure from the two new policies is available yet. Original-sweep 100k–300k results are early diagnostics, not evidence of final quality.

Reproduce a fresh trial using the exact command/manifest in the [launch record](../../../docs/juggling-smooth-longdiscount-launch.json). Regenerate the report with `.venv/bin/python -m scripts.analysis.juggle_smooth_sweep`. Checkpoints, TensorBoard events, milestone statuses, frozen bundles, NPZs and GIFs remain under the ignored run roots. Source/config hashes and commands are tracked for review. Verified Python source compilation, effective configurations, live Slurm handles, documentation links/anchors/TOCs and whitespace; no new implementation math needs additional regression tests for this launch.

The prior goal turn was a verified wait on live jobs. This turn makes new progress by launching the controlled longer-discount pair; the research goal remains active and unsolved.

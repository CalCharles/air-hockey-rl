# The strongest completed regularized 3M target enters a paired long-duration test

- **Date**: 2026-11-02 01:48 UTC submission; exact submitted01:48:47.348637, worker evaluating01:48:48.045586. Documentation time is later and recorded in the logbooks.
- **Status**: in-flight
- **Run dirs**: `runs/td3/juggle_smooth_002_seed1_target3m_long_20261102`, private `frozen_target_3m` bundle and fresh `paired_reference_100k_eval` output.
- **Configs**: copied source bundle `runs/td3/noisy_history_smooth_002_20261101_seed1/frozen_3000000_target/{args.yaml,config.yaml,rma_meta.json}`; all file hashes and exact submitted sources are in [launch provenance](../../../docs/juggling-smooth-3m-long-reference-launch.json).

## Question and selection

Does the modest sampled-motion reduction of coefficient0.002 seed1 target persist over longer juggling, and how does its duration distribution compare with the retained reference? This policy was selected from the completed four-trial3M subset using32 reused starts700000–700031 at20k. It averaged155.09375 physical juggles/10690.3125 steps and had median paired p95 acceleration/jerk ratios .954291/.885122 versus the same-role3M unregularized control. The other training seed still juggled much less. Those are selection results, not long-duration findings.

![Completed3M selection context](../../../docs/assets/juggling/smooth-first-3m-comparison-20261102.png)

Existing figure generated2026-11-02 from the [completed subset](2026-11-02_01-44_first-3m-smoothness-comparison.md); the coefficient0.002 seed1 target bar explains the candidate choice. The figure is not a result of this newly running long test. Whole-episode motion has duration/state confounding; paired-prefix results are documented in that source note. No new performance figure is available while the long test remains pending.

## Fixed protocol

CPU-only job115892,6CPUs/16GiB/4h, explicitly no GPU resources and empty CUDA_VISIBLE_DEVICES. Evaluate the frozen actor on32 reset starts9000000–9000031 with100000-step recording limits, original sensing/control recipe and established full DR. The same starts/horizon were already used for the retained3.6M target's [complete long reference](2026-11-02_01-16_long-reference-results.md). This is a **paired reference cohort**, not a globally unseen cohort. The new actor has no outcome available on it at launch. Future reuse must keep that label; later independent claims require separately identified evidence.

Private frozen target model SHA256 `0a8cb0d061acc1839c8905bc7fb36f61cd879388237db4b1c42691a1d0774601`, byte-identical to the verified3M actor-target checkpoint from the scratch seed1 trial. No learning, checkpoint warm start, pretrained/residual initialization, simulator/control/sensing/DR change or training restart. All eight new learners retain their6M budgets. This inference test is separate from them.

The historical reference model is `c8a1df78ade41cfbe6845a84f4b21480f512973be7a89f0baf66daf5d452b31b` at3.6M, training seed0. Its existing mean546.25 physical juggles/37131.03125 steps belongs to that100k protocol. It must not be directly contrasted with the new candidate's155-juggle20k selection mean as though horizons match. The new paired long test also compares different training seeds/budgets, so it tests candidate behavior rather than isolating a causal regularization effect.

All candidate episodes start in a new output directory. **No historical reference NPZ is reused as a candidate outcome.** The evaluator uses the candidate actor's normal deployment observations; physical state is read only for diagnostics. Existing standard evaluator records true/sensed trajectories, raw juggles, steps, rewards, action changes, terminal reasons and first-episode GIFs; the worker computes saved-data motion diagnostics after all episodes finish. Runtime/source/bundle hashes are checked before/after evaluation. The command and source/checkpoint paths are frozen in provenance.

## Verification and reproduction

Before submission, checked the actor hash against the verified3M checkpoint, original scratch training args (`model_path: null`, no history warm start, history mode), exact equality of saved simulator YAML with `original_sensing.yaml`, and the original launch's reviewed hashes. New worker also protects renderer/history/evaluator/model/config inputs. Worker source compiled and Slurm reported RUNNING on node006. Eleven total user GPUs remain requested including three preexisting jobs, below16; this test adds zero. No implementation change or new implementation tests for launching the existing evaluator.

```sh
.venv/bin/python -m scripts.td3.juggle_survival --run-dir runs/td3/juggle_smooth_002_seed1_target3m_long_20261102/frozen_target_3m --scenario-set training --episodes 32 --horizon 100000 --seed-base 9000000 --gif-steps 200 --out-dir /tmp/regularized-target-long-reproduction
```

This is the reproduction command with a fresh output location; do not resubmit a duplicate while115892 is live. The exact submitted command/batch/worker source, source hashes, model files, selection evidence and historical comparison path/hash are in [provenance](../../../docs/juggling-smooth-3m-long-reference-launch.json).

## Next

Collect all32 episodes and authoritative completion before reporting full-cohort means. Compare raw per-start counts/durations and physical-motion windows against the retained long reference on these same starts, preserving the finite recording limits and policy-selection limitations. Continue remaining3M/6M milestone tests for the eight learners. No new result yet establishes infinite juggling, low UR5 loads or protective-stop safety; no real robot is connected.

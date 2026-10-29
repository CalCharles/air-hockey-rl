# History 2.1M checkpoint milestone

Recorded 2026-10-29 15:19:59 -0500. The completed shared benchmark improves to9/32 genuine10k caps, but remains unreliable. No change to the simulator, sensing/control or DR; normal continuation of the existing random-initialized history learner.

Checkpoint: `runs/td3/noisy_history_20261029_seed0/train/checkpoint_2100000`. Results: `runs/td3/noisy_history_20261029_seed0/milestone_2m1_benchmark/results.json`. Same32 reused seeds300000–300031, cap10000: mean4885.28125, median3190, min264,74.34375 mean physical-trajectory juggles,23failures. All9 caps satisfy the physical repeat-juggle heuristic. Mean action delta RMS0.4165324672. Previous1.6M mean3519.375,3/32 caps. Paired mean difference1365.90625,21/32 wins, paired bootstrap10000resamples with NumPy default_rng(42),95% interval[-332.4703125,3050.4234375]. The interval includes zero; one training seed and shared starts preclude a strong reliability/generalization claim.

Reproduction command (choose a new output directory):

```sh
.venv/bin/python -m scripts.td3.juggle_survival --run-dir runs/td3/noisy_history_20261029_seed0/train/checkpoint_2100000 --out-dir NEW_OUTPUT --scenario-set training --horizon 10000 --episodes 32 --seed-base 300000 --gif-steps 0
```

A paired20k evaluation is now **running**, root `runs/td3/noisy_history_20261029_seed0/paired_2m1_20k_eval`, samecommand with horizon20000 and seed-base500000. Those seeds were independent for the1.6M test but are now reused. No result is claimed yet. All three training processes were confirmed live. No new GIF generated for this milestone; the immediately preceding E019 contains two published clips. No implementation changes/tests in this task.

[Per-start comparison](../../../docs/juggling-history-benchmark.csv), [previous experiment](../../../docs/experiment_logbook.md#e019).

## Planned replication (not launched)

Prepared `configs/td3/sustained/noisy_history_scratch_seed1.yaml`, differing from the active history recipe only in training seed1 and run name. Scratch routing and canonical environment guards pass. It will train actor, critics and encoder from random initialization; no candidate checkpoint is a source. A second training seed will test whether the architectural gain replicates. No GPU allocated yet; existing learners continue. This is preparation, not a completed experiment.

## Completed longer-test status update — 2026-10-29 15:25:44 -0500

The paired20k evaluation completed: mean7066.28125, median5100.5, minimum173,108.71875 mean physical-trajectory juggles,4/32 genuine caps and28failures. Cap seeds500014/500016/500018/500021 have295/301/333/338 physical juggles. Previous1.6M on identical starts: mean3606.6875,0/32 caps. Paired mean improvement3459.59375,95% paired bootstrap interval[1170.4515625,5843.27734375],22/32 wins (10000resamples, default_rng42). Starts are reused; no new independent acceptance claim. This supports retaining2.1M as the stronger research candidate while normal training continues. Candidate manifest now points to its SHA25683819bb1d7ef4ccb3ccb0cb276f91a4de89505d1a2293b887172e2c59f698b0b. [Per-start20k comparison](../../../docs/juggling-history-20k-comparison.csv). No completion claim:28failures remain.

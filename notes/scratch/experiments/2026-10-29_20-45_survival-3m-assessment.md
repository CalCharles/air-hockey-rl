# Survival recipe: 3M assessment

Recorded 2026-10-29 15:45:09 -0500. The combined survival-reward/long-discount/slower-actor scratch trial has a new validation best but remains weak on the shared benchmark. Training is still running, not interrupted or completed.

Root: `runs/td3/noisy_survival_20261029_seed0`. At3M, eight reused validation starts10000–10007,cap5000: mean963,median915.5,8.875physical-trajectory juggles,0/8caps. Prior best at1.3M was544.625; the3M improvement changes the earlier tentative resource plan. The proposed stop was not executed. No processes were signaled or restarted.

Completed shared benchmark300000–300031,cap10000: mean717.65625,median708,min94,5.8125physical juggles,0/32caps,32failures. Physical and noisy observed juggle counts can differ (observed mean6.25). End reasons overlap: bottom20,passed-paddle13. Mean action-delta RMS0.2695658870. Raw returns are not comparable across reward recipes.

The2.1M history checkpoint on the same starts has mean4885.28125 and9/32caps. Different architectures, trainers, rewards, discount and actor optimization prohibit treating this as an isolated reward ablation; budgets also differ. One training seed and reused starts are research comparisons, not independent acceptance.

```sh
.venv/bin/python -m scripts.td3.juggle_survival --run-dir runs/td3/noisy_survival_20261029_seed0/train/checkpoint_3000000 --out-dir runs/td3/noisy_survival_20261029_seed0/milestone_3m_benchmark --scenario-set training --horizon 10000 --episodes 32 --seed-base 300000 --gif-steps 0
```

[Per-start shared comparison](../../../docs/juggling-history-benchmark.csv); raw result at `runs/td3/noisy_survival_20261029_seed0/milestone_3m_benchmark/results.json`. [Live curves](../../../docs/juggling-noisy-progress.md). No new GIF for this weaker milestone; the immediately preceding E022 publishes two verified history20k clips.

## Resource decision and next milestones

Keep all three existing learners running for now because the survival recipe just improved and the baseline is approaching its6M target (last verified5.21M). The prepared seed1 history replication remains **not launched** and can use the baseline slot after completion. This does not claim the survival recipe will catch up. The next fixed history benchmark is3M; automatic final/selected evaluators remain attached to the live jobs.

Replication preflight was reviewed during monitoring: original noisy environment and full±25% DR, random seed1, model_path null, history warm start false, no residual and optional simulator history mode false. Re-read simulator/base-env diffs versus required merge94acd81932c59f3cd3b827d6f39a0283747a1beb: only established reward/terminal bookkeeping and default-false history code; no new simulator changes. No new training was launched.

Validation: frozen benchmark completed, live process checks, saved-data report regeneration, relative links/image existence, unique anchors/TOCs and whitespace. No implementation changes or fresh implementation-test claim in this task.

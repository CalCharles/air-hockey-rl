# Longer-discount history: first checkpoint and next fixed benchmark

Recorded 2026-10-29 16:30:02 -0500. The gamma .999 trial's first saved checkpoint confirms the intended scratch runtime and real learning updates; early validation remains weak. All three history learners continue.

Checkpoint `runs/td3/noisy_history_long_discount_20261029_seed0/train/checkpoint_100000`: global step100,000,62,500critic updates,48actor and72critic optimizer-state entries. Saved args retain gamma .999, model_path null and history warm start false. Saved environment passes the canonical comparison with only bookkeeping seed normalized; metadata modehistory,50×8latest-frame/action memory. [Portable state/hash/config/evaluation evidence](../../../docs/juggling-history-long-discount-first-checkpoint.json).

Completed validation: eight reused seeds10000–10007,cap5000,mean62.75,median58,0.5physical-trajectory juggles,0/8caps. This is a pipeline check, not evidence that longer discount improves juggling. Seed0's original history trial also learned slowly at first. Do not stop these new scratch trials based on their initial random-policy behavior.

## Queued fixed 4M benchmark

A detached watcher1322650 is confirmed live, waiting for the original history seed0's complete `train/checkpoint_4000000/survival_eval/results.json`. It will validate the saved canonical config and evaluate that exact checkpoint, cap10000,32reused starts300000–300031,output `runs/td3/noisy_history_20261029_seed0/milestone_4m_benchmark/`. It is **waiting**, not a completed experiment. Source/log/status are sibling files `milestone_4m_watch.py`, `milestone_4m_watch.log`, `milestone_4m_status.json`. Same watcher logic that successfully completed E026, changing only checkpoint/output/status identifiers. It refuses existing output and records an error if training becomes terminal before the required checkpoint validation. No new policy initialization or selection by validation.

Equivalent frozen evaluation after availability, in a new output directory:

```sh
.venv/bin/python -m scripts.td3.juggle_survival --run-dir runs/td3/noisy_history_20261029_seed0/train/checkpoint_4000000 --out-dir NEW_OUTPUT --scenario-set training --horizon 10000 --episodes 32 --seed-base 300000 --gif-steps 0
```

No new GIF from the weak100k policy. Latest verified clip remains [E025](../../../docs/experiment_logbook.md#e025); the current3M research candidate still has27/32independent20k failures (E026). No simulator, sensing, control or randomization changes and no new training launched in this task. Checks: actual checkpoint/optimizer/config inspection, completed validation output, live watcher and learner processes, saved-data report generation, relative links/images, anchors/TOCs and whitespace. Earlier implementation tests were not rerun.

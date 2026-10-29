# New history validation caps and fixed 2M tests

Recorded 2026-10-29 17:56:56 -0500. Both newer scratch history trials have genuine 5k validation caps by 1.6M. These are reused validation starts, not independent replication or evidence of a discount advantage.

| Trial | First genuine cap checkpoint | Mean at 1.6M | Genuine 5k caps at 1.6M | Training seed | Gamma |
|---|---:|---:|---:|---:|---:|
| History replication | 1.2M (1/8) | 2,836.25 | 3/8 | 1 | .995 |
| Longer discount | 1.6M (1/8) | 1,349.875 | 1/8 | 0 | .999 |

The full saved history was checked: seed 1 first capped at 1.2M with mean 906.5, before its stronger 1.6M result. Gamma .999 first capped at 1.6M. Same eight reused starts 10000–10007 and physical-trajectory heuristic, not exact contact counting. Both learners passed long weak early phases and remain running toward 6M. [Portable checkpoint hashes and per-start evidence](../../../docs/juggling-history-new-trials-caps.json). No early termination or candidate promotion follows from this small validation set.

## Waiting fixed 2M comparisons

Four live watcher processes are waiting for complete 2M validation, then will freeze/evaluate online and target weights from each run on 32 reused starts 700000–700031 at 20k. Seed 1 online/target PIDs 1360532/1360533; longer-discount online/target 1360534/1360535. [Queue metadata](../../../docs/juggling-history-new-trials-2m-watch.json), [archived source](../../../docs/juggling-history-new-trials-2m-watch-source.txt). Source syntax compiled and startup states verified; no 2M outcome yet. The archive is for inspection, with paths relative to each live run directory.

The existing guarded 5M watcher logic is reused with fixed 2M checkpoint/output names. Heavy policy imports are deferred until a checkpoint is ready, reducing idle memory use. Each evaluation runs in its own process, verifies reviewed simulator/config hashes and canonical environment, refuses existing output and records a frozen weight hash before acting. Target weights are Polyak averages inside that same scratch run; no learner or warm start is created. No new training, reward, sensing, dynamics, PID/smoothing, geometry, contacts, workspace or DR modification.

These comparisons add candidate measurements to the research selection set; they are not new independent acceptance. Budgets differ from older 3.6M/4M/5M candidates, so scores alone cannot establish a causal gamma or seed effect. A selected policy still needs a locked new-seed test. Current best reference remains the 3.6M target with 18/32 caps and 14 failures on its independent 800000 cohort.

Equivalent evaluation after readiness, choosing new outputs:

```sh
.venv/bin/python -m scripts.td3.juggle_survival --run-dir runs/td3/noisy_history_20261029_seed1/train/checkpoint_2000000 --out-dir NEW_OUTPUT --scenario-set training --horizon 20000 --episodes 32 --seed-base 700000 --gif-steps 0
# Target view maps model.pth to actor_target.pth and shares saved metadata.
# Repeat for the longer-discount run; all are inference-only.
```

Both original 5M long tests are still running and all three learners continue. No new GIF for these early validation sets; E038 has the latest verified candidate clip. Checks: first-cap history, saved summaries/weights, live queue PIDs/states, syntax/archive equality, report generation, links/images, anchors/TOCs and whitespace. No new implementation tests or expensive training reruns.

# Selected target independent result and fixed 5M queue

Recorded 2026-10-29 17:36:52 -0500. The selected 3.6M target reaches 18/32 genuine 20k caps on new starts, but 14 failures leave reliable sustained juggling unsolved. Two fixed 5M evaluations are queued while all three learners continue.

## Completed independent test

| Metric | Result |
|---|---:|
| Previously unused seed range | 800000–800031 |
| Horizon | 20,000 |
| Genuine physical-trajectory caps | 18/32 |
| Failures | 14/32 |
| Bounded mean survival | 13,761.03125 |
| Median observed | 20,000 (censored) |
| Minimum | 27 |
| Mean physical juggles | 200.34375 |
| Surviving 10k | 20/32 |

All 14 failures list passed-paddle. The shortest episode lasts only 27 steps; no starts were excluded. E038 froze the selected target/hash before test launch and checked that these seeds were unused in saved campaign results. They are now consumed research starts. Median at the cap is not a true lifetime estimate, and no same-seed online-actor control was run on this new range.

The selection cohort had 20/32 caps and mean15,427.625. The new measured level is broadly compatible but lower on a different cohort; do not claim a paired population improvement or successful independent training-seed replication. [Per-start test data](../../../docs/juggling-history-selected-target-independent.csv), [pre-test record](../../../docs/juggling-history-selected-target-independent-provenance.json), [updated unaccepted candidate](../../../docs/juggling-candidate.json). Raw output: `runs/td3/noisy_history_20261029_seed0/independent_selected_target_3m6_20k_eval/results.json`. Hash remains c8a1df78ade41cfbe6845a84f4b21480f512973be7a89f0baf66daf5d452b31b.

An [offline motion audit](../../../docs/juggling-history-selected-target-independent-motion.csv) of all 32 stored trajectories finds that every capped episode spans at least1.39334m longitudinally and0.80926m laterally, with at least98.415% of samples separated beyond combined body radii plus2cm. Minimum sampled path length793.928m. This supports substantial free flight; exact contact-event counting and indefinite reliability are not established. No simulator/RNG/action is touched by the audit.

```sh
.venv/bin/python -m scripts.td3.juggle_survival --run-dir runs/td3/noisy_history_20261029_seed0/target_actor_3m6_bundle --out-dir NEW_OUTPUT --scenario-set training --horizon 20000 --episodes 32 --seed-base 800000 --gif-steps 0
.venv/bin/python -m scripts.analysis.juggle_body_motion --results runs/td3/noisy_history_20261029_seed0/independent_selected_target_3m6_20k_eval/results.json --out docs/juggling-history-selected-target-independent-motion.csv
```

## Waiting fixed 5M evaluations

Online watcher1349083 and target watcher1349084 are confirmed live, both waiting for complete five-million-step checkpoint validation. [Queue/status/hash metadata](../../../docs/juggling-history-5m-watch.json); [archived source](../../../docs/juggling-history-5m-watch-source.txt) is for inspection, with run-directory-relative paths. The live source resides under the original history run root.

Both will evaluate exact5M weights on the reused700000–700031 selection cohort,32episodes,20k horizon. They run in separate processes so global RNG streams cannot interfere. Before evaluating, each verifies the reviewed simulator/config hashes and canonical saved environment, refuses existing output, freezes its weight hash, and records terminal-training errors. Target inference aliases only the same checkpoint's actor_target.pth and unchanged metadata; no learner or warm start is created. At recording both are **waiting**, with no5M result. Source compilation and actual waiting process/status checks passed; later execution is not yet observed.

All three existing policies still train from their original random initializations. No new training or simulator/sensing/control/DR change in this task. No new GIF: E038 already published this selected target's verified clip. Checks: completed independent handle/results, frozen hash, all caps/means, offline body motion, live queue states, report regeneration, relative links/images, unique anchors/TOCs and whitespace. Earlier implementation tests were not rerun.

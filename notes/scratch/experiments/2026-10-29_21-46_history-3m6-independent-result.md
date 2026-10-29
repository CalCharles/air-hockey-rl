# First perfect-validation checkpoint: independent result

Recorded 2026-10-29 16:46:28 -0500. The frozen3.6M history checkpoint reaches14/32genuine20k caps on previously unused starts, but18failures still preclude reliable sustained juggling. Training continues.

| Metric | Result |
|---|---:|
| Seed range | 700000–700031 |
| Evaluation horizon | 20,000 |
| Genuine physical-trajectory cap completions | 14/32 |
| Failures | 18/32 |
| Mean bounded survival | 13,099.8125 |
| Median survival | 14,790 |
| Minimum survival | 629 |
| Mean physical-trajectory juggles | 197.4375 |
| Starts surviving10k | 20/32 |
| Mean action-delta RMS | 0.3630449063 |

All14caps have265–324physical juggles and satisfy the repeated-juggle heuristic. All18failures list passed-paddle; no bottom termination in this set. Physical counts are trajectory heuristics, not exact distinct-contact instrumentation. Sensors, PID/smoothing, contacts, geometry, limits and full±25% DR remain unchanged.

## Selection and limits

Selected checkpoint3.6M because it was the first saved checkpoint with8/8reused5k validation caps. It was frozen and hashed before evaluating the new seeds; [pre-test provenance](../../../docs/juggling-history-3m6-test-provenance.json) was published while the test was still running. Model SHA256`97e1184364f9f6a0c00406c8608bbc8190a82e60629f4a593eb483125ce95351` still matches after evaluation. All prior noisy-campaign result ranges were checked before launch and none overlapped these seeds. They are now consumed research starts for future candidates.

The earlier3M independent test averaged9,116.5625 with5/32caps on seeds600000–600031. The new measured fraction/mean is higher, but these are **different cohorts**, not a paired policy comparison. Single training seed, checkpoint selection using repeatedly visited validation starts and multiple candidate tests limit inference. Do not claim successful independent replication or acceptance. The updated research pointer is descriptive, not a proof of reliable or indefinite survival.

[Per-start CSV](../../../docs/juggling-history-3m6-independent.csv), [candidate manifest](../../../docs/juggling-candidate.json). Raw result `runs/td3/noisy_history_20261029_seed0/independent_3m6_20k_eval/results.json`; saved args/config and trajectories remain local. All actors/critics/encoder originally initialized randomly in the ongoing scratch run. No checkpoint is a source for new training.

```sh
.venv/bin/python -m scripts.td3.juggle_survival --run-dir runs/td3/noisy_history_20261029_seed0/train/checkpoint_3600000 --out-dir NEW_OUTPUT --scenario-set training --horizon 20000 --episodes 32 --seed-base 700000 --gif-steps 0
```

No new GIF for this test; E031's newly published clips show3M, not3.6M. The fixed4M watcher remains waiting; all three history learners continue. No new training or implementation/simulator changes in this task. Checks: completed test handle/results, frozen hash, per-start caps/means, report regeneration, live processes, relative links/images, unique anchors/TOCs and whitespace. Earlier implementation/adapter tests are not rerun here.

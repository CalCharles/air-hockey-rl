# Fixed history 3M checkpoint: shared and independent tests

Recorded 2026-10-29 16:13:02 -0500. Both evaluations completed. The fixed 3M checkpoint is the strongest measured shared-benchmark candidate, but 27 of 32 independent starts still fail before 20,000 steps. Training and the goal remain active.

Checkpoint: `runs/td3/noisy_history_20261029_seed0/train/checkpoint_3000000`. SHA256 `1e0dda69f8148b8178d0c575a0dca54e36357bbcf2809b1b98857e9f26357d20` was recorded before the independent test. Existing history learner, random actor/critics/encoder at initialization, model_path null and no warm start or residual. No change to canonical sensing, control, physics or ±25% DR. Saved canonical environment assertion passed, normalizing only the saved top-level bookkeeping seed.

| Evaluation | Starts | Horizon | Mean steps | Median | Physical juggles | Genuine caps |
|---|---|---:|---:|---:|---:|---:|
| Fixed 3M shared benchmark | 300000–300031 | 10,000 | 6,662.65625 | 9,635.5 | 99.1875 | 14/32 |
| Previous 2.1M on same starts | same | 10,000 | 4,885.28125 | 3,190 | 74.34375 | 9/32 |
| Fixed 3M independent test | 600000–600031 | 20,000 | 9,116.5625 | 7,998 | 133.875 | 5/32 |

Shared paired mean difference 1,777.375, 18/32 wins; paired bootstrap 10,000 resamples, NumPy default_rng(42), 95% interval [-15.5625, 3582.86640625]. The interval includes zero. One training seed, repeated shared benchmark use and many inspected checkpoints limit inference. The earlier 7/8 validation caps at 5k are not independent acceptance.

Before launching the independent test, all saved noisy-campaign results were checked for overlap with 600000–600031; none overlapped. The checkpoint was locked first. This corroborates longer average survival beyond the shared starts, but no same-seed independent control was run. These seeds are **now consumed research starts**, not untouched acceptance seeds for future candidates. Independent cap seeds: 600013/600017/600026/600028/600029 with 310/290/288/292/297 physical juggles. All five satisfy the repeat-juggle heuristic; it is not exact contact instrumentation. Minimum survival 187; 12/32 last beyond 10k; end reasons overlap (passed paddle 27, bottom 1). Mean action-delta RMS 0.36666623, with unchanged PID and action smoothing.

## Sources and reproduction

[Shared per-start data](../../../docs/juggling-history-benchmark.csv), [independent data](../../../docs/juggling-history-3m-independent.csv), [current candidate manifest](../../../docs/juggling-candidate.json). Raw outputs: `milestone_3m_benchmark/results.json` and `independent_3m_20k_eval/results.json` under the run root; pre-test hash/seed audit in `independent_3m_test_provenance.json`. Local checkpoint files are not pushed. The 3M watcher completed successfully; E024's waiting status is historical and superseded by this record.

```sh
.venv/bin/python -m scripts.td3.juggle_survival --run-dir runs/td3/noisy_history_20261029_seed0/train/checkpoint_3000000 --out-dir NEW_OUTPUT --scenario-set training --horizon 20000 --episodes 32 --seed-base 600000 --gif-steps 0
# Shared test: use horizon10000 and seed-base300000, also in a new output directory.
```

No new GIF for this milestone; [E025](../../../docs/experiment_logbook.md#e025) contains verified ending clips from the earlier 2.1M checkpoint, not this model. The candidate pointer is only for frozen evaluation, never for new training initialization.

All three learners remain live. Baseline is near its 6M target; seed1 history replication is prepared, not launched. No new training or implementation changes in this task. Checks: completed evaluation status, frozen hash/config/seed audit, saved-data report regeneration, CSV consistency, relative links/image paths, unique anchors/TOCs and whitespace. Earlier implementation tests were not rerun.

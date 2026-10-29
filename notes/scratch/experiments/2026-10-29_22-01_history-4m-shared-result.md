# Fixed history 4M shared benchmark

Recorded 2026-10-29 17:01:05 -0500. The fixed 4M checkpoint reaches 16/32 genuine 10k caps on the reused shared benchmark, but half of starts still fail. A longer paired test is running; the 3.6M research candidate remains unchanged pending that result.

| Fixed checkpoint | Mean steps | Median | Physical juggles | Genuine 10k caps |
|---|---:|---:|---:|---:|
| 3M | 6,662.65625 | 9,635.5 | 99.1875 | 14/32 |
| 4M | 7,335.59375 | 9,989.5 | 107.21875 | 16/32 |

Same32starts300000–300031 and horizon10000. Four-million-step minimum44;16passed-paddle failures. Paired mean difference672.9375,17/32wins; bootstrap10000resamples, NumPy default_rng(42),95% interval[-1236.88671875,2529.1328125]. The interval includes zero. One training seed, reused starts and multiple checkpoints limit inference; this does not establish reliable survival or an unambiguous improvement.

[Per-start comparison](../../../docs/juggling-history-benchmark.csv), raw `runs/td3/noisy_history_20261029_seed0/milestone_4m_benchmark/results.json`. The fixed watcher completed successfully at2026-10-29T21:58:34.569960+00:00. The original eight-start4M validation has mean3784.75 and6/8genuine5k caps; these are different starts and horizon.

## Running longer paired comparison

Froze checkpoint4M and verified its canonical saved config before launching32starts700000–700031,horizon20000. These are **reused paired research starts** from the3.6M online/target tests, not independent acceptance. [Locked provenance](../../../docs/juggling-history-4m-test-provenance.json); raw output `paired_4m_20k_eval/` under the run root. No long-test result yet. This will compare4M with the current3.6M candidate on the same long-horizon cases, rather than ranking different cohorts/horizons. Target-actor comparison is also still running. No candidate promotion follows from the short shared result alone.

```sh
# Completed fixed benchmark; choose a new output directory.
.venv/bin/python -m scripts.td3.juggle_survival --run-dir runs/td3/noisy_history_20261029_seed0/train/checkpoint_4000000 --out-dir NEW_OUTPUT --scenario-set training --horizon 10000 --episodes 32 --seed-base 300000 --gif-steps 0
# Running paired test: same checkpoint, horizon20000, seed-base700000, new output.
```

All three learners continue. No new training, implementation or simulator/sensing/control/DR changes. No new GIF; E031 has the latest verified3M clips. Checks: completed watcher/output, paired seed alignment and bootstrap, new frozen hash/canonical guard, per-start CSV, saved-data report generation, live processes, links/images, anchors/TOCs and whitespace. Earlier implementation tests are not rerun here.

# Full-policy warm-start campaign for sustained juggling

- **Date**: 2026-10-21 18:50 UTC
- **Status**: in-flight (snapshot at approximately 300k steps of seed 0)
- **Run dirs**: `runs/td3/sustained_20261021_warm/{g0975,g099,g0995}/`
- **Configs**: `configs/new_juggle/sustained/juggle.yaml`, `configs/td3/sustained/{g0975,g099,g0995}.yaml`; run-local copies and `source_snapshot/` record provenance
- **Follows**: [scratch pilot](2026-10-21_18-44_sustained-juggle-scratch-pilot.md)

## Question

Does simulation-only training from an existing juggling actor, with corrected failure handling and critic warmup, improve long-horizon reliability without increasing control gains?

## Setup

Recipe and monitoring: [sustained-juggling guide](../../docs/training/sustained-juggling.md). Three discounts, three sequential training seeds per discount, 1M steps each. Actors initialize from the historical `hist2_motion0_v2/model.pth`; critics and optimizers are fresh. Critics start at 10k steps, actor updates at 50k. Training uses 16 critic / two actor updates per 32 steps, action-change penalty 0.005, and lower actor learning rate. Measured workspace and PID gains are preserved; physics and sensor timing are randomized. No robot operation or real-world training occurs.

All implementation ran in the main session without subagents. GPU lanes run in parallel; each lane's seeds run sequentially. Three existing A40 allocations were used (106217 / 106218 / 106219). Warmup was verified by a separate CUDA smoke run: all actor tensors stayed exactly equal to initialization while critics learned. Checkpoint survival evaluation was also exercised.

## Results so far

Eight paired validation episodes per scenario, capped at 2,000 steps. Only seed 0 has reached these checkpoints; no across-training-seed standard deviation is available. Values are mean observed episode lengths, including capped episodes when applicable.

| Actor | Nominal | Randomized | Camera 50 ms | Camera 100 ms |
|---|---:|---:|---:|---:|
| Unchanged source, **same new observations and environment** | 243.2 | 357.0 | 190.4 | 107.4 |
| gamma 0.975, 300k | 78.0 | 128.8 | 95.6 | 103.5 |
| gamma 0.99, 300k | 138.8 | 82.0 | 134.8 | 79.8 |
| gamma 0.995, 300k | 106.0 | 102.4 | 125.5 | 69.5 |

None of these episodes reached 2,000 steps. The source baseline is `baseline/{results.json,summary.md}` in this campaign. The source on legacy history sampling had a higher nominal mean (754.6, one of eight reached the cap), but that is a different observation contract and is not a fair measure of the new policies' training improvement.

The full-policy trials reduce action-change RMS relative to the source but have not improved survival. At 300k, nominal source action-change RMS was about 0.547, versus 0.333 / 0.285 / 0.258 for the three discounts. Peak simulator paddle speeds still reach the unchanged 2 m/s limit. Smoother commands alone do not establish successful juggling or safe hardware deployment.

Verification: 22 relevant targeted tests passed. Full TD3 discovery ran 141 tests with seven skips and one existing error (`test_direct_args_keys_are_applied`, missing `updates_per_second` in the real-runtime mapping). That test and implementation are byte-identical to HEAD. CUDA graph training, critic-only warmup, checkpointing, and the asynchronous evaluator passed smoke checks.

## Current conclusion

The infrastructure changes address real continuing-task and observation-contract issues. The first 300k steps do **not** establish that this training recipe improves juggling. Do not promote any of these early actors or claim indefinite survival.

## Next

The bounded campaign continues automatically through its training budget and then tests final actors for 10,000 steps on 32 fresh episodes per scenario, seeds starting at 100000. Checkpoint selection uses validation seeds starting at 10000; further model selection requires fresh final-test seeds. Independent bounded residual/CQL comparisons were added after the 200k results: [residual experiment](2026-10-21_19-00_sustained-juggle-residual-cql.md).

# Target actor: paired 3.6M result and 4M follow-up

Recorded 2026-10-29 17:06:06 -0500. The saved 3.6M target actor has higher measured cap completion than its online counterpart on the same starts, but paired uncertainty includes zero and 12 failures remain. The 4M target comparison is now running.

| 3.6M frozen policy | Bounded mean | Median observed | Physical juggles | Genuine 20k caps | Failures |
|---|---:|---:|---:|---:|---:|
| Online actor | 13,099.8125 | 14,790 | 197.4375 | 14/32 | 18 |
| TD3 target actor | 15,427.625 | 20,000 (censored) | 225.0625 | 20/32 | 12 |

Both use the same 32 previously used starts 700000–700031 and horizon 20,000. The target's mean gain is 2,327.8125; paired bootstrap 10,000 resamples, NumPy default_rng(42), 95% interval [-834.57109375, 5445.16328125]. Target wins 14 cases, loses 8, ties 10. No unambiguous population improvement or independent acceptance claim follows. Median at the cap does not establish true median lifetime. All 12 target failures are passed-paddle; minimum 1,101 steps, 26/32 last beyond 10k. Mean action-delta RMS 0.3554831264, with unchanged action smoothing/PID.

[Per-start paired data](../../../docs/juggling-history-target-comparison.csv), [pre-test source/hash record](../../../docs/juggling-history-target-actor-test-provenance.json). Raw target result `runs/td3/noisy_history_20261029_seed0/target_actor_3m6_paired_20k_eval/results.json`. This completes E035's running comparison. The target is the Polyak average within the original randomly initialized scratch run; no pretrained source, residual, new training or checkpoint warm start.

## Offline motion audit — 2026-10-29 17:09:08 -0500

[Driver](../../../scripts/analysis/juggle_body_motion.py) and [per-start data](../../../docs/juggling-history-target-3m6-motion.csv) read only the stored true body positions. Across all20capped target episodes, minimum longitudinal puck span1.393393m, lateral span0.809788m, sampled path length762.317804m, and separated fraction0.9836. Separation means center distance greater than paddle radius plus puck radius plus0.02m (threshold0.10255m). These observations support substantial free flight and rule out a stationary-contact explanation of these caps. They do not provide exact contact-event counts, prove indefinite survival or change the trajectory-heuristic limitation. No simulator, RNG or policy is touched.

```sh
.venv/bin/python -m scripts.analysis.juggle_body_motion --results runs/td3/noisy_history_20261029_seed0/target_actor_3m6_paired_20k_eval/results.json --out docs/juggling-history-target-3m6-motion.csv
```

## Running 4M comparison

An inference-only bundle `target_actor_4m_bundle/` links to checkpoint 4M's saved actor_target.pth plus unchanged config/args/RMA metadata. Source hash `452f6dbfdd43c9a17224966e1f7db8b9b6640745c53413abe9bd7d076df63ae2`, parameter L2 difference from online 0.7360034204; bundle load and canonical guard pass. [Locked provenance](../../../docs/juggling-history-target-4m-test-provenance.json). New output `target_actor_4m_paired_20k_eval/`, same reused 32 starts and 20k horizon. Its result is unknown. The existing online 4M long test is also running. This gives a matched online/target comparison across two checkpoints before choosing a policy for a new-seed test. No new candidate promotion yet.

Reproduce by constructing an inference bundle with model.pth linked to the saved target weights and args.yaml/config.yaml/rma_meta.json linked to the same checkpoint, then:

```sh
.venv/bin/python -m scripts.td3.juggle_survival --run-dir NEW_TARGET_BUNDLE --out-dir NEW_OUTPUT --scenario-set training --horizon 20000 --episodes 32 --seed-base 700000 --gif-steps 0
```

No new GIF for these aggregate tests; E031 has the latest verified 3M clips. All three learners continue. No simulator, sensing, control or randomization change; no new training starts. Checks: completed 3.6M target test, common seeds and bootstrap, distinct 4M weights/hash, canonical reload, per-start arithmetic, saved-data report generation, live processes, links/images, anchors/TOCs and whitespace. Earlier implementation tests were not rerun.

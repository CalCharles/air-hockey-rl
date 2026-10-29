# Independent target-test physics reconstruct exactly and failures span the range

- **Date**: 2026-10-29 23:24 UTC diagnostic/writeup recording time
- **Status**: done
- **Run dirs**: `runs/td3/noisy_history_20261029_seed0/independent_selected_target_3m6_20k_eval`
- **Configs**: original unchanged canonical noisy sensing and full ±25% DR

## Question
Do the retained candidate's independent failures identify a single problematic physics range?

## Setup and results
Isolated32 seeded resets without actions exactly reproduce all initial observations. All parameters remain within full DR ranges. See [E043](../../../docs/experiment_logbook.md#e043) for raw data, command and quartile counts.

## Conclusion
Failures span every sampled marginal quartile. Small post-test, confounded summaries do not isolate causes or justify changing randomization. Original18/32 caps and14 failures remain unchanged.

## Next
Continue the existing full-budget learners and guarded final evaluations; no new training launched for this diagnostic.

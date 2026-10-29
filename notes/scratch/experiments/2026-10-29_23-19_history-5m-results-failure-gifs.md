# Fixed 5M comparisons retain the 3.6M target and expose independent failures

- **Date**: 2026-10-29 23:19 UTC writeup start; raw status files carry evaluation completion times
- **Status**: done (evaluations/replays); parent training in-flight
- **Run dirs**: `runs/td3/noisy_history_20261029_seed0`, `runs/td3/noisy_history_20261029_seed1`, `runs/td3/noisy_history_long_discount_20261029_seed0`
- **Configs**: unchanged `configs/td3/sustained/noisy_history_scratch.yaml`, seed1 and longer-discount variants

## Question
Do fixed 5M online/target weights improve selection survival, and what failures remain?

## Setup and results
See [E041](../../../docs/experiment_logbook.md#e041) for the full table, paired uncertainty, frozen model hashes, replay provenance and figures. Cohort700000 is reused for selection; independent800000 is consumed. No new independent case was added by replay.

## Conclusion
Neither 5M variant displaces the retained selection winner. Independent 18/32 caps leave14 failures, so reliable sustained juggling remains unsolved. New seed1 target2M reaches7/32 selection caps, longer discount1/32; these are unfinished training trials, not final comparisons.

## Next
Continue all three scratch learners to their full6M budgets and assess later frozen models. Preserve canonical noisy sensing, physics and established DR.

# Juggling rollout GIFs

Compact copies of the experiment GIFs, at 160 × 357 pixels, 10 fps and 64 colors. Playback duration is preserved; original 20 fps GIFs remain in the local `runs/td3/juggle_iterations/` directory. These clips show individual episodes, not aggregate reliability.

| Experiment | Rollout | Mean steps over 16 validation starts | Genuine 5,000-step completions |
|---|---|---:|---:|
| Iteration 1, original reward | [GIF](iteration-1-upper.gif) | 1,011.9 | 2/16 |
| Iteration 1, survival reward | [GIF](iteration-1-survival.gif) | 1,188.6 | 1/16 |
| Iteration 1, added memory | [GIF](iteration-1-memory.gif) | 234.6 | 0/16 |
| Iteration 2, continued training | [GIF](iteration-2-continue.gif) | 4,397.1 | 14/16 |
| Iteration 2, independent seed | [GIF](iteration-2-replicate.gif) | 4,693.3 | 15/16 |
| Iteration 2, longer discount | [GIF](iteration-2-long_discount.gif) | 5,000.0 | 16/16 |

## Independent final tests

The selected longer-discount policy completed 20,000 steps (16 minutes 40 seconds at 20 Hz) with repeated juggling on 28/32 fresh starts. Four episodes failed within 334 steps. These bounded tests do not demonstrate indefinite survival. Controller gains and limits were unchanged.

- Clean simulation: [start](final-clean-start.gif), [last 20 seconds](final-clean-end.gif).
- Transfer configuration, nominal physics with sensor corruption: [GIF](final-transfer-nominal.gif). Mean duration 80.4 steps; 0/32 completed the cap.
- Transfer configuration, 100 ms camera lag plus sensor corruption: [GIF](final-transfer-camera-100ms.gif). Mean duration 78.4 steps; 0/32 completed the cap.

The clean policy is not yet robust to the simulated sensing conditions needed for real-world transfer. See [experiment setup](../../juggling-iterations.md) for configuration and evaluation details.

## Canonical noisy scratch training

[Final 2M-step rollout](noisy-scratch-final.gif): mean survival 416.1 steps over 32 fresh starts, 0/32 reaching 10,000 steps. See [E005](../../experiment_logbook.md#e005).

[Validation-selected scratch actor](noisy-scratch-selected.gif): 816.94 mean steps and 7.75 physical-trajectory juggles across 32 paired starts, 0/32 completing 10k. Final actor on those starts: 415.31 steps. Clip generated 2026-10-29, compacted to 10 fps with duration preserved; see [E008](../../experiment_logbook.md#e008).

[Reduced exploration, 500k](noisy-low-explore-500k.gif): first validation rollout; eight-start mean survival 233.125 versus matched baseline 149.75, neither completing the 5k cap. Early running-experiment milestone, not a final result; [E009](../../experiment_logbook.md#e009).

[Reduced exploration, 1M benchmark](noisy-low-explore-1m.gif): shared 32-start mean survival 695.375, 0/32 at 10k, versus selected incumbent 816.9375. Retain incumbent; matched eight-start validation lead is not a solved result. [E012](../../experiment_logbook.md#e012).

## First bounded noisy survival milestone

Baseline 1.9M, specifically selected successful validation seed 10005: [first 10 seconds](noisy-first-cap-start.gif), [final 20 seconds](noisy-first-cap-end.gif), with a time gap between clips. It completed 5k then failed at 6,859 steps (75 physical-trajectory juggles). Only 1/8 validation starts capped; 0/32 shared 10k starts capped. These 2026-10-29 clips illustrate one favorable episode, not reliability. [E015](../../experiment_logbook.md#e015).

## Scratch history milestone

[Selected episode final20seconds](noisy-history-selected-end.gif): seed300017 survived17049steps with235 physical juggles, then failed. [First benchmark episode final20seconds](noisy-history-benchmark-first.gif): different seed300000 failed at785steps. Generated2026-10-29. Shared benchmark3/32 at10k; independent test0/32 at20k. These individual episodes do not establish reliability. [E019](../../experiment_logbook.md#e019).

## Selected 20k noisy history success

History2.1M, known successful seed500014: [first10seconds](noisy-history-20k-start.gif), [final20seconds](noisy-history-20k-end.gif), with a time gap. Ends at the20000-step cap (16minutes40seconds simulated), with295physical juggles. All eight saved trajectory arrays match the earlier evaluation exactly. Generated2026-10-29, duration preserved. This selected success accompanies **4/32caps and28failures** in the reused-start20k test. On-screen Contacts counts contact timesteps. [E022](../../experiment_logbook.md#e022).

## Eventual failure after 48 simulated minutes

[Final 20 seconds](noisy-history-48min-end.gif) of selected seed 500014: 57,811 steps, 838 physical juggles, then failure. The other three 20k successes also eventually failed. All extension prefixes and the complete renderer replay match saved arrays exactly. Generated 2026-10-29; duration preserved. This is a selected long episode, not independent reliability evidence. The original 32-start cohort now has 32 observed eventual failures. [E025](../../experiment_logbook.md#e025).

## Current 3M candidate: success and failure

[Selected cap, final 20 seconds](noisy-history-3m-cap-end.gif): seed 600013, 20,000 steps, 310 physical juggles. [First test seed, final 20 seconds](noisy-history-3m-first-failure.gif): different seed 600000, failure at 8,862 steps, 133 physical juggles. Both full replays match all eight original arrays exactly. Generated 2026-10-29, duration preserved. Aggregate independent test: **5/32 caps and 27 failures**. The first clip is selected success; the second is the first predeclared seed. [E031](../../experiment_logbook.md#e031).

## Selected target-actor illustration

[Final 20 seconds](noisy-history-target-3m6-end.gif) of seed 700000 under the 3.6M target actor: 20,000 steps, 270 physical juggles, censored at the cap. This is the first reused selection-cohort case, a known success. Full selection result: **20/32 caps and 12 failures**; a separate new-seed test is running. All eight full replay arrays match byte for byte. Generated 2026-10-29, duration preserved. [E038](../../experiment_logbook.md#e038).

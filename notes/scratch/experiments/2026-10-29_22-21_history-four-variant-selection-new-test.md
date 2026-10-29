# Four frozen variants compared; selected target enters a new-seed test

Recorded 2026-10-29 17:21:46 -0500. The completed matched comparison selects the 3.6M target actor for further research evaluation. Its new-seed test is running, so reliability remains unproven.

| Frozen variant | Bounded mean steps | Median observed | Genuine 20k caps | Failures |
|---|---:|---:|---:|---:|
| Online 3.6M | 13,099.8125 | 14,790 | 14/32 | 18 |
| Target 3.6M | 15,427.625 | 20,000 (censored) | 20/32 | 12 |
| Online 4M | 13,687.09375 | 16,801.5 | 15/32 | 17 |
| Target 4M | 13,074.8125 | 16,791 | 14/32 | 18 |

All four use the same 32 starts 700000–700031, horizon 20,000, and identical effective environment configs. These starts are now a **research selection cohort**, not untouched acceptance data. Rule stated before the final variant completed: rank genuine cap count first, then bounded mean. The 3.6M target wins this observed comparison; no fresh-test outcome influenced selection.

[Full scores and paired bootstrap](../../../docs/juggling-history-four-variant-selection.json), [per-start matrix](../../../docs/juggling-history-target-comparison.csv). The 4M online gain over 3.6M online is 587.28125 steps, 95% paired interval [-2851.26, 4080.83]. The 4M target minus 4M online is -612.28125, interval [-3357.99, 2234.31]. The 3.6M target gain is 2327.8125, interval [-834.57, 5445.16]. All include zero; 10,000 paired bootstrap resamples, NumPy default_rng(42). One training seed, several inspected variants and reused selection data limit inference. Parameter averaging did not consistently improve both checkpoints.

## Running independent test

Before launch, checked every saved noisy-campaign result range: none overlapped seeds 800000–800031. Locked the selected bundle and SHA256 `c8a1df78ade41cfbe6845a84f4b21480f512973be7a89f0baf66daf5d452b31b`, then launched 32 starts with a 20,000-step horizon. [Pre-test provenance](../../../docs/juggling-history-selected-target-independent-provenance.json). Output `runs/td3/noisy_history_20261029_seed0/independent_selected_target_3m6_20k_eval/`. **Running; no aggregate result yet.** These seeds become consumed research starts after use.

[Candidate manifest](../../../docs/juggling-candidate.json) now references the inference-only target bundle, explicitly pending this test and unaccepted. The target was initially copied from the original random actor/encoder and Polyak-updated within that scratch run. No pretrained source, residual, new learner or checkpoint warm start. Saved config/args/RMA metadata are unchanged; the view's model.pth links to the saved actor_target.pth. No simulator, sensing, PID/smoothing, geometry, contacts, limits or full DR change.

```sh
.venv/bin/python -m scripts.td3.juggle_survival --run-dir runs/td3/noisy_history_20261029_seed0/target_actor_3m6_bundle --out-dir NEW_OUTPUT --scenario-set training --horizon 20000 --episodes 32 --seed-base 800000 --gif-steps 0
```

## Verified illustration

![Selected target actor, final 20 seconds of first selection-cohort case](../../../docs/assets/juggling/noisy-history-target-3m6-end.gif)

Seed 700000, the first case in the reused selection cohort, reaches 20k with 270 physical-trajectory juggles. This is a known capped case, not new independent evidence. Ends at censoring after 16 minutes 40 seconds simulated; the GIF shows only the final 20 seconds. Aggregate selection result remains 20 caps and 12 failures. On-screen Contacts counts contact timesteps. Generated 2026-10-29, 160 × 357, 10 fps, 64 colors, playback duration preserved; decoded last frame inspected. [Replay hashes](../../../docs/juggling-history-selected-target-gif-provenance.json): all eight arrays match the full original 20k trajectory byte for byte. Local replay root `target_actor_3m6_demo_seed700000/`; reproduce with the same bundle, one episode, seed700000 and gif-steps400 in a new output directory.

All three scratch history learners continue. No new training launch or implementation change in this task. Checks: all four completed results, common seeds/configs, paired arithmetic, candidate and pre-test hashes, no seed overlap, byte-exact renderer replay, GIF duration/frame, report regeneration, links/images, unique anchors/TOCs and whitespace. Earlier implementation tests were not rerun.

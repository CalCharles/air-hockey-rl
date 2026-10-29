# Longer extensions reveal persistent history-policy failures

Recorded 2026-10-29 15:58:07 -0500. All four episodes censored at 20,000 steps eventually failed when extended toward 100,000. This establishes failures during long play as well as near the start; it does not establish their cause. The policy remains an unaccepted research candidate.

Frozen checkpoint: `runs/td3/noisy_history_20261029_seed0/train/checkpoint_2100000`. These are deliberately selected successes from the reused 500000–500031 cohort, not new independent seeds.

| Seed | Failure step | Physical-trajectory juggles | End reasons |
|---|---:|---:|---|
| 500014 | 57,811 | 838 | passed paddle |
| 500016 | 38,023 | 577 | passed paddle |
| 500018 | 27,936 | 466 | bottom and passed paddle |
| 500021 | 28,246 | 479 | passed paddle |

For every extension, all eight saved arrays match the original 20,000-step prefix bitwise. No physics, observations, control or randomization changed. Only the diagnostic episode horizon increased. Raw roots are `runs/td3/noisy_history_20261029_seed0/selected_stretch_2m1_seedSEED_100k/`.

```sh
# Run once per selected seed; use a new output directory.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m scripts.td3.juggle_survival --run-dir runs/td3/noisy_history_20261029_seed0/train/checkpoint_2100000 --out-dir NEW_OUTPUT --scenario-set training --horizon 100000 --episodes 1 --seed-base 500014 --gif-steps 0
```

## Cohort accounting

The original 32 starts contained 28 observed failures below 20k and four capped episodes. Retain those 28 failures and replace the four censored lengths with the verified extensions: all 32 episodes now end in failure, with no censoring remaining. Derived mean duration 9,316.78125, median 5,100.5, maximum 57,811 and mean physical juggles 142.875. This is an update of the original cohort, **not a new independent 32-start test**. [Per-start sources](../../../docs/juggling-history-2m1-uncensored.csv), [derived summary](../../../docs/juggling-history-2m1-uncensored.json), [candidate provenance](../../../docs/juggling-candidate.json).

## Selected longest episode

![Final 20 seconds of the selected 57,811-step episode](../../../docs/assets/juggling/noisy-history-48min-end.gif)

The selected longest episode lasts 2,890.55 simulated seconds (48 minutes 10.55 seconds) and ends in failure. This clip is the final 20 seconds, not the whole episode. On-screen Contacts counts contact timesteps, not distinct juggles. Generated 2026-10-29; 160 × 357 pixels, 10 fps, 64 colors, playback duration preserved. The renderer replay in `selected_demo_2m1_seed500014_100k_gif/` matches all eight full 57,811-step arrays exactly; its last decoded frame was visually inspected. Reproduce with the command above and `--gif-steps 400` in a new directory.

All four diagnostic evaluation processes completed successfully. The three learners and fixed-3M watcher remain live; no new training was launched. [The queued fixed checkpoint](../../../docs/experiment_logbook.md#e024) will measure subsequent learning. Checks: exact trajectory arrays/prefixes, derived cohort arithmetic, GIF duration, relative links/images, unique anchors, TOC coverage and whitespace. No new implementation-test claim.

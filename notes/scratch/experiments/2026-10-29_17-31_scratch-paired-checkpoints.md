# Validation-selected scratch checkpoint outperforms the final actor on fresh paired starts

- **Date**: 2026-10-29 17:31 UTC
- **Status**: done
- **Run dirs**: `runs/td3/noisy_scratch_20261028_seed0/{selected_paired_eval,final_paired_eval}`
- **Configs**: `configs/td3/sustained/noisy_scratch.yaml`; `configs/new_juggle/sustained/original_sensing.yaml`

## Question
Does validation checkpoint selection improve the from-scratch incumbent over the final actor? The 1.8M checkpoint was selected from 19 saved checkpoints using eight validation starts (10000–10007), before these paired evaluations.

## Setup
Canonical noisy sensing/control, existing three-parameter DR, unchanged dynamics. Frozen scratch actors at 1.8M and 2M steps, same 32 fresh starts (300000–300031), 10,000-step cap. This is one training seed, with paired evaluation episodes, not multiple training replications. The policy sees only its original observations; direct Box2D body positions support diagnostic counts only. Evaluation jobs completed 2026-10-29; exact recording time is above, not an invented run-start timestamp.

## Results

| Actor | Mean steps | Median steps | Mean observed juggles | Mean physical-trajectory juggles | Genuine cap completion |
|---|---:|---:|---:|---:|---:|
| Validation-selected 1.8M | 816.9375 | 612 | 7.40625 | 7.75 | 0/32 |
| Final 2M | 415.3125 | 358.5 | 3.28125 | 3.84375 | 0/32 |

Paired mean survival gain: 401.625 steps. Percentile paired bootstrap 95% interval [174.59, 660.10] steps (10,000 resamples; NumPy default_rng seed 42). Selected actor outlasts final on 20/32 starts. Interval reflects episode sampling only, not training-seed variability. Physical-trajectory counts use the same distance/reversal heuristic as observed counts and are not a formal contact-event ground truth.

[Per-start CSV](../../../docs/juggling-paired-checkpoints.csv). Raw result files remain local under the run dirs above. [Selected-actor raw results](../../../runs/td3/noisy_scratch_20261028_seed0/selected_paired_eval/results.json), [final-actor raw results](../../../runs/td3/noisy_scratch_20261028_seed0/final_paired_eval/results.json).

![Selected scratch actor under canonical noisy sensing](../../../docs/assets/juggling/noisy-scratch-selected.gif)

This newly generated clip shows the tail of the first paired episode, not aggregate reliability. Copied from the saved evaluation GIF, reduced to 10 fps and 64 colors with playback duration preserved (20 seconds); generated 2026-10-29. Original remains in the selected evaluation directory.

## Reproduce

```sh
.venv/bin/python -u -m scripts.td3.juggle_survival --run-dir runs/td3/noisy_scratch_20261028_seed0/train/checkpoint_1800000 --scenario-set training --episodes 32 --horizon 10000 --seed-base 300000 --gif-steps 400 --out-dir <new-selected-output>
.venv/bin/python -u -m scripts.td3.juggle_survival --run-dir runs/td3/noisy_scratch_20261028_seed0/train --scenario-set training --episodes 32 --horizon 10000 --seed-base 300000 --gif-steps 0 --out-dir <new-final-output>
```

## Conclusion
The validation-selected checkpoint is the stronger scratch incumbent. Final-only reporting obscures substantial degradation late in this one run. Neither actor solves sustained juggling; the earlier original-actor baseline used other seeds and is not a matched comparator here.

## Next
Three 6M-step scratch trials continue. A detached `scripts.td3.juggle_select --root <root> --wait` evaluator is attached to each; it selects from saved validation summaries and evaluates on 32 starts from seed 400000, cap 10k, after the final-actor benchmark finishes. These starts are disjoint from checkpoint validation and the final-actor benchmark. This is still a shared cross-recipe benchmark; a winning candidate needs new untouched acceptance starts and longer horizons. No checkpoint is used to initialize training.

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

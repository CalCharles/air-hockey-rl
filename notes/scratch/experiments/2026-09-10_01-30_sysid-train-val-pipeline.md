# Puck + wall system identification with a train / validation split on the mouse dataset

- **Date**: 2026-09-10 01:30 UTC
- **Status**: done
- **Run dirs**: `sysid/auto_segments/mouse_all100_sections/` (sections) and `.../sysid/` (fit, seed 0, 20-sample windows); `.../sysid_win20`, `.../sysid_win30` (window-length checks, puck only)
- **Code**: `scripts/sysid/extract_sysid_sections.py`, `scripts/sysid/run_sysid_pipeline.py`, `scripts/sysid/helper/{sysid_dataset,puck_dynamics_fit,wall_restitution_fit}.py`
- **Doc**: `notes/docs/environments/real-world/sysid-pipeline.md`

## Question
Re-identify puck gravity / damping and wall restitution on automatically harvested sections
with recordings held out, instead of the hand-picked segments and in-sample fits used so far.

## Setup
100 recordings → 542 free-fall clips + 133 wall bounces (auto-segmented, quality-filtered).
Split by recording, 80 train / 20 val (free-fall clips 405 / 137, bounces 91 / 42). Puck:
grid search of (g, γ) on fixed-length windows, LSQ in (p0, v0) per window. Walls: Box2D
replay of each bounce from the real pre-impact fitted state with the identified g / γ, sweep of
the wall restitution to minimise the exit-speed error. Base config
`configs/new_juggle/sysid_best_params_hist2.yaml` (side 0.99, end 0.70).

## Results

**Puck dynamics** (fit rms / prediction rms in cm; prediction = fit on the first half of the window, predict the second half):

| window | n train / val | fitted (g, γ) | train | val | canonical (−0.661, 0.178) train | val |
|---|---|---|---|---|---|---|
| 10 samples (0.5 s) | 762 / 269 | −0.730, 0.100 | 0.72 / 1.41 | 0.72 / 1.45 | 0.72 / 1.42 | 0.72 / 1.46 |
| **20 samples (1 s)** | 244 / 93 | **−0.730, 0.110** | 0.79 / 1.55 | 0.76 / 1.51 | 0.85 / 1.88 | 0.83 / 1.82 |
| 30 samples (1.5 s) | 104 / 38 | −0.725, 0.090 | 0.92 / 2.10 | 0.94 / 1.90 | 1.15 / 2.99 | 1.12 / 2.74 |

The optimum is stable across window lengths; 10-sample windows are simply too short to tell
the parameters apart (grid flat to 0.1 cm). At 20 samples the fitted values cut validation
prediction error from 1.82 to 1.51 cm. Gravity is 10 % stronger than canonical and damping
40 % weaker.

**Wall restitution** (mean |exit speed sim − real|, m/s; sim from the identified g / γ):

| walls | fitted | train err | val err | canonical | train @canonical | val @canonical | bounces train / val |
|---|---|---|---|---|---|---|---|
| side (y±) | **0.900** | 0.091 | 0.079 | 0.99 | 0.110 | 0.105 | 74 / 36 |
| end (x±) | 0.650 | 0.178 | 0.269 | 0.70 | 0.180 | 0.276 | 17 / 6 |

Side walls: clean minimum at 0.90 on train and ≈ 0.85 on val; 0.99 is clearly too lively.
End walls: 17 / 6 bounces, val curve monotone in the restitution (prefers the lower edge of the
grid), several bounces below the sim's 0.25 m/s normal-speed gate — not identified.

**Apparent wall lines** (measured puck-centre apex near impact vs the sim's contact line):
x+ 0.865 vs 0.933, x− 0.933 vs 0.933, y+ 0.359 vs 0.400, y− 0.442 vs 0.400. The puck frame is
offset ≈ −4 cm in y and the far end wall sits 7 cm inside the sim's line.

## Conclusion
- Adopt for this dataset: `gravity −0.73`, `puck_damping 0.11`, `side_wall_restitution 0.90`
  (written to `sysid/auto_segments/mouse_all100_sections/sysid/sim_config_fitted.yaml`, not
  promoted into `configs/`). Keep `end_wall_restitution` at 0.70 until more clean end-wall
  bounces exist; the fitted 0.65 is within noise of it.
- The earlier wall-collision doc's "real bounces keep ~65 % of speed" came from 11 hand-picked
  clips with finite-difference velocities; on 110 side-wall bounces with fitted velocities the
  normal-speed retention is ≈ 0.90–0.95.
- Damping: the previous 0.178 came from 10 clips fitted in-sample; the 40 % lower value holds
  on held-out recordings at every window length ≥ 20.

## Next
- Fix the y offset / x+ wall position in the puck frame (homography) or add per-wall lines to
  the sim; until then replays are timing-shifted by 1–2 frames near those walls.
- Collect end-wall bounces without a human at the far end to identify `end_wall_restitution`.
- Paddle-puck restitution the same way (paddle collisions are already harvested by the segmenter).

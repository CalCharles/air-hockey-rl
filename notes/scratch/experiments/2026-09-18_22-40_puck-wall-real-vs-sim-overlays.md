# Real-vs-Box2D overlays of representative puck free-flight windows and wall bounces under the identified parameters

- **Date**: 2026-09-18 22:40 UTC
- **Status**: done (tooling + reference renders; nothing promoted, no fit changed)
- **Run dirs**: `sysid/puck_dynamics/results/mouse_dataset/overlays/` (the integrator-compensated variant is not kept; regenerate with `--integrator-compensation --out …`), `sysid/wall_collision/results/mouse_dataset/overlays/` (gitignored)
- **Code**: new `sysid/common/table_scene.py`, `sysid/puck_dynamics/code/render_overlays.py`, `sysid/wall_collision/code/render_overlays.py`
- **Fits shown**: puck g = −0.73, γ = 0.11; `side_wall_restitution` = 0.90 (end 0.55, not identified) — the 2026-09-10 mouse-dataset fits, regenerated in the new `sysid/` layout the same day ([`sysid/README.md`](../../../sysid/README.md))
- **Docs**: [`sysid/puck-free-flight.md`](../../docs/environments/real-world/sysid/puck-free-flight.md), [`sysid/puck-wall-collision.md`](../../docs/environments/real-world/sysid/puck-wall-collision.md) ("Overlays" rows)

## Question

Show, for verification and visualisation, five representative real trajectories of each kind
(free flight, side-wall bounce) with the Box2D puck under the identified parameters played on
top of the real one — and check that what the simulator does matches what the fits assumed.

## Setup

Both tools read a fit's `results.json` + `sim_config_fitted.yaml`, take the **validation**
recordings of that fit's split, and pick 5 examples at spread percentiles (10 / 30 / 50 / 70 / 90 %)
of the fit's own error distribution, so the set spans typical-to-poor cases rather than the best
ones. Output per example: GIF (half speed, 10 fps), last-frame PNG, plus `mosaic.png` and
`overlay_summary.{md,csv}`. Real puck = the env's puck sprite + black trail; Box2D puck = blue
ghost + trail; the analytic model / the pre- and post-impact fits = thin grey lines.

- **Free flight**: env from `sim_config_fitted.yaml` (noise / delays off, paddle parked in the
  workspace corner furthest from the path); puck started at the fitted position / velocity of the
  first sample of the 20-sample window and stepped **to each real sample with its real interval**
  (`time_per_step` set per step; real spacing 48.5 ms, sim nominal 50). Error = ‖Box2D − tracker‖
  per sample. Windows whose real path comes within 1 cm of a sim wall contact line are not
  candidates (11 of 93 val windows) — see finding 2.
- **Wall bounces**: the fit's own replay (`replay_bounce`: puck at the fitted state of the first
  pre-impact frame, static paddle at its real position, one step per real frame), fitted
  `side_wall_restitution`; 8 extra steps drawn after the first real post-impact frame. Error of
  the fit = |exit speed sim − real| / real; positions after the bounce are reported but shifted
  (finding 3).

## Results

**Free flight** (val windows, fit rms p10 / p50 / p90 = 0.46 / 0.69 / 1.02 cm; Box2D stepped with real intervals):

| # | clip @ start | speed (m/s) | fit rms (cm) | percentile | Box2D rms / final / max (cm) | final / distance | model rms / final (cm) | Box2D vs model final (cm) |
|---|---|---|---|---|---|---|---|---|
| 1 | `trajectory_data103_free_fall_105_136` @0 | 0.44 | 0.46 | p10 | 1.01 / 1.0 / 1.9 | 2.2 % | 0.46 / 0.6 | 1.5 |
| 2 | `trajectory_data22_free_fall_138_188` @0 | 0.56 | 0.55 | p30 | 1.05 / 1.9 / 2.3 | 6.4 % | 0.55 / 0.4 | 1.5 |
| 3 | `trajectory_data257_free_fall_47_70` @0 | 0.69 | 0.68 | p49 | 1.14 / 1.8 / 1.9 | 3.0 % | 0.68 / 0.3 | 1.5 |
| 4 | `trajectory_data196_free_fall_48_93` @20 | 0.30 | 0.85 | p70 | 1.20 / 0.6 / 2.2 | 1.1 % | 0.85 / 1.2 | 1.4 |
| 5 | `trajectory_data102_free_fall_219_241` @0 | 0.62 | 1.02 | p90 | 1.30 / 0.3 / 2.5 | 0.3 % | 1.02 / 1.3 | 1.3 |

With `--integrator-compensation` (puck started at v0 − ½·g·h) the Box2D rms equals the model rms
in every example (0.46 / 0.55 / 0.68 / 0.85 / 1.03 cm) and Box2D vs model final drops to 0.1–0.2 cm.

**Side-wall bounces** (36 val bounces, all reproduced by the sim; exit-speed rel err p10 / p50 / p90 = 3 / 7 / 34 %):

| # | clip | wall | in (m/s) | exit real / Box2D (m/s) | rel err | percentile | angle err | normal out real / Box2D | post-impact pos err (cm) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `trajectory_data124_wall_y-_191_212` | y− | 0.99 | 0.96 / 0.93 | 3 % | p11 | 0.7° | 0.57 / 0.55 | 6.3 |
| 2 | `trajectory_data122_wall_y-_239_258` | y− | 0.37 | 0.35 / 0.36 | 5 % | p29 | 24.3° | 0.35 / 0.32 | 7.5 |
| 3 | `trajectory_data257_wall_y-_151_170` | y− | 0.35 | 0.32 / 0.34 | 7 % | p51 | 24.4° | 0.32 / 0.30 | 7.9 |
| 4 | `trajectory_data122_wall_y+_66_85` | y+ | 0.59 | 0.55 / 0.62 | 14 % | p69 | 9.6° | 0.43 / 0.42 | 11.9 |
| 5 | `trajectory_data102_wall_y+_211_226` | y+ | 0.72 | 0.53 / 0.73 | 40 % | p91 | 14.3° | 0.22 / 0.46 | 5.3 |

### Findings

1. **Box2D's integrator biases free flight by ½·g·h along gravity.** Stepping the sim puck from
   the same state as the analytic model, its position leads the exact solution by ≈ 1.5 cm after
   1 s in every example, growing linearly (controlled test at the table centre: 0.4 / 0.8 / 1.4 cm
   after 5 / 10 / 20 steps of 48.5 ms, both with and against gravity). This is semi-implicit Euler
   (v += g·h, then x += v·h) at one physics step per control step: a constant velocity offset of
   ½·g·h = 0.5 · 0.73 · 0.0485 ≈ 1.8 cm/s. Starting the puck at v0 − ½·g·h removes it entirely
   (Box2D rms = model rms). So under the fitted (g, γ) the simulator's puck is the fitted model
   plus a 1.8 cm/s drift toward the robot; the same drift exists at the nominal 50 ms step in
   training. It is small next to the tracker noise (0.5–1 cm rms) but systematic.
2. **Free-flight windows near the y− / x+ walls cannot be replayed in the sim as is.** The first
   render picked a slow window that ran along y ≈ −0.39 … −0.44 m: the sim puck hit its y− wall
   (contact line at −0.400 m) and bounced while the real puck kept going (6.7 cm final error).
   The measured puck frame extends past the sim table there (apex 0.442 vs 0.400 m at y−, 0.865
   vs 0.933 at x+, from the wall fit) — the known homography / table-width mismatch. The tool now
   excludes windows within 1 cm of a sim contact line (11 of 93 val windows) and says so.
3. **Wall overlays: exit speeds match to the fitted accuracy; positions after the bounce do not,
   by design.** Because of the same wall-line offset the sim puck turns around about one frame
   earlier than the real one at y− and then leads it along the exit direction (post-impact
   position error 5–12 cm even for the 3 % exit-speed example). The fit compares exit
   *velocities*, which are unaffected; the position lead is what the offset looks like. The two
   slow examples (#2, #3, 0.35 m/s) show the 24° exit-angle error typical of this dataset: the sim
   only scales the normal component, the real bounce also loses tangential speed (documented
   limitation). #5 (p91) is a bounce whose real normal exit speed collapses (0.22 m/s from 0.52
   in) — the sim cannot reproduce it at any single restitution.
4. Step-time mismatch matters for position overlays: with the sim's nominal 50 ms step instead of
   the real 48.5 ms spacing the sim puck runs 3 % ahead in time (≈ 2 cm at 0.6 m/s after 1 s).
   The wall replay in the fit still uses 50 ms steps (velocities only; left unchanged).

## Conclusion

The overlays confirm the fitted puck model and side-wall restitution reproduce the real
validation trajectories to the accuracy the fits report (free flight ≈ 1 cm rms over 1 s, side
walls ≈ 3–15 % exit-speed error for the typical bounce), and expose three simulator-side effects
that are not part of the fits: the ½·g·h integrator drift, the puck-frame / sim-table offset at
the y− and x+ walls, and the missing tangential loss at the walls. None of these changes the
fitted values. Reference set: 5 + 5 GIFs and mosaics in the two `overlays/` folders.

## Next

- Sub-step the Box2D world (e.g. 5 × 10 ms per control step) or correct the puck's spawn velocity
  by ½·g·h if the 1.8 cm/s drift matters for sim2real; measure the effect on the juggle policy.
- Fix the puck homography / table width so real trajectories near the y− and x+ walls are inside
  the sim table; then the excluded windows and the post-bounce position lead go away.
- Add a wall tangential coefficient to the sim and fit it on the exit angle (24° on slow bounces).

# Offset (oblique) paddle–puck collisions replayed with the head-on sysid parameters — exit speed and angle errors, no fitting

- **Date**: 2026-09-18 23:30 UTC
- **Status**: done (evaluation + videos; nothing fitted, nothing promoted)
- **Run dir**: `sysid/paddle_puck_collision/results/offset_robot_data_collection_puck_collision_change_angle_20260910_1818/` (gitignored): `summary.md`, `per_trial.csv`, `all_trials.csv`, `evaluations.json`, `plots/exit_vs_offset.png`, `plots/trial_fits.png`, `videos/{gifs,mp4,png}/` + `videos/mosaic_y*.png`
- **Data**: `/data2/air_hockey/robot_data_collection_puck_collision_change_angle_20260910_1818/` (30 trials, recorded 2026-09-10 18:18–18:37 CDT, right after the head-on session; linked at `sysid/paddle_puck_collision/data/`)
- **Parameters**: the head-on fit `sysid/paddle_puck_collision/results/cmaes_20260910_puck_collision/` — e = 0.631, m_paddle/m_puck = 397.5 (gain 1.627), camera lag 102 ms, speed-estimation settings ([`2026-09-11_01-21`](2026-09-11_01-21_paddle-puck-collision-cmaes-sysid.md))
- **Code**: new `sysid/paddle_puck_collision/code/evaluate_offset_collisions.py`; `HeadOnCollider.run(..., dy=)` (lateral lane offset), `SpeedConfig.oblique` (2-D separation / re-contact gates), `CollisionRenderer(offset=True)`
- **Doc**: [`sysid/paddle-puck-collision.md`](../../docs/environments/real-world/sysid/paddle-puck-collision.md) ("Offset collisions")

## Question

Take the paddle–puck parameters identified on head-on collisions and replay the *offset*
session (same release and paddle action, paddle commanded 3–11 cm beside the puck lane) in the
sim, assuming the puck comes straight down the table into the measured contact position. How
wrong are the simulated exit speed and exit angle?

## Setup

- Session: one condition (release 1/2 ≈ 0.7 m/s at the paddle, action 0.66 ≈ 0.7 m/s paddle),
  commanded paddle y = −0.03 … −0.11 in 1 cm steps, 3–5 takes each. The puck lane is at
  y ≈ −0.02, so the lateral offset at contact `dy = y_puck − y_paddle` runs from 0.1 to 7.6 cm
  (contact distance r_paddle + r_puck = 8.25 cm). The puck's incoming direction is 1–5° off the
  table axis.
- Measurement: exactly the head-on tool (change-point split, damped free-flight fits before /
  after, contact time from their intersection, paddle from the robot at t_c − 102 ms), with the
  outgoing-angle gate opened to 90° and the separation / re-contact gates made 2-D (relative
  velocity along the contact normal; centre distance instead of x-gap). Exit angle = signed
  angle of the outgoing velocity from the paddle normal (table axis).
- Sim: the fit's collider (env from `sysid_best_params_hist2.yaml`, noise / delays off, paddle
  under its own PID at the measured speed), puck launched straight (vy = 0) at the measured
  incoming speed **in the real lane**, i.e. at the measured `dy`, meeting the paddle at the
  x-gap `sqrt(d² − dy²)`. Box2D's paddle–puck contact is frictionless; the listener applies the
  restitution impulse along the contact normal. Scored with the fitted (e, r), the other ridge
  point (r = 2.56, e = 1.26) and the canonical (e = 1.09, r = 2.56).
- Videos: camera · real on the Box2D table · sim replay per trial, as for the head-on set,
  fitted parameters, half speed.

## Results

23 of 30 trials are valid collisions. The 7 dropped are the −0.10 / −0.11 takes: `|dy|` 16–32 cm
at the found split or no reversal — the puck missed or grazed the paddle and the split landed on
a later wall bounce (`all_trials.csv`).

**All 23 valid trials (sim − real):**

| parameters | e | r | gain | exit-speed err mean ± std | rms | mean abs rel | exit-angle err mean ± std | rms | max |
|---|---|---|---|---|---|---|---|---|---|
| fitted | 0.631 | 397.5 | 1.627 | −0.109 ± 0.198 m/s | 0.226 | 13.8 % | +36.3 ± 31.8° | 48.2° | 103° |
| ridge, r canonical | 1.263 | 2.56 | 1.627 | −0.109 ± 0.198 | 0.226 | 13.8 % | +36.3 ± 31.8° | 48.2° | 103° |
| canonical | 1.091 | 2.56 | 1.504 | −0.220 ± 0.154 | 0.269 | 17.7 % | +37.6 ± 32.0° | 49.4° | 104° |

**Per offset (fitted parameters):**

| commanded y | n | dy at contact | in / paddle (m/s) | exit speed real → sim | speed err | exit angle real → sim | angle err |
|---|---|---|---|---|---|---|---|
| −0.03 | 5 | +0.6 cm | 0.62 / 0.69 | 1.57 → 1.51 | −0.06 ± 0.17 | −2° → +7° | +8.8 ± 2.2° |
| −0.04 | 3 | +1.7 | 0.70 / 0.65 | 1.30 → 1.46 | +0.16 ± 0.12 | +4° → +21° | +17.6 ± 2.1° |
| −0.05 | 3 | +2.4 | 0.70 / 0.70 | 1.52 → 1.49 | −0.03 ± 0.05 | +16° → +30° | +14.1 ± 3.7° |
| −0.06 | 3 | +3.6 | 0.68 / 0.68 | 1.34 → 1.35 | +0.01 ± 0.03 | +25° → +47° | +22.4 ± 1.4° |
| −0.07 | 3 | +5.3 | 0.72 / 0.71 | 1.29 → 1.14 | −0.15 ± 0.07 | +40° → +82° | +41.3 ± 5.2° |
| −0.08 | 3 | +6.4 | 0.74 / 0.74 | 1.25 → 0.91 | −0.34 ± 0.03 | +55° → +120° | +65.2 ± 10.1° |
| −0.09 | 3 | +7.4 | 0.69 / 0.69 | 1.09 → 0.69 | −0.39 ± 0.03 | +76° → +179° | +102.8 ± 0.4° |

Over the 17 trials with `dy ≤ 5.5 cm` (offsets the sim handles cleanly) the exit-speed error is
−0.02 ± 0.14 m/s (7.5 % abs — the same accuracy as the head-on fit) and the exit-angle error
+19 ± 12°.

### Findings

1. **Exit speed is reproduced up to ≈ 5 cm offset; exit angle is not.** With the head-on gain the
   sim's exit speed matches to 7.5 % for `dy ≤ 5.5 cm`. The exit angle is systematically too
   large: the sim deflects the puck about **1.9 × the real angle at every offset** (7° vs −2°,
   21° vs 4°, 30° vs 16°, 47° vs 25°, 82° vs 40°). The other ridge point gives identical results
   (the normal impulse of a frictionless contact depends only on the gain, so e and r stay
   degenerate for oblique hits too); the canonical gain only changes the speed.
2. **What the sim would need.** For each trial the offset at which the sim reproduces the real
   exit angle was found by bisection: it is **0.57 × the measured offset** (median; IQR
   0.49–0.65; `per_trial.csv` `dy_required_cm`), i.e. the shortfall grows with the offset
   (0.3 cm at dy 0.7, 1.7 cm at 3.6, 2.5 cm at 5.3). A constant ratio is what a tangential
   impulse gives (friction between puck and paddle rim, or rolling); a y offset between the
   puck-camera frame and the paddle frame would give a constant difference. So the evidence
   favours a missing tangential (friction) term in the frictionless Box2D contact, but the y
   calibration of the two frames has never been checked directly — the camera panel of the
   videos shows the real contact offset, and the head-on static-contact test only verified x.
3. **Grazing contacts (dy ≥ 6.5 cm) are unreliable in the sim.** At −0.08 the frictionless sim
   already sends the puck forward past the paddle (angle 120°, real 55°); at −0.09 two of three
   replays register **no contact at all**: with the contact placed 20 ms into a 50 ms step at
   1.4 m/s closing speed the Box2D time-of-impact misses a 0.7 cm-deep graze (it is found at slow
   speeds or when the contact is placed 45 ms into the step), and a static 1.7 mm overlap is
   not resolved either (below Box2D's 5 mm linear slop). Real grazes at 7.5 cm offset are solid
   hits (exit 1.1 m/s at 77°). The head-on static gap of 85.5 ± 7.4 mm also suggests the real
   contact distance is a few mm larger than the sim's 82.5 mm.
4. **Speed at large offsets.** For `dy ≥ 5 cm` the sim puck leaves 0.15–0.4 m/s too slow: the
   normal impulse shrinks with the offset and nothing else adds energy, whereas the real puck
   keeps ≈ 1.1–1.3 m/s out to 7.5 cm.

## Conclusion

The head-on parameters carry over to oblique hits for the *speed* up to about 5 cm offset, but
the frictionless normal-impulse contact deflects the puck roughly twice as much as the real
paddle does, consistently across offsets. For any use of the sim where the exit *direction*
matters (aiming, goal tasks) the paddle–puck contact needs a tangential term (friction /
tangential restitution) fitted on this session — that is a modelling change, not a parameter
change, and was out of scope here. Grazing hits beyond ≈ 6.5 cm offset are additionally limited
by the 50 ms step and Box2D's slop.

## Next

- Add a tangential coefficient to the paddle–puck contact (or Box2D friction on the paddle
  fixture) and fit it on this session's exit angles; re-check the head-on speeds afterwards.
- Verify the y offset between the puck camera frame and the robot frame with a stationary paddle
  at several y (the x offset was verified this way; y was not).
- For grazes: sub-step the contact or lower `b2_linearSlop`, and measure the real contact
  distance from the static gap.

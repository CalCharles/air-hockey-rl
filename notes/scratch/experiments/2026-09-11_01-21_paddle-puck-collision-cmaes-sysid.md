# Paddle–puck restitution and mass ratio fitted jointly with CMA-ES on the scripted head-on collision session

- **Date**: 2026-09-11 01:21 UTC
- **Status**: done (fit + tooling; nothing promoted to configs)
- **Run dirs**: `sysid/paddle_puck/cmaes_20260910_puck_collision/` (gitignored; `summary.md`, CSVs, plots)
- **Data**: `/data2/air_hockey/robot_data_collection_puck_collision_20260910_1719/` (44 files, 12 conditions)
- **Configs**: base `configs/new_juggle/sysid_best_params_hist2.yaml`; fitted `…/sim_config_fitted.yaml` (not copied into `configs/`)
- **Code**: new `scripts/sysid/paddle_puck/` (dataset, speeds, sim_collision, cmaes_fit, report, `fit_collision_cmaes.py`, tests)
- **Doc**: [`notes/docs/environments/real-world/sysid/paddle-puck-collision.md`](../../docs/environments/real-world/sysid/paddle-puck-collision.md)
- **Related**: [`2026-09-10_02-49_paddle-puck-mass-ratio.md`](2026-09-10_02-49_paddle-puck-mass-ratio.md) (predicted the degeneracy), [`2026-09-10_03-39_paddle-pid-cmaes-sysid.md`](2026-09-10_03-39_paddle-pid-cmaes-sysid.md) (same CMA-ES / validation pattern)

## Question

Identify the paddle–puck restitution `e` and the mass ratio `r = m_paddle/m_puck` of the Box2D
contact from real head-on collisions, matching only the puck's incoming and outgoing speeds
(paddle speed taken from the robot), after selecting the best three takes per condition.

## Setup

Session protocol: puck released from `top` / `3/4` / `1/2` of the table (approach ≈ 1.1 / 1.0 /
0.65 m/s at the paddle) × paddle action `delta` 0 / 0.33 / 0.66 / 1.00 (plateau ≈ 0 / 0.3 / 0.6 /
1.0 m/s), 3 clean repeats intended, extra takes kept on disk.

Measurement (see the doc): damped free-flight fits on 6 usable frames before / after a
change-point split, contact time from the crossing, speeds at contact; camera lag calibrated
from the moving-paddle trials; validity gates (≤ 40° outgoing, puck separates, paddle up to
speed, no re-contact); quality score → best 3 per condition. Sim: the env's own paddle plant
driven at the measured paddle speed, puck launched at the measured incoming speed, outgoing speed
read after the contact step. CMA-ES on train (23 collisions), one trial per condition held out
(12), 31 × 31 landscape grid.

## Results

**Data quality.** Camera lag 102 ± 23 ms (21 moving-paddle trials, no dependence on paddle
speed); stationary-paddle contact gap 85.5 ± 7.4 mm vs 82.5 mm expected → the `pose_x + 1.2`
paddle frame is right. 35 of 44 files valid and selected:

| rejected | why |
|---|---|
| 000 (top / 1.00) | trigger at `x = 0.5` (later runs 0.25): the puck hit the still-stationary paddle (`u_p = 0`, gain 1.61 — a fine `delta 0` datapoint) and was then carried by the accelerating paddle; images confirm |
| 022 (1/2 / 1.00) | same failure: hit at `t = 0.10 s` before the paddle moved |
| 009 (top / 0.33) | outgoing 71° — glancing |
| 037 (3/4 / 0), 038–040 (1/2 / 0) | outgoing 50–68°; 038–040 are the `y −0.030 / −0.060` takes that were re-done at `−0.045` (041–043) |
| 003, 013, 016 | valid 4th takes, ranked below the other three (013: hit while the paddle was still accelerating, `u_p` 0.65 vs plateau 1.0) |

`h3-4_delta0.00` keeps only 2 trials (035, 036, both 32° — the puck arrived 3 cm off the paddle
centre). `h1-2_delta0.00` keeps 041–043 at 23–29°. Every selected `delta > 0` trial is ≤ 22°.

**Measured gain** `(v_out + v_in)/(u_p + v_in)` is flat across the whole approach-speed range
0.8–2.2 m/s (selected trials: mean 1.63, spread 1.37–1.83; `plots/gain_vs_speed.png`), i.e. no
speed-dependent restitution is visible at this noise level. The `1/2 · 1.00` trials sit high
(1.62–1.83) and the `3/4 · 1.00` ones low (1.37–1.62).

**Fit** (`summary.md`):

| | e | r | gain | train RMS (m/s) | val RMS (m/s) | mean err train | mean abs rel err |
|---|---|---|---|---|---|---|---|
| canonical sim | 1.0915 | 2.56 | 1.504 | 0.223 | 0.239 | −0.175 (sim slower) | 14.4 % |
| CMA-ES best | 0.631 | 398 | **1.627** | **0.134** | **0.121** | −0.002 | 7.5 % |

576 candidates, 7 s. Selection beats 64 % of candidates on val and reaches 97 % of the val oracle
(0.117; the val landscape is flat along the ridge); canonical beats 20 %. Largest residuals:
`1/2 · 1.00` (023: −0.30, 025: −0.27) and 012 (+0.43, its outgoing 1.67 vs 2.0–2.1 for its
siblings) — a single gain cannot fit both those groups.

**Degeneracy, measured.** Train / val RMS are identical (0.134 / 0.121) at every point of the
ridge `(1+e)·r/(r+1) = 1.627`:

| r | e on the ridge |
|---|---|
| 1000 (rigid paddle) | 0.629 |
| 398 (CMA-ES) | 0.631 |
| 4.46 (real 58 g / 13 g) | 0.992 |
| 2.56 (canonical densities) | 1.263 |

The landscape (`plots/landscape.png`) is a single valley along that curve; CMA-ES candidates
scatter along it. Only the gain is identified; the sim's current `e = 1.09 · 2.56/3.56 = 1.504`
launches the puck **8 % too slowly** on average, and correcting it means raising the gain to
≈ 1.63, not choosing between `e` and `r`.

## Conclusion

- The canonical dataset is 35 head-on collisions (best 3 of every condition, one condition with
  2); the speed estimates are robust to the free-flight constants (mean gain moves ≤ 0.04) and
  the frame / lag calibration checks out against the geometry.
- The paddle–puck gain is **1.63 ± ~0.03** (RMS 0.12–0.13 m/s ≈ 7.5 % of the outgoing speed) vs
  1.50 in the sim today. Restitution and mass ratio are exactly degenerate for head-on data with
  this contact model — as predicted in the 2026-09-10 note — so "e = 0.63, r = 398" is a
  representative, not a unique answer. Picking the ridge point is a modelling choice: a large `r`
  (no paddle recoil, as the UR5's recorded speed shows) with `e ≈ 0.63`, or keeping `r = 2.56`
  with `e ≈ 1.26`. Nothing was changed in `configs/`.
- The remaining 7.5 % scatter is between conditions (the `1/2 · 1.00` and `3/4 · 1.00` groups
  pull in opposite directions), not within them, so a second parameter would have to act on
  something the head-on model lacks (paddle acceleration at contact, off-centre hits).

## Next

- Break the degeneracy with the paddle's own post-impact speed (`speed_x` through the contact
  is recorded; the sim paddle at `r = 2.56` loses ~60 % of its speed per hit, the robot none).
- Decide which ridge point to promote and re-run the canonical juggle training with it (launch
  speeds +8 %).
- The oblique takes (Δy up to 4 cm) could be used with a normal / tangential decomposition
  instead of being dropped.

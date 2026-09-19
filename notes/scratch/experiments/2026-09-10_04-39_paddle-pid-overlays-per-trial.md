# Paddle PID fit, qualitative: sim-vs-real overlays on the Box2D scene and per-trial errors

- **Date**: 2026-09-10 04:39 UTC start
- **Status**: done
- **Follows**: [`2026-09-10_03-39_paddle-pid-cmaes-sysid.md`](2026-09-10_03-39_paddle-pid-cmaes-sysid.md) (the fit)
- **Run dir**: `sysid/paddle/cmaes_20260909_paddle_motion/overlays/` (gitignored; 81 GIFs + PNGs, 27 condition mosaics, `mosaic_val.png`, `mosaic_all.png`, `per_trial_errors.md/.csv`, `plots/`)
- **Code**: `scripts/sysid/paddle/overlay.py`, `scripts/sysid/paddle/render_overlays.py`

## Question

What do the canonical (kp 9000 / ki 0 / kd 50) and CMA-ES (kp 5496 / ki 5883 / kd 0) replays
look like next to the real paddle on the table, trial by trial — and where exactly does each
gain set win or lose, rather than only in the aggregate?

## Setup

`render_overlays.py --input-dir <session> --fit-dir sysid/paddle/cmaes_20260909_paddle_motion`:
every trial replayed with both gain sets; per step the real paddle is the paddle sprite, each
sim a translucent ghost (orange canonical, blue CMA-ES) with a trail, the recorded target a
grey cross; frames cropped to the robot workspace, 360 px wide, GIF at 10 fps (half real
time). Static per-trial PNGs with the full trails, mosaics per condition (3 repeats) and for
the 27 validation trials, a per-trial table and two plots (mean error per trial by condition;
per-step error curves of all 81 trials by condition).

## Results

**Scene overlays** (`mosaic_val.png`, `mosaic_<condition>.png`, `gifs/`):

- With the canonical gains the orange ghost leaves the start ~2 steps before the real paddle
  and stays ahead for the whole move; on the fast diagonals (delta 1.00) it is 60–100 mm ahead
  at steps 6–8 and still 50–80 mm off at the end. The blue (CMA-ES) ghost sits on the real
  paddle through the onset and stays within ~1 paddle radius except on the delta-1.00 diagonals
  and the tight arc at 0.60 m/s.
- Both sims end up at the same place as the real paddle on the axis-aligned lines that hit the
  workspace clip (x delta 0.66 / 1.00: final 17–27 mm); the remaining final error there is the
  real robot's cross-axis drift (a y offset of ~2 cm builds up during a pure x move, visible as
  the real trail bending in the static overlays) plus the sim reaching the clipped target more
  slowly.
- The arcs are the cleanest match: all three arc families are within ~1 cm mean with CMA-ES
  (5–11 mm at the two lower speeds), the trails overlap the real one almost everywhere.

**Per trial** (`per_trial_errors.md`, `plots/per_trial_errors.png`):

| family (n trials) | canonical mean (mm) | CMA-ES mean (mm) |
|---|---|---|
| x+ (9) | 27.9 | 18.0 |
| x− (9) | 22.3 | 17.7 |
| y+ (9) | 17.3 | 6.9 |
| y− (9) | 15.3 | 9.4 |
| diag+ (9) | 39.3 | 21.2 |
| diag− (9) | 42.3 | 25.2 |
| arcs (27) | 24.3 | 12.9 |

- Repeats agree: for most conditions the three repeats are within 1–3 mm of each other under
  either gain set (e.g. xpos_delta0.66 canonical 28.7 / 28.1 / 27.4, CMA-ES 13.9 / 14.1 / 14.3),
  so a held-out repeat is a fair validation. Exceptions where the *robot* differs between
  repeats by 10–20 mm: xneg_delta0.66 (trial 3 responded a step later), xneg_delta1.00,
  diagneg_delta1.00, arc_medium_v0.50, arc_tight_v0.60 — all fast moves.
- CMA-ES is worse on 22 of 81 trials, almost all slow ones where both errors are small
  (x/y/diag delta 0.33, yneg 0.66 / 1.00, arc_medium at all speeds, arc_wide 0.32): there the
  large-ki plant is a little *too* sluggish (blue trails lag the real paddle mid-move, the
  error hump at steps 10–15 in `per_trial_step_errors.png`), and xneg_delta0.66 trials 1–2
  (17 → 24–26 mm) where the real robot was faster than in the x+ direction.
- Error is transient everywhere: the per-step curves peak 2–5 steps after onset (canonical
  50–100 mm on delta-1.00 lines, CMA-ES 35–75) and decay to the final offset; the fit mainly
  removes the onset error, it does not change the shape of the late-move error.

**Real robot onset latency** (first ≥ 2 mm displacement after the first non-zero action):
2–3 steps (100–150 ms) in every condition and direction, varying by one step between repeats
of the same condition (the command lands at a random phase of the 49 ms control block plus a
fixed servo delay). Peak speeds are direction-dependent: x− 0.99–1.03 m/s vs x+ 0.82–0.87 m/s
at delta 1.00 (same commanded 0.26 m step); y moves reach only 0.49 m/s (move_lims 0.12).

## Conclusion

The fitted gains fix the dominant, systematic error of the canonical plant — the sim starting
2 steps early — and match the real paddle to about a paddle radius on everything except the
fastest diagonals. What is left is not a gain problem: a 2–3-step dead time with ±1 step
jitter, a direction-dependent peak speed (x− faster than x+), and a cross-axis drift during x
moves. None of these is representable by an isotropic PID on a point mass, which is why the
optimum settles on a "slow integral" compromise and loses a few mm on the gentle moves.

## Next

- Dead-time model in the Box2D paddle (2 steps + optional jitter) — the largest remaining
  systematic term (cf. the `--action-delay-steps` diagnostic in the fit note).
- Per-axis gains (kp_x ≠ kp_y) or a direction-dependent force scale to capture the x−/x+
  asymmetry; needs the env to expose them.
- Repeat the session on another day to see whether the 2–3-step latency and the asymmetry are
  stable.

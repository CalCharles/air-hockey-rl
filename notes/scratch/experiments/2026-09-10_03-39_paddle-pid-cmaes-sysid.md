# Paddle PID gains refit with CMA-ES on the scripted paddle-motion session (mass fixed, one trial per condition held out)

- **Date**: 2026-09-10 03:39 UTC start
- **Status**: done
- **Run dirs**: `sysid/paddle/cmaes_20260909_paddle_motion/` (main), `…_split1/`, `…_split2/` (other held-out trials), `…_delay1/`, `…_delay2/` (latency diagnostic), `sysid/paddle/eval_20260909_reference_gains/` (all gitignored)
- **Data**: `/data2/air_hockey/vertical_horizontal_diagonal_arc_paddle_motion_20260909_2024` (81 trials = 27 conditions × 3 repeats, `hist_len` 1, recorded 2026-09-09 20:25–20:37 CDT)
- **Code**: `scripts/sysid/paddle/` (new; doc [`notes/docs/environments/real-world/sysid/paddle-pid.md`](../../docs/environments/real-world/sysid/paddle-pid.md))
- **Configs**: base `configs/new_juggle/sysid_best_params_hist2.yaml`; fitted copy at `sysid/paddle/cmaes_20260909_paddle_motion/sim_config_fitted.yaml` (not promoted)

## Question

Can a CMA-ES search over (kp, ki, kd) with the paddle mass fixed (`paddle_density` 3000) beat
the hand-tuned 2026-05 grid-search gains (kp 9000, kd 50, ki 0) on the per-step position
error of a new scripted paddle-motion session, and does the gain hold up on held-out trials?

## Setup

- Split: one trial of each of the 27 conditions held out (seed 0) → 54 train / 27 val.
- Replay: reset to the real pose + velocity of row 0, step the recorded actions, compare the
  sim paddle after each step with the recorded pose (robot frame). Sim at `hist_len` 1 (the
  recording's), workspace / edge / move limits from the recording, gravity 0, puck parked,
  noise / delays / terminations off. Target mapping verified (max 0.8 mm vs recorded
  `desired_pose`). 0.28 ms per sim step → one candidate (81 trials) ≈ 0.5 s.
- Metric: mean over trials of the per-trial mean over steps k ≥ 1 of ‖sim − real‖ (mm).
- CMA-ES: unit cube → kp log-uniform [500, 1e5], ki / kd `expm1` mapping (0 reachable) up to
  1e5 / 5e3; x0 = canonical, sigma0 0.3, popsize 24, tolfun 0.02 mm, 2 IPOP restarts, 16
  workers; 2472 candidates in 86 s. Validation error of every candidate recorded.

## Results

| gains | kp | ki | kd | train (mm) | **val (mm)** | val rms | val max | val final |
|---|---|---|---|---|---|---|---|---|
| canonical hist2 | 9000 | 0 | 50 | 25.99 | **27.02** | 30.2 | 43.2 | 31.8 |
| hist4 refit gains, density 3000 | 7500 | 0 | 50 | 19.79 | 19.75 | | | |
| hist4 refit gains, density 3500 (as in `sysid_best_params_hist4.yaml`) | 7500 | 0 | 50 | 28.59 | 28.33 | | | |
| **CMA-ES (this run)** | **5496** | **5883** | **0** | 15.21 | **15.23** | 17.7 | 30.5 | 21.0 |

Validation error −43.6 % relative to canonical. Both CMA-ES restarts stopped on `tolfun` at the
same point; the second (popsize 48) re-explored widely and returned to it (`plots/convergence.png`,
`plots/candidates.png`). kd sits at 0: with the fixed mass and `paddle_damping` 17 the plant
wants no derivative damping (an earlier run with a kd floor of 0.5 pinned there, hence the
`expm1` mapping).

**Held-out choice does not matter** (single session, so this is the only replication we have):

| split seed | fitted kp / ki / kd | canonical val | fitted val | selection beats candidates | val oracle |
|---|---|---|---|---|---|
| 0 | 5496 / 5883 / 0 | 27.02 | 15.23 | 91 % of 2472 | 15.21 |
| 1 | 5441 / 5954 / 0 | 26.18 | 15.88 | 93 % of 2688 | 15.86 |
| 2 | 5501 / 5699 / 0 | 25.92 | 16.38 | 87 % of 2784 | 16.22 |

The selected gains equal the validation oracle among all candidates in every split (the
percentile report's p90 candidate is already at the oracle → the landscape around the optimum
is flat along the ki–kp ridge visible in `candidates.png`, not a sign of a weak search).

**Per condition (val, mm, canonical → fitted)**: improves 22 of 27 conditions, most on y,
diagonal and arc moves (ypos_delta1.00 31.7 → 9.7, yneg_delta1.00 32.3 → 10.1,
diagpos_delta0.66 40.4 → 19.5, arc_wide_v0.45 31.8 → 11.0, arc_tight_v0.42 31.5 → 10.7); worse
on xneg_delta0.66 (17.1 → 26.0), arc_medium_v0.35 (14.3 → 18.7), ypos_delta0.33 (2.8 → 6.0),
arc_medium_v0.50 (+2.3), yneg_delta0.66 (+2.2). Largest remaining errors are the fastest
trials: xpos_delta1.00 37.9, diagpos_delta1.00 37.4, arc_tight_v0.60 34, diagneg_delta1.00 30.5.
Full table in `summary.md`; trajectories in `plots/val_trajectories.png`.

**What limits the fit** (`plots/val_trajectories.png`, `plots/val_step_error_profile.png`):

1. **Command latency.** After the first non-zero action (step 3) the real paddle does not move
   for ~1.5 steps (≈ 75 ms; e.g. xpos_delta1.00: 0.2 / 1.3 / 15 mm of x travel in steps 4 / 5 /
   6) and then accelerates to a higher peak speed (0.81 m/s) than the canonical sim, which
   starts moving in step 4 (35 mm). The error profile peaks right after onset for both gain
   sets. The fitted (kp↓, ki≫0, kd 0) plant imitates the dead time with a ramping integral
   force, which is why ki is large.
   Diagnostic — shifting the replayed actions by n steps (`--action-delay-steps`):

   | action delay | canonical val (mm) | CMA-ES val (mm) | fitted kp / ki / kd |
   |---|---|---|---|
   | 0 (this run) | 27.02 | 15.23 | 5496 / 5883 / 0 |
   | 1 step | 15.98 | 13.23 | 7336 / 2967 / 0 |
   | 2 steps | 13.77 | 13.07 | 11987 / 1651 / 1113 |

   A one-step delay alone brings the *canonical* gains to 16 mm; with a delay the optimum
   moves back to a stiff, damped PD-like controller. A real latency model (not just a
   shifted action) is the next lever — but it changes the sim's control interface, so it is a
   separate decision.
2. **Cross-axis drift** on the robot (≈ +19 mm y during a 0.4 m pure-x move at delta 1.00, not
   in the sim) and the y-axis steady-state lag in fast y moves.
3. dt: the recording steps at 0.0492 s mean (std 2.8 ms), the sim at 0.05 s (not corrected).

## Conclusion

CMA-ES over (kp, ki, kd) at fixed mass is a working replacement for the hand-tuned grid:
27.0 → 15.2 mm per-step validation error (−44 %) on the 2026-09-09 session, the same optimum
from three held-out choices and two restarts, and the selection is the validation oracle among
~2.5k candidates. The gains (kp ≈ 5500, ki ≈ 5900, kd 0) are qualitatively different from the
canonical (kp 9000, kd 50, ki 0) because they emulate a ~75 ms command latency the sim does not
model. Not promoted to the canonical configs: the earlier hist4 episode showed a paddle refit
can hurt policy learning, and the recording is `hist_len` 1 while training runs `hist_len` 2
(the gains describe the servo, not the smoothing, so they should transfer — untested).
Single session, single robot state; no repeat session yet.

## Next

- Add an explicit command-latency model to the Box2D paddle (dead time of 1–2 steps) and refit
  — expected ≈ 13 mm and a stiff PD optimum.
- Fit `paddle_density` jointly (mass ↔ kp are partly degenerate; the fixed-mass constraint was
  deliberate here) and check the paddle–puck collision implication
  ([`2026-09-10_02-49_paddle-puck-mass-ratio.md`](2026-09-10_02-49_paddle-puck-mass-ratio.md)).
- Train the juggle / puck_vel tasks with `sim_config_fitted.yaml` gains vs canonical to see
  whether the better plant fit helps or hurts learning (cf. the hist4 episode).
- Record a second session (different day) to validate across sessions rather than trials.

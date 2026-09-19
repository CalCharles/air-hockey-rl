# Paddle PID gains refit on the pooled lines / arcs + reversal-jerk sessions — the lines-only optimum overshoots reversals, the pooled optimum is promoted

- **Date**: 2026-09-19 01:35 UTC
- **Status**: done (gains promoted into `configs/new_juggle/sysid_v2_hist2.yaml`)
- **Supersedes** (as the reference gains): [`2026-09-10_03-39_paddle-pid-cmaes-sysid.md`](2026-09-10_03-39_paddle-pid-cmaes-sysid.md) — that fit stays valid for its own session
- **Run dirs** (gitignored): `sysid/paddle_pid/results/cmaes_20260919_paddle_motion_and_jerk/` (main, + `overlays/` of the 45 val trials), `…/cmaes_20260910_reversal_jerk_only/`, `…/eval_20260919_reference_gains_pooled/`, `…/eval_20260919_reference_gains_paddle_motion/`, `…/eval_reversal_jerk_reference_gains/`
- **Data**: `/data2/air_hockey/vertical_horizontal_diagonal_arc_paddle_motion_20260909_2024` (81 trials, lines + arcs) + **new** `/data2/air_hockey/reversal_jerk_20260910_1921` (63 trials, 2026-09-10 19:43–20:33 CDT; `jerk_<idx>_<cond>_delta<d>_out<n>_slow<m>.hdf5`: out, reverse, slow down; `up_down` x-moves of 10 steps that saturate at the workspace limit, `right_left` y-moves of 20–22 steps; hist_len 1, same lims)
- **Code**: `sysid/paddle_pid/code/dataset.py` (jerk layout, several sessions pooled, aborted trials from file attrs — the session manifest only lists the last restart), `fit_pid_cmaes.py` / `evaluate_gains.py` / `render_overlays.py` take `--input-dir A B …`; per-session errors in the summary; `report.plot_step_errors` handles unequal trial lengths
- **Doc**: [`sysid/paddle-pid.md`](../../docs/environments/real-world/sysid/paddle-pid.md)

## Question

Do the CMA-ES paddle gains fitted on straight lines and arcs (kp 5496 / ki 5883 / kd 0, 15.2 mm)
hold up on a session of hard reversals, and what single gain set fits both sessions?

## Setup

Same replay, metric and search as the first fit (one trial per condition held out, seed 0; CMA-ES
popsize 24, sigma0 0.3, 2 IPOP restarts, 16 workers, mass fixed at density 3000, sim hist_len 1
like the recordings). Three fits / evaluations:

- pooled: 144 trials, 53 conditions → 99 train / 45 val (27 + 18 val), 1656 candidates, 116 s;
- jerks only: 63 trials, 26 conditions → 45 / 18, 1608 candidates, 59 s;
- every gain set scored on every session with `evaluate_gains.py`.

## Results

Mean per-step paddle position error on the held-out trials (mm):

| gains | kp | ki | kd | pooled val | lines / arcs val | jerks val | pooled all |
|---|---|---|---|---|---|---|---|
| canonical hist2 (`sysid_best_params_hist2.yaml`) | 9000 | 0 | 50 | 23.23 | 27.02 | 17.55 | 24.77 |
| lines-only CMA-ES (2026-09-10) | 5496 | 5883 | 0 | 24.99 | **15.23** | 39.68 | 25.85 |
| jerks-only CMA-ES | 8339 | 1423 | 0 | 19.86 | 25.91 | **11.14** | 21.11 |
| **pooled CMA-ES** | **7532** | **1929** | **0** | **17.45** | 19.66 | 14.14 | 18.94 |

Pooled fit: train 19.62 mm, val 17.45 mm (−24.9 % vs canonical); both restarts stopped on
`tolfun` at the same point; the selection beats 78 % of the 1656 candidates and reaches 99 % of
the validation oracle (17.33 mm at kp 7463 / ki 1774 / kd 0.3; the canonical gains beat 27 %).
Per condition (val): every line / arc family improves except the smallest x−/y+ moves (±0.5 mm);
delta-1.00 lines stay the worst (46 mm on xpos, the onset-latency limit). Overlays
(`overlays/mosaic_*.png`, 45 GIFs): on `right_left_delta1.00_out20_slow03` the lines-only ghost
runs 250 mm past the real paddle at the reversal (mean 79 mm), the pooled ghost stays within
23 mm mean / 54 mm max, canonical 28 / 53.

## Findings

1. **The lines-only gains do not generalise.** Their large integral term — which reproduces the
   ~75 ms command latency on straight moves — winds up over a 20-step move and drives the paddle
   far past the reversal point: 39.7 mm on the jerks, worse than canonical (17.6). Any session
   with only monotone moves cannot see this.
2. **No PID gain set fits both.** The single-session optima differ (ki 5883 vs 1423) and the pooled
   optimum (kp 7532, ki 1929, kd 0) loses 4.4 mm on the lines and 3.0 mm on the jerks relative to
   each specialist, while beating canonical on both. The remaining gap is structural: the onset
   latency (a delay, not a gain) and, at reversals, the robot's deceleration limit.
3. kd stays at 0 in all three fits (as before: with the fixed mass and `paddle_damping` 17 the
   plant is already overdamped).

## Conclusion

The pooled gains kp 7531.5 / ki 1928.7 / kd 0 are the paddle plant of sysid v2
(`configs/new_juggle/sysid_v2_hist2.yaml`): −25 % error vs canonical over both kinds of motion,
without the reversal overshoot of the first fit. Any future paddle-gain fit must include reversal
data.

## Next

- Explicit command latency in the Box2D paddle (1–2 steps) and a refit on the pooled data — the
  first fit's diagnostic showed canonical gains reach 16 mm with a 1-step delay.
- A joint fit of `paddle_density` (mass) with the gains, now that reversals constrain the inertia.

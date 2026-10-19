# Real start states + the real 8-step zero-action hold, in sim

- **Date**: 2026-10-18 07:44 UTC
- **Status**: done
- **Follows**: [`2026-10-18_07-38_real-start-states-sim2sim.md`](2026-10-18_07-38_real-start-states-sim2sim.md). Same policies, starts, seeds and config; only the hold is added.
- **Output**: `runs/td3/sysid_v2_real_workspace/real_start_sim2sim_hold8/`
- **Command**: `scripts/td3/extras/eval_real_start_sim2sim.py --hold-steps 8` (other arguments as in the previous note)

## Setup

On the real robot, the first 8 steps of every episode send `policy_action = 0`. The desired pose equals the current pose, so the paddle sits still for about 0.4 s while the puck moves. Those steps count toward the episode.

The new `--hold-steps 8` option does the same in sim, in all three conditions:
- it sends a zero action (zero displacement target, so the PID holds the paddle)
- the 8 steps count toward return and length
- when the policy takes over, its last-action input is zero

## Results

Return is the mean ± standard error across the 5 hist4 policies, each over 40 paired episodes.

| | Sim reset | Real puck start | Real puck + paddle start |
|---|---|---|---|
| **With 8-step hold** | **108.5 ± 7.3** | **128.3 ± 7.4** | **125.6 ± 6.1** |
| Δ vs sim reset (paired) | – | +19.9 ± 6.1 | +17.2 ± 5.0 |
| Without hold (previous note) | 110.6 ± 6.2 | 120.8 ± 5.9 | 124.0 ± 4.1 |

| Policy (with hold) | Sim reset | Real puck | Real puck + paddle |
|---|---|---|---|
| hist4_seed0 | 106.2 ± 7.6 | 139.5 ± 6.4 | 133.3 ± 7.1 |
| hist4_seed1 | 108.9 ± 7.2 | 140.9 ± 6.1 | 137.1 ± 6.4 |
| hist4_seed2 | 125.3 ± 6.6 | 125.1 ± 7.5 | 130.1 ± 7.0 |
| hist4_seed3 | 82.9 ± 8.9 | 100.8 ± 8.0 | 102.7 ± 7.6 |
| hist4_seed4 | 119.0 ± 7.7 | 135.3 ± 7.6 | 125.0 ± 6.0 |

## Conclusion

- **The hold does not hurt.** For each condition, the results with and without the hold agree within their standard errors.
- **The real starts stay easier than the sim reset**: +17 to +20 with the hold. After the reset strike the puck is still moving away during the 0.4 s hold, so the policy loses nothing by waiting.
- **The start conditions, including the hold, are ruled out as a cause of the real-robot drop** (130 in sim → 48 real).
- **Confidence.** Per-policy standard errors are about ±6–9 at 40 episodes, and only one seed set was run. The changes in individual policies between the hold and no-hold runs (e.g. seed2 sim reset 110.8 → 125.3) are within the noise, because the hold changes the trajectory from step 0.

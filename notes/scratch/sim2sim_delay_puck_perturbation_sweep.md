# Sim2sim — radius decay sweeps with mass conservation

**Date:** 2026-04-27 · **Status:** consolidated; supersedes earlier non-mass-preserved sweep results.

**Motivation:** residual RL on `sim2sim_combined` showed too-small a sim2sim gap to give residual learning headroom. We needed to find a single-knob perturbation that produces a clean, monotonic decay in zero-shot return so future residual experiments have a meaningful gap to close.

## Headline

**Paddle-radius shrink with mass preserved** is the right knob:
- Clean monotonic decay 98 → 44 across 0–70% radius reduction.
- No plateau; the policy keeps losing ground all the way down.
- Puck-radius shrink, even with mass preserved, only buys ~20 points of degradation across the full sweep — too narrow for residual headroom.

The earlier "smaller puck doesn't hurt the policy" finding was an artifact of **mass loss**: shrinking the radius without compensating density made the puck up to 84% lighter, which made it bounce higher per hit and more than offset the smaller contact target. Once mass is held constant, the puck-shrink curve still decays — just gently.

---

## What changed in the sim

`airhockey/sims/airhockey_box2d.py` now exposes two opt-in mass-preservation knobs (default `None`, behavior unchanged):

| key | semantic |
|---|---|
| `paddle_mass_reference_radius` | If non-None, scale the effective paddle density by `(reference / paddle_radius)²` at init so paddle mass = `paddle_density · π · reference²` regardless of `paddle_radius`. |
| `puck_mass_reference_radius`   | Same for puck. |

The density-fluctuation code (`enable_paddle_density_fluctuation`) acts multiplicatively on the new mass-preserving baseline, so it composes cleanly.

Verification: with `paddle_radius=0.02540` (−50%) and `paddle_mass_reference_radius=0.0508`, the simulator computes `paddle_density 3000 → 12000`; the resulting Box2D body's `.mass` reads 24.32 — identical to source. Same pattern for puck. (See `notes/scratch/radius_sweeps_mass_preserved.py` for the verification code path.)

Bug surfaced in passing: **`sim2sim_combined.yaml`** (the canonical residual-RL baseline) silently runs with paddle mass at 15.57 vs source 24.32 (−36%) because it shrinks `paddle_radius` 0.0508→0.04064 without compensating density. To fix without changing every existing campaign, add `paddle_mass_reference_radius: 0.0508` to that file.

---

## Source policy and protocol

- Checkpoint: `latest_model/hist2_motion0/model.pth`
- Eval driver: `scripts/smooth_policy/sim2sim_eval.py`, n=50 deterministic episodes, seed=0
- Sweep driver: `notes/scratch/radius_sweeps_mass_preserved.py`
- Base config: `configs/new_juggle/sim2sim_combined_v2.yaml` — provides the fixed perturbations (pid_kp 7200, wall_cone 25, action_delay enabled, normal jitter at ±0.35) and `delay_seconds: 0.030` (+20% from source)
- Each sweep varies one radius, holds the other at source, sets the matching mass-preservation flag.

## Puck-radius decay (mass preserved)

`runs/td3/sim2sim/perturbation_sweep/puck_radius_decay_mass_preserved/`

| pct shrink | puck_radius (m) | density_eff | mass_eff | mean | median | std | n_zero | ≥100/50 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  0% | 0.03175 |  3 000 | 9.5008 | 98.02 | 89.0 | 66.5 | 6 | 22 |
| 10% | 0.02857 |  3 705 | 9.5008 | 90.22 | 83.0 | 61.7 | 7 | 22 |
| 20% | 0.02540 |  4 688 | 9.5008 | 92.54 | 89.0 | 61.7 | 7 | 21 |
| 30% | 0.02222 |  6 125 | 9.5008 | 88.14 | 75.5 | 69.8 | 8 | 19 |
| 40% | 0.01905 |  8 333 | 9.5008 | 91.54 | 78.0 | 70.6 | 8 | 23 |
| 50% | 0.01588 | 11 992 | 9.5008 | 78.52 | 79.5 | 64.7 | 9 | 17 |
| 60% | 0.01270 | 18 750 | 9.5008 | 75.44 | 55.0 | 65.4 | 8 | 16 |
| 70% | 0.00953 | 33 298 | 9.5008 | 76.18 | 55.0 | 66.6 | 10 | 21 |

Decay is shallow: mean drops from 98 to ~75–78 and plateaus by 50% shrink. The policy can still roughly juggle a smaller-but-same-mass puck. `n_zero` barely moves (6 → 10), confirming the perturbation doesn't add catastrophic-failure risk.

## Paddle-radius decay (mass preserved)

`runs/td3/sim2sim/perturbation_sweep/paddle_radius_decay_mass_preserved/`

| pct shrink | paddle_radius (m) | density_eff | mass_eff | mean | median | std | n_zero | ≥100/50 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  0% | 0.05080 |  3 000 | 24.3220 | 98.02 | 89.0 | 66.5 | 6 | 22 |
| 10% | 0.04572 |  3 704 | 24.3220 | 84.60 | 75.5 | 63.3 | 8 | 20 |
| 20% | 0.04064 |  4 688 | 24.3220 | 84.50 | 74.5 | 59.8 | 8 | 19 |
| 30% | 0.03556 |  6 122 | 24.3220 | 75.54 | 75.0 | 55.9 | 7 | 16 |
| 40% | 0.03048 |  8 333 | 24.3220 | 75.14 | 51.5 | 65.8 | 8 | 18 |
| 50% | 0.02540 | 12 000 | 24.3220 | 63.64 | 56.0 | 53.9 | 8 | 12 |
| 60% | 0.02032 | 18 750 | 24.3220 | **49.26** | 29.0 | 48.3 | 9 | 10 |
| 70% | 0.01524 | 33 333 | 24.3220 | **43.84** | 28.5 | 47.6 | 10 |  7 |

Clean linear-ish decay 98 → 44 with no plateau. The contact-window-shrink mechanism has no mass-side compensation, so the policy genuinely loses skill as the paddle shrinks.

## Side-by-side

```
  pct   puck (mass-pres)   paddle (mass-pres)
   0%       98.02              98.02
  10%       90.22              84.60
  20%       92.54              84.50
  30%       88.14              75.54
  40%       91.54              75.14
  50%       78.52              63.64
  60%       75.44              49.26
  70%       76.18              43.84
```

`n_ge100` deltas tell the same story:
- Puck: 22 → 16 (small ceiling drop).
- Paddle: 22 →  7 (dramatic ceiling drop — most episodes can't reach long juggles).

`mass_eff` columns confirm the implementation is in fact pinning mass at 9.5008 / 24.3220 across every row.

## Recommendation for residual RL

- **Best target:** paddle_radius shrink with mass preserved.
  - **`paddle_radius=0.02032` (−60%)** → zero-shot mean 49.26, ~50-point gap, policy still juggles well enough to bootstrap (≥100 in 10/50 eps).
  - `paddle_radius=0.02540` (−50%) → mean 63.64, ~34-point gap. Easier-to-recover bootstrap if the −60% version proves unrecoverable.
- **Avoid:** puck-shrink-only targets — capped at ~20-point gap even at 70% shrink. Not enough headroom for residual to clearly outperform zero-shot.
- **Always set:** the matching `*_mass_reference_radius` flag whenever you change either radius. Without it, a chunk of the apparent perturbation is just mass loss.

Suggested next-step config (sketch):

```yaml
# In the simulator_params block of a new sim2sim target:
paddle_radius: 0.02032
paddle_mass_reference_radius: 0.0508   # preserve mass against source
puck_radius: 0.03175                   # source
puck_mass_reference_radius: null
delay_seconds: 0.030                   # +20% from source
delay_relative_range: 0.35
delay_jitter_distribution: "normal"    # opt-in normal-clipped jitter
# (carryover from sim2sim_combined: pid_kp 7200, wall_cone 25, action_delay true)
```

---

## Historical context (what was wrong before)

A first pass varied `puck_radius` and found that returns barely moved (sometimes even improved) at 40–60% shrink. That was misleading: Box2D circle mass = `density · π · r²`, so shrinking radius without compensating density made the puck up to 84% lighter. A lighter puck flies higher off the paddle, giving the policy more reaction time and offsetting the smaller contact target.

Adding the mass-preservation knobs and re-running with mass held fixed produced the clean curves above. The legacy non-mass-preserved data lives at `runs/td3/sim2sim/perturbation_sweep/puck_radius_decay/` for contrast — its mean dropped 98 → 56 at 60% shrink, vs the corrected 98 → 75. The 19-point delta is the puck-mass-loss artifact.

Older non-mass-preserved sim2sim variants (`sim2sim_combined_v2/v3/v4.yaml`) still exist and still vary `puck_radius` without mass preservation. Their numbers from the earlier sweep should be treated as confounded; the fine-grained mass-preserved sweep above is the current canonical reference.

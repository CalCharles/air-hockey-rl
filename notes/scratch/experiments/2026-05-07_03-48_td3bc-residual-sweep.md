# TD3+BC sweep on residual fine-tune (post-Polyak-fix collapse rescue)

- **Date**: 2026-05-07 03:48 UTC start
- **Status**: in-flight
- **Supersedes**: nothing — this is the first follow-up to
  [`2026-05-06_18-29_post-polyak-fix-rerun.md`](2026-05-06_18-29_post-polyak-fix-rerun.md)
  per its "If 1M is unsatisfactory" protocol.
- **Run dirs**: `runs/td3/sim2sim/post_polyak_fix_1M/fix_td3bc_lam{05,1,2}/seed0/`
- **Configs**: `configs/td3/sim2sim/paddle50/post_polyak_fix/fix_td3bc_lam{05,1,2}.yaml`
- **Launcher**: `bash scripts/smooth_policy/run_post_polyak_fix.sh <gpu_id> _bc`

## Question

The 2026-05-06 post-Polyak-fix 1M sweep showed **all 5 cells collapse to ~30–45 by
700k**, well below the paddle50 zero-shot baseline (~67.5). The Polyak fix exposed
the residual-drift mechanism that had been mitigated by frozen targets in the v25–v30
era. None of the cells satisfied any of the 4 acceptance criteria from the parent
doc (above-zs floor, no cliff, reasonable band width, mean-above-zs).

**TD3+BC** (Fujimoto & Gu 2021) adds a behavior-cloning penalty
`λ‖π(s) − π_source(s)‖²` to the actor loss, which directly counteracts the failure
mode (residual head exploits Q overestimation and drifts). In residual mode (where
π = π_source + residual) this reduces to `λ‖residual‖²`, which is **already
implemented** as the `residual_action_l2` knob at `td3_training.py:2069`.

In the broken-target era, λ=1.0 (v22) and λ=10.0 (v23) were tested and rejected
("kills peak", "too aggressive"). That was confounded: with frozen targets the Q
gradient was noise, so any anchor-to-zero penalty would dominate. **Re-testing
post-fix is a natural priority** because the failure modes are genuinely opposite
(too-stable in v22/v23, too-drifting now).

## Setup

Sweep λ ∈ {0.5, 1.0, 2.0} on the canonical `fix_v27_baseline_1M` config (N=5, q=1,
no exploration, 1M steps, residual_scale=0.15). Only `residual_action_l2` differs
between cells. 1 seed each.

| Run | λ | GPU | Base config |
|---|---:|---:|---|
| `fix_td3bc_lam05` | 0.5 | 0 | v27 baseline (N=5, q=1, no expl) |
| `fix_td3bc_lam1`  | 1.0 | 3 | same |
| `fix_td3bc_lam2`  | 2.0 | 1 | same |

GPU 2 stays untouched (running `fix_v30_lite_1M` from the parent sweep).

### Why λ ∈ {0.5, 1.0, 2.0}

- TD3+BC paper recommends λ ≈ 2.5 / mean(|Q|) for offline RL where Q is unnormalized.
- Our `actor_objective` already includes `(1−γ) ≈ 0.015` normalization (with γ=0.985),
  so the policy-gradient term has magnitude ~0.015 × Q ≈ 1.5 at peak Q ≈ 100.
- BC term magnitude is `λ × ‖residual‖²` ≤ `λ × residual_scale² = λ × 0.0225`.
- λ=1 gives BC ≈ 0.02 vs PG ≈ 1.5 (BC ≈ 1.5% of PG) — too weak in absolute terms,
  but the BC gradient is stronger because it's linear in residual while PG diffuses.
- λ ∈ {0.5, 1, 2} is a multiplicative sweep that should reveal the right scale.
  If the entire range over-anchors (kills peak), drop to {0.05, 0.1, 0.2} next.
  If under-anchors (still drifts), bump to {5, 10, 20}.

## Acceptance criteria (carried over from parent)

A cell is **satisfactory** if all hold:

1. Back-half (500k–1M) hold-band lower edge ≥ zs + 10 ≈ 77.5
2. No cliff in the back half
3. Band width ≤ 30 points
4. Last-100k mean ≥ zs + 10 ≈ 77.5

Coupled goal: BC should reduce drift WITHOUT killing the peak. Watch for both
"BC too weak: still drifts" (BC term doesn't bite) and "BC too strong: stays at
zero-shot forever" (residual head pinned to zero, no improvement).

## Results

| Run | λ | Peak | 0-200k mean[band] | 500-700k mean[band] | 900k-1M mean[band] | Cliff? | Above zs? | Verdict |
|---|---:|---:|---|---|---|---|---|---|
| `fix_v27_baseline_1M` (control, no BC) | 0 | 110 @ 131k | 68 [51,90] | 46 [34,62] | 37 [30,46] | yes | no | drift / fails |
| `fix_td3bc_lam05` | 0.5 | 114 @ 708k | 62 [47,78] | 63 [49,81] | 60 [49,73] | **no** | **at-zs, not above** | drift fixed; not improved |
| `fix_td3bc_lam1`  | 1.0 | 109 @ 623k (still running, 835k of 1M at last check) | 62 [46,80] | 64 [48,82] | (TBD) | no so far | similar | (in flight, same pattern) |
| `fix_td3bc_lam2`  | 2.0 | 112 @ 469k (still running, 720k of 1M at last check) | 63 [48,80] | 63 [49,78] | (TBD) | no so far | similar | (in flight, same pattern) |

### Diagnosis (after lam05 landed; same pattern in mid-flight lam1/lam2)

The hold band's mean (60-63) sits ~5 points *below* deterministic zs (67.5). This is
exactly consistent with **the source policy under ε=0.05 training-rollout noise**:
deterministic zs minus a small Gaussian-noise dilution. In other words, BC at λ=0.5
through 2.0 has **pinned the residual head to zero** and the policy is the source
policy, holding flat for 1M.

This is a **partial win**:
- ✅ Drift is fixed: tight band, no cliff, flat hold across 0-1M.
- ✅ The peak (110+) shows the system *can* find above-zs behavior episodically.
- ❌ The mean isn't improved over zs — the BC penalty is too strong, residual can't learn.

That all three λ are within noise of each other says we're in the "BC dominates,
residual ≈ 0" regime. To find improvement, the residual needs more room.

### Next iteration: lower-λ sweep

Drop λ to {0.05, 0.1, 0.2} — give the residual head 5–40× more freedom. The hypothesis:
there's a sweet spot where BC is still strong enough to prevent the §8.13 drift
mechanism but weak enough to let the residual head learn target-sim-specific
corrections. If this range *also* pins the residual at zero, fall back to CQL or LN
critic per the parent doc's menu.

If the lower-λ range *drifts* (collapses like the no-BC control), then BC is wholly
binary in this regime — either pin or drift — and we need an orthogonal mechanism
(CQL or LN critic) anyway.

## Conclusion

(Pending lam1 / lam2 final.) Provisional read: BC at λ ∈ {0.5, 1.0, 2.0} pins the
residual to zero, achieving stability but not improvement. Next iteration: lower-λ
sweep {0.05, 0.1, 0.2} on freed GPUs as lam1/lam2 finish.

## Conclusion

(Pending.) Decision tree once results land:

- **If a λ produces a cell that meets all 4 criteria**: declare TD3+BC the post-fix
  recipe. Update `notes/docs/training/residual-rl-recipe.md` (big-gap section) and
  the `project_residual_drift_fix_in_flight.md` memory entry to point at TD3+BC
  with the winning λ. Demote v27 ensemble5 below this.
- **If all λ kill the peak (anchor too strong everywhere)**: sweep down to
  {0.05, 0.1, 0.2} in a follow-up file.
- **If all λ still drift (anchor too weak everywhere)**: try the next known method
  (CQL or layer-norm critic per the parent doc's menu).
- **If results are mixed** (some cells partial wins, some collapse): write up,
  pick the most promising direction for the next sweep.

## Next

- Wait for the 3 BC runs (~5h each, all 4 GPUs busy until ~9h from now when
  v30_lite finishes — though v30_lite is in the orig sweep, not this one).
- Look at trajectories holistically per the 4 criteria.
- Iterate per the decision tree.

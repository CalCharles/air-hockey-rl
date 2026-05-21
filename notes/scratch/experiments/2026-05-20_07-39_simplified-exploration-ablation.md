# Can a 2-primitive exploration subset match the canonical 4-primitive DR recipe?

- **Date**: 2026-05-20 07:39 UTC start (runs launched ~2026-05-19 evening, finished 2026-05-20 morning)
- **Status**: done
- **Run dirs**: `runs/td3/zeroshot_paramrand/expl_compare/{baseline_seed0,baseline_seed1,simple_seed0,simple_seed1}`
- **Configs**: baseline = `configs/td3/zeroshot_paramrand/td3_paramrand_pm25.yaml`; simplified = `configs/td3/zeroshot_paramrand/td3_paramrand_pm25_simple_expl.yaml`
- **Analysis script**: `scripts/analysis/analyze_expl_compare.py` → `runs/td3/zeroshot_paramrand/expl_compare/expl_compare_trajectory.png`

## Question

The TD3 exploration-primitive machinery has 4 primitives (stand_still, same_direction,
y_aligned, target_position_directional) and takes a lot of code/config surface. Is a small
subset — **stand_still + same_direction (random unit direction × uniform magnitude, held 5
steps)** — enough to match the canonical recipe on the zero-shot ±25% physics-DR task?

## Setup

- Trainer: `scripts/td3/td3_training_dr.py`, env `configs/new_juggle/zeroshot_ablations/sim_paramrand_pm25.yaml`, 2M steps.
- 2 arms × 2 seeds = 4 runs, one per GPU, ~12h wall clock.
- **Only variable changed**: primitive weights. Simplified drops y_aligned + target_position
  to weight 0 and bumps same_direction 1.0→3.0, so total weight (3.2) and stand_still:others
  ratio (0.2:3.0 = 1:15 → 6.25% / 93.75%) are preserved. Trigger-chance schedule
  (0.15→0.05 over 200k), takeover_steps=5, q-updates, PER, network all identical.
- Eval: per-checkpoint (25k interval, 79 ckpts/run) `multi_env_eval.json` = 5 fixed
  seed-sampled dynamics overlays × 4 episodes. **eval_envs.json md5-identical across all 4
  runs** → apples-to-apples. Mid-run sanity confirmed target_position applied 0% on simple
  arms vs ~12% on baseline.

## Results

Headline: **the 2-primitive subset matches the 4-primitive canonical — no regression on any
of the 5 eval envs; if anything it trends slightly higher, but within seed noise.**

Back-half (1.5M–2M, ~20 ckpts/run) mean_return:

| run | back-half mean ± sd | back-half succ | peak (step) | final ckpt |
|---|---|---|---|---|
| baseline_seed0 | 66.9 ± 12 | 0.78 | 92.2 (1.55M) | 55.1 |
| baseline_seed1 | 80.1 ± 12 | 0.87 | 110.5 (1.98M) | 110.5 |
| simple_seed0 | 89.7 ± 13 | 0.88 | 112.6 (1.23M) | 109.2 |
| simple_seed1 | 79.9 ± 10 | 0.84 | 101.0 (0.85M) | 89.8 |

Arm-level back-half mean (n=2 seeds):
- **baseline = 73.5** (seeds 66.9, 80.1; SE 6.6)
- **simple   = 84.8** (seeds 89.7, 79.9; SE 4.9)
- **simple − baseline = +11.3, pooled SE ≈ 8.2** → ~1.4 SE, **not significant** at n=2.

Per-env back-half mean (averaged over 2 seeds) — simplified ≥ baseline on **every** env:

| arm | env0 | env1 | env2 | env3 | env4 |
|---|---|---|---|---|---|
| baseline | 78.7 | 91.1 | 60.6 | 67.0 | 70.0 |
| simple | 87.4 | 98.0 | 79.8 | 80.3 | 78.6 |
| diff | +8.7 | +6.9 | +19.2 | +13.3 | +8.6 |

## Trajectory shape (don't reduce to one number)

Both arms are noisy across the whole run (typical for this DR task) and ramp similarly through
~0.5M. In the back-half the simple seed-mean sits at or slightly above baseline; success-rate
curves essentially overlap. Note baseline_seed0 peaks at 1.55M (92.2) then sags to 55.1 final —
a single-checkpoint final number would have understated it, hence the back-half mean. simple_seed0
peaks early (1.23M) and holds ~109. No collapse in either arm.

## Conclusion

For this zero-shot ±25% physics-DR task, **stand_still + same_direction is sufficient** — the
two dropped primitives (y_aligned, target_position_directional) contribute nothing measurable on
top, and removing them did not hurt any eval env. Caveat: n=2 seeds, so the +11.3 edge is within
noise; the honest claim is "matches, no regression," not "better."

**Not yet acted on** (per user): exploration code left fully in place. Pending user decision on
whether to (a) make the 2-primitive set the canonical config and prune y_aligned /
target_position_directional from `exploration_selector.py` + `exploration_primitives.py` + the
arg knobs in `td3_training.py`, or (b) add more seeds first to firm up the comparison.

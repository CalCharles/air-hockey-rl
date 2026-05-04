# Residual RL — adaptation-phase exploration plan

Date: 2026-05-01
Branch: `online_learning`

User hypothesis: residual head plateaus / decays past a point on paddle50 because
the rollout distribution stays narrow around the frozen base trajectory. The
recipe that *trained the base policy* (`td3_recommended.yaml`) used non-trivial
primitive exploration + 2× higher Gaussian noise — neither is present in the
current residual or "from-scratch" paddle50 configs.

## Diagnosis: what's missing

| Knob | `td3_recommended` (made base) | v27 residual / `td3_from_scratch_1M_paddle50` | gap |
|---|---|---|---|
| `exploration_noise` | **0.1** | 0.05 | ½× |
| `exploration_primitive_chance` (steady) | **0.05** | 0 | off |
| `exploration_primitive_chance_start` | **0.15** | 0 | off |
| `exploration_primitive_chance_anneal_steps` | **200000** | 50000 | tighter |
| `weight_stand_still` | **0.2** | 0 | off |
| `weight_same_direction` | **1.0** | 1.0 (chance=0 → unused) | off |
| `weight_y_aligned` | **1.0** | 0 | off |
| `weight_target_position_directional` | **1.0** | 0 | off |
| `q_updates` | **25** (a:q≈0.24, N*=31) | 1 (residual) / 4 (from_scratch_1M) | far less |

In `td3_training.py`, `primitive_selector.apply()` (line ≈1427) overrides
`action_tensor` with the primitive's raw [-1, 1] action for the takeover window.
In residual mode, `actor` is `ResidualActor`, so `action_tensor` is already
base + residual_scale·residual. Replacing it with a primitive means the env
sees the primitive directly — base **and** residual are bypassed for those
steps. That broadens the data distribution the residual head learns from.
This is exactly the override semantics used during base-policy training.

Code path runs unconditionally for non-eval rollouts. **No code changes
required** — the residual configs simply zero the knobs. Flipping YAML is
enough.

## Plan

### Part A — true from-scratch on paddle50 with original-recipe exploration

Goal: a fair "what does proper from-scratch training look like *on this env*"
reference. The previous `td3_from_scratch_1M_paddle50.yaml` was strictly
compute-matched to full_ft (q_updates=4, primitives=0) and plateaued at ~21,
which doesn't tell us whether the env is learnable from scratch.

Create `…/paddle50/td3_from_scratch_1M_paddle50_full_explore.yaml`:
- Clone `td3_recommended.yaml` knobs: q_updates=25, actor_updates_per_iteration=6,
  exploration_noise=0.1, full primitive set with chance 0.15→0.05 over 200k,
  learning_starts=20000, buffer_size=100000, batch_size=512, q_weight_decay=1e-4
- Override env `config:` → `sim2sim_combined.yaml` (paddle50 target)
- 1M timesteps, checkpoint_interval=20000 (50 ckpts)
- `model_path: null`, `full_checkpoint_load: fine_tune` (ignored when null)

Estimated 16+ hours under contention.

### Part B — residual RL with adaptation-phase exploration (v30 family)

Three variants, 300k each, 2 seeds each (run sequentially after Part A).
All keep v27's winning knobs: `num_critics=5` Maxmin, `q_updates=1`,
`success_top_fraction=0.15`, `priority_age_decay=1e-4`, `residual_scale=0.15`,
`buffer_size=20000`.

1. **v30_explore_full**: full original primitive set, chance 0.15→0.05 over 50k
   (residual is shorter than 1M base training); weights
   stand_still=0.2, same_direction=1.0, y_aligned=1.0, target_position_directional=1.0;
   exploration_noise=0.1.
2. **v30_explore_lite**: chance 0.10→0.03 over 50k (base already produces
   sensible behavior so don't over-disturb); same primitive weights as v30_explore_full;
   exploration_noise=0.1.
3. **v30_explore_directional_only**: only `target_position_directional=1.0` and
   `same_direction=1.0` (drop stand_still + y_aligned, since base already
   handles vertical alignment); chance 0.10→0.03; exploration_noise=0.1.

Eval: `eval_all_ckpts_residual.sh` against `sim2sim_combined.yaml`
(zero-shot 67.54), n=50, seed=0, deterministic.

## Decisions / open questions

1. **Order**: Part A first (only 1 GPU realistically free → can't run both at
   once). Part B variants queued behind it.
2. **q_updates=25 for Part A** — actual "original args", not the
   compute-matched 4 from before.
3. **Primitive override semantics in residual** — exact match to original
   training (replace combined action with primitive). Alternative ("primitive
   only replaces the residual portion, base still acts under primitive
   guidance") requires code changes; **not pursued** unless v30 family
   underperforms.

## What's NOT in scope here

- No new code path; reusing existing `primitive_selector.apply()`.
- No changes to `async_td3_real.py` / real-world rollout.
- No changes to v27's other ablation knobs.

## Reference

- Recipe doc: `notes/docs/training/residual-rl-recipe.md`
- Earlier exploration ablations: `notes/docs/training/td3-exploration-ablations.md`
- Paddle50 log (current results, v1–v29): `notes/scratch/residual_rl_paddle50_log.md`
- Recommended base config: `scripts/smooth_policy/amp_history/configs/td3/td3_recommended.yaml`
- v27 winning residual: `…/td3/sim2sim/paddle50/td3_residual_v27_ensemble5.yaml`
- Old (non-explore) from-scratch: `…/paddle50/td3_from_scratch_1M_paddle50.yaml`

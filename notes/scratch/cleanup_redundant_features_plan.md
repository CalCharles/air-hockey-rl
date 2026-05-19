# Cleanup Plan — Remove Redundant Features

Branch: `cleanup-redundant-features` (off `main`, 2026-05-19)
Goal: produce the **simplest codebase** that still runs every active workflow without behavior change.

---

## Active workflows we must not break

Every cleanup item below is judged against this exact list. Anything not referenced by these stays untouched.

| # | Workflow | Entry | Canonical config(s) |
|---|----------|-------|---------------------|
| 1 | Sim training (canonical source policy, no DR) | `scripts/td3/td3_training.py` | `configs/td3/td3_recommended_top50_hist2.yaml` |
| 2 | Sim source policy with env-param randomization | `scripts/td3/td3_training_dr.py` | `configs/td3/zeroshot_paramrand/td3_paramrand_pm25.yaml` |
| 3 | Sim2sim residual (small gap) | `td3_training.py` | `configs/td3/sim2sim/td3_sim2sim_residual.yaml` |
| 4 | Sim2sim residual (big gap, warp075 family) | `td3_training.py` | `configs/td3/sim2sim/warp075_p30_residual/{phaseC_actor2_1M, phaseD_actor2_p10_1M, phaseD_actor4_w10_1M}.yaml` |
| 5 | Real-world async training (canonical CQL recipe) | `scripts/td3/extras/async_td3_real.py` + `helper/real_td3_runtime.py` | `configs/td3_real_world/td3_residual_cql.yaml` |
| 6 | Real-world v27 baseline (no CQL, regression keep) | same | `configs/td3_real_world/td3_residual.yaml` |
| 7 | Real-world fixed-policy eval | `scripts/td3/extras/async_td3_real_eval.py` | — |
| 8 | Real-world teleop user-study eval | `scripts/td3/extras/async_td3_real_teleop_eval.py` | — |
| 9 | GAT sim2sim research | `scripts/td3/td3_training_gat.py`, `gat_trainer.py` | `configs/td3/gat/` |
| 10 | Box2D env (all registered tasks; only `puck_juggle_upper_half_reward` has a current config) | `airhockey/` | `configs/new_juggle/sysid_best_params_hist2.yaml` |

**Obs type kept:** `history` (30-dim) is the only one used by the workflows above, but the others are kept since the registered non-juggle tasks (and `airhockey/utils.py`) reference them — see §3.

**Replay kept:** PER + uniform `TD3ReplayBuffer` for sim, `SharedTD3Replay` for real-world. Verified: every active sim/sim2sim config has `per_enabled: true`.

---

## §1 — Motion rewards + dual-head TD3 critic (BIG)

**Why kill:** user always sets motion rewards to 0; multi-objective RL is fragile; head is dead weight. Removing it is the single biggest win for codebase simplicity.

### 1A. Files to delete outright
- `scripts/td3/helper/dual_head_q.py` — `TD3DualHeadQNetwork` class
- `scripts/td3/helper/real_motion_rewards.py` — entire file (used only by `td3_training.py:1612` and `real_policy_runner.py:565`)
- `scripts/td3/helper/motion_magnitudes.py` — entire file

### 1B. Files to surgically edit

**`scripts/td3/td3_training.py`** (~2516 LOC, large file)
- Remove import of `TD3DualHeadQNetwork`; replace with a single-head critic. Build new minimal `TD3QNetwork` (single trunk + scalar head) and adopt it everywhere a critic is constructed.
- Drop `motion_reward_weight` (line ~527) and all per-component motion weights: `stand_still_reward_weight`, `temporal_alignment_reward_weight`, `axis_alignment_reward_weight`, `velocity_reward_weight`, `jerk_reward_weight`, `stand_still_threshold`, `velocity_at_*`, `jerk_at_*` (~10 fields).
- Drop `_compute_motion_reward_components()` callsite (line ~1612).
- Critic loss (lines 1847-1899): collapse `min_next_task`/`min_next_motion` to a single `min_next_q`; drop `bellman_target_motion_original`/`next_q_motion_h`; remove the second MSE accumulation.
- Actor loss (lines 2103-2117): `actor_objective = q1` (drop `+ motion_reward_weight * q1_motion`). Also drop `task_reward_weight` if it was only there to weight the sum; if it's still scaling the single critic, keep it.
- Drop TensorBoard scalars: `losses/q_motion_loss`, `losses/q1_motion_mean`, `losses/actor_norm_motion_mean`, `debug/bellman_target_motion_original_mean`, `debug/next_q_motion_h_mean`, `rewards/sampled_motion_reward_mean`, all `rewards/{temporal_valid_fraction, temporal_alignment_reward_*, axis_alignment_reward_*, velocity_reward_*, jerk_reward_*, stand_still_reward_*}` (lines ~1625-1640, 1885, 1889, 1991, 1994, 2015, 2137, 2499, 2503).
- Drop `motion/avg_velocity_magnitude`, `motion/avg_acceleration_magnitude` logging.
- Drop tracking lists `velocity_magnitudes`, `acceleration_magnitudes`.

**`scripts/td3/helper/real_td3_runtime.py`** (large)
- Drop `motion_reward_weight` (line ~725) and import of motion reward helpers.
- Drop motion branch in critic / actor loss (lines ~1956, 2117) — mirror the td3_training.py edits.
- Drop `rolling50_motion_reward_*` (line ~919).

**`scripts/td3/helper/real_policy_runner.py`**
- Drop call to `_compute_motion_reward_components()` (line ~565) and the resulting `motion_reward_total` channel.
- Drop `episode_motion_reward` (line ~728).

**Replay buffers** — all three:
- `scripts/td3/helper/replay_buffer.py` — drop `self.motion_rewards`, `motion_rewards` arg to `.add()`, `motion_rewards` key in `.sample()` output.
- `scripts/td3/helper/shared_replay.py` — same drop.
- `scripts/td3/helper/prioritized_replay_buffer.py` — same drop.
- Audit every `.add(...)` and `.sample(...)` callsite; the sampler in `td3_replay_sampling.py` likely needs an edit too.

**`scripts/td3/helper/td3_checkpointing.py`**
- Drop serialize/deserialize of `velocity_magnitudes`, `acceleration_magnitudes`.
- Confirm old checkpoints still load (backward-compat: tolerate missing keys, do not require them).

**`scripts/td3/helper/real_collector_metrics.py`** + `real_eval_stats.py`
- Drop any motion-reward rolling stat keys.

**`airhockey/airhockey_base.py`**
- Drop reward fields: `diagonal_motion_rew` (line 193), `stand_still_rew` (line 194), and the per-step accumulation (lines 832-834 — `vel_mags`, `acc_mags`, `jerk_mags`).
- Drop `episode_motion_data` dict (lines 436, 469, 902).
- Drop `velocity_penalty_coeff` (lines 92, 198, 775-777) and `jerk_penalty_coeff` (line 197) — both default to 0 in all canonical configs.

### 1C. Config cleanup
Every config that sets `motion_reward_weight: 0.025` or similar — delete the key. Most likely candidates: `configs/td3_real_world/*.yaml`, `configs/td3/sim2sim/**/*.yaml`. Audit with `grep -r motion_reward_weight configs/`.

### 1D. Risk
Medium. Many call sites, but the changes are mechanical and the user has explicitly confirmed motion reward weight is always 0. Verification plan in §10.

---

## §2 — Exploration primitives (MEDIUM)

**Status today:** 6 primitives defined; weights configured per recipe.

| Prim | Name | Active in canonical configs? |
|------|------|------------------------------|
| 0 | stand_still | Source-policy only (0.2 in `td3_recommended_top50_hist2.yaml` + `td3_paramrand_pm25.yaml`); 0 in residual & real-world |
| 1 | same_direction | **All five** active configs at weight 1.0 — keep |
| 2 | y_aligned | Source-policy only (1.0); 0 in residual & real-world |
| 3 | policy_takeover | **Never** — delete |
| 4 | target_position_directional | Source-policy only (1.0); 0 in residual & real-world |
| 5 | pre_contact_hit_variant | **Never** — delete |

### 2A. Files
- `scripts/td3/helper/exploration_primitives.py` (~6.5 KB) — drop prim-3 and prim-5 helpers; keep `stand_still_actions`, `sample_directions_from_angle_range`, `sample_uniform_magnitude`, `sample_target_distances`, `project_displacement_to_action_box`.
- `scripts/td3/helper/exploration_selector.py` (~39 KB) — significant simplification. Drop `policy_takeover` branch entirely (config knob `exploration_policy_takeover_enabled`, weight `exploration_primitive_weight_policy_takeover`, and policy-loading code path). Drop `pre_contact_hit_variant` branch (knob `exploration_pre_contact_hit_variant_chance` and all `exploration_pre_contact_hit_variant_*` thresholds).
- Reduce default `weights` list from 6→4 entries (or 6→3 if user wants to nuke prims 0/2/4 — see §2C).

### 2B. Config cleanup
Drop dead config keys across all canonical configs and `configs/td3*/`:
- `exploration_policy_takeover_enabled`, `exploration_primitive_weight_policy_takeover`, related policy-load paths
- `exploration_pre_contact_hit_variant_chance`, `exploration_pre_contact_hit_variant_*`

### 2C. Open question for review (not auto-executed)
The user said keep "a simple subset." Options:
- **Conservative**: drop only prims 3 and 5 (always-zero). 4 primitives remain.
- **Aggressive**: also drop prims 0, 2, 4 if residual/real-world is the primary developer focus. Source-policy training would have to switch to plain Gaussian noise + same_direction only.

→ **Recommend conservative.** Drop only the two truly-dead primitives. Saves ~30% of `exploration_selector.py` complexity (policy-takeover branch is the heaviest) without losing source-policy capability.

### 2D. Risk
Low. The deleted primitives have weight 0 in every config, so behavior is unchanged.

---

## §3 — Velocity / acceleration magnitude features (SMALL)

These turned out to be **logging-only**, not observation features or reward inputs. The cleanup is bundled into §1 because the data path is identical (env writes `episode_motion_data`, training script reads it for TB scalars).

After §1 there is nothing left in this category — confirmed not part of the obs (active obs is `history`, 30-dim; velocity is implicit `obs[27:29]−obs[15:17]`).

---

## §4 — Replay buffer variants (NO CHANGES)

Investigation conclusion: all three buffer classes are load-bearing.

- `TD3ReplayBuffer` — used when `per_enabled: false` (rare in active configs but kept as fallback).
- `TD3PrioritizedReplayBuffer` — used by all 7 active sim/sim2sim configs (`per_enabled: true` in every one). PER is canonical, not optional.
- `SharedTD3Replay` — required by real-world async (multiprocess-safe, lock-protected, CPU-only).

**Decision: keep all three.** Only change is dropping the `motion_rewards` channel from each (already in §1).

---

## §5 — Dead observation types

`airhockey/utils.py` lines 53-200 define 15 obs types; canonical workflows use only `history`. However, **user said keep all the tasks**, and several tasks depend on other obs types:

- `paddle` → `paddle_reach_position`
- `negative_regions_paddle` → `paddle_reach_position_negative_regions`
- `single_block_*` / `many_blocks_*` → block tasks
- `multipuck_*` → most recent commit "adding back multipuck"
- `paddle_acceleration_*` → if acceleration task is used

**Decision: keep all obs types.** This preserves the option to use any registered task. Revisit only if user reverses the "keep all tasks" call.

If the user later changes their mind, the deletable subset (obs types referenced by no remaining task) would be: `vel`, `pos`, `negative_regions_puck_vel`, `negative_regions_puck_history`. The block / multipuck / acceleration variants are tied to registered tasks.

---

## §6 — Dead reward fields in `airhockey_base.py`

These default to 0 and are never set by any canonical config, but some are used by other registered tasks:

| Field | Used by | Cleanup decision |
|-------|---------|------------------|
| `wall_bumping_rew` | Pinball / strike_crowd tasks? | **Keep** (other tasks) |
| `direction_change_rew` | unknown | Audit task files; if no task sets it, drop |
| `horizontal_vel_rew` | unknown | Audit; if no task sets it, drop |
| `truncate_rew` | termination shaping | Audit; if always zero in registered tasks, drop |
| `enable_survival_bonus`, `survival_bonus_per_step` | unknown | Audit |
| `dense_goal`, `goal_selector`, `puck_goal_success_bonus`, `paddle_puck_success_bonus` | Goal-position tasks | **Keep** |
| `num_positive_reward_regions`, `num_negative_reward_regions`, region rectangle params | Negative-regions tasks | **Keep** |
| `velocity_penalty_coeff` | none (always 0) | **Drop in §1** |
| `jerk_penalty_coeff` | none (always 0) | **Drop in §1** |

**Action:** before deleting any of the "unknown" rows, audit `airhockey/airhockey_tasks/*.py` to see which tasks set them in their `task_config`. Anything no task sets is safe to delete. Estimate ~3-5 deletable fields after audit.

---

## §7 — Side workflows / scripts

Per user's call:
- **Reset-policy training (DELETE):** `scripts/td3/extras/async_td3_real_reset_policy.py` (~28 KB). Audit imports — confirm no active script imports it. Also delete any `scripts/real/rollout_reset_policy_*.py` files that exist solely to feed reset-policy training (the rollout scripts themselves are kept per "keep real-robot scripts").
- **v27 baseline (KEEP):** `configs/td3_real_world/td3_residual.yaml`. Already noted in CLAUDE.md.
- **All tasks (KEEP):** no deletions in `airhockey/airhockey_tasks/`, no unregistration in `airhockey/__init__.py`, no deletions of `configs/baseline_box2d/`.
- **Real-robot scripts (KEEP):** all of `scripts/real/`, `scripts/visualization/`, `scripts/analysis/`, `scripts/td3/run_density_sweep.sh`.

### 7A. Reset-policy delete checklist
- Delete `scripts/td3/extras/async_td3_real_reset_policy.py`.
- Grep for any imports of it (`grep -r async_td3_real_reset_policy scripts/ configs/`). Drop dead references.
- Check `notes/scratch/reset_policy_redesign.md` — mark as historical (don't delete; this is research history).
- Confirm `scripts/real/rollout_reset_policy_real.py` and `scripts/real/rollout_reset_policy_hybrid.py` are the *real-robot rollout* scripts (user said keep these), not reset-policy *training*. Inspect first.

---

## §8 — Documentation updates

After code edits land:
- `notes/docs/training/architecture.md` — remove the dual-head critic section.
- `notes/docs/training/td3-algorithm.md` — drop motion-reward branch from algorithm description.
- `notes/docs/training/network-architecture.md` — single-head critic only.
- `notes/docs/training/reward-shaping.md` — drop motion-reward components, jerk/velocity penalty.
- `notes/docs/training/monitoring.md` — drop the ~10 deleted TB scalars from the reference table.
- `notes/docs/exploration/td3-primitives.md` — drop prim 3, prim 5.
- `notes/docs/training/replay-and-episodes.md` — drop `motion_rewards` from the buffer schema; note PER is canonical.
- `CLAUDE.md` — no edits needed (it already reflects current state; the changes don't invalidate it).

---

## §9 — Things explicitly NOT touched

For the record (and so future Claude doesn't undo this):
- GAT code (`td3_training_gat.py`, `gat_trainer.py`, `configs/td3/gat/`) — active research.
- All registered Box2D tasks (`airhockey/airhockey_tasks/*.py` and `airhockey/__init__.py`).
- All `configs/baseline_box2d/` configs.
- v27 baseline real-world config.
- All real-robot scripts (`scripts/real/`).
- Visualization and analysis scripts.
- PER replay (in use by every active sim/sim2sim recipe).
- Multipuck and other obs types (preserved for non-juggle tasks).
- `notes/scratch/` history files — frozen, do not edit.

---

## §10 — Execution sequence (when authorized)

Suggested order, one commit per step, all on `cleanup-redundant-features`:

1. **Drop reset-policy trainer** (`§7A`) — smallest, isolated. Verifies branch hygiene.
2. **Drop dead exploration primitives** (`§2`, conservative variant) — clear win, no behavior change.
3. **Drop motion rewards + dual head — env side** (`§1` part `airhockey_base.py` only). Run sim training smoke test (a few episodes); verify reward trajectory unchanged.
4. **Drop motion rewards — replay buffers** (`§1` replay edits + all callsites). Heaviest mechanical change but well-contained.
5. **Drop motion rewards — sim trainer** (`§1` `td3_training.py`). Smoke test: short sim run + load an existing checkpoint with backward-compat.
6. **Drop motion rewards — real-world trainer** (`§1` `real_td3_runtime.py`, `real_policy_runner.py`). Cannot live-test on hardware from here — confirm checkpoint resume works on dry-run.
7. **Drop config keys** (`§1` config sweep + `§2` exploration keys). Lint-check every active config still loads.
8. **Audit + drop unused reward fields** (`§6`).
9. **Documentation refresh** (`§8`).

### Verification checklist per step
- `python scripts/td3/td3_training.py --config configs/td3/td3_recommended_top50_hist2.yaml` runs for ≥50 episodes with no exceptions and producing the same per-episode return on a fixed seed.
- `python scripts/td3/td3_training.py --config configs/td3/sim2sim/td3_sim2sim_residual.yaml` resumes from canonical `latest_models/canonical/hist2_motion0_v2/` checkpoint without error.
- `python scripts/td3/td3_training_dr.py --config configs/td3/zeroshot_paramrand/td3_paramrand_pm25.yaml` runs for ≥50 episodes.
- `python -c "from scripts.td3.extras import async_td3_real"` imports clean.
- `python -c "from scripts.td3.extras import async_td3_real_eval, async_td3_real_teleop_eval"` imports clean.
- All deleted-key configs `yaml.safe_load()` without error.
- `pytest scripts/td3/tests/` (if tests exist) still passes.

### Backward compat
- Old checkpoints written with the dual-head critic: provide a one-time conversion (drop the `motion_head` weights, rename `task_head` → `head`), or document the breaking change. **Recommend:** drop conversion shim, document the cut. The pre-cleanup canonical source policy lives in `latest_models/canonical/hist2_motion0_v2/` and may need re-export if anyone wants to resume it.

---

## §11 — Estimated impact

| Category | LOC removed (rough) | Risk |
|----------|---------------------|------|
| Motion rewards + dual head | 600-900 | Medium (many callsites) |
| Exploration prims 3 & 5 | 200-400 | Low |
| Reset-policy trainer | ~700 (one file) | Low (isolated) |
| Vel/accel logging | (bundled in §1) | — |
| Dead reward fields | 30-80 | Low (after audit) |
| Config key cleanup | ~50 | Low |
| Doc updates | — | None |

**Total: ~1500-2000 LOC removed**, no behavior change on any active workflow.

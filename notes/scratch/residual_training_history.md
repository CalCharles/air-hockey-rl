# Residual fine-tuning on top of the actor: what has been tried

Compiled 2026-11-05 from the git history and the experiment notes. This is a pointer page; the linked notes are the
source of truth.

Every version freezes the trained source actor and trains a small new head that adds a bounded correction to its
action. A fresh critic is trained alongside it. There were two rounds.

## Round 1, April–May 2026: sim-to-sim, plus real-robot plumbing (Da Liu, Carl Qi)

This work is in the history of `main` and most other branches. It started on the old `smooth_policy` layout and was
moved to `scripts/td3/` on 2026-05-11.

| Commit | Date / author | What it added |
|---|---|---|
| `d4acdd9` | 04-25 Da Liu | First residual implementation (`scripts/smooth_policy/residual_agent.py`) and the small-gap config |
| `c126f86` | 04-25 Da Liu | More experiments; `notes/scratch/residual_rl_plan.md` |
| `1bdf361` | 04-27 Da Liu | "Recipe is now fixed": the drift fix (`success_top_fraction: 0.5`), the recipe doc, `residual_rl_drift_fix_log.md` |
| `9bc0d93` | 04-27 Carl Qi | Residual training for the async real-robot TD3 stack, with tests and `td3_residual.yaml` |
| `f34c0e9` | 04-30 Da Liu | Large paddle50 sweep (v10–v30 configs); `residual_rl_paddle50_log.md` |
| `c72d462`, `7f16992` | 05-08 Da Liu / Carl Qi | CQL experiments on sim-to-sim |
| `b9b2f02` | 05-08 Carl Qi | Real-robot residual training "mostly done" |
| `10bd9cf` | 05-11 Carl Qi | Real-robot residual + CQL config (`td3_residual_cql.yaml`); tip of `origin/old_main` |
| `964b33e`, `6af0175` | 05-11 Da Liu | Cleanup: code moved to `scripts/td3/`, the paddle50 configs deleted |

### What was tried and what happened

All runs are single seed. "Zero-shot" means the source policy run on the target with no fine-tuning; the numbers are
episode returns.

- **paddle50 target** (paddle −50 %):
  - Early wins turned out to be a bug. The 2026-05-06 Polyak bug (the target computation used the actor instead of the
    actor-target) made all the v25–v30 results unreliable.
  - After the fix, every 1M run collapsed to 30–45, against 67.5 zero-shot.
  - The target was later dropped because training from scratch can't solve it either (3.85M steps: peak 84, mean 47).
- **TD3+BC** (an L2 anchor to the base action) didn't rescue it.
- **CQL α=20 is what worked.**
  - On paddle50 it held the return near zero-shot.
  - On the redesigned `warp075_p30` target it plateaued at **95 vs 48 zero-shot** at 1M, with a peak of 155.
  - Stacking BC, exploration, or N=10 critics on top of CQL hurt.
- **The 2026-05-08 campaign** found that `actor_updates_per_iteration` = 2 was the knob that mattered (4 for the harder
  warp). That gave the final recipe:
  - residual_scale 0.15, CQL α=20, 5 critics, no BC, no exploration, 1M steps.
  - +47 to +68 over zero-shot on the mild and canonical targets.
  - It **fails** at warp 0.125.
- **Full-network fine-tuning** (`2026-05-10_00-55`) is the reverse: CQL is fatal there, while plain low-LR fine-tuning
  peaks early and then drifts.
- **Real robot:** the configs (`configs/td3_real_world/td3_residual.yaml`, `td3_residual_cql.yaml`), the async runtime
  branch and the tests exist. No experiment note with real-robot residual results was found, so it looks like it was
  built but never run or written up here.

### Write-ups

- Canonical summary: [`notes/docs/training/residual-rl-recipe.md`](../docs/training/residual-rl-recipe.md)
- Experiment notes:
  - [`2026-05-06_18-29_post-polyak-fix-rerun.md`](experiments/2026-05-06_18-29_post-polyak-fix-rerun.md)
  - [`2026-05-07_03-48_td3bc-residual-sweep.md`](experiments/2026-05-07_03-48_td3bc-residual-sweep.md)
  - [`2026-05-07_09-13_cql-residual-sweep.md`](experiments/2026-05-07_09-13_cql-residual-sweep.md)
  - [`2026-05-07_21-30_residual-on-warp075-p30.md`](experiments/2026-05-07_21-30_residual-on-warp075-p30.md)
  - [`2026-05-08_02-17_cql-campaign.md`](experiments/2026-05-08_02-17_cql-campaign.md)
  - [`2026-05-10_00-55_full-ft-cql-on-warp075.md`](experiments/2026-05-10_00-55_full-ft-cql-on-warp075.md) (the
    full fine-tuning contrast)
- Older read-only logs: `notes/scratch/residual_rl_{plan,drift_fix_log,paddle50_log,async_real_plan}.md`,
  `residual_exploration_plan.md`

## Round 2, October 2026: sustained juggling in sim (Max Rudolph)

- **Commit:** `4a64790` (2026-10-21), "Fix continuing-task training and add sustained juggling experiments".
- **Branches:** `codex/sustained-juggling`, `sustained_juggling_sim2real` and `world_model` (also on the remotes). It is
  **not on `main`**.
- **Note:**
  [`2026-10-21_19-00_sustained-juggle-residual-cql.md`](experiments/2026-10-21_19-00_sustained-juggle-residual-cql.md)
- **What it tried:** the round 1 recipe adapted (bounded 0.15, CQL α=20, 5 critics) with discounts 0.975, 0.99 and
  0.995. The goal was to avoid the early collapse seen when fine-tuning the whole actor.
- **What it found:**
  - At 100k steps the residual kept juggling better than full fine-tuning.
  - Results across test conditions were mixed: γ=0.975 was best in nominal and randomized conditions (345 / 438 vs
    243 / 357 for the source), but worse at 50 ms camera delay.
  - No episode reached the 2,000-step cap.
- **Status:** the note is still marked in-flight and no follow-up note was written. The run directory
  `runs/td3/sustained_20261021_residual/` isn't on this machine, so the final 500k results are missing.

## Not yet done

None of this residual work has been done on the current new-sim policies (`latest_models/ceiling/newsim_scratch_cap7000_swa`)
or on the real-plant sim (`configs/new_juggle/sysid_v2_hist4_realplant.yaml`).

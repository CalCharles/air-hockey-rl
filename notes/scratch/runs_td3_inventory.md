# `runs/td3/` inventory — what each experiment group is, and what's still live

Snapshot taken **2026-09-03**. 19 top-level groups, **337 real run directories**
(9,423 `args.yaml` files total; the rest are per-checkpoint copies), **248 GB**.
Newest run in the tree is 2026-05-20; nothing has trained since.

`runs/` is gitignored — this file is the only durable record of what's on disk.

Method: every `args.yaml` was parsed for `config` / `args_file` / `run_name` /
`total_timesteps` / `model_path`, cross-referenced against (a) which sim + TD3
configs still exist in `configs/`, (b) which run dirs are cited in `notes/` and
`latest_models/`, and (c) the two hard invalidation events below.

## Two invalidation events that define the tiers

| Date | Event | Invalidates |
|---|---|---|
| **2026-05-06** | Polyak-averaging bug fixed (Q-target was using the actor, not the actor-target) | every residual / full-FT number produced before this date |
| **2026-05-11** | Engineered randomization removed from `airhockey/sims/airhockey_box2d.py` (collision direction/strength, action-force attenuation, delay jitter, paddle-density fluctuation, occlusion bad-region) | every run whose recipe used those knobs — **not reproducible**, checkpoints only |

Verified: `enable_action_force_attenuation`, `paddle_puck_direction_randomization`,
`wall_direction`, `randomize_delay` return **zero hits** in `airhockey/`. Only
`random_occlusion_rate` survives.

Corroborating signal: `configs/` has already been pruned to the live set.
`pid_noise_constant_upper_half_custom_sim_params.yaml`, `sim_real_world_adaptation.yaml`,
`sysid_best_params_hist3/hist5.yaml`, `sim2sim_warp075_p00/p20.yaml`,
`sim2sim_warp125_p30.yaml`, and every `sim_no_*` / `sim_only_*` / `sim_all_sysid_no_rand*`
ablation config are **gone**. Runs pointing at a deleted config cannot be re-run.

---

# Tier A — live (keep)

~61 GB. Current canonical lineage, or promoted into `latest_models/`.

### `hist_motion_collision/` — 13 GB, 9 runs, 2026-04-20 → 05-05
History-length × motion-reward grid: `hist{2,3,4,5}_motion{0,0p01}`, 1M steps each on
`sysid_best_params_hist{N}.yaml`. Settled `hist_len=2`, `motion_reward_weight=0`.

`hist2_motion0_v2/seed0` (05-05, `td3_hist2_motion0_v2.yaml`, best ckpt 850k, eval mean
169.72) is the **sim-to-real ground-truth source policy** → promoted to
`latest_models/canonical/hist2_motion0_v2/`. `hist2_motion0` is its predecessor
(mean 148.08), kept for reproducibility. Both are the base policy for every sim2sim
residual campaign below, and are cited ~150× across `notes/`.

Caveat: per CLAUDE.md, `hist2_motion0_v2` is **historical only for new sim2real work** —
it predates the 05-11 deprecation. Keep it as the residual campaigns' base; retrain
for any new deployment. The hist3/hist5 cells are unreproducible (configs deleted).

### `zeroshot_ablations_700k/` — 8.5 GB, 16 runs, 2026-05-09/10
700k continuations of the 500k ablation set (11 `full_resume`d from `zeroshot_ablations/`,
`no_obs_delay_randomization` fresh, `all_sysid_no_rand_v2` fresh). 700k means span 88–122
vs 73–116 at 500k. **All 16 promoted into `latest_models/ablations/`** as the
CoRL-2026 deployment-ready checkpoints (`checkpoint_675000` — the trainer's off-by-one
means there is no `checkpoint_700000`).

Composition: `baseline`, `sysid_off`, 9 single-knob `no_*` / `start_100_*` cells,
and 4 isolation cells (`all_sysid_no_rand_v2`, `only_obs_noise_occlusion`,
`only_action_attenuation`, `only_action_attenuation_obs_noise_occlusion`).

**Keep as paper evidence** — but they ablate mechanisms that no longer exist in the env,
and their sim configs are deleted. Checkpoints only; not re-runnable.

### `zeroshot_paramrand/` — 11 GB, 5 runs, 2026-05-19/20
The current canonical sim2real DR line (`td3_training_dr` + `sim_paramrand_pm25.yaml`,
paddle_density / puck_damping / gravity ±25 % per reset).

- `expl_compare/{baseline,simple}_seed{0,1}` — 4 × 2M, the **newest runs in the tree**.
  2-primitive vs 4-primitive exploration ablation; simple ≥ baseline on all 5 eval envs
  (back-half 84.8 vs 73.5, n.s. at n=2).
- `paramrand_pm25/seed0` — **⚠ emptied.** Contains only `eval_envs.json` (8 KB). This is
  the 2M run whose `checkpoint_1000000` became `latest_models/ablations/paramrand_pm25/`
  (rolling-5 peak 132.7). The training artifacts are gone; only the promoted
  `training_state.pth` survives.
- `paramrand_pm25/seed0r1` — 800-step abort.

### `sim2sim_redesign/` — 28 GB, 35 runs, 2026-05-07/08
The campaign that produced the **current canonical residual-RL recipe**.

- `warp075_p{00,10,20,30}/seed0` — 400k trainability screen; all 4 cells passed, picked
  `warp075_p30` (paddle −30 % + sine-y warp 0.075, zs = 48) as the canonical big-gap target,
  replacing paddle50.
- `residual_warp075_p30/redesign_{raw,cql,cql_lite,cql_full,cql_n10,cql_bc05,cql_bc1,
  cql_lite_bc1,cql_1M}` (05-07) — round 1+2. Established CQL α=20 alone; BC stacking hurts.
- `residual_warp075_p30/phase{A,B,C,D}_*` (05-08, 12-hour campaign) — Phase A env difficulty
  (p00/p10/w10/w125), Phase B hyperparam grid (α ∈ {2,5,10,40}, actor2, q{2,4,8}),
  Phase C/D 1M extensions. **Result: `actor_updates_per_iteration` is the load-bearing knob**
  — actor=2 for warp ≤ 0.075, actor=4 for warp 0.10, warp 0.125 intractable.
  Best: actor=2 on p10 → 1M end-mean 117 [94, 142], peak 177.
- Three of these are still shipped as configs: `phaseC_actor2_1M`, `phaseD_actor2_p10_1M`,
  `phaseD_actor4_w10_1M` in `configs/td3/sim2sim/warp075_p30_residual/`.
- Stubs: `phaseC_actor2_1M/seed0r{1,2}` (2000 / 1500 steps, 05-11 smoke tests).

### `sim2sim_full_ft_warp075_p30/` — 7.3 GB, 12 runs, 2026-05-09
Does the residual recipe port to **full-network** FT? 4 cells × 500k
(`A_baseline` vanilla lr÷10, `B_cql20`, `C_cql20_actor2_n5`, `D_..._fulllr`) × targets
p30 (2 seeds) and p10 (1 seed). Most-cited group in `notes/` (C_cql20_actor2_n5 ×159,
A_baseline ×74). All 12 trained clean; **9 of 12 never got deterministic eval** — the
run was killed once the headline was clear from training curves.

### `recommended_top50_hist2/` — 1.5 GB, 3 runs, 2026-04-27
`seed0` = 1M reference run of the canonical TD3 args (`td3_recommended_top50_hist2.yaml`,
2-layer, q=25/a=6) on `sysid_best_params_hist2.yaml`. **Keep seed0.**
`seed0r1` (1500 steps) and `seed0r2` (empty) are smoke tests.

### `smoke_paramrand_500kr1/` — 688 MB, 2026-05-12
500k smoke test of the `td3_training_dr` paramrand entrypoint. Sibling
`smoke_paramrand_500k/` is **empty**. Marginal — superseded by the 2M `expl_compare` runs.

### `sim2sim/` eval-only dirs — ~97 MB total, no `args.yaml`
- `zs_warp_paddle_sweep/` (288 KB) — the Phase-1 zero-shot heatmap (paddle p00–p30 × warp
  0–0.25) that selected warp075_p30. **Cited; keep.**
- `hist2_motion0_to_sweeps/` (172 KB) — per-axis zs sweeps (wall_cone_deg, action_delay, pid_kp).
- `perturbation_sweep_reps_verify/` (89 MB) + `_n10/` (7.6 MB) — paddle/puck perturbation reps
  with rollouts. Keep the `summary.{json,md}`, the `eval_rollouts/` are droppable.
- `hist2_motion0_to_source/` (12 KB), `sim2sim_combined_paddle50_verify/` (8 KB) — single
  `metrics.json` each.

---

# Tier B — superseded, still lineage-bearing

~61 GB. Conclusions are settled and written up; the checkpoints are the only cost.

### `zeroshot_ablations/` — 8.1 GB, 12 runs, 500k, 2026-05-05/06
Direct predecessor of the 700k set — 11 of the 700k runs `full_resume`d from these
`training_state.pth`. Includes the broken `no_obs_delay` cell (flatlined at mean 17;
an env-side coupling between obs delay and puck-history density, later replaced by
`no_obs_delay_randomization`). Deleting these breaks the resume chain but not the record.

### `sysid_params/` — 19 GB, 17 runs + 6 MB logs, 2026-04-16/17
Post-sysid recipe sweeps on `sysid_best_params.yaml`, 1M each unless noted:
`upd_sweep{,r1,r2}` (UTD), `ratio_sweep{,r1,r2}` (q/actor ratio), `ablate_l2/l3/ablater1`
(layer count), `delay{,r1}`, `hist_len_{3,4,5}`, and `expl_{no_bootstrap,no_warmstart,
warmstart_heavy}` at 500k. These produced the canonical `top50_hist2` recipe. Cited in docs.

### `sysid_best/` — 614 MB, 2 runs, 2026-04-16
`delay_and_force{,r1}` — 1M attempt, only r1 got to 400k. Same era.

### `sim2sim/post_polyak_fix/` — 2.4 GB, 5 runs, 300k, 2026-05-06
First reruns after the Polyak fix: `fix_v27_baseline`, `fix_v27_q4`, `fix_twin`,
`fix_redq10`, `fix_v30_lite`. Found **Maxmin-N inverted** (N=2 ≥ N=10 ≥ N=5), while
UTD-1 and "exploration is harmful" survived.

### `sim2sim/post_polyak_fix_1M/` — 32 GB, 22 runs, 1M, 2026-05-07
The big post-fix sweep on the **old `sim2sim_combined` target**: `fix_cql_alpha{0.1,1,5,10,20}`,
`fix_cql_alpha5_rs{030,050}`, `fix_cql20_{bc01,bc1,twin,redq10}`, `fix_td3bc_lam{0.01…2}`,
`fix_{twin,redq10,v27_baseline,v27_q4,v30_lite}_1M`. Established that **CQL is the winning
post-fix mechanism** — then was superseded the same week by `sim2sim_redesign` on warp075.
`sim2sim_combined.yaml` is still the small-gap recipe target in CLAUDE.md, so the target
itself is live; these particular sweeps are not. **32 GB for a settled sweep — top Tier-B
deletion candidate.**

---

# Tier C — deprecated

~120 GB. Invalidated by the Polyak bug, the randomization removal, or a dead config.

### `sim2sim/hist2_motion0_to_paddle50/` — **50 GB, 63 runs**, 2026-04-29 → 05-04
`residual_v1`–`v30` (age-decay, top-k recency, buffer size, action-L2, q-weight-decay,
q_updates, ensemble3/5, REDQ5/10, exploration lite/full/directional), `full_ft_v32/v33`,
and `from_scratch_{300k,1M,1M_full_explore,5M}`. **Triple-deprecated**: pre-Polyak-fix
numbers unreliable; the paddle50 target was declared untrainable; memory record explicitly
marks all v25–v30 paddle50 recipes deprecated. **Largest single deletion candidate in the tree.**

### `sim2sim/hist2_motion0_to_combined/` — 13 GB, 44 runs, 2026-04-25/26
The 100k residual drift study + `residual_diagnose/` (lower_qlr, lower_utd, bigger_buffer,
low_success_frac, wd1e2/1e3, no_per, EMA decay, action-L2, recency top50/top99/window100,
scale schedules) + 400k extensions + `from_scratch_{400k,1M_resume}`. All pre-Polyak-fix.
Superseded by `post_polyak_fix_1M`, then by `sim2sim_redesign`.

### `newest_runs/` — 2.2 GB, 6 runs, 350k, 2026-04-06/07
`force_attenuation`, `puck_delay{,r1,r2}`, `both_additions`, `heavier` — probes of
action-force attenuation + puck delay interpolation. **Both mechanisms removed from the env
2026-05-11.** Sim config deleted.

### `force_attenuation/` — 489 MB, 1 run, 350k, 2026-04-04
Standalone predecessor of the above. Same deprecation.

### `updated_training/` — 5.4 GB, 19 runs, 350k, 2026-03-23 → 04-03
`motion_weight{00,0001,001,002,005}` × `{standard, heavy}` with repeats, plus
`density_sweep/d{2250,3000,3750,4500,6000}` (paddle density). Pre-sysid config
(`pid_noise_constant_upper_half_custom_sim_params.yaml`, deleted). Its one surviving
conclusion — `paddle_density: 3000` — is baked into `sysid_best_params.yaml`.

### `final_tuning/` — **26 GB, 34 runs**, 2026-02-28 → 03-02
Staged motion-reward-weight curricula, each stage resuming the previous:
`motion_staged_0p1` (10 stages → 1.54M cumulative), `motion_staged_0p2` (5),
`motion_staged_lower_alignment` (9 → 2.1M), `motion_staged_no_alignment` (8),
`motion_staged_one_step` (2). Dead pre-sysid config, and the entire motion/alignment
reward line was dropped — canonical args now carry `motion_reward_weight: 0.0` and
`axis_alignment_reward_weight: 0.0`. **Second-largest deletion candidate.**

### `final2/` — 13 GB, 2 runs, 1M, 2026-02-28 / 03-19
`mp_0p1_little_guidance`, `task_only_little_guidance`. Same dead config + reward line.

### `final/` — 9.6 GB, 3 entries (2 with args), 1M, 2026-02-26
`motion_w0p1_gpu1r1`, `task_only` (+ a `motion_w0p1_gpu1` dir). Same deprecation.

### `eval/` — 63 MB, 5 runs, 2026-03-24
`motion_weight001r1_sim_real_10k{,r1..r4}` — 5k/10k real-world adaptation fine-tunes off a
March policy, `eval_mode_model.pth` only, no checkpoints. Config `sim_real_world_adaptation.yaml`
deleted. Superseded by the whole `scripts/td3/extras/async_td3_real*` stack and
`configs/td3_real_world/`.

---

## Loose ends

- **Empty / stub dirs**: `smoke_paramrand_500k/`, `recommended_top50_hist2/seed0r2/`,
  `zeroshot_paramrand/paramrand_pm25/seed0/` (⚠ this one is data loss, not a stub —
  see Tier A), plus `seed0r1` / `seed0r2` aborts at 800–2000 steps in
  `recommended_top50_hist2`, `zeroshot_paramrand`, `sim2sim_redesign/…/phaseC_actor2_1M`,
  and `sim2sim/hist2_motion0_to_combined/residual/seed0`.
- **Stray logs at the group root**: `recommended_top50_hist2.log` (1.3 MB),
  `recommended_top50_hist2_eval.log`, `hist_motion_collision/logs/` (11 MB),
  `sysid_params/logs/` (6 MB), `hist_motion_collision/hist2_motion0_eval.log`.
- **Path drift**: every `args.yaml` records configs under
  `scripts/smooth_policy/amp_history/configs/…`, a path that no longer exists. Map to
  `configs/…` when reading provenance.

## If you clean up

| Action | Frees | Risk |
|---|---:|---|
| Delete all of Tier C | ~120 GB | none — invalidated or unreproducible, conclusions already in `notes/` |
| + `post_polyak_fix_1M/` | +32 GB | low — superseded the same week; write-up exists |
| + `sysid_params/` + `sysid_best/` + `post_polyak_fix/` + `zeroshot_ablations/` | +30 GB | low; only breaks the 500k→700k resume chain |
| Keep Tier A | 61 GB | — |

Before deleting anything in Tier A/B, confirm nothing needed has been promoted only
in-place: `latest_models/` currently holds `canonical/hist2_motion0{,_v2}/` and 16
`ablations/*/training_state.pth`, and those are the only copies that matter.

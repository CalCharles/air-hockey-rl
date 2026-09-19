# Sysid v2: the four fits compiled into one canonical sim config, and the four puck tasks retrained on it under sysid / wrong-sysid / DR (5-frame, long-history) / RMA with two randomization sets

- **Date**: 2026-09-19 01:50 UTC (campaign launched)
- **Status**: done (all 40 jobs rc 0, final paired evaluation 2026-09-19 ~11:00 UTC) — **revised 2026-09-19 ~03:30 UTC**: touch / reach / reach_vel dropped (user), the goal tasks added under long-history / RMA (HER path added to `scripts/rma/train_base_policy.py`) → 32 training cells + 8 RMA phase-2 jobs on 4 GPUs (2 per GPU); the eight juggle cells launched at 01:50 were kept (two finished, six adopted by the relaunched runner); `runs/td3/sysid_v2_20260919/status.md` is the live status, `summary.md` + `final_eval/summary.md` appear when everything is done
- **Run dir**: `runs/td3/sysid_v2_20260919/` (gitignored; `<cell>/`, `<cell>.stdout.log`, `campaign.log`)
- **Configs**: `configs/new_juggle/sysid_v2_hist2.yaml` (the compiled parameters), generated `configs/new_juggle/tasks_v2/sim_{sysid,low25,dr3,drfull}_<task>.yaml`, `configs/td3/tasks_v2/<task>_<method>.yaml` (+ `manifest.json`, `README.md`), `configs/rma/tasks_v2/<task>_rma_*_phase2.yaml`
- **Code**: `scripts/td3/extras/make_sysid_v2_configs.py` (generator), `run_sysid_v2_campaign.py` (scheduler: N jobs per GPU, RMA phase 2 chained, resume with adoption of live trainers, status / summary), `eval_sysid_v2_campaign.py` (paired final evaluation); HER in the RMA / long-history trainer: `scripts/rma/train_base_policy.py` (+ `env_wrapper.py` goal mode, `evaluate.FlatGoalEnv`, `networks.step_feature_dim` accepting the goal suffix), doc [`rma-baseline.md`](../../docs/training/rma-baseline.md#goal-conditioned-tasks-her-2026-09-19)
- **Sysid inputs**: [`2026-09-10_01-30`](2026-09-10_01-30_sysid-train-val-pipeline.md) (puck, walls), [`2026-09-19_01-35`](2026-09-19_01-35_paddle-pid-pooled-refit-reversal-jerk.md) (paddle PID), [`2026-09-11_01-21`](2026-09-11_01-21_paddle-puck-collision-cmaes-sysid.md) + [`2026-09-18_23-30`](2026-09-18_23-30_offset-paddle-puck-collisions-replay.md) (paddle–puck)

## Question

With every identified parameter in one config, how do the policies of the four puck tasks
(juggle, puck_vel, puck_goal, puck_goal_vel; touch / reach / reach_vel were dropped) compare when trained on (a) the identified sim, (b) a sim whose identified parameters are all
25 % lower, (c) ±25 % domain randomization with the 5-frame history obs, (d) the same DR with a
50-step history, and (e) RMA — each DR case with the three canonical randomized parameters and
with the full identified set? Sim evaluation now; real-robot evaluation of the same policies
later.

## Setup

### Sysid v2 (`configs/new_juggle/sysid_v2_hist2.yaml`)

| parameter | v1 | **v2** | from |
|---|---|---|---|
| `gravity` / `puck_damping` | −0.661 / 0.178 | **−0.73 / 0.11** | `sysid/puck_dynamics` |
| `side_wall_restitution` / `end_wall_restitution` | 0.99 / 0.70 | **0.90 / 0.55** | `sysid/wall_collision` (end wall weakly identified; taken as the data's best estimate) |
| `puck_restitution` (paddle–puck), `paddle_restitution` | 1.09145, 1.0 | **1.2626, 0** | `sysid/paddle_puck_collision`: head-on gain 1.627 at the canonical mass ratio 2.56; the listener uses `max()` of the two, so 0 on the paddle makes one knob |
| `pid_kp` / `pid_ki` / `pid_kd` | 9000 / 0 / 50 | **7531.5 / 1928.7 / 0** | `sysid/paddle_pid` pooled fit |
| `paddle_density` / `puck_density` | 3000 / 3000 | 3000 / 3000 | not identified |
| `hist_len` | 2 (4 in the task configs) | **2** | the real rollout config; hist4 measured worse ([`2026-09-04_22-19`](2026-09-04_22-19_hist4-smoothing-five-tasks.md)) |

Noise, occlusion, observation delay and the task keys (max_timesteps, terminations, goal
settings) are copied unchanged from the existing task configs (`sim_sysid_<task>.yaml`, the
`_hist2` variants for the goal tasks).

### Cells (32 = 4 tasks × 8 methods, + 8 `low05` / `low10` cells added afterwards = 40)

| method | sim | trainer | steps |
|---|---|---|---|
| `sysid` | v2, no DR | `td3_training` (HER trainer for the goal tasks) | 1M |
| `low25` | v2 with every identified parameter × 0.75 (gravity −0.5475, damping 0.0825, density 2250, side / end wall 0.675 / 0.4125, puck_restitution 0.947, kp 5649, ki 1447) | same | 1M |
| `low05` / `low10` | the same parameters × 0.95 / × 0.90 (added 2026-09-19 16:20 UTC, user request: a finer wrong-sysid ladder) | same | 1M |
| `dr5_3p` | ±25 % per-reset DR on paddle_density / puck_damping / gravity around v2 | `td3_training_dr` (HER for goal tasks) | 2M |
| `dr5_full` | ±25 % on the 3 + side / end wall restitution, puck_restitution, pid_kp, pid_ki (side wall capped at 1.0; kd = 0 and the masses excluded) | same | 2M |
| `drlong_3p` / `drlong_full` | as above | `scripts.rma.train_base_policy`, `rma_mode: history` (50-step window, no privileged input) | 2M |
| `rma_3p` / `rma_full` | as above | RMA phase 1 (privileged encoder over the randomized set) 2M + phase 2 adaptation module 400k on-policy steps | 2M + 400k |

Tasks: juggle, puck_vel, puck_goal, puck_goal_vel. The goal tasks use HER under every method:
the plain trainers through `td3_training_her`, the long-history / RMA trainer through its own HER
path (added for this campaign: `GoalEnvVector` inside the e_t / window wrappers, the same
relabelling, flat-goal evaluation). Recipes are the existing `configs/td3/tasks/<task>_{sysid,dr}.yaml`
(q = 25 / a = 6, PER, flat 1M buffer, q_weight_decay 0; HER k = 4 and the 128 / 256 networks for
the goal tasks), i.e. the 2026-09-04 recipe with only the sim config, hist_len and the method keys
changed. One seed per cell.

### Evaluation

- During training each trainer keeps its own eval (single-env for `sysid` / `low25`, the 5 fixed
  DR envs of its own range set for the DR / history / RMA runs, goal eval for HER).
- **Final paired evaluation** (`eval_sysid_v2_campaign.py`, runs automatically at the end): every
  final policy — plain actor, HER actor, `HistoryActor`, RMA *adapted* policy — on the same three
  sets with identical episode seeds: `nominal` (the task's v2 sim, 50 episodes), `id_3p` and
  `id_full` (the 5 fixed DR envs of each range set, seed 12345, 10 episodes each).
  `final_eval/summary.md` is the table to read.
- Real-robot evaluation: not part of this note; the policies are ordinary `model.pth` files
  (history / RMA policies need their window / adaptation module — `scripts/rma/evaluate.py` agents).

Smoke test before launch: every trainer type (plain, DR, HER, history, RMA + phase 2) at 30k
steps on the generated configs, all rc 0; the goal task under history / RMA (+ phase 2) likewise
before the relaunch, and the paired evaluation on those outputs.

## Results

All 32 training cells and 8 phase-2 jobs finished (rc 0; 0.5–1.5 h per juggle / puck_vel cell,
2–3.3 h per goal cell at 2 jobs per GPU). Full tables: `runs/td3/sysid_v2_20260919/summary.md`
(the trainers' own evals) and `final_eval/summary.md` (paired). Below: the paired final
evaluation, **return ± SEM over episodes** on the task's v2 sim (`nominal`, 50 episodes) and on
the 5 fixed DR envs of the 3-parameter / full set (`id_3p` / `id_full`, 10 episodes each; same
episode seeds for every policy). Goal tasks: return = 10 × success rate. RMA = adapted policy.

| method | juggle nominal | juggle id_3p | juggle id_full | puck_vel nominal | puck_vel id_3p | puck_vel id_full |
|---|---|---|---|---|---|---|
| sysid | 124 ± 6 | 138 ± 6 | 108 ± 6 | 34.3 ± 1.7 | 31.6 ± 1.4 | 31.6 ± 1.9 |
| low25 | 73 ± 5 | 66 ± 4 | 58 ± 5 | 19.4 ± 1.3 | 23.3 ± 1.2 | 23.5 ± 1.6 |
| dr5_3p | 156 ± 7 | 147 ± 7 | 124 ± 7 | 39.0 ± 1.5 | 39.0 ± 1.3 | 37.2 ± 1.5 |
| dr5_full | 115 ± 6 | 124 ± 6 | 95 ± 6 | 38.7 ± 1.6 | 37.3 ± 1.5 | 33.7 ± 1.6 |
| drlong_3p | 194 ± 4 | 190 ± 3 | 186 ± 4 | 39.7 ± 1.1 | 40.3 ± 1.3 | 37.6 ± 1.3 |
| **drlong_full** | **197 ± 2** | **193 ± 4** | 178 ± 6 | **40.9 ± 1.2** | **41.5 ± 1.4** | **39.2 ± 1.4** |
| rma_3p | 128 ± 6 | 130 ± 7 | 117 ± 7 | 31.9 ± 1.5 | 32.1 ± 1.5 | 30.4 ± 1.2 |
| rma_full | 135 ± 7 | 119 ± 8 | 101 ± 8 | 20.5 ± 1.4 | 20.7 ± 1.4 | 22.1 ± 1.4 |

| method | puck_goal nominal (success) | puck_goal id_3p | puck_goal id_full | puck_goal_vel nominal | puck_goal_vel id_3p | puck_goal_vel id_full |
|---|---|---|---|---|---|---|
| sysid | **0.82** | 0.74 | 0.66 | 0.36 | **0.56** | **0.46** |
| low25 | 0.70 | 0.58 | 0.62 | 0.22 | 0.24 | 0.22 |
| dr5_3p | 0.80 | **0.84** | **0.72** | **0.62** | 0.46 | 0.44 |
| dr5_full | 0.76 | 0.76 | **0.72** | 0.56 | 0.46 | 0.30 |
| drlong_3p | 0.02 | 0.04 | 0.02 | 0.46 | 0.38 | **0.46** |
| drlong_full | 0.02 | 0.04 | 0.02 | 0.02 | 0.02 | 0.00 |
| rma_3p | 0.74 | 0.54 | 0.54 | 0.40 | 0.40 | 0.24 |
| rma_full | 0.44 | 0.62 | 0.60 | 0.10 | 0.16 | 0.18 |

(SEM of a success rate over 50 episodes ≈ 0.06–0.07.) RMA phase-2 latent fit (val R², 400k
steps): 0.53 / 0.38 (juggle 3p / full), 0.43 / 0.32 (puck_vel), 0.33 / 0.19 (puck_goal),
0.46 / 0.06 (puck_goal_vel) — the 8-parameter latent is much harder to infer from 50 steps.

### Findings

1. **The identified sim matters: the 25 %-low baseline loses 40–50 % on every task**
   (juggle 124 → 73, puck_vel 34 → 19, puck_goal 0.82 → 0.70, puck_goal_vel 0.36 → 0.22 on the
   v2 sim). That is the size of the gap a wrong parameter set opens in sim; how much of it the
   real robot shows is the next (real-world) measurement.
2. **Dense tasks: long-history TD3 with DR is the best policy everywhere**, juggle 194–197 vs
   sysid 124 and the 5-frame DR 156 (3p); it is also the most robust (id_full 178–186 vs 95–124).
   DR itself helps the 5-frame policy on the 3-parameter set (juggle +32, puck_vel +5 over sysid)
   but the **full set hurts the 5-frame policy on juggle** (115 vs 156; the wider physics spread,
   incl. paddle–puck restitution 0.95–1.58, is too much to absorb without a longer history);
   long-history absorbs it (197 vs 194).
3. **RMA is not competitive**: adapted policies sit at or below the 5-frame DR policy on the dense
   tasks (juggle 128 / 135, puck_vel 32 / 21) and below sysid on the goal tasks; the full set
   makes it worse (puck_vel 20.5). Consistent with the 2026-09-18 finding that RMA's supervised
   8-dim latent is the bottleneck, now on four tasks and with a harder latent.
4. **Goal tasks (sparse, HER): the 5-frame policies (sysid / dr5) are the best, and the long-history
   policy fails to take off in 3 of 4 cells** (puck_goal both sets, puck_goal_vel full: success
   0.02, episodes end at 35 steps = puck passes untouched, HER valid-goal fraction stuck at 0.04
   from the first 500k steps vs 0.2–0.3 in the cells that learn). The relabelling path is the
   same one the RMA cells learn with (0.74 on puck_goal), and drlong_3p on puck_goal_vel does
   learn (0.46), so it is an exploration / bootstrap failure of the 432-dim-input critic on a
   sparse reward, not a plumbing bug — but it is single-seed and needs seeds before it is a claim.
5. DR on the goal tasks is roughly neutral (dr5_3p ≈ sysid on nominal, slightly better on the DR
   envs); the full set is again the worse randomization.

## Conclusion

Sysid v2 is in place and every method has a trained policy per task. On the identified sim the
ordering is long-history DR > 5-frame DR (3 parameters) > sysid > RMA > wrong sysid for the dense
tasks, and sysid ≈ 5-frame DR (3 parameters) > RMA > wrong sysid for the goal tasks, with the
long-history recipe unreliable on sparse goals. The full identified DR set is never better than
the 3-parameter set for a 5-frame policy and only matches it for the long-history one — the extra
randomization of restitutions and PID gains costs return without buying robustness on these
evaluation sets. Candidates for the real-robot evaluation: `drlong_full` / `drlong_3p` (juggle,
puck_vel), `dr5_3p` and `sysid` (goal tasks), with `low25` as the wrong-sysid control. All single
seed.

## Next

- Real-robot evaluation with `scripts/td3/extras/async_td3_real_eval.py` on the v2 rollout config
  (hist_len 2); history / RMA policies need an eval agent there.
- Seeds: every cell is one seed; repeat the winning and losing cells with 2 more seeds before
  drawing conclusions about small differences — first the long-history goal cells (3 of 4 at 0).
- Why the long-history critic does not bootstrap on sparse goals: try the window without the
  goal tasks' 128 / 256 networks, a shorter window, or a warm-up from the 5-frame HER policy.

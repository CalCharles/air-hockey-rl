# Air Hockey RL — Agent Context

Shared context for AI agents (Claude Code, Cursor, etc.). Read this before making changes.

---

## What this project is

Reinforcement learning for a physical air-hockey robot (UR5 arm + paddle). The agent learns to juggle a puck in a Box2D simulator, then transfers the policy to the real robot. The training algorithm is **TD3 with single-head critic and transformed Bellman targets**.

See [`notes/docs/repo/project-goal-and-safety.md`](notes/docs/repo/project-goal-and-safety.md) for safety policy (real-robot e-stops, protective stops).

---

## Repo layout

```
airhockey/         — Box2D + real-UR5 env package; tasks registered in __init__.py
scripts/
├── td3/           — TD3 training, helpers, real-world entrypoints, tests
├── real/          — real-robot rollout / teleop / calibration helpers
├── visualization/ — trajectory rendering / teleop-segment helpers
├── analysis/      — standalone analysis tools (occlusion patterns, etc.)
└── utils.py       — small shared helpers (e.g., save_tensorboard_plots)
configs/           — all YAMLs
├── new_juggle/    — sim env configs (sysid_best_params*, sim2sim targets)
├── td3/           — TD3 training args (canonical: td3_recommended_top50_hist2.yaml)
├── td3_real_world/— real-robot residual fine-tune args
└── real_configs/  — real-robot rollout / mouse-teleop configs
latest_models/canonical/ — sim-pretrained source policies (hist2_motion0_v2/, hist2_motion0/)
latest_models/ablations/ — CoRL-2026 deployment-ready ablation checkpoints
```

## Active code paths

| What | Where |
|------|-------|
| **Training entrypoint (source-sim only)** | `scripts/td3/td3_training.py` — 2026-09-03: CUDA-graph updates, CPU rollout, async checkpoint eval, reduced logging (~10× faster training phase). Read [`notes/docs/training/training-throughput.md`](notes/docs/training/training-throughput.md) before touching the loop. **Batch runner for a set of configs across GPUs (one job per GPU): `scripts/td3/run_experiments.py --mode dr|nodr --configs ... --gpus ...`.** Old-vs-new comparison wrapper: `scripts/td3/extras/throughput_bench.py` (defaults to `configs/td3/tasks`). **Recipe since 2026-09-04: `q_weight_decay: 0` + `single_replay_buffer: true` (one flat 1M buffer) in every canonical TD3 config.** |
| **Training entrypoint (goal-conditioned tasks, HER)** | `scripts/td3/td3_training_her.py` — same recipe with the goal appended to the history obs and hindsight relabelling of every episode (`scripts/td3/helper/td3_her.py`). Tasks `puck_goal_position_sparse` / `puck_goal_position_speed_sparse` (`airhockey/airhockey_tasks/puck_goal_sparse.py`, sparse +10 at the goal after a paddle contact, episode ends), configs `configs/td3/her/*.yaml`, batch via `run_experiments.py --mode her`. Read [`notes/docs/training/her.md`](notes/docs/training/her.md). Results 2026-09-10 ([`2026-09-10_06-30_her-puck-goal-tasks.md`](notes/scratch/experiments/2026-09-10_06-30_her-puck-goal-tasks.md)): position 79 % final success (hist2; 3 % without HER), speed 40 % (best ckpt 60 %); a full velocity-vector goal stayed ≤ 6 % and was dropped (code at commit 467da75); policies in `latest_models/her/`. **2026-09-10: `td3_training.py` / `td3_training_dr.py` and their helper chain were restored to the last working loop (commit 3e3df6f + the `save_replay_buffer_intermediate` fix) after the merged wandb/transformer edits left them unrunnable; the transformer work under `scripts/transformer/` is not wired into the canonical trainers.** |
| **Training entrypoint (canonical sim2sim / sim2real)** | `scripts/td3/td3_training_dr.py` — wraps `td3_training.py` with per-reset env-parameter randomization. Used with `configs/td3/zeroshot_paramrand/td3_paramrand_pm25.yaml` (sim config `configs/new_juggle/zeroshot_ablations/sim_paramrand_pm25.yaml`). |
| **RMA sim2real baseline (paper baseline, separate package)** | `scripts/rma/` — Rapid Motor Adaptation (Kumar et al. 2021) on the same ±25 % physics DR and TD3 recipe as `td3_training_dr`. Phase 1 `scripts/rma/train_base_policy.py` (π(x, a₋₁, μ(e)) + privileged encoder, asymmetric critic), phase 2 `scripts/rma/train_adaptation_module.py` (50-step conv adaptation module, on-policy supervised), paired eval `scripts/rma/evaluate.py` / `scripts/rma/eval_paired.py`, tables `scripts/rma/summarize.py`; **fair control = `rma_mode: history`** (`configs/rma/history_juggle_dr.yaml`: the deployable RMA architecture with the 50-step window trained end-to-end without privileged input — plain `td3_training_dr` only sees the 5-frame obs); configs `configs/rma/`; batch via `run_experiments.py --mode rma`. Goal-conditioned (`return_goal_obs`) sim configs get HER inside this trainer too (2026-09-19). **Read [`notes/docs/training/rma-baseline.md`](notes/docs/training/rma-baseline.md) first** (includes the TD3-instead-of-PPO assessment). Results: [`2026-09-18_21-00_rma-fair-baselines-long-history-control.md`](notes/scratch/experiments/2026-09-18_21-00_rma-fair-baselines-long-history-control.md) — paired sim eval: long-history TD3 192 > RMA adapted 150 > plain DR 133 (ID); RMA's edge over plain DR is mostly the 50-step window, not the privileged latent. Latent-fit analysis in [`2026-09-11_01-05_rma-td3-baseline.md`](notes/scratch/experiments/2026-09-11_01-05_rma-td3-baseline.md). |
| **Canonical sim config (sysid ground truth)** | **`configs/new_juggle/sysid_v2_hist2.yaml` (sysid v2, 2026-09-19: the four `sysid/` fits compiled — g −0.73 / damping 0.11 / side wall 0.90 / end wall 0.55 / paddle–puck `puck_restitution` 1.2626 / pid 7531.5 · 1928.7 · 0, hist_len 2). Task configs derived from it: `configs/new_juggle/tasks_v2/` + `configs/td3/tasks_v2/` (generated by `scripts/td3/extras/make_sysid_v2_configs.py`).** Pre-v2: `sysid_best_params.yaml` / `sysid_best_params_hist2.yaml` (v1, kept for the older runs and configs) |
| **Canonical TD3 args** | `configs/td3/td3_recommended_top50_hist2.yaml` — 2-layer, q=25/a=6, references `sysid_best_params_hist2.yaml` |
| **Sim-to-real ground truth source policy** | `latest_models/canonical/hist2_motion0_v2/` (predecessor `hist2_motion0/` kept on disk for reproducibility; don't reference in new work) |
| **Residual RL recipe (sim2sim/sim2real fine-tune)** | Canonical big-gap recipe: CQL α=20 + `actor_updates_per_iteration=2` (=4 for warp ≥ 0.10), no BC, no exploration, N=5, residual_scale=0.15. Configs: `configs/td3/sim2sim/warp075_p30_residual/{phaseC_actor2_1M,phaseD_actor2_p10_1M,phaseD_actor4_w10_1M}.yaml` (sim targets in `configs/new_juggle/sim2sim_warp075_p30.yaml`, `sim2sim_warp075_p10.yaml`, `sim2sim_warp100_p30.yaml`). Small-gap recipe: `configs/td3/sim2sim/td3_sim2sim_residual.yaml` (sim target `configs/new_juggle/sim2sim_combined.yaml`). Real-world canonical big-gap CQL recipe: `configs/td3_real_world/td3_residual_cql.yaml` (cql_alpha=20, actor_updates_per_iteration=2; reuses `td3_residual_train_args.yaml`). v27 baseline (no CQL, kept for regression / non-residual eval comparison): `configs/td3_real_world/td3_residual.yaml`. CQL is wired into `scripts/td3/helper/real_td3_runtime.py` as a default-off branch (gated on `cql_alpha > 0`) so v27 launches are bit-identical to before. **Read [`notes/docs/training/residual-rl-recipe.md`](notes/docs/training/residual-rl-recipe.md) before running.** |
| **Env entrypoint** | `airhockey/` (`AirHockeyEnv`) |
| **Real-world rollout entrypoint** | `scripts/td3/extras/async_td3_real.py` (orchestrator + `__main__`; thin file driving the per-concern runners). The shared runtime library — `Args`, `TrainArgs`, `LearnerRuntimeState`, args-file parsing, checkpoint helpers, the synchronous learner step — lives at `scripts/td3/helper/real_td3_runtime.py` alongside the other modular helpers (`real_policy_runner`, `real_reset_runner`, `real_transition_hold`, …). Plus `scripts/real/` for non-training rollout helpers. |
| **Real-world fixed-policy eval** | `scripts/td3/extras/async_td3_real_eval.py` (frozen actor, no learner / replay / checkpointing — emits `eval_summary.json` + `eval_per_episode.jsonl`). Two pluggability layers: `--agent {td3,sgcrl,…}` selects how the actor is built (dispatch in `scripts/td3/helper/real_eval_agents.py`); the env config's `task:` selects a `TaskEvalHooks` from `scripts/td3/helper/real_task_eval_hooks.py` (juggle tasks → `JuggleEvalHooks` with `min_timesteps=50` + juggles/contacts metrics; everything else → `GenericEvalHooks`, `min_timesteps=10`). Full reference: [`notes/docs/training/real-world-eval-pipeline.md`](notes/docs/training/real-world-eval-pipeline.md). |
| **Human-baseline teleop eval (paper user study)** | `scripts/td3/extras/async_td3_real_teleop_eval.py` (mouse-controlled paddle running the same task / termination / juggle counter / output schema as the policy eval; auto-detects puck-in-upper-half between episodes; phase banner window with colored borders for RESET / HANDOFF / USER CONTROL / EPISODE OVER). Read [`notes/docs/training/teleop-eval-baseline.md`](notes/docs/training/teleop-eval-baseline.md) before running. |

The config file passed to `td3_training.py` has a `config:` key pointing to the sim config and a `model_path:` key for resuming.

---

## Observation and action spaces

**Active obs type**: `history` (30-dim). Configured via `obs_type: history` in sim config.

```
[  0:15]  paddle history — 5 × [x, y, valid_flag], oldest (t-4) → newest (t)
[ 15:30]  puck   history — 5 × [x, y, valid_flag], oldest (t-4) → newest (t)
```

Key slices: paddle pos = `[12:14]`, puck pos = `[27:29]`, puck vel proxy = `obs[27:29] − obs[15:17]`.

With `use_last_action_in_policy_state: true` (default), the **actor receives 32 dims**: raw obs (30) + last action (2).

**Action space**: `Box([-1,1], shape=(2,))` — normalised displacement target fed to a PID controller.
- `action[0] * 0.26 m` = max x-step, `action[1] * 0.12 m` = max y-step.

Full details: [`notes/docs/environments/observation-action-spaces.md`](notes/docs/environments/observation-action-spaces.md)

---

## System-ID best-fit parameters

**Sysid v2 (2026-09-19, `configs/new_juggle/sysid_v2_hist2.yaml`) is the parameter set to work off from now on.** It compiles the four fits in `sysid/` (start at [`sysid/README.md`](sysid/README.md)):

| Parameter | v1 (`sysid_best_params_hist2.yaml`) | **v2** | Source (all in `sysid/`) |
|-----------|----------------|------------|--------|
| `gravity` / `puck_damping` | -0.661 / 0.178 | **-0.73 / 0.11** | `puck_dynamics`: 542 free-flight clips of the mouse dataset, held-out prediction 1.82 → 1.51 cm / 0.5 s |
| `side_wall_restitution` / `end_wall_restitution` | 0.99 / 0.70 | **0.90 / 0.55** | `wall_collision`: Box2D replay of real bounces (end wall weakly identified) |
| `puck_restitution` (paddle–puck; `paddle_restitution` 0) | 1.09145 (with paddle 1.0) | **1.2626** | `paddle_puck_collision`: head-on gain 1.627 at the canonical mass ratio 2.56 (e / r degenerate; oblique hits show the frictionless contact deflects 1.9× too much) |
| `pid_kp` / `pid_ki` / `pid_kd` | 9000 / 0 / 50 | **7531.5 / 1928.7 / 0** | `paddle_pid`: CMA-ES on the pooled lines / arcs + reversal-jerk sessions, val 23.2 → 17.5 mm |
| `paddle_density` / `puck_density` | 3000 / 3000 | 3000 / 3000 | unchanged (masses not identified) |

Policy campaign on v2 (2026-09-19, **done**): `scripts/td3/extras/run_sysid_v2_campaign.py` → `runs/td3/sysid_v2_20260919/` — juggle, puck_vel, puck_goal, puck_goal_vel × {sysid, 25 %-low sysid baseline, DR 5-frame, DR long-history, RMA} with the 3-parameter and the full identified DR sets (32 cells, one seed each; the goal tasks use HER under every method). Paired sim result: 25 %-low sysid loses 40–50 % everywhere; long-history DR is best on juggle (197 vs sysid 124) and puck_vel; on the goal tasks sysid ≈ 5-frame DR (3 params) are best and the long-history policy fails to take off in 3 of 4 cells; RMA never wins; the full identified DR set is never better than the 3-parameter set ([`2026-09-19_01-50`](notes/scratch/experiments/2026-09-19_01-50_sysid-v2-compiled-params-policy-campaign.md)); `final_eval/summary.md` there is the paired comparison. The v1 values came from the 2026-04/05 grid searches: [`environments/real-world/puck-system-id.md`](notes/docs/environments/real-world/puck-system-id.md) · [`environments/real-world/teleop-system-id.md`](notes/docs/environments/real-world/teleop-system-id.md)

---

## Tasks

The canonical task is `puck_juggle_upper_half_reward` — keep this name. **Seven canonical tasks** have ready-to-run configs under `configs/td3/tasks/<task>_{sysid,dr}.yaml` (sim configs `configs/new_juggle/tasks/sim_{sysid,dr}_<task>.yaml`): `juggle` (`puck_juggle_upper_half_reward`), `touch` (`puck_touch`, +1 on contact), `reach` (`paddle_reach_position`, +10 at the goal), `reach_vel` (`paddle_reach_position_velocity`, +10 at goal position *and* velocity), `puck_vel` (`puck_velocity`, 10 × upward puck displacement per step), and the two goal-conditioned puck tasks added 2026-09-11, `puck_goal` (`puck_goal_position_sparse`, +10 when the puck reaches a goal position in the upper half after a paddle contact) and `puck_goal_vel` (`puck_goal_position_speed_sparse`, +10 when it does so at the goal *speed*; goals sampled from simulated shots). The two goal tasks train with `scripts/td3/td3_training_her.py` (HER; `run_experiments.py --mode auto` picks it from `her_k`), see [`notes/docs/training/her.md`](notes/docs/training/her.md). `_sysid` = sysid physics, 1M steps; `_dr` = ±25 % physics randomization, 2M steps.

**Smoothing: `hist_len: 4` (set 2026-09-04 — but measured WORSE than hist2; see the caveat below).** All ten task sim configs run a 4-timestep moving-average low-pass on the PID target (`_filter_update()` in `airhockey/sims/airhockey_box2d.py`), paired with the hist4-specific paddle sysid refit — `pid_kp: 7500`, `paddle_density: 3500` (vs the hist2 fit `9000` / `3000`), measured against real teleop data re-recorded under `hist_len: 4` (`configs/new_juggle/sysid_best_params_hist4.yaml`). The `_dr` paddle_density range is recentred to 2625–4375. The previous `hist_len: 2` baseline for the identical TD3 recipe is preserved at `runs/td3/tasks_20260904/{sysid,dr}/summary.md`.

> **hist4 adoption is NOT validated — it measured worse.** The 2026-09-04 like-for-like head-to-head ([`2026-09-04_22-19_hist4-smoothing-five-tasks.md`](notes/scratch/experiments/2026-09-04_22-19_hist4-smoothing-five-tasks.md), runs `runs/td3/tasks_hist4_20260904/`) found hist4 loses on every task with reward headroom, in both sysid and DR: juggle DR back-half **94.5 vs 116.8**, puck_vel DR **25.0 vs 57.7** (max 31 vs 80), juggle sysid **64.3 vs 83.4**; under DR it also delays reach saturation from 200k to ~1.8M steps. touch / reach_vel / reach-sysid sit at their reward ceiling and are indifferent. Single seed per cell, but consistent across all four independent comparisons and across the whole trajectory. Smoothing and the paddle-gain refit moved together, so the cause is not yet isolated. **Revert to `hist_len: 2` (+ kp 9000 / density 3000) unless hist4 is required for real-robot transfer.**
 Run them all with `scripts/td3/run_experiments.py --mode auto --configs configs/td3/tasks/*.yaml --gpus ...`. **Since 2026-09-19 the sysid-v2 versions of juggle / puck_vel / puck_goal / puck_goal_vel live in `configs/td3/tasks_v2/<task>_<method>.yaml` (methods sysid / low25 / dr5_3p / dr5_full / drlong_3p / drlong_full / rma_3p / rma_full; sim configs `configs/new_juggle/tasks_v2/`, hist_len 2) and are run with `scripts/td3/extras/run_sysid_v2_campaign.py`; regenerate them with `scripts/td3/extras/make_sysid_v2_configs.py` after any change to `sysid_v2_hist2.yaml`.** The `configs/td3/tasks/` files remain the v1 (hist4) configs. The reward scales are part of the task definitions (reward classes), not config knobs — see `notes/scratch/experiments/2026-09-04_01-05_sparse-task-collapse-diagnosis.md` for why the sparse tasks need ×10. Other tasks (`puck_juggle`, `puck_strike`, `puck_score`, `puck_goal_position*`, `move_block`, `strike_crowd`, etc.) remain registered in `airhockey/__init__.py` and are callable from Python, but no config files target them.

---

## Documentation

Formal docs live in `notes/docs/`. Start at [`notes/docs/index.md`](notes/docs/index.md).

Key docs:
- Architecture & algorithm: [`training/architecture.md`](notes/docs/training/architecture.md) · [`training/td3-algorithm.md`](notes/docs/training/td3-algorithm.md)
- **Residual RL recipe**: [`training/residual-rl-recipe.md`](notes/docs/training/residual-rl-recipe.md) — winning data-balance recipe for sim2sim/sim2real fine-tuning
- Configs: [`training/td3-configs.md`](notes/docs/training/td3-configs.md) · [`training/sim-env-configs.md`](notes/docs/training/sim-env-configs.md)
- **External trainer quickstart** (someone else bringing their own RL algo, using the same Box2D env / canonical hist2 sim config): [`training/box2d-env-usage.md`](notes/docs/training/box2d-env-usage.md)
- Monitoring (TensorBoard layout, scalar reference, console output): [`training/monitoring.md`](notes/docs/training/monitoring.md)
- **Training throughput** (profile of the loop, CUDA-graph / CPU-rollout / async-eval design, knobs, remaining costs): [`training/training-throughput.md`](notes/docs/training/training-throughput.md)
- Rewards: [`training/reward-shaping.md`](notes/docs/training/reward-shaping.md)
- Networks: [`training/network-architecture.md`](notes/docs/training/network-architecture.md)
- Replay / episodes: [`training/replay-and-episodes.md`](notes/docs/training/replay-and-episodes.md)
- **Resume async real-world training from a checkpoint**: [`training/checkpointing.md#resuming-real-world-async-training`](notes/docs/training/checkpointing.md#resuming-real-world-async-training) — exact resume command, required `include_non_vital_training_state_fields: true` flag, replay-source semantics, multi-run `episode_summaries.jsonl` stitching
- **Goal-conditioned tasks + HER**: [`training/her.md`](notes/docs/training/her.md) — sparse puck-goal position / position+speed tasks, hindsight relabelling, eval
- Sim2sim transfer testing: [`training/sim2sim.md`](notes/docs/training/sim2sim.md)
- **Sim2sim / sim2real source-policy training (canonical post-2026-05-11)**: use `scripts/td3/td3_training_dr.py` with [`configs/td3/zeroshot_paramrand/td3_paramrand_pm25.yaml`](configs/td3/zeroshot_paramrand/td3_paramrand_pm25.yaml). Per-reset **environment-parameter randomization** (paddle_density / puck_damping / gravity, ±25 % of sysid) on top of a minimal baseline (sysid physics + observation delay + puck noise + plain uniform occlusion). The earlier engineered per-collision / action-attenuation / delay-jitter / paddle-density-fluctuation stack was deprecated and the mechanisms removed from the env. The pre-deprecation source `latest_models/canonical/hist2_motion0_v2/` is historical only — retrain for any new sim2sim / sim2real deployment.
- Box2D env: [`environments/box2d/simulator-essentials.md`](notes/docs/environments/box2d/simulator-essentials.md)
- Real-world stack: [`environments/real-world/overview.md`](notes/docs/environments/real-world/overview.md)
- **Real-world clamping coverage gap**: [`environments/real-world/paddle-clamping-coverage-gap.md`](notes/docs/environments/real-world/paddle-clamping-coverage-gap.md) — sync-only `apply_negative_z_force` + UR's ~2 s `forceMode` timeout means the paddle goes compliant-free during reset/idle phases. Read before debugging any "robot stops clamping mid-reset" report.
- System ID: [`environments/real-world/puck-system-id.md`](notes/docs/environments/real-world/puck-system-id.md) · [`environments/real-world/teleop-system-id.md`](notes/docs/environments/real-world/teleop-system-id.md) · **All sysid code, data and results live in `sysid/` (one folder per fit: `puck_dynamics/`, `wall_collision/`, `paddle_pid/`, `paddle_puck_collision/`, each `code/` + `data/` + `results/`; shared library + tools in `sysid/common/`) — start at [`sysid/README.md`](sysid/README.md).** · **Auto-segmentation of recordings into free-fall / wall / paddle clips**: [`environments/real-world/trajectory-auto-segmentation.md`](notes/docs/environments/real-world/trajectory-auto-segmentation.md) (`sysid/common/segment_trajectories.py`; read its frame-calibration section before using `pose` from `shared/mouse_state_data_*` geometrically — puck and paddle are logged in different frames) · **Train/val sysid pipeline** (recordings → sections → gravity / damping / wall restitution via Box2D replay), one command `sysid/common/run_puck_wall_sysid.py --input-dir … --name <name>` (writes `sysid/{puck_dynamics,wall_collision}/{data,results}/<name>/`): [`environments/real-world/sysid-pipeline.md`](notes/docs/environments/real-world/sysid-pipeline.md) (general: stages, split, percentile validation of each search, replication checklist) + per-section pages [`sysid/puck-free-flight.md`](notes/docs/environments/real-world/sysid/puck-free-flight.md) and [`sysid/puck-wall-collision.md`](notes/docs/environments/real-world/sysid/puck-wall-collision.md) (data, model, metrics, results, reproduce); directory index `scripts/sysid/README.md`; smoke test `pytest sysid/common/tests -q` · **Paddle PID sysid (CMA-ES, 2026-09-10 / 19)**: `sysid/paddle_pid/code/fit_pid_cmaes.py --input-dir <session> [<session> …] --out sysid/paddle_pid/results/<name>` fits kp / ki / kd (mass fixed) to scripted paddle-motion recordings (lines / arcs `traj_*` and reversal-jerk `jerk_*` sessions pooled) with one trial per condition held out — [`sysid/paddle-pid.md`](notes/docs/environments/real-world/sysid/paddle-pid.md); pooled result kp 7532 / ki 1929 / kd 0, val 23.2 → 17.5 mm, promoted into sysid v2 (the lines-only fit 5496 / 5883 overshot reversals): [`2026-09-19_01-35_paddle-pid-pooled-refit-reversal-jerk.md`](notes/scratch/experiments/2026-09-19_01-35_paddle-pid-pooled-refit-reversal-jerk.md), first fit [`2026-09-10_03-39_paddle-pid-cmaes-sysid.md`](notes/scratch/experiments/2026-09-10_03-39_paddle-pid-cmaes-sysid.md) · **Paddle–puck collision sysid (CMA-ES, 2026-09-11)**: `sysid/paddle_puck_collision/code/fit_collision_cmaes.py --input-dir <puck_collision session> --out sysid/paddle_puck_collision/results/<name>` fits restitution + mass ratio to scripted head-on collisions (best 3 takes per condition, puck speeds from free-flight fits); `render_collisions.py --fit-dir …` makes side-by-side camera · real · sim videos per collision — [`sysid/paddle-puck-collision.md`](notes/docs/environments/real-world/sysid/paddle-puck-collision.md); head-on data only pin the gain `(1+e)·r/(r+1)` = **1.63** vs 1.50 in the sim (puck launched 8 % too slow), e / r degenerate along that ridge, not promoted; `evaluate_offset_collisions.py` replays the offset (oblique) session with those parameters (no fit): exit speed to 7.5 % up to 5 cm offset but exit angle 1.9 × too large — the frictionless contact lacks a tangential term ([`2026-09-18_23-30`](notes/scratch/experiments/2026-09-18_23-30_offset-paddle-puck-collisions-replay.md)): [`2026-09-11_01-21_paddle-puck-collision-cmaes-sysid.md`](notes/scratch/experiments/2026-09-11_01-21_paddle-puck-collision-cmaes-sysid.md)
- Exploration primitives: [`exploration/td3-primitives.md`](notes/docs/exploration/td3-primitives.md)

---

## Project conventions

- **New docs** → `notes/docs/*.md`. **Scratch/plans** → `notes/scratch/`.
- **Experiment writeups** → `notes/scratch/experiments/YYYY-MM-DD_HH-MM_<topic-slug>.md` — one new file per experiment, never edit prior ones. **Read [`notes/scratch/experiments/README.md`](notes/scratch/experiments/README.md) before writing experiment notes.** This convention exists to avoid git merge conflicts when multiple agents append to the same long-lived log file. The long-form logs (`notes/scratch/residual_rl_paddle50_log.md`, `notes/scratch/residual_rl_drift_fix_log.md`, etc.) are now **read-only history** — historical context only, do not append. New experiments go in dated files; cross-link instead of merging; update [`notes/scratch/experiments/INDEX.md`](notes/scratch/experiments/INDEX.md) (additive only) when each experiment lands. Stable conclusions from a finished experiment can still be reflected in the canonical docs (`notes/docs/training/residual-rl-recipe.md`, this file) — but reference the experiment file as the source of truth, don't restate the data.
- **GIFs for qualitative changes** to Box2D env: use `AirHockeyRenderer`, BGR→RGB, resize width to 160, fps 20. See `.cursor/rules/box2d-environment.mdc`.
- **Default Box2D config** for one-off scripts: `configs/new_juggle/sysid_v2_hist2.yaml` (sysid v2, hist_len 2). `sysid_best_params_hist2.yaml` is v1 (the pre-2026-09-19 runs); `sysid_best_params_hist4.yaml` matches the hist4 task configs under `configs/new_juggle/tasks/`.
- **Virtual env**: check for `.venv/` or `pyproject.toml` before running code.
- Prefer editing existing files over creating new ones — **except for experiment writeups, which always go in new dated files** (see above).

---

## Run-artifact directories (gitignored)

These directories live on disk for local use but are not tracked: `runs/`, `results/`, `trained_models/`, `eval_gifs/`, `real_runs/`, `shared/`, `sysid/*/data/`, `sysid/*/results/`, `sysid/common/runs/`, `dataset_management/`, `tests/`, `wandb/`, `gifs/`, `plots/`, `datasets/`. Clean them up locally as needed; don't add them to git.

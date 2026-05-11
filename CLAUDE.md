# Air Hockey RL — Agent Context

Shared context for AI agents (Claude Code, Cursor, etc.). Read this before making changes.

---

## What this project is

Reinforcement learning for a physical air-hockey robot (UR5 arm + paddle). The agent learns to juggle a puck in a Box2D simulator, then transfers the policy to the real robot. The training algorithm is **TD3 with dual-head critics and transformed Bellman targets**.

See [`notes/docs/repo/project-goal-and-safety.md`](notes/docs/repo/project-goal-and-safety.md) for safety policy (real-robot e-stops, protective stops).

---

## Active code paths

| What | Where |
|------|-------|
| **Training entrypoint** | `scripts/smooth_policy/amp_history/amp_training/td3/td3_training.py` |
| **Canonical sim config (sysid ground truth)** | `scripts/smooth_policy/amp_history/configs/new_juggle/sysid_best_params.yaml` (or `sysid_best_params_hist2.yaml` for hist_len=2) |
| **Canonical TD3 args** | `scripts/smooth_policy/amp_history/configs/td3/td3_recommended_top50_hist2.yaml` — 2-layer, q=25/a=6, references `sysid_best_params_hist2.yaml` |
| **Sim-to-real ground truth source policy** | `latest_models/canonical/hist2_motion0_v2/` (predecessor `hist2_motion0/` kept on disk for reproducibility; don't reference in new work) |
| **Residual RL recipe (sim2sim/sim2real fine-tune)** | Canonical big-gap recipe: CQL α=20 + `actor_updates_per_iteration=2` (=4 for warp ≥ 0.10), no BC, no exploration, N=5, residual_scale=0.15. Configs: `scripts/smooth_policy/amp_history/configs/td3/sim2sim/warp075_p30_residual/{phaseC_actor2_1M,phaseD_actor2_p10_1M,phaseD_actor4_w10_1M}.yaml` (sim targets in `configs/new_juggle/sim2sim_warp075_p30.yaml`, `sim2sim_warp075_p10.yaml`, `sim2sim_warp100_p30.yaml`). Small-gap recipe: `…/td3/sim2sim/td3_sim2sim_residual.yaml` (sim target `configs/new_juggle/sim2sim_combined.yaml`). Real-world canonical big-gap CQL recipe: `…/td3_real_world/td3_residual_cql.yaml` (cql_alpha=20, actor_updates_per_iteration=2; reuses `td3_residual_train_args.yaml`). v27 baseline (no CQL, kept for regression / non-residual eval comparison): `…/td3_real_world/td3_residual.yaml`. CQL is wired into `helper/real_td3_runtime.py` as a default-off branch (gated on `cql_alpha > 0`) so v27 launches are bit-identical to before. **Read [`notes/docs/training/residual-rl-recipe.md`](notes/docs/training/residual-rl-recipe.md) before running.** |
| **Env entrypoint** | `airhockey/` (`AirHockeyEnv`) |
| **Real-world rollout entrypoint** | `scripts/smooth_policy/amp_history/amp_training/td3/extras/async_td3_real.py` (orchestrator + `__main__`; thin file driving the per-concern runners). The shared runtime library — `Args`, `TrainArgs`, `LearnerRuntimeState`, args-file parsing, checkpoint helpers, the synchronous learner step — lives at `scripts/smooth_policy/amp_history/amp_training/td3/helper/real_td3_runtime.py` alongside the other modular helpers (`real_policy_runner`, `real_reset_runner`, `real_transition_hold`, …). Plus `scripts/real/` for non-training rollout helpers. |
| **Real-world fixed-policy eval** | `scripts/smooth_policy/amp_history/amp_training/td3/extras/async_td3_real_eval.py` (frozen actor, no learner / replay / checkpointing — emits `eval_summary.json` + `eval_per_episode.jsonl`). |
| **Human-baseline teleop eval (paper user study)** | `scripts/smooth_policy/amp_history/amp_training/td3/extras/async_td3_real_teleop_eval.py` (mouse-controlled paddle running the same task / termination / juggle counter / output schema as the policy eval; auto-detects puck-in-upper-half between episodes; phase banner window with colored borders for RESET / HANDOFF / USER CONTROL / EPISODE OVER). Read [`notes/docs/training/teleop-eval-baseline.md`](notes/docs/training/teleop-eval-baseline.md) before running. |

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

Real-world system identification found these optimal sim parameters (captured in `sysid_best_params.yaml`):

| Parameter | Standard config | Sysid best | Source |
|-----------|----------------|------------|--------|
| `gravity` | -0.650 | **-0.661** | Puck grid search (10 trajectory segments) |
| `puck_damping` | 0.250 | **0.178** | Puck grid search |
| `paddle_density` | 1000 | **3000** | Paddle 3D PID+density grid search (8 teleop categories) |
| `pid_kp` | 5000 | **9000** | Paddle grid search |
| `pid_kd` | 200 | **50** | Paddle grid search |
| `pid_ki` | 0.0 | 0.0 | Ki sweep (no benefit) |

Full details: [`environments/real-world/puck-system-id.md`](notes/docs/environments/real-world/puck-system-id.md) · [`environments/real-world/teleop-system-id.md`](notes/docs/environments/real-world/teleop-system-id.md)

---

## Tasks

The canonical task is `puck_juggle_upper_half_reward` — keep this name. Other tasks (`puck_juggle`, `puck_touch`, `puck_strike`, `puck_score`, `puck_goal_position*`, `paddle_reach_position*`, `move_block`, `strike_crowd`, etc.) remain registered in `airhockey/__init__.py` and are callable from Python, but no config files target them. Add a sim config under `scripts/smooth_policy/amp_history/configs/new_juggle/` if you want to train on a non-juggle task.

---

## Documentation

Formal docs live in `notes/docs/`. Start at [`notes/docs/index.md`](notes/docs/index.md).

Key docs:
- Architecture & algorithm: [`training/architecture.md`](notes/docs/training/architecture.md) · [`training/td3-algorithm.md`](notes/docs/training/td3-algorithm.md)
- **Residual RL recipe**: [`training/residual-rl-recipe.md`](notes/docs/training/residual-rl-recipe.md) — winning data-balance recipe for sim2sim/sim2real fine-tuning
- Configs: [`training/td3-configs.md`](notes/docs/training/td3-configs.md) · [`training/sim-env-configs.md`](notes/docs/training/sim-env-configs.md)
- **External trainer quickstart** (someone else bringing their own RL algo, using the same Box2D env / canonical hist2 sim config): [`training/box2d-env-usage.md`](notes/docs/training/box2d-env-usage.md)
- Monitoring (TensorBoard layout, scalar reference, console output): [`training/monitoring.md`](notes/docs/training/monitoring.md)
- Rewards: [`training/reward-shaping.md`](notes/docs/training/reward-shaping.md)
- Networks: [`training/network-architecture.md`](notes/docs/training/network-architecture.md)
- Replay / episodes: [`training/replay-and-episodes.md`](notes/docs/training/replay-and-episodes.md)
- **Resume async real-world training from a checkpoint**: [`training/checkpointing.md#resuming-real-world-async-training`](notes/docs/training/checkpointing.md#resuming-real-world-async-training) — exact resume command, required `include_non_vital_training_state_fields: true` flag, replay-source semantics, multi-run `episode_summaries.jsonl` stitching
- Sim2sim transfer testing: [`training/sim2sim.md`](notes/docs/training/sim2sim.md)
- Box2D env: [`environments/box2d/simulator-essentials.md`](notes/docs/environments/box2d/simulator-essentials.md)
- Real-world stack: [`environments/real-world/overview.md`](notes/docs/environments/real-world/overview.md)
- **Real-world clamping coverage gap**: [`environments/real-world/paddle-clamping-coverage-gap.md`](notes/docs/environments/real-world/paddle-clamping-coverage-gap.md) — sync-only `apply_negative_z_force` + UR's ~2 s `forceMode` timeout means the paddle goes compliant-free during reset/idle phases. Read before debugging any "robot stops clamping mid-reset" report.
- System ID: [`environments/real-world/puck-system-id.md`](notes/docs/environments/real-world/puck-system-id.md) · [`environments/real-world/teleop-system-id.md`](notes/docs/environments/real-world/teleop-system-id.md)
- Exploration primitives: [`exploration/td3-primitives.md`](notes/docs/exploration/td3-primitives.md)

---

## Project conventions

- **New docs** → `notes/docs/*.md`. **Scratch/plans** → `notes/scratch/`.
- **Experiment writeups** → `notes/scratch/experiments/YYYY-MM-DD_HH-MM_<topic-slug>.md` — one new file per experiment, never edit prior ones. **Read [`notes/scratch/experiments/README.md`](notes/scratch/experiments/README.md) before writing experiment notes.** This convention exists to avoid git merge conflicts when multiple agents append to the same long-lived log file. The long-form logs (`notes/scratch/residual_rl_paddle50_log.md`, `notes/scratch/residual_rl_drift_fix_log.md`, etc.) are now **read-only history** — historical context only, do not append. New experiments go in dated files; cross-link instead of merging; update [`notes/scratch/experiments/INDEX.md`](notes/scratch/experiments/INDEX.md) (additive only) when each experiment lands. Stable conclusions from a finished experiment can still be reflected in the canonical docs (`notes/docs/training/residual-rl-recipe.md`, this file) — but reference the experiment file as the source of truth, don't restate the data.
- **GIFs for qualitative changes** to Box2D env: use `AirHockeyRenderer`, BGR→RGB, resize width to 160, fps 20. See `.cursor/rules/box2d-environment.mdc`.
- **Default Box2D config** for one-off scripts: `scripts/smooth_policy/amp_history/configs/new_juggle/sysid_best_params_hist2.yaml`.
- **Virtual env**: check for `.venv/` or `pyproject.toml` before running code.
- Prefer editing existing files over creating new ones — **except for experiment writeups, which always go in new dated files** (see above).

---

## Run-artifact directories (gitignored)

These directories live on disk for local use but are not tracked: `runs/`, `results/`, `trained_models/`, `eval_gifs/`, `real_runs/`, `shared/`, `sysid/`, `dataset_management/`, `tests/`, `wandb/`, `gifs/`, `plots/`, `datasets/`. Clean them up locally as needed; don't add them to git.

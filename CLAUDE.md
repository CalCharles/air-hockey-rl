# Air Hockey RL — Agent Context

Shared context for AI agents (Claude Code, Cursor, etc.). Read this before making changes.

---

## What this project is

Reinforcement learning for a physical air-hockey robot (UR5 arm + paddle). The agent learns to juggle/hit a puck in a Box2D simulator, then transfers the policy to the real robot. The active training algorithm is **TD3 with dual-head critics and transformed Bellman targets**. PPO/SAC code exists but is legacy.

See [`notes/docs/repo/project-goal-and-safety.md`](notes/docs/repo/project-goal-and-safety.md) for safety policy (real-robot e-stops, protective stops).

---

## Active code paths

| What | Where |
|------|-------|
| **Training entrypoint** | `scripts/smooth_policy/amp_history/amp_training/td3/td3_training.py` |
| **Canonical sim config (sysid ground truth)** | `scripts/smooth_policy/amp_history/configs/new_juggle/sysid_best_params.yaml` |
| **Legacy sim config** | `…/pid_noise_constant_upper_half_custom_sim_params.yaml` (pre-sysid; still used by some TD3 args YAMLs) |
| **TD3 training configs** | `scripts/smooth_policy/amp_history/configs/td3/` |
| **Recommended TD3 default** | `…/td3/td3_recommended.yaml` (2-layer, q=25/a=6, hist_len=4 via `sysid_best_params_hist4.yaml`, no bootstrap forcing, no external warmstart — see [depth/update ablations](notes/docs/training/td3-ablations-updates-and-depth.md) and [exploration ablations](notes/docs/training/td3-exploration-ablations.md)) |
| **Residual RL recipe (sim2sim/sim2real fine-tune)** | Small-gap (<10% zs drop): `…/td3/sim2sim/td3_sim2sim_residual.yaml` (`recency_top50`). **Big-gap (>20% zs drop, paddle50) — canonical default: `…/td3/sim2sim/paddle50/td3_residual_v27_ensemble5.yaml`** (Maxmin-5 critics; 1M-verified; build all future residual sim2sim/sim2real work off this). Alternative for fire-and-forget 300k deployment only: `…/paddle50/td3_residual_v30_explore_lite.yaml` (tighter cross-seed last5 std at lower peak). **Real-world (v27 Maxmin-5)**: `…/td3_real_world/td3_residual.yaml` (args-file) + `…/td3_real_world/td3_residual_train_args.yaml` (train-args supplying `num_critics: 5`); launch via `async_td3_real_modular.py`. **Read [`notes/docs/training/residual-rl-recipe.md`](notes/docs/training/residual-rl-recipe.md#real-world-residual--v27-canonical) before running.** |
| **Env entrypoint** | `airhockey/` (`AirHockeyEnv`) |
| **Real-world rollout entrypoint** | `scripts/smooth_policy/amp_history/amp_training/td3/extras/async_td3_real_modular.py` (only entrypoint; the non-modular `async_td3_real.py` is now a shared library — `Args`, `LearnerRuntimeState`, helpers). Plus `scripts/real/` for non-training rollout helpers. |
| **Real-world fixed-policy eval** | `scripts/smooth_policy/amp_history/amp_training/td3/extras/async_td3_real_eval.py` (frozen actor, no learner / replay / checkpointing — emits `eval_summary.json` + `eval_per_episode.jsonl`). |
| **Human-baseline teleop eval (paper user study)** | `scripts/smooth_policy/amp_history/amp_training/td3/extras/async_td3_real_teleop_eval.py` (mouse-controlled paddle running the same task / termination / juggle counter / output schema as the policy eval; auto-detects puck-in-upper-half between episodes; phase banner window with colored borders for RESET / HANDOFF / USER CONTROL / EPISODE OVER). Read [`notes/docs/training/teleop-eval-baseline.md`](notes/docs/training/teleop-eval-baseline.md) before running. |

The config file passed to `td3_training.py` (e.g., `td3_no_alignment.yaml`) has a `config:` key pointing to the sim config and a `model_path:` key for resuming.

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

## Documentation

Formal docs live in `notes/docs/`. Start at [`notes/docs/index.md`](notes/docs/index.md).

Key docs:
- Architecture & algorithm: [`training/architecture.md`](notes/docs/training/architecture.md) · [`training/td3-algorithm.md`](notes/docs/training/td3-algorithm.md)
- **Residual RL recipe**: [`training/residual-rl-recipe.md`](notes/docs/training/residual-rl-recipe.md) — winning data-balance recipe for sim2sim/sim2real fine-tuning
- Configs: [`training/td3-configs.md`](notes/docs/training/td3-configs.md) · [`training/sim-env-configs.md`](notes/docs/training/sim-env-configs.md)
- Monitoring (TensorBoard layout, scalar reference, console output): [`training/monitoring.md`](notes/docs/training/monitoring.md)
- Rewards: [`training/reward-shaping.md`](notes/docs/training/reward-shaping.md)
- Networks: [`training/network-architecture.md`](notes/docs/training/network-architecture.md)
- Replay / episodes: [`training/replay-and-episodes.md`](notes/docs/training/replay-and-episodes.md)
- **Resume async real-world training from a checkpoint**: [`training/checkpointing.md#resuming-real-world-async-training`](notes/docs/training/checkpointing.md#resuming-real-world-async-training) — exact resume command, required `include_non_vital_training_state_fields: true` flag, replay-source semantics, multi-run `episode_summaries.jsonl` stitching
- Sim2sim transfer testing: [`training/sim2sim.md`](notes/docs/training/sim2sim.md)
- Box2D env: [`environments/box2d/simulator-essentials.md`](notes/docs/environments/box2d/simulator-essentials.md)
- Real-world stack: [`environments/real-world/overview.md`](notes/docs/environments/real-world/overview.md)
- System ID: [`environments/real-world/puck-system-id.md`](notes/docs/environments/real-world/puck-system-id.md) · [`environments/real-world/teleop-system-id.md`](notes/docs/environments/real-world/teleop-system-id.md)
- Exploration primitives: [`exploration/td3-primitives.md`](notes/docs/exploration/td3-primitives.md)

---

## Project conventions

- **New docs** → `notes/docs/*.md`. **Scratch/plans** → `notes/scratch/`.
- **GIFs for qualitative changes** to Box2D env: use `AirHockeyRenderer`, BGR→RGB, resize width to 160, fps 20. See `.cursor/rules/box2d-environment.mdc`.
- **Default Box2D config** for one-off scripts: `scripts/smooth_policy/amp_history/configs/new_juggle/pid_noise_constant_upper_half_custom_sim_params.yaml`.
- **Virtual env**: check for `venv/`, `.venv/`, or `pyproject.toml` before running code.
- Prefer editing existing files over creating new ones.

---

## What is legacy / undocumented

- `scripts/domain_adaptation/`, `scripts/gat/`, `scripts/curriculum/`, `scripts/rl/` — older research paths, not part of active TD3 pipeline.
- `scripts/trainers/` — structured trainer abstraction (SAC, PPO), not used by current TD3 training.
- `offline_rl_algorithms/` — offline RL experiments, not active.
- `airhockey/sims/airhockey_robosuite.py` — MuJoCo/robosuite backend, **legacy** (not used for training). Basic functionality restored 2026-05-01 (ten interlocking bugs fixed across two passes: paddle_history, IK pose, OSC frame mismatch, pedestal/gripper collisions, translate_action z-bug, puck-under-tilted-table, RoundGripper registration, robot config keys + return value, puck free joint, gravity flag honored). Yellow round paddle (`RoundGripper`) is attached when `simulator_params.gripper_types: 'RoundGripper'` is set. Puck juggle init now spawns puck on tilted table and slides under gravity. Documented in [`notes/docs/environments/robosuite/overview.md`](notes/docs/environments/robosuite/overview.md). Diagnostic renders: `scripts/render_robosuite_views.py` (multi-camera) and `scripts/render_puck_juggle.py` (puck-juggle init).

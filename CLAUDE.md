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
| **Env entrypoint** | `airhockey/` (`AirHockeyEnv`) |
| **Real-world rollout** | `scripts/real/` + `scripts/smooth_policy/amp_history/amp_training/td3/extras/async_td3_real/` |

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
- Configs: [`training/td3-configs.md`](notes/docs/training/td3-configs.md) · [`training/sim-env-configs.md`](notes/docs/training/sim-env-configs.md)
- Rewards: [`training/reward-shaping.md`](notes/docs/training/reward-shaping.md)
- Networks: [`training/network-architecture.md`](notes/docs/training/network-architecture.md)
- Replay / episodes: [`training/replay-and-episodes.md`](notes/docs/training/replay-and-episodes.md)
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
- `airhockey/sims/airhockey_robosuite.py` — MuJoCo/robosuite backend, not currently used for training.

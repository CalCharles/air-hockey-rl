# Training docs

TD3 is the active training algorithm. PPO/SAC paths exist as legacy (see `architecture.md` for the boundary). Real-robot training uses the async pipeline; sim training uses the synchronous vec-env script.

> **Sim2sim / sim2real training strategy (2026-05-11 onward):** new source policies that need to transfer are trained with **environment-parameter randomization** via `scripts/td3/td3_training_dr.py` ([`configs/td3/zeroshot_paramrand/td3_paramrand_pm25.yaml`](../../../configs/td3/zeroshot_paramrand/td3_paramrand_pm25.yaml)). The earlier engineered-randomization stack has been deprecated and removed from the env. See [`sim2sim.md`](sim2sim.md) for the strategy overview. The residual fine-tune recipe ([`residual-rl-recipe.md`](residual-rl-recipe.md)) remains valid as a complementary adaptation step on top of a paramrand-trained source.

## Overview / reference

| Doc | What it covers |
|-----|----------------|
| [`architecture.md`](architecture.md) | Active code paths, legacy folders, where each topic lives |
| [`td3-algorithm.md`](td3-algorithm.md) | h-transform, dual-head critics, actor objective |
| [`network-architecture.md`](network-architecture.md) | `ResidualMLPTrunk`, `DualHeadQ`, `DeterministicAgent` shapes |

## Runtime behavior

| Doc | What it covers |
|-----|----------------|
| [`reward-shaping.md`](reward-shaping.md) | Task + motion reward composition (5 components, weights) |
| [`replay-and-episodes.md`](replay-and-episodes.md) | PER, success/failure partitions, episode staging |
| [`checkpointing.md`](checkpointing.md) | Schema, resume vs fine-tune, migrations, real-world async resume |
| [`monitoring.md`](monitoring.md) | TensorBoard scalar reference, console layout, rolling windows |

## Configs

A reader picking which YAML to use should start with the doc closest to their goal:

| Doc | Scope |
|-----|-------|
| [`td3-configs.md`](td3-configs.md) | Sim TD3 args YAMLs (`configs/td3/`) |
| [`td3-real-world-configs.md`](td3-real-world-configs.md) | Real-robot async configs (`configs/td3_real_world/`) |
| [`sim-env-configs.md`](sim-env-configs.md) | Box2D sim env YAMLs (`configs/new_juggle/`) |

## Recipes (end-to-end procedures)

| Doc | When to use |
|-----|-------------|
| [`residual-rl-recipe.md`](residual-rl-recipe.md) | Residual fine-tune for sim2sim or sim2real (must read before launching) |
| [`sim2sim.md`](sim2sim.md) | Cross-sim transfer testing protocol |
| [`real-world-eval-pipeline.md`](real-world-eval-pipeline.md) | Frozen-policy eval: agent dispatch (`--agent td3 / sgcrl`), task hooks, output schema |
| [`teleop-eval-baseline.md`](teleop-eval-baseline.md) | Human-baseline mouse-paddle eval for the paper user study |

## Ablation reports

| Doc | Findings |
|-----|----------|
| [`td3-ablations-updates-and-depth.md`](td3-ablations-updates-and-depth.md) | Update count and network depth — the basis for `td3_recommended_top50_hist2.yaml` defaults |
| [`td3-exploration-ablations.md`](td3-exploration-ablations.md) | Warm-start and bootstrap-forcing exploration variants |

## Other

| Doc | Use |
|-----|-----|
| [`box2d-env-usage.md`](box2d-env-usage.md) | External-user guide: bring your own RL algo, use the Box2D env directly |

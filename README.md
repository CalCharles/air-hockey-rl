# Air Hockey RL

Reinforcement learning for a physical air-hockey robot (UR5 arm + paddle). The agent learns to juggle a puck in a Box2D simulator, then transfers the policy to the real robot. Active algorithm: TD3 with dual-head critics and transformed Bellman targets.

Project context for agents and humans: [`CLAUDE.md`](CLAUDE.md). Formal documentation: [`notes/docs/index.md`](notes/docs/index.md).

## Repo layout

```
airhockey/             — Box2D + real-UR5 env package; tasks registered in __init__.py
scripts/
├── td3/               — TD3 training, helpers, real-world entrypoints, tests
├── real/              — real-robot rollout / teleop / calibration helpers
├── visualization/     — trajectory rendering / teleop-segment helpers
├── analysis/          — standalone analysis tools (occlusion patterns, etc.)
└── utils.py           — small shared helpers
configs/               — all YAMLs
├── new_juggle/        — sim env configs
├── td3/               — TD3 training args
├── td3_real_world/    — real-robot residual fine-tune args
└── real_configs/      — real-robot rollout / mouse-teleop configs
latest_models/canonical/ — sim-pretrained source policies
latest_models/ablations/ — CoRL-2026 deployment-ready ablation checkpoints
notes/docs/            — formal docs
notes/scratch/         — experiment log files
paper/                 — CoRL 2026 LaTeX
```

## Installation

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create the venv and install deps (base + training)
uv sync --extra train
```

`pyproject.toml` declares the dependencies; `uv.lock` pins versions.

## Quickstart

Every TD3 run takes two YAMLs:

- **TD3 args file** (`configs/td3/...`) — algorithm hyperparameters (learning rates, replay buffer, network size, training schedule). Passed via `--args-file`. References a sim env config via its `config:` field.
- **Sim env config** (`configs/new_juggle/...`) — Box2D environment definition (task, physics, sim-to-real-gap features, per-reset randomization, reward weights). Loaded indirectly via the args file.

You edit the args file; the env config is loaded for you.

### Sim TD3 training (juggling source policy — for sim2sim / sim2real transfer)

| Role | Path |
|---|---|
| Trainer entrypoint | `scripts/td3/td3_training_dr.py` |
| TD3 args | `configs/td3/zeroshot_paramrand/td3_paramrand_pm25.yaml` |
| Sim env config (referenced by args) | `configs/new_juggle/zeroshot_ablations/sim_paramrand_pm25.yaml` |

```bash
.venv/bin/python -m scripts.td3.td3_training_dr \
  --args-file configs/td3/zeroshot_paramrand/td3_paramrand_pm25.yaml
```

For a source-sim-only policy (ablations, no transfer): `scripts/td3/td3_training.py --args-file configs/td3/td3_recommended_top50_hist2.yaml --num-envs 1`.

See [`notes/docs/training/sim2sim.md`](notes/docs/training/sim2sim.md) for the strategy overview.

### Sim2sim residual fine-tune

A residual recipe takes a trained source policy (above) and fine-tunes it on a perturbed target sim. Pick the recipe whose target matches your gap; before launching, edit the recipe YAML's `config:`, `model_path:` (source ckpt), `log_parent_dir:`, `run_name:`, and `seed:`.

| Target | TD3 args | Sim env config |
|---|---|---|
| **warp 0.075 · paddle −30% — canonical sim2sim** | **`configs/td3/sim2sim/warp075_p30_residual/phaseC_actor2_1M.yaml`** | **`configs/new_juggle/sim2sim_warp075_p30.yaml`** |
| warp 0.075 · paddle −10% | `configs/td3/sim2sim/warp075_p30_residual/phaseD_actor2_p10_1M.yaml` | `configs/new_juggle/sim2sim_warp075_p10.yaml` |
| warp 0.10 · paddle −30% | `configs/td3/sim2sim/warp075_p30_residual/phaseD_actor4_w10_1M.yaml` | `configs/new_juggle/sim2sim_warp100_p30.yaml` |
| paddle / dynamics only, no warp (small gap) | `configs/td3/sim2sim/td3_sim2sim_residual.yaml` | `configs/new_juggle/sim2sim_combined.yaml` |

```bash
.venv/bin/python -m scripts.td3.td3_training \
  --args-file configs/td3/sim2sim/warp075_p30_residual/phaseC_actor2_1M.yaml \
  --num-envs 1
```

Recipe selection guide: [`notes/docs/training/residual-rl-recipe.md`](notes/docs/training/residual-rl-recipe.md). Run ≥ 3 seeds.

### Real-robot residual training

See [`notes/docs/recent-commands.md`](notes/docs/recent-commands.md) for the canonical `async_td3_real` invocation, and [`notes/docs/environments/real-world/overview.md`](notes/docs/environments/real-world/overview.md) for the real-robot stack.

Boot the UR5 through the touchpad (power → external_control.urp) before launching any real-robot script. Hold `q` to terminate trajectories.

### Real-robot frozen-policy eval

```bash
python -m scripts.td3.extras.async_td3_real_eval \
  --config configs/real_configs/rollout_config_residual.yaml \
  --args-file configs/td3_real_world/td3_residual.yaml \
  --model-path <path_to_training_state.pth> \
  --train-args <path_to_args.yaml> \
  --collector-device cpu \
  --learner-device cuda:0 \
  --data-root-dir real_runs/eval_run
```

### Human-baseline teleop eval (user study)

```bash
python -m scripts.td3.extras.async_td3_real_teleop_eval --config configs/real_configs/mouse_config.yaml --args-file configs/td3_real_world/td3_residual.yaml --data-root-dir runs/teleop_user_study
```

See [`notes/docs/training/teleop-eval-baseline.md`](notes/docs/training/teleop-eval-baseline.md) for protocol.

## Testing

```bash
.venv/bin/python -m pytest scripts/td3/tests/
```

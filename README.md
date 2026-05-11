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

### Train a TD3 policy in sim

```bash
.venv/bin/python -m scripts.td3.td3_training \
  --args-file configs/td3/td3_recommended_top50_hist2.yaml \
  --run-name my_run \
  --num-envs 1
```

The trainer is single-env-collection only — pass `--num-envs 1`. Output lands under `runs/td3/<run-name>/` (gitignored).

See [`notes/docs/training/td3-configs.md`](notes/docs/training/td3-configs.md) for what's in the recommended config, and [`notes/docs/training/architecture.md`](notes/docs/training/architecture.md) for the code layout.

### Residual sim2sim fine-tune

```bash
.venv/bin/python -m scripts.td3.td3_training \
  --args-file configs/td3/sim2sim/warp075_p30_residual/phaseC_actor2_1M.yaml \
  --num-envs 1
```

Set `config:`, `model_path:`, `log_parent_dir:`, `run_name:`, and `seed:` in the recipe YAML before launching. See [`notes/docs/training/residual-rl-recipe.md`](notes/docs/training/residual-rl-recipe.md) for the recipe selection guide.

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
python -m scripts.td3.extras.async_td3_real_teleop_eval \
  --config configs/real_configs/mouse_config.yaml \
  --args-file configs/td3_real_world/td3_residual.yaml \
  --data-root-dir real_runs/teleop_eval
```

See [`notes/docs/training/teleop-eval-baseline.md`](notes/docs/training/teleop-eval-baseline.md) for protocol.

## Testing

```bash
.venv/bin/python -m pytest scripts/td3/tests/
```

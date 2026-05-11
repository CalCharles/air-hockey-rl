# Training architecture

Primary training code lives under [`scripts/td3`](../../../scripts/td3). See [`scripts/td3/README.md`](../../../scripts/td3/README.md) for run configs, examples, and real-robot TD3 notes.

## Active paths

### TD3 (current development)

- Entrypoint: [`scripts/td3/td3_training.py`](../../../scripts/td3/td3_training.py)
- **Does not use AMP** — no discriminator, no demo tensor for style matching; twin critics with task/motion heads and transformed Bellman targets (see module docstring in that file).
- Implementation helpers: [`scripts/td3/helper/`](../../../scripts/td3/helper/) (replay, dual-head Q, checkpointing, exploration primitives, metrics, etc.).
- Extras (staged runs, async real collector, visualization): [`scripts/td3/extras/`](../../../scripts/td3/extras/)
- Args YAML examples: [`configs/td3/`](../../../configs/td3/), real/async: [`configs/td3_real_world/`](../../../configs/td3_real_world/)

## Legacy / low-use training folders

These remain in the tree for older experiments but are **not** the current workflow. TD3 above is the active path; everything below is preserved for reference.

### PPO + AMP (legacy)

- Entrypoint: [`scripts/amp_training/amp_training.py`](../../../scripts/amp_training/amp_training.py) — PPO with optional least-squares AMP discriminator (position / position+action / puck-augmented features).
- Shared AMP building blocks in [`amp_training/`](../../../scripts/amp_training): [`discriminator.py`](../../../scripts/amp_training/discriminator.py), [`feature_processing.py`](../../../scripts/amp_training/feature_processing.py), [`demo_loader_position_history.py`](../../../scripts/amp_training/demo_loader_position_history.py), [`replay_buffer.py`](../../../scripts/amp_training/replay_buffer.py), [`normalizer.py`](../../../scripts/amp_training/normalizer.py), [`running_stats.py`](../../../scripts/amp_training/running_stats.py).
- Typical env/args configs: [`configs/pid/`](../../../configs/pid/) (e.g. AMP vs no-AMP args YAML).
- Discriminator data pipeline: [`amp_data/`](../../../scripts/amp_training/amp_data/) prepares windowed paddle/action/puck tensors as `.pt` for `--demo_data_path`.

### Other legacy folders

| Folder | Role |
|--------|------|
| [`amp_training/sac/`](../../../scripts/amp_training/sac/) | SAC + optional AMP (`amp_training_sac.py`) |
| [`amp_training/rma/`](../../../scripts/amp_training/rma/) | RMA-style AMP and adaptation scripts |
| [`amp_training/self_supervised/`](../../../scripts/amp_training/self_supervised/) | Self-supervised / SSL AMP variant |

Prefer extending **TD3** unless a task explicitly revives one of these paths.

## Detailed topics

| Topic | Doc |
|-------|-----|
| TD3 algorithm (h-transform, dual-head critics, actor objective) | [`td3-algorithm.md`](td3-algorithm.md) |
| PPO+AMP discriminator (modes, features, demo loader, auxiliary rewards) | [`ppo-amp-discriminator.md`](ppo-amp-discriminator.md) |
| Reward shaping (task + motion reward composition) | [`reward-shaping.md`](reward-shaping.md) |
| Network architecture (ResidualMLPTrunk, DualHeadQ, DeterministicAgent) | [`network-architecture.md`](network-architecture.md) |
| Replay buffers and episode handling (PER, success/failure, staging) | [`replay-and-episodes.md`](replay-and-episodes.md) |
| Checkpoint system (schema, resume vs fine-tune, migrations) | [`checkpointing.md`](checkpointing.md) |

## Guidance for edits

- Keep **AMP-specific** logic (discriminator, demo loading, feature windows) coherent with `amp_training.py` and `amp_data/`.
- Keep **TD3** concerns inside `td3/` (helpers, extras, tests) so the non-AMP stack stays isolated from discriminator code paths.

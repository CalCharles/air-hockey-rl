# Training architecture

Primary training code lives under [`scripts/smooth_policy`](../../../scripts/smooth_policy). See [`scripts/smooth_policy/README.md`](../../../scripts/smooth_policy/README.md) for run configs, examples, and real-robot TD3 notes.

## Active paths

### TD3 (current development)

- Entrypoint: [`scripts/smooth_policy/amp_history/amp_training/td3/td3_training.py`](../../../scripts/smooth_policy/amp_history/amp_training/td3/td3_training.py)
- **Does not use AMP** — no discriminator, no demo tensor for style matching; twin critics with task/motion heads and transformed Bellman targets (see module docstring in that file).
- Implementation helpers: [`scripts/smooth_policy/amp_history/amp_training/td3/helper/`](../../../scripts/smooth_policy/amp_history/amp_training/td3/helper/) (replay, dual-head Q, checkpointing, exploration primitives, metrics, etc.).
- Extras (staged runs, async real collector, visualization): [`scripts/smooth_policy/amp_history/amp_training/td3/extras/`](../../../scripts/smooth_policy/amp_history/amp_training/td3/extras/)
- Args YAML examples: [`scripts/smooth_policy/amp_history/configs/td3/`](../../../scripts/smooth_policy/amp_history/configs/td3/), real/async: [`configs/td3_real_world/`](../../../scripts/smooth_policy/amp_history/configs/td3_real_world/)

### PPO + AMP (current AMP training)

- Entrypoint: [`scripts/smooth_policy/amp_history/amp_training/amp_training.py`](../../../scripts/smooth_policy/amp_history/amp_training/amp_training.py) — PPO with optional least-squares AMP discriminator (position / position+action / puck-augmented features).
- Shared AMP building blocks in [`amp_training/`](../../../scripts/smooth_policy/amp_history/amp_training): [`discriminator.py`](../../../scripts/smooth_policy/amp_history/amp_training/discriminator.py), [`feature_processing.py`](../../../scripts/smooth_policy/amp_history/amp_training/feature_processing.py), [`demo_loader_position_history.py`](../../../scripts/smooth_policy/amp_history/amp_training/demo_loader_position_history.py), [`replay_buffer.py`](../../../scripts/smooth_policy/amp_history/amp_training/replay_buffer.py), [`normalizer.py`](../../../scripts/smooth_policy/amp_history/amp_training/normalizer.py), [`running_stats.py`](../../../scripts/smooth_policy/amp_history/amp_training/running_stats.py).
- Typical env/args configs: [`scripts/smooth_policy/amp_history/configs/pid/`](../../../scripts/smooth_policy/amp_history/configs/pid/) (e.g. AMP vs no-AMP args YAML).

### `amp_data` — discriminator data pipeline

- [`scripts/smooth_policy/amp_history/amp_training/amp_data/`](../../../scripts/smooth_policy/amp_history/amp_training/amp_data/) prepares **datasets for the AMP discriminator** from trajectory HDF5 (e.g. [`prepare_position_dataset.py`](../../../scripts/smooth_policy/amp_history/amp_training/amp_data/prepare_position_dataset.py), [`prepare_position_dataset_split.py`](../../../scripts/smooth_policy/amp_history/amp_training/amp_data/prepare_position_dataset_split.py)): windowed paddle (and optional action/puck) tensors saved as `.pt` for `--demo_data_path` on PPO+AMP runs.

## Legacy / low-use training folders

These remain in the tree for older experiments but are **not** the main workflows now:

| Folder | Role |
|--------|------|
| [`amp_training/sac/`](../../../scripts/smooth_policy/amp_history/amp_training/sac/) | SAC + optional AMP (`amp_training_sac.py`) |
| [`amp_training/rma/`](../../../scripts/smooth_policy/amp_history/amp_training/rma/) | RMA-style AMP and adaptation scripts |
| [`amp_training/self_supervised/`](../../../scripts/smooth_policy/amp_history/amp_training/self_supervised/) | Self-supervised / SSL AMP variant |

Prefer extending **TD3** or **PPO (`amp_training.py`)** unless a task explicitly revives one of these paths.

## Guidance for edits

- Keep **AMP-specific** logic (discriminator, demo loading, feature windows) coherent with `amp_training.py` and `amp_data/`.
- Keep **TD3** concerns inside `td3/` (helpers, extras, tests) so the non-AMP stack stays isolated from discriminator code paths.

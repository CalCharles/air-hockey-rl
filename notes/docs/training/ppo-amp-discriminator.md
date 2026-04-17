# PPO + AMP discriminator

> **Legacy.** TD3 is the active training path (see [`architecture.md`](architecture.md) and [`td3-algorithm.md`](td3-algorithm.md)). This doc describes the PPO+AMP stack that remains in the tree for reference.

Adversarial Motion Prior (AMP) discriminator used by PPO training to encourage expert-like motion style.

Entrypoint: [`amp_training.py`](../../../scripts/smooth_policy/amp_history/amp_training/amp_training.py).

## Discriminator network

**Code:** [`discriminator.py`](../../../scripts/smooth_policy/amp_history/amp_training/discriminator.py)

MLP that scores consecutive state pairs `[s_t, s_{t+1}]`. Configurable depth/width and activation (LeakyReLU default, slope 0.2).

- **Loss:** least-squares GAN (MSE on `[-1, 1]` targets: +1 expert, -1 agent) rather than binary cross-entropy.
- **Gradient penalty:** `compute_grad_penalty` provides Lipschitz regularization on the discriminator inputs.
- **Input normalizer:** [`normalizer.py`](../../../scripts/smooth_policy/amp_history/amp_training/normalizer.py) maintains running mean/std with clipping (default 10.0) to prevent scale drift.
- **Agent replay buffer:** [`replay_buffer.py`](../../../scripts/smooth_policy/amp_history/amp_training/replay_buffer.py) stores agent-generated observations to prevent catastrophic forgetting during discriminator updates.

## Discriminator modes

Three input configurations, toggled via CLI flags:

| Mode | Flags | Input dim | Content |
|------|-------|-----------|---------|
| Position-only | (default) | 8D | Paddle position history |
| + Actions | `--use_action_discriminator` | 16D | + 4-step action history |
| + Puck | `--use_puck_discriminator` | +3D | + puck features |

Modes are combinable. Dimension is computed dynamically via the demo loader's `get_obs_dim()`.

## Feature processing

**Code:** [`feature_processing.py`](../../../scripts/smooth_policy/amp_history/amp_training/feature_processing.py)

Discriminator inputs are normalized before scoring:

- **Paddle positions:** converted to relative deltas from first position -> `(T-1) * 2` features.
- **Actions:** unit-normalized and flattened -> 8D.
- **Puck features** (3D): `direction_sign` (net vertical motion sign), `downward_speed_bin` (3 bins), `vertical_pos_bin_5` (5-bin quantized position).

Supports **bucketed temporal sampling** for long trajectory windows via `sample_bucketed_indices_torch` (stratified sampling with endpoints + interior bins).

## Demo loader

**Code:** [`demo_loader_position_history.py`](../../../scripts/smooth_policy/amp_history/amp_training/demo_loader_position_history.py)

Loads `.pt` tensor datasets built by the `amp_data/` pipeline. Provides windowed position history (default 5-step) with optional action and puck features. Supports bucketed sampling for long temporal windows (`bucket_window_len`, `bucket_num_bins`, `bucket_samples_per_bin`).

## PPO auxiliary reward terms

Two optional reward shaping terms independent of the discriminator:

- `--temporal_alignment_reward_scale` -- encourages policy action timing to match demonstration timing.
- `--action_magnitude_reward_scale` -- encourages policy action magnitudes to match demonstrations.

These are added directly to the PPO reward signal, weighted by their respective scales.

## Reference state initialization

`ReferenceStateWrapper` (used in `amp_training.py` and `evaluate.py`) resets episodes from demonstration states rather than the default initial state distribution. Provides demo-conditioned starting poses for more targeted training.

## Related docs

- [Training architecture](architecture.md) -- where PPO+AMP fits in the training stack
- [Network architecture](network-architecture.md) -- `Agent` (stochastic PPO policy)
- [Reward shaping](reward-shaping.md) -- TD3 reward composition (separate system)

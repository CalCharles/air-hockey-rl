# Network architecture

Policy and critic network designs used by TD3 training.

## Shared backbone: `ResidualMLPTrunk`

**Code:** [`scripts/td3/agent.py`](../../../scripts/td3/agent.py)

All networks use `ResidualMLPTrunk` as their core feature extractor. It stacks `num_residual_blocks` instances of `ResidualDenseNormSwishBlock`:

```
ResidualDenseNormSwishBlock:
  for i in 1..units_per_block (default 4):
    Linear -> LayerNorm -> SiLU (Swish)
  output = block_output + skip_projection(input)
```

Each block has a skip connection; if the input dimension differs from `hidden_layer_size`, a learned linear projection is used for the residual path. All linear layers are initialized with orthogonal weights via `layer_init(std=sqrt(2))`.

The total network depth (linear layers) is `num_residual_blocks * units_per_block`.

## Deterministic actor: `DeterministicAgent`

**Code:** [`scripts/td3/deterministic_agent.py`](../../../scripts/td3/deterministic_agent.py)

```
obs -> ResidualMLPTrunk -> Linear (actor_mean_head) -> tanh -> scale + bias -> action
```

- Input: observation vector (optionally augmented with last action)
- Output: continuous action in `[-1, 1]` (the `action_scale`/`action_bias` buffers are hardcoded to 1.0/0.0 for the canonical 2D action space; residual policies use `residual_scale` as the head's `action_scale`).
- The `actor` and `actor_mean_head` attribute names are intentionally compatible with the stochastic `Agent` class, allowing direct weight transfer between PPO-trained and TD3 policies.

| Parameter | Default | Role |
|-----------|---------|------|
| `hidden_layer_size` | 64 | Width of each residual block |
| `num_hidden_layers` | 2 | Number of residual blocks |

## Stochastic actor: `Agent`

**Code:** [`scripts/td3/agent.py`](../../../scripts/td3/agent.py)

Used by PPO training; shares the same `ResidualMLPTrunk` backbone. Adds a log-std parameter head for Gaussian exploration. Not used during TD3 training but weight-compatible with `DeterministicAgent` for warm-starting.

## Critic: `TD3QNetwork`

**Code:** [`scripts/td3/helper/q_network.py`](../../../scripts/td3/helper/q_network.py)

```
cat(obs, action) -> ResidualMLPTrunk -> head (Linear -> scalar)
```

- Input: concatenation of observation and action vectors
- Output: Q-value in h-transformed space
- The output head uses small initialization (`std=0.01`) to keep initial Q estimates near zero
- `num_critics` instances (default 2 for twin TD3) plus target copies

| Parameter | Default | Role |
|-----------|---------|------|
| `hidden_layer_size` | 128 | Width of residual blocks (typically wider than actor) |
| `num_hidden_layers` | 2 | Number of residual blocks |

## Environment encoder: `EnvEncoder`

**Code:** [`scripts/td3/encoder.py`](../../../scripts/td3/encoder.py)

Compact MLP (Dense -> Tanh layers) that maps environment variable vectors to a latent conditioning code. Present in the codebase for domain-randomization experiments; not used by the canonical TD3 training path.

| Parameter | Default | Role |
|-----------|---------|------|
| `env_var_dim` | (required) | Dimension of the environment variable input |
| `latent_dim` | 8 | Output latent code size |
| `hidden_size` | (128, 128) | Hidden layer widths (tuple or int) |

## Weight initialization: `layer_init`

```python
def layer_init(layer, std=sqrt(2), bias_const=0.0):
    orthogonal_(layer.weight, std)
    constant_(layer.bias, bias_const)
```

Used everywhere: trunk blocks (`std=sqrt(2)`), skip projections (`std=1.0`), output heads (`std=0.01` for critics, `std=1` for actor).

## Related docs

- [TD3 algorithm](td3-algorithm.md) -- how these networks are trained
- [Checkpointing](checkpointing.md) -- how network state dicts are saved/restored

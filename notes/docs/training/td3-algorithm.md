# TD3 training algorithm

Core training loop for both simulation ([`td3_training.py`](../../../scripts/td3/td3_training.py)) and real-world ([`extras/async_td3_real.py`](../../../scripts/td3/extras/async_td3_real.py); shared runtime library [`helper/real_td3_runtime.py`](../../../scripts/td3/helper/real_td3_runtime.py)).

## Transformed Bellman targets (h / h-inverse)

Q-values are regressed in a **transformed space** using a symlog-style squashing function. This compresses large value magnitudes, stabilizing critic learning when rewards span multiple scales:

```
h(x)      = sign(x) * (sqrt(|x| + 1) - 1) + eps * x
h_inv(x)  = sign(x) * ( ((sqrt(1 + 4*eps*(|x| + 1 + eps)) - 1) / (2*eps))^2 - 1 )
```

`eps` (default `1e-3`) adds a small linear term that keeps `h` invertible everywhere. Critics output values in transformed space; Bellman targets are computed in original space, then mapped back through `h`.

**Code:** `h_transform` / `h_inverse` in `td3_training.py` and duplicated in `helper/real_td3_runtime.py`.

## Single-head critic

Each of the `num_critics` critic networks ([`TD3QNetwork`](../../../scripts/td3/helper/q_network.py)) has a residual trunk and one scalar output head predicting transformed Q for the environment reward stream.

See also: [network-architecture.md](network-architecture.md) for the `TD3QNetwork` structure.

## Critic update

Standard TD3 twin-critic procedure (generalized to `num_critics >= 2`):

1. **Target actions:** query the target actor with the next observation, add clipped Gaussian noise (`policy_noise=0.2`, `noise_clip=0.5`), clamp to action bounds.
2. **Min target:** element-wise minimum across the target critics in transformed space (or a sampled subset of size `target_critic_subset_size` for REDQ-style training), then inverted via `h_inv`.
3. **Bellman target:** `target = reward + (1 - done) * gamma * min_next_q`, then mapped through `h`.
4. **Loss:** weighted MSE in transformed space (weights come from PER importance sampling when enabled): `q_total_loss = sum_i q_i_loss`.
5. **CQL (optional):** when `cql_alpha > 0`, an offline-RL penalty `cql_alpha * (logsumexp_a Q(s, a_rand) - Q(s, pi(s)))` is added to each critic loss (canonical residual recipe sets `cql_alpha=20`).
6. **Priority update:** when PER is active, the TD error for priority updates is the mean of the absolute errors across all critics.

## Actor update

The actor maximizes `Q_1(s, pi(s))` normalized by `(1 - gamma)` to keep it on a per-step scale:

```
norm_q          = (1 - gamma) * h_inv(Q1(s, pi(s)))
actor_loss      = -mean(norm_q)
```

In residual mode an optional `residual_action_l2` term is added to keep the residual head small.

## Target network updates

Polyak averaging of every critic/target pair + the actor target with `tau=0.005`:

```
target_param = tau * param + (1 - tau) * target_param
```

Applied every `target_network_frequency` (default 1) critic update steps.

## Training cadence

Updates trigger at **episode boundaries** (not every step), after `learning_starts` steps:

| Parameter | Default | Role |
|-----------|---------|------|
| `learning_starts` | 5000 | Steps before any training begins |
| `q_updates` | 1 | Critic update iterations per episode end |
| `actor_updates_per_iteration` | 1 | Actor update iterations per episode end |
| `batch_size` | 256 | Samples per update |

In the real-world script, the learner runs after each collected episode rather than per-step, matching the hardware's episodic collection cadence.

## Observation augmentation

When `use_last_action` is enabled, the previous action is concatenated to the observation before being fed to the actor:

```
policy_obs = cat([observation, last_action], dim=-1)
```

The critic always receives the raw observation (without last-action augmentation). This gives the actor access to its own recent output for smoother temporal behavior without increasing the critic's input dimensionality.

## Exploration noise

During collection, Gaussian noise is added to the deterministic actor output:

```
action = actor(obs) + N(0, exploration_noise)
```

`exploration_noise` defaults to `0.1`. This is separate from the target policy smoothing noise used in critic targets (`policy_noise=0.2`, `noise_clip=0.5`).

Primitive exploration (stand-still, directional, target-position) can override the noisy policy action for configurable multi-step windows. See [td3-primitives.md](../exploration/td3-primitives.md).

## Key hyperparameters

| Parameter | Default | Role |
|-----------|---------|------|
| `gamma` | 0.975 | Discount factor |
| `policy_lr` | 3e-4 | Actor learning rate (Adam) |
| `q_lr` | 1e-3 | Critic learning rate (Adam) |
| `q_weight_decay` | 1e-4 | Critic weight decay |
| `buffer_size` | 1e6 | Replay buffer capacity |
| `tau` | 0.005 | Polyak averaging coefficient |
| `h_transform_eps` | 1e-3 | Epsilon for h-transform invertibility |
| `num_critics` | 2 | Number of critics (twin TD3 with 2; REDQ-style ensemble for >2) |
| `cql_alpha` | 0.0 | CQL penalty weight (canonical residual recipe uses 20.0) |

## Related docs

- [Reward shaping](reward-shaping.md)
- [Network architecture](network-architecture.md) -- `TD3QNetwork`, `DeterministicAgent`, `ResidualMLPTrunk`
- [Replay and episodes](replay-and-episodes.md) -- buffer types, PER, episode staging
- [Checkpointing](checkpointing.md) -- save/resume schema
- [Exploration primitives](../exploration/td3-primitives.md)

# TD3 training algorithm

Core training loop for both simulation ([`td3_training.py`](../../../scripts/smooth_policy/amp_history/amp_training/td3/td3_training.py)) and real-world ([`async_td3_real.py`](../../../scripts/smooth_policy/amp_history/amp_training/td3/extras/async_td3_real.py)).

## Transformed Bellman targets (h / h-inverse)

Q-values are regressed in a **transformed space** using a symlog-style squashing function. This compresses large value magnitudes, stabilizing critic learning when rewards span multiple scales:

```
h(x)      = sign(x) * (sqrt(|x| + 1) - 1) + eps * x
h_inv(x)  = sign(x) * ( ((sqrt(1 + 4*eps*(|x| + 1 + eps)) - 1) / (2*eps))^2 - 1 )
```

`eps` (default `1e-3`) adds a small linear term that keeps `h` invertible everywhere. Both critics output values in the transformed space; Bellman targets are computed in original space, then mapped back through `h`.

**Code:** `h_transform` / `h_inverse` in `td3_training.py` (lines 72-81) and duplicated in `async_td3_real.py`.

## Dual-head critics

Each of the two critic networks ([`TD3DualHeadQNetwork`](../../../scripts/smooth_policy/amp_history/amp_training/td3/helper/dual_head_q.py)) has a shared residual trunk and two independent scalar output heads:

- **Task head** -- predicts transformed Q for the environment/task reward stream
- **Motion head** -- predicts transformed Q for the auxiliary motion reward stream

This dual-head design allows separate discount factors and reward scales while sharing representation:

| Parameter | Default | Role |
|-----------|---------|------|
| `task_gamma` | 0.975 | Discount factor for task rewards |
| `motion_gamma` | 0.8 | Discount factor for motion rewards (shorter horizon, faster credit assignment) |

See also: [network-architecture.md](network-architecture.md) for the `TD3DualHeadQNetwork` structure.

## Critic update

Standard TD3 twin-critic procedure, extended for dual heads:

1. **Target actions:** query the target actor with the next observation, add clipped Gaussian noise (`policy_noise=0.2`, `noise_clip=0.5`), clamp to action bounds.
2. **Min-of-two targets:** for each head, take the element-wise minimum across the two target critics (in transformed space), then invert via `h_inv`.
3. **Bellman targets:** computed in original space per head:
   - `target_task = task_reward + (1 - done) * task_gamma * min_next_task`
   - `target_motion = motion_reward + (1 - done) * motion_gamma * min_next_motion`
4. **Transform back:** `h(target_task)`, `h(target_motion)` become the regression targets.
5. **Loss:** weighted MSE in transformed space (weights come from PER importance sampling when enabled):
   - `q_total_loss = q1_task_loss + q2_task_loss + q1_motion_loss + q2_motion_loss`
6. **Priority update:** when PER is active, the TD error for priority updates is the mean of the absolute errors across all four head-critic combinations.

## Actor update

The actor maximizes a weighted combination of the two Q streams, each normalized by `(1 - gamma)` to put them on comparable scales:

```
norm_task   = (1 - task_gamma) * h_inv(Q1_task(s, pi(s)))
norm_motion = (1 - motion_gamma) * h_inv(Q1_motion(s, pi(s)))
actor_objective = task_reward_weight * norm_task + motion_reward_weight * norm_motion
actor_loss = -mean(actor_objective)
```

The `(1 - gamma)` normalization converts discounted sums to per-step-equivalent values, preventing the higher-gamma task stream from dominating the objective.

**Code:** actor loss block in `td3_training.py` (lines 1626-1642).

## Target network updates

Polyak averaging of all three network pairs (actor, qf1, qf2) with `tau=0.005`:

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

Primitive exploration (stand-still, directional, target-position, etc.) can override the noisy policy action for configurable multi-step windows. See [td3-primitives.md](../exploration/td3-primitives.md).

## Key hyperparameters

| Parameter | Default | Role |
|-----------|---------|------|
| `policy_lr` | 3e-4 | Actor learning rate (Adam) |
| `q_lr` | 1e-3 | Critic learning rate (Adam) |
| `q_weight_decay` | 1e-4 | Critic weight decay |
| `buffer_size` | 1e6 | Replay buffer capacity |
| `tau` | 0.005 | Polyak averaging coefficient |
| `h_transform_eps` | 1e-3 | Epsilon for h-transform invertibility |

## Related docs

- [Reward shaping](reward-shaping.md) -- task and motion reward composition
- [Network architecture](network-architecture.md) -- `TD3DualHeadQNetwork`, `DeterministicAgent`, `ResidualMLPTrunk`
- [Replay and episodes](replay-and-episodes.md) -- buffer types, PER, episode staging
- [Checkpointing](checkpointing.md) -- save/resume schema
- [Exploration primitives](../exploration/td3-primitives.md) -- primitive takeover behavior

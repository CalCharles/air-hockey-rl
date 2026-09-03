# TD3 Args Reference

Per-field documentation for the `Args` dataclass in [`scripts/td3/td3_training.py`](../../../scripts/td3/td3_training.py). For which configs to use, see [`td3-configs.md`](td3-configs.md). For the recipe-level discussion, see [`residual-rl-recipe.md`](residual-rl-recipe.md) and [`td3-algorithm.md`](td3-algorithm.md).

Args are accepted from a YAML via `--args-file` and from the CLI. CLI flags override the YAML.

---

## Run mode

| Field | Default | Notes |
|---|---|---|
| `eval_mode` | `False` | Evaluation-only mode — no exploration, no replay writes, no gradient updates. `total_timesteps` becomes the rollout horizon. |
| `total_timesteps` | `1_000_000` | Total env-steps to train (or to roll out, in eval mode). |
| `num_envs` | `1` | Vector-env count. Currently only `1` is supported. |

## TD3 core

| Field | Default | Notes |
|---|---|---|
| `buffer_size` | `1_000_000` | Unused leftover; see `success_buffer_size` / `failure_buffer_size`. |
| `gamma` | `0.975` | Discount factor. |
| `tau` | `0.005` | Polyak averaging coefficient. |
| `batch_size` | `256` | Minibatch size for critic + actor updates. |
| `learning_starts` | `5000` | Env-steps to collect before any gradient updates. |
| `policy_lr` | `3e-4` | Adam learning rate for the actor. |
| `q_lr` | `1e-3` | Adam learning rate for each critic. |
| `q_weight_decay` | `1e-4` | Adam weight decay on critic parameters. |
| `q_frequency` | `1` | (Vestigial — `q_updates` and `episode_finished`-gated cadence supersede.) |
| `q_updates` | `1` | Critic update steps per training cycle (once an episode finishes). |
| `policy_frequency` | `2` | TD3 delayed-policy-update interval (vestigial under current loop structure). |
| `target_network_frequency` | `1` | Polyak averaging happens every Nth completed critic update. |
| `actor_updates_per_iteration` | `1` | Actor update steps per training cycle. Load-bearing for residual RL (see [residual recipe](residual-rl-recipe.md)). |
| `exploration_noise` | `0.1` | Gaussian noise added to actions during rollouts. |
| `policy_noise` | `0.2` | Noise added to the target action when computing the Bellman target. |
| `noise_clip` | `0.5` | Clip range for `policy_noise`. |
| `h_transform_eps` | `1e-3` | Epsilon for the reward-rescaling transform `h`/`h_inverse`. |

## Critic ensemble (REDQ-style; Chen et al., ICLR 2021)

| Field | Default | Notes |
|---|---|---|
| `num_critics` | `2` | `2` reproduces vanilla TD3. `>2` enables Maxmin-N (min over all N targets). |
| `target_critic_subset_size` | `None` | When `num_critics > 2` and this is `M < N`, the target Q is a min over a random M-subset (REDQ-N-M). |

## Prioritized experience replay

| Field | Default | Notes |
|---|---|---|
| `per_enabled` | `True` | Toggles PER. When false, uniform replay sampling. |
| `per_alpha` | `0.6` | Priority exponent. |
| `per_beta_start` / `per_beta_end` | `0.4` / `1.0` | Importance-sampling exponent, annealed linearly. |
| `per_beta_anneal_steps` | `200_000` | Steps over which `per_beta` anneals from start to end. |
| `per_eps` | `1e-6` | Priority floor added to every sample. |
| `priority_age_decay` | `0.0` | Age-weighted PER: multiplies sample priorities by `exp(-priority_age_decay * age_in_slots)` before alpha-scaling. `age_in_slots = 0` for the most recently added transition, growing linearly with eviction order. Orthogonal to FIFO eviction and TD-error PER. Reasonable values: `1e-5` (half-life ≈ 70k slots), `1e-4` (≈7k), `1e-3` (≈700). `0.0` disables. |

## Replay buffer split + sampling mix

| Field | Default | Notes |
|---|---|---|
| `success_buffer_size` | `200_000` | Slots for high-return ("success") episodes. |
| `failure_buffer_size` | `800_000` | Slots for low-return ("failure") episodes. |
| `success_top_fraction` | `0.2` | An episode is "success" if its return ranks in the top fraction of `recent_episode_returns`. |
| `recent_episode_window_size` | `500` | Sliding window of episode returns used to compute the success threshold. |
| `critic_per_fraction` | `0.7` | Fraction of each critic minibatch drawn from PER. |
| `critic_uniform_fraction` | `0.3` | Fraction drawn uniformly. Must sum to 1.0 with `critic_per_fraction`. |
| `critic_success_sample_fraction` | `0.3` | Fraction of each critic minibatch drawn from the success buffer. |
| `critic_failure_sample_fraction` | `0.7` | Fraction drawn from the failure buffer. Must sum to 1.0 with `critic_success_sample_fraction`. |

## Primitive exploration takeover

When the policy hands off to a scripted primitive (stand-still, same-direction, y-aligned, target-position-directional). See [`td3-primitives.md`](../exploration/td3-primitives.md).

| Field | Default | Notes |
|---|---|---|
| `exploration_primitive_chance` | `0.05` | Steady-state per-step probability that a primitive takes over. |
| `exploration_primitive_chance_start` | `0.5` | Initial probability; anneals linearly to `exploration_primitive_chance`. |
| `exploration_primitive_chance_pre_learning_starts` | `None` | If set, overrides the chance while `global_step < learning_starts`. |
| `exploration_pre_learning_action_source` | `"random"` | `"random"` or `"policy"` actions during the pre-learning rollout phase. |
| `exploration_primitive_chance_anneal_steps` | `50_000` | Step count over which `chance_start → chance` anneals. |
| `exploration_primitive_steps` | `3` | Default takeover horizon (env-steps the primitive holds). |
| `exploration_primitive_weight_{stand_still,same_direction}` | `0.5, 0.5` | Steady-state primitive-selection weights. Each must be ≥ 0. |
| `exploration_primitive_weight_anneal_{stand_still,same_direction}` | `0.3, 0.7` | Annealing-phase weights (used while `global_step < exploration_primitive_chance_anneal_steps`). |
| `exploration_direction_y_component_weight` | `1.5` | Bias factor on the y-component for the same_direction legacy sampling path. |
| `exploration_action_delta_x` / `_delta_y` | `0.26` / `0.12` | Action box used to project same_direction simulator-space displacements. |
| `exploration_same_direction_{min,max}_angle_deg` | `None` | Optional simulator-space-range override for same_direction angle bounds. All four (min/max angle + min/max magnitude) must be set together. |
| `exploration_same_direction_{min,max}_magnitude` | `None` | Optional simulator-space-range override for same_direction magnitude bounds. |

## Checkpointing

| Field | Default | Notes |
|---|---|---|
| `checkpoint_interval` | `25_000` | Save a checkpoint every N env-steps. |
| `save_replay_buffer` | `True` | Whether to dump the success/failure buffers into the `training_state.pth` file. |
| `checkpoint_eval_async` | `True` | Run the per-checkpoint `evaluate_agent` in a CPU-only background subprocess (`scripts/td3/checkpoint_eval.py`) instead of blocking the loop. Output files are identical; see [training-throughput.md](training-throughput.md). |

## Paths and checkpoint loading

| Field | Default | Notes |
|---|---|---|
| `config` | `configs/new_juggle/sysid_best_params.yaml` | Air-hockey sim config. Canonical is `sysid_best_params_hist2.yaml` (see [`sim-env-configs.md`](sim-env-configs.md)). |
| `args_file` | `None` | YAML file to load as default `Args`. CLI flags override. |
| `model_path` | `None` | Path to a `model.pth` (actor-only) or `training_state.pth` (full state). |
| `full_checkpoint_load` | `"full_resume"` | How to interpret `model_path`: `"full_resume"` restores full runtime state; `"weights_only"` restores network weights only; `"residual"` loads the source actor as a frozen base and builds a fresh residual + fresh critic. |
| `log_parent_dir` | `None` | Override the auto-generated `runs/default_training/<task>/<run_name>_<timestamp>` log directory. |
| `run_name` | `"default"` | Sub-directory name for the run. |

## Residual RL

Active only when `full_checkpoint_load == "residual"`. See [`residual-rl-recipe.md`](residual-rl-recipe.md).

| Field | Default | Notes |
|---|---|---|
| `residual_scale` | `0.25` | Max magnitude of the residual action component. Combined action is clipped to env action bounds, so `residual_scale > 0` caps `|residual|_inf` via tanh. |
| `residual_weight_decay` | `0.0` | L2 weight decay on the residual actor's parameters (Adam `weight_decay`). `> 0` keeps the residual head close to zero even when the critic encourages large corrections — counteracts long-horizon drift at `residual_scale=0.15`. |

## CQL (Kumar et al., 2020)

| Field | Default | Notes |
|---|---|---|
| `cql_alpha` | `0.0` | Conservative-Q penalty coefficient. `> 0` adds `cql_alpha * (logsumexp_a Q(s,a) - Q(s, pi(s)))` to each critic's loss. The canonical fix for Q-overestimation drift (see [`residual-rl-recipe.md`](residual-rl-recipe.md) §big-gap recipe). |
| `cql_n_random` | `10` | Number of uniform `[-1,1]^act_dim` action samples used to approximate the logsumexp. |

## Runtime

| Field | Default | Notes |
|---|---|---|
| `device` | `"cuda:0"` | Torch device for the networks, replay buffers and updates. |
| `rollout_device` | `"cpu"` | Device that drives the env: batch-`num_envs` actor inference, exploration selector, trajectory staging. CPU is ~3× faster than the GPU for the 64-wide actor at batch 1 and avoids per-step syncs. |
| `use_cuda_graphs` | `True` | Capture each critic / actor update (replay sampling included) in a CUDA graph (`helper/td3_graphed_update.py`). GPU only; auto-disabled when `target_critic_subset_size < num_critics`. |
| `compile_update` | `True` | `torch.compile` the loss forward/backward inside the graphs (~30 % fewer kernels). Falls back to uncompiled graphs if compilation fails. One-time ~15 s. |
| `compile_rollout_actor` | `True` | `torch.compile` the CPU rollout actor (273 → 140 µs per step). Falls back to eager. |
| `torch_num_threads` | `1` | `torch.set_num_threads` for the process; rollout tensors are tiny. |
| `train_metrics_log_interval` | `20` | Training cycles between writes of the `losses/`, `debug/`, `replay/`, `charts/SPS` scalars. |
| `stats_log_interval` | `5000` | Env steps between rolling-window stat writes + the console line. |
| `seed` | `0` | RNG seed. |

## Network architecture

| Field | Default | Notes |
|---|---|---|
| `agent_hidden_layer_size` | `64` | Width of each residual block in the actor MLP. |
| `agent_num_hidden_layers` | `2` | Number of residual blocks in the actor. |
| `q_hidden_layer_size` | `128` | Width of each residual block in each critic MLP. |
| `q_num_hidden_layers` | `2` | Number of residual blocks per critic. |

## Policy observation

| Field | Default | Notes |
|---|---|---|
| `use_last_action_in_policy_state` | `False` | When `True`, the actor sees `concat(obs, last_action)` (32-dim for `hist2`). When `False`, just `obs` (30-dim). |

## Episode GIF recording

| Field | Default | Notes |
|---|---|---|
| `watch_ring_size` | `10` | Number of GIFs in the rotating `watch/` directory. |
| `watch_episode_interval` | `50` | Record one episode every N completed episodes. |
| `sample_gif_interval` | `10_000` | Also persist a recorded episode into `samples/` every N env-steps. |
| `sample_gif_max_storage_mb` | `50.0` | Storage cap on `samples/`; older files evicted FIFO. |

## Multi-env evaluation

Used only by the `td3_training_dr.py` wrapper. When `eval_param_seed` is `None`, behavior is unchanged. When set, the wrapper monkey-patches `evaluate_agent` to roll `eval_eps_per_env` episodes through each of `eval_n_envs` fixed seed-sampled environments and aggregate; per-env stats are dumped to `<ckpt_dir>/multi_env_eval.json`.

| Field | Default | Notes |
|---|---|---|
| `eval_param_seed` | `None` | Seed for sampling the eval-env parameters. |
| `eval_n_envs` | `1` | Number of distinct eval environments to sample. |
| `eval_eps_per_env` | `4` | Episodes per eval environment per checkpoint. |

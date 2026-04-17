# TD3 Simulator Config Files

Config files live in `scripts/smooth_policy/amp_history/configs/td3/`.
All are used with `scripts/smooth_policy/amp_history/amp_training/td3/td3_training.py`.

## Recommended default — `td3_recommended.yaml`

**New training runs should start from this config.** Distilled from the update-count, network-depth, and actor:Q-ratio ablations in [`td3-ablations-updates-and-depth.md`](td3-ablations-updates-and-depth.md).

Key choices and why:

| Arg | Value | Why |
|---|---|---|
| `agent_num_hidden_layers` | **2** | 2-layer nets give up only ~3 pts peak vs 3-layer and run ~20% faster; 5-layer is strictly dominated. |
| `q_num_hidden_layers` | **2** | Same as above. |
| `q_updates` | **25** | Halving updates from (50, 12) to (25, 6) preserves the peak (max ≈ 157) with +13% throughput. Below 15 total updates/episode, learning starves. |
| `actor_updates_per_iteration` | **6** | Holds actor:Q ratio ≈ 0.24, which clearly beats 0.07 / 0.48 / 2.10 on peak return (156 vs 129–141). |
| `total_timesteps` | **1_000_000** | 1M converges well; peaks typically land between 500k–700k. |
| `config` | `sysid_best_params.yaml` | Real-world-tuned physics. See [`environments/real-world/puck-system-id.md`](../environments/real-world/puck-system-id.md) and [`teleop-system-id.md`](../environments/real-world/teleop-system-id.md). |
| `enable_puck_delay_interpolation` | `true` | Matches the real-world puck-delay behavior our sysid was done against. |
| `target_network_frequency` | 10 | Inherited from `td3_no_alignment.yaml`; not independently re-ablated. |

All other fields (reward weights, PER, exploration primitives) are inherited from `td3_no_alignment.yaml` unchanged — the alignment reward terms are zero, motion terms use the standard values.

Launch:

```bash
python scripts/smooth_policy/amp_history/amp_training/td3/td3_training.py \
  --args-file scripts/smooth_policy/amp_history/configs/td3/td3_recommended.yaml \
  --run-name my_run --device cuda:0
```

CLI flags still override the config. To vary e.g. network depth or update count for a one-off, pass `--agent-num-hidden-layers`, `--q-updates`, etc.

## Current configs

### `td3_standard.yaml` — Baseline
The reference config. 1M timesteps, `q_updates: 100`, `target_network_frequency: 20`, `actor_updates_per_iteration: 50`. Motion reward weight is `0.0` (task-only) but all motion component weights are set to `0.5`. Uses sim config `pid_noise_constant_upper_half_custom_sim_params.yaml`.

### `td3_no_alignment.yaml` — No temporal/axis alignment
350K timesteps, more aggressive critic updates (`q_updates: 200`, `target_network_frequency: 10`). Zeros out `temporal_alignment_reward_weight` and `axis_alignment_reward_weight`; keeps `stand_still: 0.1`, `velocity: 0.5`, `jerk: 0.5`. Same sim config as standard. Purpose: ablate alignment reward terms.

### `td3_no_alignment_real_world_mirror.yaml` — Sim mirror of real-world online training
Very short run (120K steps), minimal update budget (`q_updates: 20`, `actor_updates_per_iteration: 5`). Exploration is fixed with no annealing (`anneal_steps: 0`, constant `primitive_chance: 0.025`), no warm-start policy takeover. Adds fine-grained angle/magnitude bounds on exploration primitives. Uses `sim_real_world_adaptation.yaml` sim config and `full_checkpoint_load: fine_tune`. Purpose: mirror the online real-world training loop in simulation.

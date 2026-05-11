# TD3 Simulator Config Files

Config files live in `scripts/smooth_policy/amp_history/configs/td3/`.
All are used with `scripts/smooth_policy/amp_history/amp_training/td3/td3_training.py`.

## Recommended default — `td3_recommended_top50_hist2.yaml`

**New training runs should start from this config.** Distilled from the update-count, network-depth, and actor:Q-ratio ablations in [`td3-ablations-updates-and-depth.md`](td3-ablations-updates-and-depth.md) and the exploration sweep in [`td3-exploration-ablations.md`](td3-exploration-ablations.md).

> **Note (2026-05-04):** The original recommended default was `td3_recommended.yaml` (hist_len=4 via `sysid_best_params_hist4.yaml`, `success_top_fraction: 0.2`). It is preserved at `td3/legacy/td3_recommended.yaml` along with the `new_juggle/legacy/sysid_best_params_hist{3,4,5}.yaml` sim variants for reproducing past ablations. The new active default keeps every other knob the same and changes:
> - `config:` → `sysid_best_params_hist2.yaml` (hist_len=2 PID-target filter — matches `latest_models/canonical/hist2_motion0/config.yaml` and the real-env default in `airhockey/sims/airhockey_box2d.py`)
> - `success_top_fraction: 0.5` (the "top50" in the file name; from-scratch comparison written up in [`residual-rl-recipe.md`](residual-rl-recipe.md))

Key choices and why:

| Arg | Value | Why |
|---|---|---|
| `agent_num_hidden_layers` | **2** | 2-layer nets give up only ~3 pts peak vs 3-layer and run ~20% faster; 5-layer is strictly dominated. |
| `q_num_hidden_layers` | **2** | Same as above. |
| `q_updates` | **25** | Halving updates from (50, 12) to (25, 6) preserves the peak (max ≈ 157) with +13% throughput. Below 15 total updates/episode, learning starves. |
| `actor_updates_per_iteration` | **6** | Holds actor:Q ratio ≈ 0.24, which clearly beats 0.07 / 0.48 / 2.10 on peak return (156 vs 129–141). |
| `total_timesteps` | **1_000_000** | 1M converges well; peaks typically land between 500k–700k. |
| `config` | **`sysid_best_params_hist2.yaml`** | Real-world-tuned physics **with `hist_len=2`** 2-timestep low-pass filter on the PID target (see `_filter_update` in `airhockey/sims/airhockey_box2d.py` — this is the env's default `hist_len`). Base sysid values documented in [`puck-system-id.md`](../environments/real-world/puck-system-id.md) and [`teleop-system-id.md`](../environments/real-world/teleop-system-id.md). |
| `success_top_fraction` | **0.5** | "top50" PER mix; from-scratch comparison in [`residual-rl-recipe.md`](residual-rl-recipe.md#top50-from-scratch-ablation). |
| `enable_puck_delay_interpolation` | `true` | Matches the real-world puck-delay behavior our sysid was done against. |
| `exploration_primitive_chance_pre_learning_starts` | **`null`** | E4 ablation: bootstrap forcing (`=1.0`) was actively harmful. Leaving it `null` falls back to the annealing schedule (`chance_start=0.15`) during pre-learning, gaining +15 pts ret@500k. |
| `exploration_primitive_weight_policy_takeover` | **`0.0`** | E2 ablation: no external warmstart policy. Removes dependence on a demo checkpoint; marginal steady-state improvement at the cost of some early-learning speed. |
| `exploration_primitive_weight_anneal_policy_takeover` | **`0.0`** | Same as above during the annealing phase. |
| `exploration_policy_takeover_enabled` | **`false`** | No demo model loaded. |
| `target_network_frequency` | 10 | Inherited from `td3_no_alignment.yaml`; not independently re-ablated. |

All other fields (reward weights, PER, non-warmstart exploration primitives) are inherited from `td3_no_alignment.yaml` unchanged — the alignment reward terms are zero, motion terms use the standard values.

Launch:

```bash
python scripts/smooth_policy/amp_history/amp_training/td3/td3_training.py \
  --args-file scripts/smooth_policy/amp_history/configs/td3/td3_recommended_top50_hist2.yaml \
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

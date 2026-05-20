# TD3 Training Args (`configs/td3/`)

TD3 args YAMLs at [`configs/td3/`](../../../configs/td3/). Source-only training is launched via `scripts/td3/td3_training.py`; for **sim2sim / sim2real transfer**, the canonical training command is `scripts/td3/td3_training_dr.py` with the env-parameter-randomization config in [`configs/td3/zeroshot_paramrand/`](../../../configs/td3/zeroshot_paramrand/) (see [`sim2sim.md`](sim2sim.md) for the strategy overview).

## Canonical env-randomization training (sim2real source) — `zeroshot_paramrand/td3_paramrand_pm25.yaml`

The recommended source-policy training for any policy that will transfer to a perturbed sim or the real robot. Wraps `td3_training` via `td3_training_dr.py` to add per-reset randomization of paddle_density / puck_damping / gravity (±25 % of sysid) plus a 5-env eval overlay at every checkpoint. Sim config: [`configs/new_juggle/zeroshot_ablations/sim_paramrand_pm25.yaml`](../../../configs/new_juggle/zeroshot_ablations/sim_paramrand_pm25.yaml).

```bash
.venv/bin/python -m scripts.td3.td3_training_dr \
  --args-file configs/td3/zeroshot_paramrand/td3_paramrand_pm25.yaml
```

This replaces the earlier "engineered randomization" source-training recipe (`td3_recommended_top50_hist2.yaml` is still kept for source-only sim runs and ablations, but should not be used for new sim2sim / sim2real source policies).

## Canonical source-sim-only training — `td3_recommended_top50_hist2.yaml`

The canonical config for new sim TD3 runs. Distilled from the update-count, network-depth, and actor:Q-ratio ablations in [`td3-ablations-updates-and-depth.md`](td3-ablations-updates-and-depth.md) and the exploration sweep in [`td3-exploration-ablations.md`](td3-exploration-ablations.md).

Key choices and why:

| Arg | Value | Why |
|-----|-------|-----|
| `config` | [`configs/new_juggle/sysid_best_params_hist2.yaml`](../../../configs/new_juggle/sysid_best_params_hist2.yaml) | Real-world-tuned physics with `hist_len=2` 2-timestep low-pass filter on the PID target. |
| `agent_num_hidden_layers` | **2** | 2-layer nets give up only ~3 pts peak vs 3-layer and run ~20% faster; 5-layer is strictly dominated. |
| `q_num_hidden_layers` | **2** | Same as above. |
| `q_updates` | **25** | Halving updates from (50, 12) to (25, 6) preserves the peak (max ≈ 157) with +13% throughput. Below 15 total updates/episode, learning starves. |
| `actor_updates_per_iteration` | **6** | Holds actor:Q ratio ≈ 0.24, which clearly beats 0.07 / 0.48 / 2.10 on peak return (156 vs 129–141). |
| `target_network_frequency` | 10 | Standard. |
| `total_timesteps` | **1_000_000** | Peaks typically land between 500k–700k. |
| `success_top_fraction` | **0.5** | Median-split PER mix; from-scratch ablation in [`residual-rl-recipe.md`](residual-rl-recipe.md). |
| `enable_puck_delay_interpolation` | `true` | Matches the real-world puck-delay behavior the sysid was tuned against. |
| `exploration_primitive_chance_pre_learning_starts` | `null` | Bootstrap-forcing (`=1.0`) was actively harmful; leaving null uses the annealing schedule. |

Launch:

```bash
.venv/bin/python -m scripts.td3.td3_training \
  --args-file configs/td3/td3_recommended_top50_hist2.yaml \
  --run-name my_run \
  --num-envs 1
```

CLI flags still override the YAML; pass `--q-updates`, `--actor-updates-per-iteration`, `--agent-num-hidden-layers`, etc., for one-off ablations.

## Sim2sim residual recipes

See [`residual-rl-recipe.md`](residual-rl-recipe.md) for the full canonical big-gap and small-gap recipes. Four configs ship:

| Config | Target | actor_updates |
|--------|--------|---------------|
| [`configs/td3/sim2sim/warp075_p30_residual/phaseC_actor2_1M.yaml`](../../../configs/td3/sim2sim/warp075_p30_residual/phaseC_actor2_1M.yaml) | canonical big-gap (paddle −30%, warp 0.075) | **2** |
| [`configs/td3/sim2sim/warp075_p30_residual/phaseD_actor2_p10_1M.yaml`](../../../configs/td3/sim2sim/warp075_p30_residual/phaseD_actor2_p10_1M.yaml) | mild paddle (−10%) | 2 |
| [`configs/td3/sim2sim/warp075_p30_residual/phaseD_actor4_w10_1M.yaml`](../../../configs/td3/sim2sim/warp075_p30_residual/phaseD_actor4_w10_1M.yaml) | harder warp (0.10) | **4** |
| [`configs/td3/sim2sim/td3_sim2sim_residual.yaml`](../../../configs/td3/sim2sim/td3_sim2sim_residual.yaml) | small-gap (<10% zs drop) | default |

## Real-world configs

See [`td3-real-world-configs.md`](td3-real-world-configs.md). The canonical real-world args YAML is [`configs/td3_real_world/td3_residual.yaml`](../../../configs/td3_real_world/td3_residual.yaml).

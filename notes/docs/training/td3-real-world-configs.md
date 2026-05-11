# TD3 Real-World Config Files

Config files live in `configs/td3_real_world/`. Used for training, evaluation, teleoperation, and the user study on the real UR5 robot.

`async_td3_real.py` takes **two** YAMLs: `--train-args` (architecture, ensemble) and `--args-file` (online behavior). The residual configs split this way deliberately — the train-args YAML is shared between the CQL and no-CQL recipes (architecture is identical), only the args-file differs.

## Residual fine-tuning (canonical real-robot training path)

These are the live training configs for sim2real residual fine-tuning. Read [`residual-rl-recipe.md`](residual-rl-recipe.md) before launching either.

> **Source-policy training note (2026-05-11 onward).** The sim source policy fed into these real-world residual configs should be trained with **environment-parameter randomization** (paddle_density / puck_damping / gravity DR'd per-reset) — see [`sim2sim.md`](sim2sim.md). The previous engineered-randomization stack has been removed from the env, so old source policies (e.g. `latest_models/canonical/hist2_motion0_v2/`) reflect a deprecated regime and should be retrained for new real-world deployments.

Real-world entrypoints all accept `--args-file <this yaml>`:
- Training: `scripts/td3/extras/async_td3_real.py`
- Frozen-policy evaluation: `scripts/td3/extras/async_td3_real_eval.py`
- Human-baseline teleop / user study: `scripts/td3/extras/async_td3_real_teleop_eval.py`
- Reset-policy training variant: `scripts/td3/extras/async_td3_real_reset_policy.py`

### `td3_residual_cql.yaml` — Canonical big-gap residual + CQL (default)
The 2026-05-08 winner. v27 Maxmin-5 base + `cql_alpha: 20.0` + `actor_updates_per_iteration: 2`. CQL penalty (Conservative-Q on the task head) is the load-bearing addition over v27 — pushes Q down on OOD residual actions and up on the current policy action. `cql_n_random: 10`. Everything else (success_top_fraction=0.15, residual_scale=0.15, no exploration, no BC anchor, q_weight_decay=1e-3, q_updates=4) matches v27. Ships with `learning_starts_fresh_steps: 2000` and an empty warm-start; pass `--warm-start-hdf5-dirs <prior-real-run>/episode_hdf5 --learning-starts-fresh-steps 0` to seed the buffer with prior-launch HDF5s instead. Pair with `td3_residual_train_args.yaml`.

### `td3_residual.yaml` — No-CQL baseline / regression test (v27 Maxmin-5)
Pre-CQL canonical (v27). Bit-identical critic kernel to `td3_residual_cql.yaml` when its `cql_alpha = 0` — kept on disk so v27-vs-CQL comparisons stay reproducible and so launches that explicitly want zero conservatism (e.g. small-gap residual where CQL is unnecessary) have a clean starting point. Same architecture file (`td3_residual_train_args.yaml`); same launch / resume flag set; only the args-file path differs.

### `td3_residual_train_args.yaml` — Shared architecture + ensemble spec
`--train-args` companion for both residual configs. Carries `agent_hidden_layer_size: 64`, `agent_num_hidden_layers: 2`, `q_*_layers: 2`, `use_last_action_in_policy_state: true`, `num_critics: 5`, `target_critic_subset_size: null` (Maxmin-5 — set to 2 for REDQ-5-2). Architecture must match the source actor in `model_path`; if your base is the legacy 5-layer model, flip both `num_hidden_layers` to 5 here AND in the args-file's mirrored block.

## Non-residual policy eval

### `td3_online.yaml` — Online-behavior defaults for a non-residual policy
Used when running a sim-pretrained policy on the real robot **without** the residual head — typically frozen-policy evaluation of a pre-residual checkpoint via `async_td3_real_eval.py`. Carries the real-world TD3 defaults: low LRs (`policy_lr: 5e-5`, `q_lr: 1e-4`), `q_updates: 20`, `min_replay_size_before_learning: 0`, primitive exploration at `chance: 0.025`. For eval use set `--min-replay-size-before-learning 999999999` to disable gradient updates. Points at `configs/real_configs/rollout_td3_config.yaml` for the rollout config.

## Rollout configs (referenced from the args YAMLs above)

In `configs/real_configs/`:
- `rollout_config_residual.yaml` — referenced by `td3_residual.yaml` and `td3_residual_cql.yaml`.
- `rollout_td3_config.yaml` — referenced by `td3_online.yaml`; generic real-world rollout (task `puck_juggle_upper_half_reward`, `simulator: real`).
- `rollout_config.yaml` — alternate real-world rollout config.
- `mouse_config.yaml` — mouse-paddle teleop config (used by the teleop eval entrypoint).

> **Note on warm-start path layout.** `extras/async_td3_real.py` writes per-episode artifacts under `<data_root_dir>/data_<YYYYMMDD-HHMMSS>/{episode_hdf5,reset_hdf5,episode_gifs,episode_camera_videos}/` (see [`td3-async-replay.md`](../environments/real-world/td3-async-replay.md#launch-commands)). For fresh runs point `warm_start_hdf5_dirs` at the new nested location — the loader is recursive when `warm_start_hdf5_recursive: true`, so pointing at `real_runs/online_run/` works.

## Key axes
- **CQL on/off** (residual fine-tuning): `td3_residual_cql` (default — canonical big-gap) vs. `td3_residual` (no-CQL regression baseline). Same train-args.
- **Residual vs. non-residual eval**: `td3_residual*` (residual head on top of frozen base) vs. `td3_online` (full policy, no residual structure).

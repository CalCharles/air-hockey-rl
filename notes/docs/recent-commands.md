# Recent Training Commands

Commands used recently for launching training runs. Add new entries at the top.

---

## 2026-04-20 — Real robot eval-only run of `latest_model/hist3_motion0`

Runs the hist3 motion-collision-ablation checkpoint on hardware in eval-only mode (no gradient updates, no replay warm-start, no checkpoint writes). Exercises the new two-args-file convention introduced by the rollout refactor: `--train-args` for architecture, `--args-file` for online-behavior defaults.

Key differences from the stock config:

1. **Matches training `hist_len`.** hist3_motion0 was trained with `hist_len: 3` (see its `sysid_best_params_hist3.yaml`). The stock `configs/real_configs/rollout_td3_config.yaml` doesn't set `hist_len` and the real env defaults to 2, giving a wrong obs dim → state-dict load failure. Use the committed `configs/real_configs/rollout_td3_config_hist3.yaml` variant instead — it's a verbatim copy of the stock rollout config with a single added line `hist_len: 3` under `simulator_params`. For other history lengths, make a similar variant.

2. **Use the training run's args.yaml as `--train-args`.** `latest_model/hist3_motion0/args.yaml` encodes the architecture (`agent_hidden_layer_size: 64`, `agent_num_hidden_layers: 2`, `action_scale: 1.0`, `use_last_action_in_policy_state: true`, Q-head sizes). `--args-file` still points at the online-behavior YAML.

```bash
python -m scripts.smooth_policy.amp_history.amp_training.td3.extras.async_td3_real \
  --config configs/real_configs/rollout_td3_config_hist3.yaml \
  --model-path latest_model/hist3_motion0/training_state.pth \
  --train-args latest_model/hist3_motion0/args.yaml \
  --args-file scripts/smooth_policy/amp_history/configs/td3_real_world/td3_online.yaml \
  --collector-device cpu \
  --learner-device cuda:0 \
  --episode-artifact-dir real_runs/online_run/episode_hdf5 \
  --episode-gif-dir real_runs/online_run/episode_gifs \
  --reset-artifact-dir real_runs/online_run/reset_hdf5 \
  --min-replay-size-before-learning 999999999 \
  --no-enable-periodic-checkpointing \
  --no-load-replay-from-checkpoint \
  --warm-start-hdf5-dirs
```

Expect `[args_file] ignored unsupported keys: ...` to list td3_training-only fields from `td3_online.yaml` that aren't part of the async `Args` schema — this is the canonical-names-only behavior from the refactor and is expected. The `[train_args]` line above it confirms the architecture that got wired into the actor/critic.

For other checkpoints, swap the three `latest_model/hist3_motion0/...` paths (and the matching hist_len in the rollout config if the checkpoint was trained with a different history length).

---

## 2026-04-04 — Real robot online learning (no learning, data collection only)

Runs `async_td3_real` on the physical robot in data-collection-only mode (`--min-replay-size-before-learning 999999999` effectively disables gradient updates). Artifacts (HDF5 episodes, GIFs, reset data) are saved under `real_runs/online_run/`.

> For the **collect + train** and **resume training** variants, see [td3-async-replay → Launch commands](environments/real-world/td3-async-replay.md#launch-commands).

**Heavy model (checkpoint 100k) — no latency profiling:**

```bash
python -m scripts.smooth_policy.amp_history.amp_training.td3.extras.async_td3_real \
  --config configs/real_configs/rollout_td3_config.yaml \
  --model-path ex_model/heavy_td3_model/checkpoint_100000/training_state.pth \
  --train-args ex_model/heavy_td3_model/checkpoint_100000/args.yaml \
  --args-file scripts/smooth_policy/amp_history/configs/td3_real_world/td3_online.yaml \
  --collector-device cpu \
  --learner-device cuda:0 \
  --episode-artifact-dir real_runs/online_run/episode_hdf5 \
  --episode-gif-dir real_runs/online_run/episode_gifs \
  --reset-artifact-dir real_runs/online_run/reset_hdf5 \
  --min-replay-size-before-learning 999999999 \
  --no-enable-periodic-checkpointing \
  --no-load-replay-from-checkpoint
```

**New model (checkpoint 325k) — with latency profiling:**

Same as above but uses the newer model and enables per-step latency measurement, writing profiles to `real_runs/online_run/latency/`.

```bash
python -m scripts.smooth_policy.amp_history.amp_training.td3.extras.async_td3_real \
  --config configs/real_configs/rollout_td3_config.yaml \
  --model-path ex_model/new_td3_model/checkpoint_325000/training_state.pth \
  --train-args ex_model/new_td3_model/checkpoint_325000/args.yaml \
  --args-file scripts/smooth_policy/amp_history/configs/td3_real_world/td3_online.yaml \
  --collector-device cpu \
  --learner-device cuda:0 \
  --episode-artifact-dir real_runs/online_run/episode_hdf5 \
  --episode-gif-dir real_runs/online_run/episode_gifs \
  --reset-artifact-dir real_runs/online_run/reset_hdf5 \
  --enable-latency-profiling \
  --latency-profile-output-dir real_runs/online_run/latency \
  --min-replay-size-before-learning 999999999 \
  --no-enable-periodic-checkpointing \
  --no-load-replay-from-checkpoint
```

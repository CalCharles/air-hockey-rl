# Recent Training Commands

Commands used recently for launching training runs. Add new entries at the top.

---

## 2026-04-04 — Real robot online learning (no learning, data collection only)

Runs `async_td3_real` on the physical robot in data-collection-only mode (`--min-replay-size-before-learning 999999999` effectively disables gradient updates). Artifacts (HDF5 episodes, GIFs, reset data) are saved under `real_runs/online_run/`.

> For the **collect + train** and **resume training** variants, see [td3-async-replay → Launch commands](environments/real-world/td3-async-replay.md#launch-commands).

**Heavy model (checkpoint 100k) — no latency profiling:**

```bash
python -m scripts.smooth_policy.amp_history.amp_training.td3.extras.async_td3_real \
  --config configs/real_configs/rollout_td3_config.yaml \
  --model-path ex_model/heavy_td3_model/checkpoint_100000/training_state.pth \
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

# Recent Training Commands

Commands used recently for launching training runs. Add new entries at the top.

---

## 2026-04-04 — Real robot online learning (no learning, data collection only)

Runs `async_td3_real` on the physical robot in data-collection-only mode (`--min-replay-size-before-learning 999999999` effectively disables gradient updates). Artifacts (HDF5 episodes, GIFs, reset data) are saved under `real_runs/online_run/`.

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

## 2026-04-04 — Sim TD3 training, heavy config, motion weight 0.01

Sim training run using the heavy config (heavier puck/paddle) with a low motion reward weight (0.01). No alignment reward. Logs to `runs/td3/updated_training/motion_weight001_heavy`.

```bash
python scripts/smooth_policy/amp_history/amp_training/td3/td3_training.py \
  --args-file scripts/smooth_policy/amp_history/configs/td3/td3_no_alignment_heavy.yaml \
  --motion-reward-weight 0.01 \
  --device "cuda:0" \
  --log-parent-dir runs/td3/updated_training/motion_weight001_heavy \
  --run-name td3_no_align_mw001_heavy
```

---

## 2026-04-04 — TD3 fixed density + force attenuation

Single run with paddle/puck density fixed at 3000, motion weight 0.01, puck delay interpolation, and stochastic force attenuation (30% chance of 0.25–0.75x force scale).

```bash
python scripts/smooth_policy/amp_history/amp_training/td3/td3_training.py \
  --args-file scripts/smooth_policy/amp_history/configs/td3/td3_no_alignment_heavy.yaml \
  --motion-reward-weight 0.01 \
  --paddle-density 3000 \
  --puck-density 3000 \
  --enable-puck-delay-interpolation True \
  --enable-action-force-attenuation True \
  --action-force-attenuation-prob 0.30 \
  --action-force-attenuation-min 0.25 \
  --action-force-attenuation-max 0.75 \
  --device cuda:0 \
  --log-parent-dir runs/td3/force_attenuation \
  --run-name mw001_d3000_fattn
```

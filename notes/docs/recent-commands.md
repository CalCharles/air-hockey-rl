# Canonical training commands

Quick reference for the current canonical workflows. The trainer accepts CLI overrides for any field in the args YAML; the snippets below show the minimum required flags.

## Sim TD3 training — for sim2sim / sim2real transfer (canonical)

Source-policy training with **environment-parameter randomization** (paddle_density / puck_damping / gravity DR'd per-reset). Use this for any policy that will need to transfer to a perturbed sim or the real robot.

```bash
.venv/bin/python -m scripts.td3.td3_training_dr \
  --args-file configs/td3/zeroshot_paramrand/td3_paramrand_pm25.yaml
```

See [`training/sim2sim.md`](training/sim2sim.md) for the strategy overview. The earlier "engineered randomization" approach was deprecated and its mechanisms removed from the env on 2026-05-11.

## Sim TD3 training — source-sim only (ablations, no transfer)

```bash
.venv/bin/python -m scripts.td3.td3_training \
  --args-file configs/td3/td3_recommended_top50_hist2.yaml \
  --run-name my_run \
  --num-envs 1
```

`--num-envs 1` is required (the trainer is single-env-collection only). Use this for source-sim ablations or any run that does not need to transfer.

## Sim2sim residual fine-tune

```bash
.venv/bin/python -m scripts.td3.td3_training \
  --args-file configs/td3/sim2sim/warp075_p30_residual/phaseC_actor2_1M.yaml \
  --num-envs 1
```

Before launching, edit the recipe YAML's `config:`, `model_path:`, `log_parent_dir:`, `run_name:`, and `seed:`. See [`training/residual-rl-recipe.md`](training/residual-rl-recipe.md) for recipe selection.

## Real-robot residual fine-tune

```bash
python -m scripts.td3.extras.async_td3_real \
  --config configs/real_configs/rollout_config_residual.yaml \
  --args-file configs/td3_real_world/td3_residual.yaml \
  --model-path <path_to_training_state.pth> \
  --train-args <path_to_args.yaml> \
  --collector-device cpu \
  --learner-device cuda:0 \
  --data-root-dir real_runs/online_run
```

`--data-root-dir` is the single root for collected per-episode artifacts. The script creates `<data_root_dir>/data_<YYYYMMDD-HHMMSS>/{episode_hdf5,reset_hdf5,episode_gifs,episode_camera_videos}/` at startup.

## Real-robot frozen-policy eval

Frozen actor against the real robot — no learner / replay / checkpointing. `--agent` selects how the actor is built (default `td3`); the env config's `task:` selects the per-episode metrics via [`real_task_eval_hooks.py`](../../scripts/td3/helper/real_task_eval_hooks.py). See [`training/real-world-eval-pipeline.md`](training/real-world-eval-pipeline.md).

### TD3 (canonical juggle)

```bash
python -m scripts.td3.extras.async_td3_real_eval \
  --agent td3 \
  --config configs/real_configs/rollout_config_residual.yaml \
  --args-file configs/td3_real_world/td3_residual.yaml \
  --model-path <path_to_training_state.pth> \
  --train-args <path_to_args.yaml> \
  --collector-device cpu \
  --learner-device cuda:0 \
  --data-root-dir real_runs/eval_run
```

### SGCRL on `puck_goal_position`

```bash
python -m scripts.td3.extras.async_td3_real_eval \
  --agent sgcrl \
  --config configs/gcrl/gcrl.yaml \
  --model-path gcrl/03500032_sgcrl_AirHockeyPuckGoalPosition-v0.pkl \
  --collector-device cpu \
  --eval-episodes 20
```

`--train-args` / `--args-file` are not required — the SGCRL architecture lives inside the `.pkl` and the policy-state contract is synthesized.

Emits `eval_summary.json` + `eval_per_episode.jsonl` (schema varies by task hook).

## Human-baseline teleop eval (user study)

```bash
python -m scripts.td3.extras.async_td3_real_teleop_eval \
  --config configs/real_configs/mouse_config.yaml \
  --args-file configs/td3_real_world/td3_residual.yaml \
  --data-root-dir real_runs/teleop_eval
```

Mouse-controlled paddle with the same task / termination / juggle counter / output schema as the frozen-policy eval. See [`training/teleop-eval-baseline.md`](training/teleop-eval-baseline.md).

## Resume a real-world run from a checkpoint

```bash
python -m scripts.td3.extras.async_td3_real \
  --config configs/real_configs/rollout_config_residual.yaml \
  --args-file configs/td3_real_world/td3_residual.yaml \
  --model-path real_runs/checkpoints/default/checkpoint_step_<step>/training_state.pth \
  --train-args real_runs/checkpoints/default/checkpoint_step_<step>/args.yaml \
  --collector-device cpu \
  --learner-device cuda:0 \
  --data-root-dir real_runs/online_run \
  --load-replay-from-checkpoint \
  --include-non-vital-training-state-fields
```

`include_non_vital_training_state_fields: true` is needed to preserve RNG / optimizer state across the resume.

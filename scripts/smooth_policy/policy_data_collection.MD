## Policy Data Collection

Use `collect_policy_data.py` to run a trained smooth policy and export trajectory / per-step diagnostics.

### Command

```bash
python scripts/smooth_policy/collect_policy_data.py \
  --model runs/amp_with_actions/ppo/test_with_history_scale_fixed/runr1/checkpoint_300/model.pth \
  --config-path runs/amp_with_actions/ppo/test_with_history_scale_fixed/runr1/config.yaml \
  --save-dir runs/policy_data_collection/example_run \
  --num-episodes 2 \
  --total-timesteps 1000 \
  --agent-hidden-size 64 \
  --device cpu
```

Stop behavior is **whichever comes first** between `--num-episodes` and `--total-timesteps`.
At least one of those limits must be provided.

### Exported artifacts

- `per_timestep.csv`
  - One row per environment step
  - Includes:
    - Raw trajectory vectors (`obs_*`, `action_*`)
    - `global_timestep`, `episode_index`, `episode_step`
    - `reward`, `terminated`, `truncated`
    - Paddle: position / velocity / acceleration
    - Puck: position / velocity / acceleration
    - Target position (`target_pos_x`, `target_pos_y`), computed as `paddle_position + action`

- `trajectory.npz`
  - Compact arrays for the same trajectory data:
    - `observations`, `actions`, `rewards`, `terminated`, `truncated`
    - `episode_index`, `episode_step`, `global_timestep`
    - `paddle_position`, `target_position`, `paddle_velocity`, `paddle_acceleration`
    - `puck_position`, `puck_velocity`, `puck_acceleration`

- `metadata.yaml`
  - Input paths, requested limits, action scale loaded from model, collected totals, tensor dimensions, and output filenames

- `model_used.pth`, `config_used.yaml`
  - Exact copies of model/config used for the data run

- `example_episode.gif`
  - One rendered episode capped at 250 timesteps

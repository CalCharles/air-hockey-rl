# Air Hockey Reinforcement Learning Environment

This contains an air hockey simulation environment powered by Box2D. It is fast (C++ back-end), capable of self-play, 1v1 play, and easy goal-conditioned reinforcement learning, resulting in a rich testbed for various algorithms.


Policy Trained for Upward Puck Velocity |  Goal-Conditioned RL
:-------------------------:|:-------------------------:
![](assets/puck_vel.gif)  |  ![](assets/puck_goal_pos.gif)

## Installation

### Using uv
```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Option A: sync with lock file
```bash
# Create virtual environment and sync dependencies from lock file
uv sync
# For training dependencies
uv sync --extra train
```

#### Option B: Install directly
```bash
# create uv virtual environment and activate
uv venv
source .venv/bin/activate

# Install the package in development mode
uv pip install -e .

# Or if you need training too:
uv pip install -e ".[train]"
```

### Using pip (legacy)
```bash
# Install with training dependencies
pip install -e .[train]

# Or just the base package
pip install -e .
```


## Other

- Project notes and formal docs (architecture, Cursor rule mirrors): [`notes/docs/index.md`](notes/docs/index.md)

#### Having this issue?
AttributeError: 'MjRenderContextOffscreen' object has no attribute 'con'
`echo 'export MUJOCO_GL="glx"' >> ~/.bashrc`
`source ~/.bashrc`

## How to Run
Most of the files use a configuration file (--cfg cmd argument), but is defaulted to one from `configs/`. Please see there to tune parameters for various scripts.
#### What the files do
- `airhockey2d.py`: base gym environment for air hockey
- `render.py`: renders the air hockey environment
- `train.py`: trains an agent via stable-baselines3 PPO.

Legacy:
- `demonstrate.py`: user plays a self-play air hockey environment using keyboard
- `play_trained_agent`: run after training, you can play against the trained agent

## Running on the Physical UR5
- Boot up the robot through the touchpad
    - Press physical power button
    - Press red power on touchpad in bottom left corner
    - power on the robot with touch button in the middle
    - open program "external_control.urp"
- run desired script in scripts/real
    - ex: python scripts/real/teleoperate.py --cfg configs/baseline_configs/puck_vel_real.yaml
- When prompted in the terminal, run the program using the play button in the bottom middle of the touchpad
- follow prompts on the terminal. Hold 'q' to end trajectories

### TD3 Real-World Commands

All commands below use `async_td3_real`, which handles collection, resets, and (optionally) training. Two YAML files are required:

- `--train-args <train_run>/args.yaml` — the **training** run's args.yaml. Supplies architecture only (`agent_hidden_layer_size`, `agent_num_hidden_layers`, `q_hidden_layer_size`, `q_num_hidden_layers`, `action_scale`, `use_last_action_in_policy_state`) so the rebuilt actor/critic layers match the saved checkpoint exactly. Architecture is not CLI-overridable.
- `--args-file <td3_online.yaml>` — **online-behavior** defaults (replay, exploration, reward weights, checkpointing, etc.). CLI flags override values from this file. Legacy alias fields (`agent_hidden_size`, `q_hidden_size`, `learning_starts`, `device`) are no longer remapped — use the canonical names. Architecture fields in this file are ignored.

#### Eval only (run policy, no training)
```bash
python -m scripts.td3.extras.async_td3_real \
  --config configs/real_configs/rollout_td3_config.yaml \
  --model-path ex_model/new_td3_model/checkpoint_325000/training_state.pth \
  --train-args ex_model/new_td3_model/checkpoint_325000/args.yaml \
  --args-file configs/td3_real_world/td3_online.yaml \
  --collector-device cpu \
  --learner-device cuda:0 \
  --data-root-dir real_runs/online_run \
  --min-replay-size-before-learning 999999999 \
  --no-enable-periodic-checkpointing \
  --no-load-replay-from-checkpoint \
  --warm-start-hdf5-dirs
```

`--data-root-dir` is the single root for collected per-episode artifacts. The script creates `<data_root_dir>/data_<YYYYMMDD-HHMMSS>/{episode_hdf5,reset_hdf5,episode_gifs,episode_camera_videos}/` at startup.

#### Online training from a pretrained checkpoint
```bash
python -m scripts.td3.extras.async_td3_real \
  --config configs/real_configs/rollout_td3_config.yaml \
  --model-path ex_model/td3_model/checkpoint_1515000/training_state.pth \
  --train-args ex_model/td3_model/checkpoint_1515000/args.yaml \
  --args-file configs/td3_real_world/td3_online.yaml \
  --collector-device cpu \
  --learner-device cuda:0 \
  --data-root-dir real_runs/online_run
```

#### Resume training from a previous online run
```bash
python -m scripts.td3.extras.async_td3_real \
  --config configs/real_configs/rollout_td3_config.yaml \
  --model-path real_runs/checkpoints/default/checkpoint_step_100000/training_state.pth \
  --train-args real_runs/checkpoints/default/checkpoint_step_100000/args.yaml \
  --args-file configs/td3_real_world/td3_online.yaml \
  --collector-device cpu \
  --learner-device cuda:0 \
  --data-root-dir real_runs/online_run \
  --load-replay-from-checkpoint \
  --include-non-vital-training-state-fields
```

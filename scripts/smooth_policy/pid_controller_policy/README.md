# PID Controller Policy Training

This directory contains the implementation of a reinforcement learning agent that learns to control an air hockey paddle using a PID controller. The key difference from the default training approach is that the agent outputs **delta target positions** instead of direct position deltas, and a PID controller moves the paddle toward the target.

## Architecture Overview

### Key Concept
- **Agent Output**: Delta to apply to the current target position (not direct paddle control)
- **Target Position**: Maintained across timesteps, updated by adding the agent's output
- **Action to Environment**: The updated target position (which the PID controller tracks)

### Flow
```
Observation + Current Target Position 
    ↓
[Agent Neural Network]
    ↓
Delta Target Position
    ↓
New Target Position = Old Target + Delta
    ↓
[Environment with PID Controller]
    ↓
Next Observation
```

## File Structure

- `agent.py` - Neural network agent (Actor-Critic architecture)
- `training.py` - Main training script with PPO algorithm
- `evaluate.py` - Evaluation script for trained models
- `test_dimensions.py` - Dimension validation test
- `test_args.yaml` - Example configuration for quick testing
- `README.md` - This file

## Key Implementation Details

### 1. Augmented Observation Space

The agent receives a concatenated observation:
```python
augmented_obs = [environment_observation, current_target_position]
```

If the environment observation is 4D (e.g., `[paddle_x, paddle_y, puck_x, puck_y]`) and the action space is 2D (e.g., `[target_x, target_y]`), then the augmented observation is 6D.

### 2. Target Position Tracking

Target positions are tracked separately for each parallel environment:
```python
# Initialize at episode start
target_positions = next_obs[:, :act_dim].clone()

# Update each step
target_positions = target_positions + delta_target

# Reset when episode ends
if episode_done:
    target_positions[env_idx] = next_obs[env_idx, :act_dim]
```

### 3. Differences from Default Training

| Aspect | Default Training | PID Controller Training |
|--------|-----------------|------------------------|
| Agent Input | Observation only | Observation + Target Position |
| Agent Output | Position delta | Target position delta |
| Action to Env | Position delta | Target position (absolute) |
| State Tracking | None | Target positions per env |
| Reset Behavior | N/A | Reset target to paddle position |

### 4. No CAPS Loss

Unlike the default training, this implementation does not include CAPS (Consecutive Action Proximity Smoothing) loss. The smoothness is instead achieved through the PID controller's natural behavior.

## Usage

### Training

Basic training:
```bash
cd /home/air-hockey/daliu/air-hockey-rl
source .venv/bin/activate
python scripts/smooth_policy/pid_controller_policy/training.py \
    --config scripts/smooth_policy/configs/puck_touch/default_config.yaml \
    --run-name my_experiment \
    --num-envs 8 \
    --num-iterations 100
```

Training with custom arguments file:
```bash
python scripts/smooth_policy/pid_controller_policy/training.py \
    --args-file scripts/smooth_policy/pid_controller_policy/test_args.yaml
```

### Evaluation

Evaluate a trained model:
```bash
python scripts/smooth_policy/pid_controller_policy/evaluate.py \
    --model runs/pid_training/task_name/run_name_timestamp/model.pth \
    --save-dir runs/pid_training/task_name/run_name_timestamp/eval \
    --config-path runs/pid_training/task_name/run_name_timestamp/config.yaml \
    --n-eps 10 \
    --n-gifs 3
```

Using parent directory:
```bash
python scripts/smooth_policy/pid_controller_policy/evaluate.py \
    --use-parent-log-dir True \
    --parent-log-dir runs/pid_training/task_name/run_name_timestamp
```

### Testing

Run dimension validation test:
```bash
source .venv/bin/activate
python scripts/smooth_policy/pid_controller_policy/test_dimensions.py
```

Quick training test (2 envs, 2 iterations):
```bash
python scripts/smooth_policy/pid_controller_policy/training.py \
    --args-file scripts/smooth_policy/pid_controller_policy/test_args.yaml
```

## Hyperparameters

Key hyperparameters in `Args` dataclass:

- `num_envs`: Number of parallel environments (default: 8)
- `num_steps`: Steps per environment per iteration (default: 512)
- `learning_rate`: Adam learning rate (default: 1e-4)
- `num_iterations`: Total training iterations (default: 100)
- `gamma`: Discount factor (default: 0.99)
- `gae_lambda`: GAE lambda (default: 0.95)
- `clip_coef`: PPO clipping coefficient (default: 0.2)
- `ent_coef`: Entropy coefficient (default: 0.0)
- `vf_coef`: Value function coefficient (default: 0.5)

## Output Structure

Training creates the following structure:
```
runs/pid_training/task_name/run_name_timestamp/
├── config.yaml              # Environment configuration
├── args.yaml                # Training arguments
├── events.out.tfevents.*    # TensorBoard logs
├── model.pth                # Final model checkpoint
├── checkpoint_10/           # Periodic checkpoints
│   ├── model.pth
│   └── episode_*.gif
└── eval/                    # Final evaluation
    ├── episode_*.gif
    └── eval_stats.txt
```

## TensorBoard Metrics

View training progress:
```bash
tensorboard --logdir runs/pid_training
```

Tracked metrics:
- `charts/episodic_return` - Episode returns
- `charts/avg_episodic_return` - Average return per iteration
- `charts/avg_success_rate` - Success rate
- `losses/policy_loss` - PPO policy loss
- `losses/value_loss` - Value function loss
- `losses/entropy` - Policy entropy
- `motion/avg_velocity_magnitude` - Paddle velocity
- `motion/avg_acceleration_magnitude` - Paddle acceleration
- `motion/avg_jerk_magnitude` - Paddle jerk
- `motion/avg_delta_target_magnitude` - Target position changes

## Agent Architecture

The agent uses a simple MLP architecture:

**Actor (Policy)**:
- Input: Augmented observation (obs_dim + act_dim)
- Hidden: 64 → 64 (Tanh activation)
- Output: Delta target position (act_dim)
- Distribution: Gaussian with learnable std

**Critic (Value)**:
- Input: Augmented observation (obs_dim + act_dim)
- Hidden: 64 → 64 (Tanh activation)
- Output: State value (1D)

## Common Issues

### Dimension Mismatch
If you see dimension errors, verify:
1. Agent is initialized with `obs_dim + act_dim` as input dimension
2. Observations are properly concatenated with target positions
3. Target positions are initialized from the first `act_dim` elements of observation

### Target Position Not Resetting
Ensure reset logic handles each environment individually:
```python
for env_idx in range(num_envs):
    if next_done[env_idx]:
        target_positions[env_idx] = next_obs[env_idx, :act_dim]
```

### PID Controller Not Active
Check that the environment configuration uses the PID controller. The Box2D simulator should have the PID controller enabled in the config.

## Future Improvements

Potential enhancements:
- [ ] Add target position bounds checking
- [ ] Implement adaptive action scaling based on task
- [ ] Add visualization of target positions vs actual positions
- [ ] Implement curriculum learning for target position deltas
- [ ] Add metrics for target tracking error


# AMP (Adversarial Motion Priors) Training

This directory implements Adversarial Motion Priors (AMP) for learning smooth, natural motion policies in the air hockey environment.

## Overview

AMP combines reinforcement learning with imitation learning by using a discriminator to distinguish between expert demonstrations and agent-generated trajectories. The discriminator reward encourages the agent to produce motions that look similar to expert demonstrations, leading to smoother and more natural behavior.

## Architecture

```
┌─────────────────┐
│  Policy (Actor) │
│   + Critic      │
└────────┬────────┘
         │
         ├──────────────────────────────┐
         │                              │
         ▼                              ▼
  ┌─────────────┐              ┌──────────────┐
  │ Task Reward │              │ Disc. Reward │
  └──────┬──────┘              └──────┬───────┘
         │                            │
         └──────────┬─────────────────┘
                    ▼
         Combined Reward Signal
```

## Components

### 1. **discriminator.py**
MLP network that classifies state transitions as expert vs agent-generated.
- Input: Consecutive state pairs `[s_t, s_{t+1}]` (8D: position + velocity)
- Output: Single logit value
- Architecture: 2 hidden layers with 256 units each

### 2. **normalizer.py**
Running statistics normalizer for discriminator observations.
- Tracks mean and standard deviation
- Normalizes inputs with clipping to prevent extreme values
- Essential for discriminator convergence

### 3. **replay_buffer.py**
Circular buffer for storing past agent observations.
- Prevents discriminator overfitting to recent data
- Mixes current and past agent data during training
- Default capacity: 100,000 observations

### 4. **demo_loader.py**
Loads and samples expert demonstration data.
- Reads `.pt` files created by `prepare_amp_dataset.py`
- Provides random sampling for discriminator training
- Expected format: `[N, 2, 4]` state pairs

### 5. **amp_training.py**
Main training script integrating all AMP components with PPO.

## Discriminator Observations

The discriminator operates on **consecutive state pairs**:

```
disc_obs = [s_t, s_{t+1}]
         = [x_t, y_t, vx_t, vy_t, x_{t+1}, y_{t+1}, vx_{t+1}, vy_{t+1}]
```

Where:
- `x, y`: Paddle position
- `vx, vy`: Paddle velocity

This captures the motion dynamics and allows the discriminator to learn what constitutes "natural" motion.

## Usage

### 1. Prepare Demonstration Data

First, create the demonstration dataset from real robot trajectories:

```bash
cd scripts/smooth_policy/amp_data
python prepare_amp_dataset.py --output-path amp_dataset.pt
```

This creates a PyTorch file containing expert state pairs.

### 2. Train with AMP

Using the default configuration:

```bash
python scripts/smooth_policy/amp/amp_training.py
```

Using a custom configuration file:

```bash
python scripts/smooth_policy/amp/amp_training.py \
    --args-file scripts/smooth_policy/amp/example_amp_args.yaml
```

Override specific parameters:

```bash
python scripts/smooth_policy/amp/amp_training.py \
    --args-file example_amp_args.yaml \
    --num_iterations 500 \
    --disc_reward_weight 0.7 \
    --task_reward_weight 0.3
```

### 3. Test Components

Verify all AMP components work correctly:

```bash
cd scripts/smooth_policy/amp
python test_amp_components.py
```

## Key Hyperparameters

### Reward Weighting
- `task_reward_weight` (default: 0.5): Weight for task-specific reward
- `disc_reward_weight` (default: 0.5): Weight for discriminator reward
- **Tuning**: If task performance drops, increase `task_reward_weight`. If motion is jerky, increase `disc_reward_weight`.

### Discriminator Training
- `disc_batch_size` (default: 512): Batch size for discriminator updates
- `disc_learning_rate` (default: 1e-4): Learning rate for discriminator
- `disc_grad_penalty` (default: 5.0): Gradient penalty coefficient (Lipschitz constraint)
- `disc_logit_reg` (default: 0.01): Regularization on output layer weights
- `disc_weight_decay` (default: 0.0001): Weight decay regularization

### Discriminator Reward
- `disc_reward_scale` (default: 2.0): Scaling factor for discriminator rewards
- Formula: `r_disc = -log(1 - P(expert)) * disc_reward_scale`

### Replay Buffer
- `disc_replay_buffer_size` (default: 100,000): Maximum buffer capacity
- `disc_replay_samples` (default: 1,024): Number of samples to store per iteration

## Monitoring Training

### TensorBoard Metrics

**AMP-specific metrics:**
- `amp/disc_loss`: Total discriminator loss
- `amp/disc_agent_acc`: Discriminator accuracy on agent data (target: ~60-70%)
- `amp/disc_demo_acc`: Discriminator accuracy on demo data (target: ~60-70%)
- `amp/disc_reward_mean`: Average discriminator reward
- `amp/task_reward_mean`: Average task reward
- `amp/combined_reward_mean`: Average combined reward
- `amp/disc_grad_penalty`: Gradient penalty value
- `amp/replay_buffer_size`: Current replay buffer size

**Expected discriminator accuracy:**
- Too high (>80%): Discriminator too strong, agent struggles to imitate
- Too low (<50%): Discriminator too weak, not providing useful signal
- Good range: 60-75%

Launch TensorBoard:
```bash
tensorboard --logdir runs/default_training/
```

## Output Files

After training, the following files are saved:

```
runs/default_training/{task}/{run_name}_{timestamp}/
├── model.pth                    # Policy network weights
├── discriminator.pth            # Discriminator network weights
├── amp_components.pth           # Normalizer and replay buffer state
├── config.yaml                  # Environment configuration
├── args.yaml                    # Training arguments
├── events.out.tfevents.*        # TensorBoard logs
└── checkpoint_{iter}/
    ├── model.pth
    ├── discriminator.pth
    ├── amp_components.pth
    └── evaluation results...
```

## Troubleshooting

### Issue: Discriminator accuracy at 100%

**Cause**: Discriminator is too strong, easily distinguishes agent from expert.

**Solutions**:
- Reduce `disc_loss_weight` (e.g., from 1.0 to 0.5)
- Increase `disc_grad_penalty` (e.g., from 5.0 to 10.0)
- Reduce `disc_learning_rate`
- Increase `disc_reward_weight` to give agent stronger signal

### Issue: Task performance degrades

**Cause**: Discriminator reward dominating task reward.

**Solutions**:
- Increase `task_reward_weight` (e.g., from 0.5 to 0.7)
- Reduce `disc_reward_weight` (e.g., from 0.5 to 0.3)
- Reduce `disc_reward_scale`

### Issue: Training is unstable

**Cause**: Discriminator or policy updates too aggressive.

**Solutions**:
- Reduce learning rates for both policy and discriminator
- Increase gradient clipping (`max_grad_norm`)
- Increase `disc_grad_penalty` for better discriminator regularization
- Reduce batch sizes

### Issue: Demo data mismatch

**Cause**: Demo observations don't match agent's state extraction.

**Solutions**:
- Verify demo data shape is `[N, 2, 4]`
- Check that observation extraction in training matches demo format
- Ensure both use same state representation (position + velocity)

## Theory

AMP learns a reward function from demonstrations:

```
r_total = w_task * r_task + w_disc * r_disc
```

Where:
- `r_task`: Task-specific reward (e.g., puck touching reward)
- `r_disc = -log(1 - D(s))`: Discriminator reward
  - `D(s)`: Discriminator output (probability of being expert)
  - Higher when agent motion resembles expert

The discriminator is trained adversarially:
- Maximize probability for expert demonstrations
- Minimize probability for agent-generated trajectories

This encourages the agent to "fool" the discriminator by producing expert-like motions.

## References

- AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control (Peng et al., 2021)
- DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills (Peng et al., 2018)

## Implementation Notes

- **State pairs**: Using consecutive states `[s_t, s_{t+1}]` captures motion dynamics
- **Normalization**: Essential for discriminator convergence; without it, discriminator may overfit to scale differences
- **Replay buffer**: Prevents catastrophic forgetting when agent policy changes rapidly
- **Gradient penalty**: Enforces Lipschitz constraint for stable discriminator training
- **Reward scaling**: `disc_reward_scale` balances discriminator reward magnitude with task reward

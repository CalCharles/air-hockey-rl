# AMP Triplet Training

This directory implements Adversarial Motion Priors (AMP) using **triplets of consecutive states** `(s_t, s_{t+1}, s_{t+2})` instead of pairs, with **translation-only normalization** (no rotation).

## Overview

This extension of the standard AMP implementation captures richer motion dynamics including acceleration patterns by considering three consecutive states. The translation-only normalization preserves directional information while removing absolute position dependency.

## Key Differences from Standard AMP

| Feature | Standard AMP (`amp/`) | Triplet AMP (`amp_triplet/`) |
|---------|----------------------|------------------------------|
| State representation | Pairs `[s_t, s_{t+1}]` | Triplets `[s_t, s_{t+1}, s_{t+2}]` |
| Normalization | Translation + Rotation | Translation only |
| Discriminator input | 6D | 10D |
| Captures | Velocity changes | Velocity + Acceleration patterns |

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
                    │
                    ▼
         ┌──────────────────┐
         │  Triplet States  │
         │ [s_t, s_{t+1},   │
         │     s_{t+2}]     │
         └──────────────────┘
```

## Components

### 1. **discriminator.py**
MLP network that classifies state transition triplets as expert vs agent-generated.
- Input: Triplet state representation (10D: positions + velocities, translation-normalized)
- Output: Single logit value
- Architecture: 2 hidden layers with 128 units each
- Default input dimension: 10

### 2. **demo_loader.py**
Loads and samples expert demonstration triplet data.
- Reads `.pt` files created by `prepare_amp_triplet_dataset.py`
- Provides random sampling for discriminator training
- Expected format: `[N, 3, 4]` or pre-normalized `[N, 10]`
- Applies translation-only normalization

### 3. **normalizer.py** & **replay_buffer.py**
Copied from `amp/` - these are shape-agnostic and work with any observation dimension.

### 4. **amp_training_triplet.py**
Main training script integrating all AMP triplet components with PPO.
- Tracks two previous paddle states for triplet construction
- Uses 10D discriminator observations
- LSGAN objective for stable training

## Discriminator Observations

The discriminator operates on **consecutive state triplets** with translation-only normalization:

### Raw Format
```
triplet = [s_t, s_{t+1}, s_{t+2}]
where each state = [x, y, vx, vy]
```

### Normalized Format (10D)
After translation normalization:
```
disc_obs = [vel1_x, vel1_y,           # First velocity (2D)
            rel_pos2_x, rel_pos2_y,   # Relative position 2 (2D)
            rel_vel2_x, rel_vel2_y,   # Second velocity (2D)
            rel_pos3_x, rel_pos3_y,   # Relative position 3 (2D)
            rel_vel3_x, rel_vel3_y]   # Third velocity (2D)
```

**Key properties:**
- First position is always at origin (0, 0) - not included
- No rotation applied - preserves directional information
- Velocities kept as-is in world frame
- Captures both velocity and acceleration patterns

## Usage

### 1. Prepare Demonstration Data

First, create the triplet demonstration dataset from real robot trajectories:

```bash
cd scripts/smooth_policy/amp_triplet_data
python prepare_amp_triplet_dataset.py \
    --data-dir /nfs/data/airhockey \
    --output-path amp_triplet_dataset.pt
```

This creates a PyTorch file containing expert state triplets of shape `[N, 3, 4]`.

### 2. Normalize the Dataset (Optional)

You can pre-normalize the dataset for faster loading:

```bash
python normalize_triplet_dataset.py \
    --input-path amp_triplet_dataset.pt \
    --output-path amp_triplet_dataset_normalized.pt \
    --validate
```

This applies translation-only normalization and outputs shape `[N, 10]`.

### 3. Train with AMP Triplet

Using the default configuration:

```bash
python scripts/smooth_policy/amp_triplet/amp_training_triplet.py \
    --demo-data-path scripts/smooth_policy/amp_triplet_data/amp_triplet_dataset.pt
```

Using a custom configuration file:

```bash
python scripts/smooth_policy/amp_triplet/amp_training_triplet.py \
    --args-file scripts/smooth_policy/amp/example_amp_args.yaml \
    --demo-data-path scripts/smooth_policy/amp_triplet_data/amp_triplet_dataset.pt
```

Override specific parameters:

```bash
python scripts/smooth_policy/amp_triplet/amp_training_triplet.py \
    --demo-data-path scripts/smooth_policy/amp_triplet_data/amp_triplet_dataset.pt \
    --num_iterations 500 \
    --disc_reward_weight 0.7 \
    --task_reward_weight 0.3
```

## Key Hyperparameters

Same as standard AMP, but note:
- `disc_batch_size` (default: 512): Batch size for discriminator updates
- `disc_learning_rate` (default: 1e-5): May need tuning for 10D input
- `disc_grad_penalty` (default: 5.0): Gradient penalty coefficient

## Monitoring Training

### TensorBoard Metrics

All standard AMP metrics apply, plus component-wise statistics for the 10D observation:

```bash
tensorboard --logdir runs/default_training/
```

**Key metrics:**
- `amp_components/agent_vel1_x_mean`: First velocity X component (agent)
- `amp_components/agent_rel_pos2_x_mean`: Relative position 2 X component (agent)
- `amp_components/demo_*`: Corresponding demo statistics
- `amp_components/diff_*`: Differences between agent and demo distributions

## Normalization Details

### Translation-Only Normalization

Unlike standard AMP which also applies rotation, triplet AMP uses **translation-only** normalization:

```python
# Step 1: Translate all states so first position is at origin
pos2_relative = pos2 - pos1
pos3_relative = pos3 - pos1

# Step 2: Keep all velocities unchanged (no rotation)
output = [vel1, pos2_relative, vel2, pos3_relative, vel3]
```

**Why translation-only?**
- Preserves directional information (important for tasks with preferred directions)
- Simpler normalization reduces information loss
- Still removes absolute position dependency
- Captures full velocity and acceleration patterns

### What's Preserved

✓ **All velocities**: Kept in world frame  
✓ **Relative positions**: Distances and directions from first state  
✓ **Acceleration patterns**: Implicitly captured through velocity changes  
✗ **Absolute position**: Removed by translation  

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

## Comparison with Standard AMP

### Advantages of Triplet AMP

1. **Richer dynamics**: Captures acceleration patterns, not just velocity changes
2. **Better temporal context**: Three timesteps provide more information
3. **Direction-aware**: Translation-only normalization preserves directional preferences
4. **Smoother trajectories**: Encourages physically plausible acceleration

### Trade-offs

1. **More data required**: Each trajectory yields fewer triplets than pairs (T-2 vs T-1)
2. **Larger discriminator input**: 10D vs 6D (more parameters to train)
3. **Computational cost**: Slightly higher due to larger observation dimension
4. **State tracking**: Must maintain two previous states instead of one

## Implementation Notes

### State History Tracking

The training script maintains two previous paddle states:

```python
prev_paddle_state_1  # t-1
prev_paddle_state_2  # t-2
current_paddle_state # t

# Construct triplet
triplet = [prev_paddle_state_2, prev_paddle_state_1, current_paddle_state]
```

### Episode Boundaries

When an episode ends, both previous states are reset to the new initial state to avoid mixing data across episodes.

### First Two Steps

For the first two steps of each episode, the previous states are initialized with the current state. This means the first few triplets have repeated states, but this is acceptable as the discriminator learns to handle this.

## Troubleshooting

### Issue: Discriminator accuracy stuck at 50%

**Cause**: Discriminator can't distinguish between agent and expert triplets.

**Solutions**:
- Check that demo data is properly normalized
- Verify state extraction matches demo format
- Increase discriminator capacity (hidden dimensions)
- Adjust learning rate

### Issue: Agent motion is jerky despite high discriminator reward

**Cause**: Translation-only normalization may not penalize abrupt direction changes enough.

**Solutions**:
- Increase `disc_reward_weight`
- Add smoothness regularization (CAPS)
- Consider using rotation normalization (standard AMP)

### Issue: Training is slower than standard AMP

**Cause**: 10D discriminator input is larger than 6D.

**Solutions**:
- Reduce batch sizes slightly
- Use gradient accumulation
- Optimize data loading

## Theory

Triplet AMP extends the standard AMP reward function to include richer temporal context:

```
r_total = w_task * r_task + w_disc * r_disc

where r_disc is based on D(s_t, s_{t+1}, s_{t+2})
```

The discriminator learns to identify **temporally extended motion patterns** including:
- Velocity profiles
- Acceleration patterns
- Trajectory curvature
- Motion smoothness

## References

- AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control (Peng et al., 2021)
- Least Squares GAN (Mao et al., 2017)

## Related Files

- **Data preparation**: `scripts/smooth_policy/amp_triplet_data/`
  - `prepare_amp_triplet_dataset.py`: Extract state triplets from HDF5 files
  - `normalize_triplet_dataset.py`: Apply translation-only normalization
- **Training**: `scripts/smooth_policy/amp_triplet/`
  - `amp_training_triplet.py`: Main training script
  - `demo_loader.py`: Load and sample demonstration triplets
  - `discriminator.py`: 10D discriminator network

---

*Created: February 2026*  
*Based on standard AMP implementation*

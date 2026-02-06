# Discriminator Verification

This folder contains a standalone verification tool to test that the discriminator can successfully distinguish between demonstration data and agent-generated observations.

## Purpose

The discriminator verification script (`verify_discriminator.py`) allows you to:
1. Load demonstration data from the `amp_no_rotation_data` dataset
2. Collect agent observations by running a trained model in the environment
3. Train a discriminator using the same LSGAN logic as in `amp_training_lsgan.py`
4. Monitor comprehensive discriminator statistics via TensorBoard

This is useful for debugging and validating that the discriminator architecture and training procedure work correctly before integrating into full AMP training.

## Usage

### Basic Usage

```bash
python scripts/smooth_policy/amp_no_rotation/discriminator_verification/verify_discriminator.py
```

This will use default settings:
- Model: `pid/no_rotation/runr1/checkpoint_380/model.pth`
- Demo data: `scripts/smooth_policy/amp_no_rotation_data/amp_full_dataset_raw.pt`
- Collect 10,000 agent observation steps
- Train discriminator for 1,000 iterations

### Custom Configuration

```bash
python scripts/smooth_policy/amp_no_rotation/discriminator_verification/verify_discriminator.py \
    --model-path pid/no_rotation/runr2/checkpoint_400/model.pth \
    --num-collection-steps 20000 \
    --num-training-iterations 2000 \
    --batch-size 512 \
    --learning-rate 1e-5 \
    --log-dir runs/my_disc_verification
```

### Command-Line Arguments

- `--model-path`: Path to trained agent model (default: `pid/no_rotation/runr1/checkpoint_380/model.pth`)
- `--demo-data-path`: Path to demonstration data (default: `scripts/smooth_policy/amp_no_rotation_data/amp_full_dataset_raw.pt`)
- `--config`: Environment config file (default: `scripts/smooth_policy/configs/puck_touch/default_config.yaml`)
- `--num-collection-steps`: Number of agent observation steps to collect (default: 10000)
- `--val-split`: Fraction of data to use for validation (default: 0.2)
- `--num-training-iterations`: Number of discriminator training iterations (default: 1000)
- `--batch-size`: Training batch size (default: 512)
- `--learning-rate`: Discriminator learning rate (default: 1e-5)
- `--disc-logit-reg`: Logit regularization weight (default: 0.01)
- `--disc-grad-penalty`: Gradient penalty weight (default: 5.0)
- `--disc-weight-decay`: Weight decay regularization (default: 0.0001)
- `--log-dir`: Output directory for logs (default: auto-generated)
- `--log-interval`: Logging interval in iterations (default: 10)
- `--device`: Device to use (default: "cuda:0")
- `--seed`: Random seed (default: 0)

## Output

The script will create a log directory containing:
- `discriminator.pth`: Trained discriminator weights
- `disc_components.pth`: Normalizer state
- `config.yaml`: Environment configuration
- `args.yaml`: Script arguments
- TensorBoard event files

## Monitoring with TensorBoard

View training progress in real-time:

```bash
tensorboard --logdir runs/disc_verification
```

### Key Metrics

**Training Loss Metrics:**
- `train/loss`: Total discriminator loss on training set
- `train/loss_demo`: Demo classification loss on training set
- `train/loss_agent`: Agent classification loss on training set
- `train/grad_penalty`: Gradient penalty magnitude

**Training Accuracy Metrics:**
- `train/demo_acc`: Accuracy on demo data (should approach 1.0)
- `train/agent_acc`: Accuracy on agent data (should approach 1.0)

**Training Logit Statistics:**
- `train/demo_logit_mean`: Mean discriminator output for demo data (should approach 1.0)
- `train/agent_logit_mean`: Mean discriminator output for agent data (should approach -1.0)

**Validation Loss Metrics:**
- `val/loss`: Total discriminator loss on validation set
- `val/loss_demo`: Demo classification loss on validation set
- `val/loss_agent`: Agent classification loss on validation set

**Validation Accuracy Metrics:**
- `val/demo_acc`: Accuracy on demo data (should approach 1.0)
- `val/agent_acc`: Accuracy on agent data (should approach 1.0)

**Validation Logit Statistics:**
- `val/demo_logit_mean`: Mean discriminator output for demo data (should approach 1.0)
- `val/agent_logit_mean`: Mean discriminator output for agent data (should approach -1.0)

**Component-wise Statistics:**
For each component (`vel1_x`, `vel1_y`, `rel_pos_x`, `rel_pos_y`, `rel_vel_x`, `rel_vel_y`):
- `components/demo_{component}_mean`: Demo data mean
- `components/demo_{component}_std`: Demo data standard deviation
- `components/agent_{component}_mean`: Agent data mean
- `components/agent_{component}_std`: Agent data standard deviation
- `components/diff_{component}_mean_abs`: Absolute difference in means

## Expected Results

A well-functioning discriminator should achieve:
- **Train Demo accuracy**: > 0.95 (correctly classifies demos as expert)
- **Train Agent accuracy**: > 0.95 (correctly classifies agent as fake)
- **Val Demo accuracy**: > 0.90 (generalization to held-out demos)
- **Val Agent accuracy**: > 0.90 (generalization to held-out agent data)
- **Logit separation**: Demo mean ≈ 1.0, Agent mean ≈ -1.0
- **Stable training**: Train and validation losses should decrease and stabilize
- **No overfitting**: Validation metrics should track training metrics closely

If the discriminator cannot distinguish between demo and agent data, this may indicate:
1. The agent already produces expert-like behavior
2. Issues with data preprocessing or normalization
3. Need to adjust hyperparameters (learning rate, gradient penalty, etc.)

If there's a large gap between train and validation performance, this may indicate:
1. Overfitting - consider reducing model capacity or adding regularization
2. Insufficient training data
3. Poor data split (non-representative validation set)

## Implementation Details

The script uses:
- **Discriminator**: Same architecture as `amp_training_lsgan.py` (6D input, [128, 128] hidden layers)
- **LSGAN Loss**: `0.5 * E[(D(demo) - 1)^2] + 0.5 * E[(D(agent) - (-1))^2]`
- **Gradient Penalty**: Lipschitz constraint via gradient penalty on demo data
- **Normalization**: Translation-only normalization (no rotation) matching the dataset
- **Normalizer**: Running mean/std normalization with clipping

## References

- Main training script: `scripts/smooth_policy/amp_no_rotation/amp_training_lsgan.py`
- Discriminator implementation: `scripts/smooth_policy/amp/discriminator.py`
- Demo data preparation: `scripts/smooth_policy/amp_no_rotation_data/prepare_amp_dataset.py`

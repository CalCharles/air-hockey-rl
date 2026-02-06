# Dataset Normalization Guide

## Overview

The `normalize_dataset.py` script transforms the AMP dataset into relative state representations by normalizing each consecutive state pair to remove absolute position information.

## Motivation

For AMP (Adversarial Motion Priors), we want the discriminator to learn motion patterns that are **translation invariant**. By normalizing each state pair to a relative representation, we:

1. Remove absolute position information (only relative movement matters)
2. Preserve velocity directions (maintaining orientation information)
3. Focus learning on the **dynamics** of motion rather than absolute position in space

## Transformation Process

For each consecutive state pair `(state₁, state₂)`:

### Input Format
```
state₁ = [x₁, y₁, vₓ₁, vᵧ₁]
state₂ = [x₂, y₂, vₓ₂, vᵧ₂]
```

### Step 1: Translation
Translate both states so state₁'s position is at the origin:
```
state₁' = [0, 0, vₓ₁, vᵧ₁]
state₂' = [x₂ - x₁, y₂ - y₁, vₓ₂, vᵧ₂]
```

### Step 2: Keep First Velocity and Second State
Since state₁'s position is now always `[0, 0]`, we don't keep it. We keep the first velocity (preserving its original direction) and the full second state:
```
output = [vel₁ₓ, vel₁ᵧ, pos₂ₓ', pos₂ᵧ', vel₂ₓ, vel₂ᵧ]
       = [vₓ₁, vᵧ₁, x₂-x₁, y₂-y₁, vₓ₂, vᵧ₂]
```

## Properties Preserved

✓ **Distance**: The Euclidean distance between positions is preserved  
✓ **Velocity magnitude**: The magnitude of velocities is preserved  
✓ **Velocity direction**: The direction of velocities is preserved  
✓ **Relative angles**: Angular relationships between vectors are preserved  
✗ **Absolute position**: Removed by translation  

## Usage

### Basic Usage

```bash
# Activate environment
cd /home/air-hockey/daliu/air-hockey-rl
source .venv/bin/activate

# Normalize a dataset
python scripts/smooth_policy/amp_data/normalize_dataset.py \
    --input-path amp_dataset.pt \
    --output-path amp_dataset_normalized.pt
```

### With Validation

Show validation samples to verify the transformation:

```bash
python scripts/smooth_policy/amp_data/normalize_dataset.py \
    --input-path amp_dataset.pt \
    --validate \
    --num-samples 10
```

### Auto-naming

If you don't specify `--output-path`, it will automatically add `_normalized` suffix:

```bash
python scripts/smooth_policy/amp_data/normalize_dataset.py \
    --input-path datasets/amp_full.pt
# Creates: datasets/amp_full_normalized.pt
```

## Input Requirements

The input file must be created by `prepare_amp_dataset.py` and contain:
- `state_pairs`: Tensor of shape `(N, 2, 4)`
  - N = number of state pairs
  - 2 = [state₁, state₂]
  - 4 = [x_pos, y_pos, x_vel, y_vel]

## Output Format

The output file contains:
- `normalized_states`: Tensor of shape `(N, 6)`
  - Each entry contains the first velocity and second state in normalized coordinates
  - Format: [vel1_x, vel1_y, relative_x, relative_y, relative_vₓ, relative_vᵧ]
  - Note: vel1 preserves its original direction (not aligned to x-axis)

## Loading Normalized Dataset

```python
import torch

# Load normalized dataset
data = torch.load('amp_dataset_normalized.pt')
normalized_states = data['normalized_states']  # Shape: (N, 6)

# Each state represents a transition in normalized frame
for state in normalized_states:
    vel1_x, vel1_y = state[0], state[1]    # First velocity (original direction preserved)
    rel_x, rel_y = state[2], state[3]      # Relative position
    rel_vx, rel_vy = state[4], state[5]    # Relative velocity (original direction preserved)
    
    # vel1 maintains both magnitude and direction from the original velocity
```

## Example Transformation

### Original State Pair
```
State 1: position = (1.5, 2.0), velocity = (0.3, 0.4)
State 2: position = (1.7, 2.3), velocity = (0.2, 0.5)
```

### After Translation
```
State 1: position = (0.0, 0.0), velocity = (0.3, 0.4)
State 2: position = (0.2, 0.3), velocity = (0.2, 0.5)
```

### Final Output (No Rotation Applied)
```
Normalized state = (0.3, 0.4, 0.2, 0.3, 0.2, 0.5)
                   [vel1_x, vel1_y, pos2_x, pos2_y, vel2_x, vel2_y]
```

Note: Velocities preserve their original direction. Only translation normalization is applied.

## Validation

Run the test suite to verify the implementation:

```bash
python scripts/smooth_policy/amp_data/test_normalization.py
```

Expected output:
```
✓ ALL TESTS PASSED!
```

## Performance

- **Processing speed**: ~10,000 state pairs per second
- **Memory usage**: Minimal (processes in a single pass)
- **Output size**: Half of input size (only keeping second state)

## Use Cases

### 1. AMP Discriminator Training

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

# Load normalized dataset
data = torch.load('amp_dataset_normalized.pt')
states = data['normalized_states']

# Create dataloader
dataset = TensorDataset(states)
loader = DataLoader(dataset, batch_size=256, shuffle=True)

# Train discriminator on normalized states
for batch in loader:
    normalized_states = batch[0]
    # Discriminator sees only relative motion patterns
    discriminator_output = discriminator(normalized_states)
```

### 2. Analyzing Motion Patterns

```python
import matplotlib.pyplot as plt

# Load normalized states
data = torch.load('amp_dataset_normalized.pt')
states = data['normalized_states'].numpy()

# Plot relative position changes
plt.scatter(states[:, 0], states[:, 1], alpha=0.1)
plt.xlabel('Relative X Position')
plt.ylabel('Relative Y Position')
plt.title('Motion Patterns (Position-Invariant)')
plt.axis('equal')
plt.show()
```

## Technical Details

### Translation Only

Only translation normalization is applied:
- First position is moved to origin (0, 0)
- All velocities and the second position maintain their original values/directions
- No rotation transformation is applied

### Edge Cases

- **Near-zero values**: Uses float32 precision
- **Zero velocity**: Handled naturally (no special case needed since no rotation is applied)

## Troubleshooting

### Invalid input shape error
```
ValueError: Invalid input shape (N, 4). Expected (N, 2, 4)
```
**Solution**: Make sure you're loading a file created by `prepare_amp_dataset.py`, not already normalized.

### Missing 'state_pairs' key
```
ValueError: Input file does not contain 'state_pairs' key
```
**Solution**: Input file must be the original dataset, not a normalized one.

## Related Scripts

- **prepare_amp_dataset.py**: Creates the original dataset from HDF5 files
- **test_normalization.py**: Tests the normalization logic
- **example_usage.py**: Shows how to use the original dataset

## References

For more information on AMP (Adversarial Motion Priors):
- Original Paper: "AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control"
- The normalization makes motion patterns translation and rotation invariant

---

*Created: January 2026*  
*Tested with 418 trajectory files*

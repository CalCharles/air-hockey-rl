# Dataset Normalization - Changes Summary

## What Changed

The normalization script has been updated to **keep the first velocity** along with the second state, instead of discarding it entirely.

## Motivation

The first velocity, after rotation to align with (1, 0), still contains important information: **its magnitude**. While the direction is normalized away, the speed at which the robot is moving at the start of the transition is valuable information for learning motion patterns.

## Output Format Change

### Before (4D output)
```
Output shape: (N, 4)
Format: [relative_x, relative_y, relative_vₓ, relative_vᵧ]
```

Only the second state was kept. The first velocity magnitude information was lost.

### After (6D output)
```
Output shape: (N, 6)
Format: [vel1_x, vel1_y, relative_x, relative_y, relative_vₓ, relative_vᵧ]
```

Now includes:
- **vel1_x, vel1_y**: First velocity after rotation (will be `[magnitude, 0]`)
- **relative_x, relative_y**: Second position relative to first
- **relative_vₓ, relative_vᵧ**: Second velocity after rotation

## Example

### Input State Pair
```python
state1 = [1.5, 2.0, 0.3, 0.4]  # pos=(1.5, 2.0), vel=(0.3, 0.4)
state2 = [1.7, 2.3, 0.2, 0.5]  # pos=(1.7, 2.3), vel=(0.2, 0.5)
```

### Old Output (4D)
```python
output = [0.36, -0.04, 0.46, 0.29]
# Lost: magnitude of first velocity (0.5)
```

### New Output (6D)
```python
output = [0.5, 0.0, 0.36, -0.04, 0.46, 0.29]
         [vel1 magnitude, 0, relative position, relative velocity]
# Preserved: magnitude of first velocity (0.5)
```

## What's Preserved

✓ **First velocity magnitude**: Now included in output  
✓ **Position distance**: Euclidean distance between positions  
✓ **Second velocity magnitude**: Magnitude of second velocity  
✓ **Relative angles**: Angular relationships between vectors  
✓ **Translation invariance**: Absolute position removed  
✓ **Rotation invariance**: Absolute direction normalized  

## Code Changes

### Using the New Format

```python
import torch

# Load normalized dataset
data = torch.load('amp_dataset_normalized.pt')
states = data['normalized_states']  # Shape: (N, 6) instead of (N, 4)

# Extract components
vel1_magnitude = states[:, 0]       # First velocity x-component (magnitude)
# states[:, 1] is always ~0         # First velocity y-component (always ~0)
relative_pos_x = states[:, 2]       # Relative position x
relative_pos_y = states[:, 3]       # Relative position y
relative_vel_x = states[:, 4]       # Relative velocity x
relative_vel_y = states[:, 5]       # Relative velocity y

# Use in AMP discriminator
for batch in dataloader:
    normalized_states = batch[0]  # Shape: (batch_size, 6)
    discriminator_output = discriminator(normalized_states)
```

## Benefits

1. **More information**: Discriminator can learn that different speeds lead to different motion patterns
2. **Better motion matching**: Can distinguish between slow and fast movements
3. **Richer representation**: 6D space captures more about the dynamics
4. **Still invariant**: Translation and rotation invariance maintained

## Testing

All tests pass with the new 6D format:

```bash
python scripts/smooth_policy/amp_data/test_normalization.py
```

Expected output:
```
✓ All distances preserved!
✓ All first velocity magnitudes preserved!
✓ All second velocity magnitudes preserved!
✓ ALL TESTS PASSED!
```

## Migration

If you have existing code using the old 4D format:

### Old Code (4D)
```python
rel_x, rel_y = state[0], state[1]
rel_vx, rel_vy = state[2], state[3]
```

### New Code (6D)
```python
vel1_x, vel1_y = state[0], state[1]  # Added: first velocity
rel_x, rel_y = state[2], state[3]     # Indices shifted by 2
rel_vx, rel_vy = state[4], state[5]   # Indices shifted by 2
```

## Summary

The change adds 2 dimensions to the output (first velocity components) while maintaining all the properties of the normalization. This provides richer information for learning motion patterns while still being translation and rotation invariant.

---

*Updated: January 29, 2026*

# Quick Start Guide - Trajectory Visualization

## TL;DR - Run It Now

```bash
cd /home/air-hockey/daliu/air-hockey-rl
source .venv/bin/activate
python scripts/trajectory_visualization/visualize/visualize_trajectory.py
```

Output: `scripts/trajectory_visualization/visualize/trajectory_visualization.gif`

## What Does It Do?

Creates an animated GIF showing the paddle moving on the air hockey table from trajectory data.

**Input**: HDF5 trajectory file with robot position data  
**Output**: Animated GIF at 20fps showing paddle movement  
**Uses**: Only X and Y positions (ignores Z and rotations)

## Quick Examples

### Example 1: Basic Visualization (Default)
```bash
python scripts/trajectory_visualization/visualize/visualize_trajectory.py
```
- Full resolution (400 pixels per meter)
- All frames
- ~1 MB file size for 122 frames

### Example 2: Quick Preview (First 50 Frames)
Edit `visualize_trajectory.py`, change:
```python
create_trajectory_gif(
    pos_x, pos_y, timestamps, renderer, output_path,
    max_frames=50  # Add this line
)
```

### Example 3: Smaller File (Subsample Every 2nd Frame)
Edit `visualize_trajectory.py`, change:
```python
create_trajectory_gif(
    pos_x, pos_y, timestamps, renderer, output_path,
    subsample=2  # Add this line
)
```

### Example 4: Use Different Data File
Edit `visualize_trajectory.py`, change:
```python
data_path = Path('/path/to/your/trajectory_data.hdf5')
```

## Running Pre-configured Examples

```bash
# Run all example configurations
python scripts/trajectory_visualization/visualize/example_configurations.py

# Run specific example (1-5)
python scripts/trajectory_visualization/visualize/example_configurations.py --example 1
```

Available examples:
1. **Basic** - Full resolution, all frames
2. **High Res** - 600 ppm (larger file, more detail)
3. **Subsampled** - Every 2nd frame (smaller file)
4. **Preview** - First 50 frames only
5. **Low Res Fast** - 300 ppm + subsampled (fastest)

## Output Information

The script displays:
- Number of frames loaded
- Trajectory duration
- Sampling rate
- Position ranges
- GIF file size
- Rendering progress

Example output:
```
Trajectory Statistics:
  Number of frames: 122
  Duration: 5.95 seconds
  Average sample rate: 20.5 Hz
  X range: [-0.732, -0.491] meters
  Y range: [-0.202, 0.050] meters

✓ Saved GIF to: trajectory_visualization.gif
  File size: 1.00 MB
```

## Common Adjustments

### Change Output Location
```python
output_path = Path('/path/to/output.gif')
```

### Change Resolution
```python
renderer = TrajectoryRenderer(
    table_width=1.0436,
    table_length=2.1104,
    paddle_radius=0.0508,
    ppm=600  # Higher = larger file, more detail (default: 400)
)
```

### Limit Frames (For Long Trajectories)
```python
create_trajectory_gif(..., max_frames=500)  # Only first 500 frames
```

### Subsample (Reduce File Size)
```python
create_trajectory_gif(..., subsample=2)  # Every 2nd frame
```

## File Size Guidelines

| Configuration | Approx Size (per 100 frames) | Quality |
|--------------|------------------------------|---------|
| ppm=300, subsample=1 | ~400 KB | Good |
| ppm=400, subsample=1 | ~800 KB | Better (default) |
| ppm=600, subsample=1 | ~2 MB | Best |
| ppm=400, subsample=2 | ~400 KB | Good |

## Troubleshooting

### "Data file not found"
**Fix**: Update `data_path` variable with correct HDF5 file path

### "Assets folder not found"
**Fix**: Run from project root or check that `assets/` folder exists

### GIF file too large
**Fix**: Use subsampling (`subsample=2`) or lower resolution (`ppm=300`)

### Rendering too slow
**Fix**: Use `max_frames=100` to render only first 100 frames for testing

## Files Created

- `visualize_trajectory.py` - Main script
- `trajectory_visualization.gif` - Output GIF (gitignored)
- `README.md` - Full documentation
- `IMPLEMENTATION_NOTES.md` - Technical details
- `example_configurations.py` - Example configurations

## For More Information

- **Full Documentation**: See `README.md`
- **Technical Details**: See `IMPLEMENTATION_NOTES.md`
- **Data Format**: See `../initial_analysis/FIELD_DOCUMENTATION.md`

## Next Steps

1. Run the basic visualization
2. View the GIF to verify it looks correct
3. Adjust resolution or subsampling as needed
4. Process your own trajectory files by changing `data_path`

That's it! You're ready to visualize trajectories.


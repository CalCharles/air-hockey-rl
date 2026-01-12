# Trajectory Visualization

This directory contains tools for visualizing trajectory data from the air hockey robot as animated GIFs.

## Overview

The `visualize_trajectory.py` script reads HDF5 trajectory data and creates an animated GIF showing the paddle moving on the air hockey table. It uses only the X and Y end-effector positions from the trajectory data, ignoring Z components and rotations as requested.

## Features

- **Data Loading**: Reads trajectory data from HDF5 files
- **2D Visualization**: Renders paddle position using X and Y coordinates only
- **Real-world Dimensions**: Uses actual table dimensions (1.0436m x 2.1104m) and paddle radius (0.0508m)
- **Coordinate Transformation**: Properly transforms robot frame coordinates to rendering coordinates
- **20Hz Playback**: Matches the 20Hz sampling rate of the trajectory data
- **Frame Information**: Displays frame number, elapsed time, and current position
- **Configurable**: Supports subsampling and frame limiting for large trajectories

## Usage

### Basic Usage

```bash
# Make sure you're in the project root with virtual environment activated
cd /home/air-hockey/daliu/air-hockey-rl
source .venv/bin/activate

# Run the visualization script
python scripts/trajectory_visualization/visualize/visualize_trajectory.py
```

### Configuration

Edit the `main()` function in `visualize_trajectory.py` to customize:

```python
# Data source
data_path = Path('/nfs/data/airhockey/trajectory_data434.hdf5')

# Output location
output_path = output_dir / 'trajectory_visualization.gif'

# Rendering parameters
renderer = TrajectoryRenderer(
    table_width=1.0436,   # Table width in meters
    table_length=2.1104,  # Table length in meters
    paddle_radius=0.0508, # Paddle radius in meters
    ppm=400               # Pixels per meter (higher = larger file)
)

# GIF generation options
create_trajectory_gif(
    pos_x, pos_y, timestamps, renderer, output_path,
    max_frames=500,  # Limit to first 500 frames (None = all frames)
    subsample=2      # Use every 2nd frame (1 = all frames, 2 = half, etc.)
)
```

### Performance Tips

For large trajectories, you can:

1. **Limit frames**: Set `max_frames=500` to only render the first 500 frames
2. **Subsample**: Set `subsample=2` to render every other frame
3. **Reduce resolution**: Lower the `ppm` value (e.g., `ppm=300`)
4. **Resize output**: Uncomment the resize code in `create_trajectory_gif()`

## Output

The script generates:
- **GIF file**: `trajectory_visualization.gif` in the same directory
- **Console output**: Statistics about the trajectory and rendering progress

### Example Output

```
================================================================================
TRAJECTORY VISUALIZATION
================================================================================
Loading data from: /nfs/data/airhockey/trajectory_data434.hdf5
Data shape: (122, 32)
Number of timesteps: 122
Number of fields per timestep: 32

Trajectory Statistics:
  Number of frames: 122
  Duration: 5.95 seconds
  Average sample rate: 20.5 Hz
  X range: [-0.732, -0.491] meters
  Y range: [-0.202, 0.050] meters

Renderer initialized:
  Table: 1.0436m x 2.1104m
  Render size: 417px x 844px
  Paddle radius: 0.0508m (20px)

Generating GIF with 122 frames...
  Total trajectory frames: 122
  Subsample factor: 1
  Duration: 5.95 seconds

Saving GIF...
  Playback FPS: 20.0
  Duration per frame: 50ms

✓ Saved GIF to: trajectory_visualization.gif
  File size: 1.00 MB

================================================================================
✓ Visualization complete!
================================================================================
```

## Technical Details

### Data Fields

The script uses the following fields from the HDF5 trajectory data:
- **Field 0**: `cur_time` - Unix timestamp (seconds)
- **Field 5**: `pose_x` - End-effector X position (meters)
- **Field 6**: `pose_y` - End-effector Y position (meters)

See `FIELD_DOCUMENTATION.md` for complete field specifications.

### Coordinate Transformations

The script performs the following coordinate transformations:

1. **Robot Frame → Render Frame**: `(x, y) → (y, -x)`
2. **Meters → Pixels**: Position × pixels_per_meter
3. **Origin Shift**: Centers coordinates on the table

This matches the transformation logic in `airhockey/renderers/render.py`.

### Dependencies

- `h5py`: HDF5 file reading
- `numpy`: Array operations
- `opencv-python` (cv2): Image rendering and manipulation
- `imageio`: GIF creation

All dependencies should be installed in the project virtual environment.

## Files

- `visualize_trajectory.py`: Main visualization script
- `trajectory_visualization.gif`: Generated output (not in version control)
- `README.md`: This documentation file

## Related Files

- `../initial_analysis/FIELD_DOCUMENTATION.md`: Field specifications
- `../acceleration_analysis/analyze_acceleration.py`: Example of data loading
- `../../utils.py`: Example of GIF creation (see `save_task_gif`)
- `../../../airhockey/renderers/render.py`: Rendering reference implementation

## Troubleshooting

### Data file not found
```
Error: Data file not found: /nfs/data/airhockey/trajectory_data434.hdf5
```
**Solution**: Update the `data_path` variable in `main()` with the correct path.

### Assets not found
```
FileNotFoundError: Assets folder not found at ...
```
**Solution**: Ensure the `assets/` folder exists with `air_hockey_table.png` and `paddle.png`.

### Large file size
```
File size: 10.50 MB
```
**Solution**: 
- Reduce `ppm` (e.g., from 400 to 300)
- Enable subsampling (e.g., `subsample=2`)
- Uncomment the resize code in `create_trajectory_gif()`
- Limit frames (e.g., `max_frames=500`)

## Future Enhancements

Possible improvements:
- Add trajectory path overlay (showing past positions)
- Visualize velocity vectors
- Show multiple trajectories simultaneously
- Add puck visualization (if puck data available)
- Export as MP4 video format
- Interactive visualization with slider controls


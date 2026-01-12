# Trajectory Visualization Implementation Notes

## Summary

Successfully implemented a trajectory visualization tool that reads HDF5 trajectory data and generates animated GIFs showing the paddle moving on the air hockey table.

**Implementation Date**: November 28, 2025  
**Script**: `visualize_trajectory.py`  
**Status**: ✓ Complete and tested

## What Was Implemented

### Core Functionality

1. **Data Loading** (`load_trajectory_data`)
   - Reads HDF5 files containing trajectory data
   - Extracts the `train_vals` dataset

2. **Position Extraction** (`extract_positions`)
   - Extracts X and Y end-effector positions (Fields 5-6)
   - Extracts timestamps (Field 0)
   - Ignores Z components and rotations as requested

3. **Rendering System** (`TrajectoryRenderer` class)
   - Loads air hockey table and paddle assets
   - Implements coordinate transformation (robot frame → render frame)
   - Renders paddle at specified positions with alpha blending
   - Adds text overlays (frame number, time, position)

4. **GIF Generation** (`create_trajectory_gif`)
   - Creates animated GIF at 20fps (matching data sampling rate)
   - Supports subsampling for large trajectories
   - Supports frame limiting
   - Reports file size and statistics

### Key Features

- ✓ Uses real-world dimensions from the codebase:
  - Table: 1.0436m × 2.1104m
  - Paddle radius: 0.0508m
  
- ✓ Proper coordinate transformations matching `render.py`
  
- ✓ Alpha blending for smooth paddle overlay
  
- ✓ Configurable rendering resolution (pixels per meter)
  
- ✓ Performance optimizations (subsampling, frame limiting)
  
- ✓ Comprehensive error handling and user feedback

## Implementation Details

### Reference Files Used

1. **`analyze_acceleration.py`** - Used as template for:
   - HDF5 data loading pattern
   - Field extraction logic
   - Output directory structure

2. **`render.py`** - Used as reference for:
   - Coordinate transformation: `convert_to_render_coords_sys()`
   - Asset loading (table, paddle images)
   - Alpha blending for paddle overlay
   - Pixel coordinate calculation

3. **`utils.py`** - Used as reference for:
   - GIF creation with `imageio`
   - Frame rate calculation
   - Text overlay approach

4. **`FIELD_DOCUMENTATION.md`** - Used for:
   - Field indices (Field 5: X, Field 6: Y, Field 0: time)
   - Data structure understanding

5. **`air_hockey_osc.py`** - Used for:
   - Real-world table dimensions (2.1104m × 1.0436m)
   - Paddle radius (0.0508m)

### Coordinate Transformation

The script implements the same coordinate transformation as `render.py`:

```python
def convert_to_render_coords(self, pos):
    """Robot frame (x,y) → Render frame (y,-x)"""
    return np.array((pos[1], -pos[0]))
```

This is followed by:
1. Offset to table center: `pos + (width/2, length/2)`
2. Swap axes: `(y, x)`
3. Scale to pixels: `* ppm`

### Data Flow

```
HDF5 File → load_trajectory_data()
           ↓
         train_vals[:, 5:7] → extract_positions()
           ↓
         (pos_x, pos_y, timestamps)
           ↓
         TrajectoryRenderer.render_frame() → frames
           ↓
         create_trajectory_gif() → GIF file
```

## Test Results

Successfully tested with `/nfs/data/airhockey/trajectory_data434.hdf5`:

```
Trajectory Statistics:
  Number of frames: 122
  Duration: 5.95 seconds
  Average sample rate: 20.5 Hz
  X range: [-0.732, -0.491] meters
  Y range: [-0.202, 0.050] meters

Output:
  File: trajectory_visualization.gif
  Size: 1.00 MB
  Frames: 122
  FPS: 20.0
```

## Differences from `create_image_gif.py`

The existing `create_image_gif.py` script:
- Extracts camera images from the HDF5 file (`train_img` dataset)
- Creates GIF directly from recorded images
- Shows what the robot camera saw

Our new `visualize_trajectory.py` script:
- Reads robot position data from HDF5 file (`train_vals` dataset, Fields 5-6)
- **Renders** the trajectory by drawing paddle on table
- Shows paddle movement from robot's perspective
- Only uses X and Y positions (ignores Z and rotations)

Both are complementary tools for different visualization purposes.

## Usage Examples

### Basic Usage
```bash
cd /home/air-hockey/daliu/air-hockey-rl
source .venv/bin/activate
python scripts/trajectory_visualization/visualize/visualize_trajectory.py
```

### With Different Data File
Edit `main()` function:
```python
data_path = Path('/path/to/other/trajectory_data.hdf5')
```

### For Large Trajectories
```python
create_trajectory_gif(
    pos_x, pos_y, timestamps, renderer, output_path,
    max_frames=500,  # Only first 500 frames
    subsample=2      # Every other frame
)
```

### Higher Resolution
```python
renderer = TrajectoryRenderer(
    table_width=1.0436,
    table_length=2.1104,
    paddle_radius=0.0508,
    ppm=600  # Higher resolution (default: 400)
)
```

## File Structure

```
scripts/trajectory_visualization/visualize/
├── visualize_trajectory.py          # Main implementation (NEW)
├── create_image_gif.py              # Existing camera image GIF creator
├── README.md                        # User documentation (NEW)
├── IMPLEMENTATION_NOTES.md          # This file (NEW)
└── trajectory_visualization.gif     # Generated output (gitignored)
```

## Configuration Options

### In `TrajectoryRenderer.__init__()`:
- `table_width`: Table width in meters (default: 1.0436)
- `table_length`: Table length in meters (default: 2.1104)
- `paddle_radius`: Paddle radius in meters (default: 0.0508)
- `ppm`: Pixels per meter for rendering (default: 400)

### In `create_trajectory_gif()`:
- `max_frames`: Maximum frames to render (default: None = all)
- `subsample`: Subsample factor (default: 1 = all frames)

### In `main()`:
- `data_path`: Path to HDF5 trajectory file
- `output_path`: Where to save the GIF

## Performance Considerations

### File Size vs Quality Trade-offs

| Configuration | File Size | Quality | Use Case |
|--------------|-----------|---------|----------|
| ppm=300, subsample=1 | ~0.5 MB | Good | Quick preview |
| ppm=400, subsample=1 | ~1.0 MB | Better | Standard use |
| ppm=600, subsample=1 | ~2.5 MB | Best | High quality |
| ppm=400, subsample=2 | ~0.5 MB | Good | Long trajectories |

### Rendering Speed

- ~2-3 frames per second on typical hardware
- 122 frames takes ~40-60 seconds to render
- Consider subsampling for trajectories > 500 frames

## Potential Enhancements

Future improvements that could be added:

1. **Trajectory Path Overlay**
   - Draw trail showing previous paddle positions
   - Color-code by time or velocity

2. **Velocity Vectors**
   - Draw arrows showing paddle velocity
   - Similar to acceleration arrows in `render.py`

3. **Multi-trajectory Comparison**
   - Overlay multiple trajectories
   - Side-by-side comparison

4. **Puck Visualization**
   - Add puck if data available (Fields 32-34)
   - Show puck-paddle interactions

5. **Video Export**
   - MP4 format option
   - Higher quality, better compression

6. **Interactive Mode**
   - Slider to scrub through time
   - Pause/play controls
   - Zoom and pan

7. **Statistics Overlay**
   - Real-time velocity display
   - Distance traveled
   - Path smoothness metrics

## Maintenance Notes

- Script is self-contained with no external dependencies beyond standard packages
- Assets path is computed relative to script location (robust to moves)
- All hardcoded values are documented and configurable
- Error messages guide users to fix common issues

## Related Documentation

- `README.md` - User-facing documentation
- `../initial_analysis/FIELD_DOCUMENTATION.md` - Field specifications
- `../acceleration_analysis/README.md` - Related analysis tools

## Conclusion

The implementation successfully meets all requirements:
- ✓ Reads trajectory data from HDF5
- ✓ Visualizes paddle using X and Y positions only
- ✓ Ignores Z components and rotations
- ✓ Respects 20Hz data rate
- ✓ Uses rendering approach from `render.py`
- ✓ Creates GIF similar to `utils.py` approach
- ✓ Follows data loading pattern from `analyze_acceleration.py`

The tool is ready for use and can be easily extended for future needs.


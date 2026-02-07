#!/usr/bin/env python3
"""
Visualize real robot trajectory data from HDF5 files as GIFs matching Box2D simulator style.

This script reads trajectory data collected from the real UR5e robot and renders it
using the same visual style as the Box2D simulator, ensuring exact visual alignment.

Key features:
- Loads HDF5 trajectory files from /nfs/data/airhockey/
- Transforms robot frame coordinates to simulation table frame
- Uses the same rendering assets and logic as Box2D simulator
- Generates GIFs with paddle position and velocity visualization

Usage:
    python visualize_real_trajectory.py /nfs/data/airhockey/trajectory_data434.hdf5
    python visualize_real_trajectory.py /nfs/data/airhockey/trajectory_data434.hdf5 --output-dir ./output
    python visualize_real_trajectory.py /nfs/data/airhockey/trajectory_data434.hdf5 --subsample 2 --max-frames 100
"""

import h5py
import numpy as np
import cv2
import imageio
from pathlib import Path
import argparse
import os
import sys


def load_trajectory_data(filepath):
    """
    Load trajectory data from HDF5 file.
    
    Args:
        filepath: Path to HDF5 trajectory file
        
    Returns:
        numpy.ndarray: Train values array (N, 32)
    """
    print(f"Loading trajectory data from: {filepath}")
    with h5py.File(filepath, 'r') as f:
        train_vals = f['train_vals'][:]
        print(f"  Shape: {train_vals.shape}")
        print(f"  Timesteps: {train_vals.shape[0]}")
    return train_vals


def extract_paddle_data(train_vals):
    """
    Extract paddle position, velocity, and timestamps from trajectory data.
    
    According to FIELD_DOCUMENTATION.md:
    - Field 0: cur_time (Unix timestamp)
    - Field 5: pose_x (X position in meters, robot frame)
    - Field 6: pose_y (Y position in meters, robot frame)
    - Field 11: speed_vx (X velocity in m/s)
    - Field 12: speed_vy (Y velocity in m/s)
    
    Args:
        train_vals: Array of trajectory data (N, 32)
        
    Returns:
        dict: Dictionary with keys 'pos_x', 'pos_y', 'vel_x', 'vel_y', 'timestamps'
    """
    data = {
        'pos_x': train_vals[:, 5],      # Robot frame X position
        'pos_y': train_vals[:, 6],      # Robot frame Y position
        'vel_x': train_vals[:, 11],     # X velocity
        'vel_y': train_vals[:, 12],     # Y velocity
        'timestamps': train_vals[:, 0]  # Unix timestamps
    }
    return data


class RealTrajectoryRenderer:
    """
    Renderer for real robot trajectory visualization matching Box2D simulator style.
    
    This renderer transforms real robot coordinates to simulation coordinates and
    uses the same rendering logic as the Box2D AirHockeyRenderer.
    """
    
    def __init__(self, 
                 table_length=1.9304,
                 table_width=0.8636,
                 paddle_radius=0.0508,
                 render_size=360,
                 robot_x_offset=1.2,
                 orientation='vertical'):
        """
        Initialize renderer with simulation-matching parameters.
        
        Args:
            table_length: Length of air hockey table in meters (simulation default: 1.9304m)
            table_width: Width of air hockey table in meters (simulation default: 0.8636m)
            paddle_radius: Radius of paddle in meters (0.0508m)
            render_size: Render size in pixels (default: 360, matching simulation)
            robot_x_offset: Robot base offset from table center in X (real world: ~1.2m)
            orientation: Render orientation ('vertical' or 'horizontal')
        """
        self.length = table_length
        self.width = table_width
        self.paddle_radius = paddle_radius
        self.robot_x_offset = robot_x_offset
        self.orientation = orientation
        
        # Calculate pixels per meter and render dimensions (matching render.py logic)
        self.ppm = render_size / self.width
        self.render_width = int(render_size)
        self.render_length = int(self.ppm * self.length)
        
        print(f"Renderer initialized:")
        print(f"  Table dimensions: {self.length}m x {self.width}m")
        print(f"  Render size: {self.render_length}px x {self.render_width}px")
        print(f"  Pixels per meter: {self.ppm:.1f}")
        print(f"  Paddle radius: {self.paddle_radius}m ({int(self.paddle_radius * self.ppm)}px)")
        print(f"  Robot X offset: {self.robot_x_offset}m")
        print(f"  Orientation: {self.orientation}")
        
        # Load assets (matching render.py)
        self._load_assets()
        
    def _load_assets(self):
        """Load table and paddle images from assets folder."""
        # Find assets folder relative to this script
        script_dir = Path(__file__).parent
        assets_folder = script_dir / '../../../assets'
        assets_folder = assets_folder.resolve()
        
        if not assets_folder.exists():
            raise FileNotFoundError(f"Assets folder not found at {assets_folder}")
        
        print(f"  Loading assets from: {assets_folder}")
        
        # Load table image
        table_path = assets_folder / 'air_hockey_table.png'
        self.table_img = cv2.imread(str(table_path))
        if self.table_img is None:
            raise FileNotFoundError(f"Could not load table image from {table_path}")
        
        # Rotate and resize table image (matching render.py lines 63-65)
        self.table_img = cv2.rotate(self.table_img, cv2.ROTATE_90_CLOCKWISE)
        self.table_img = cv2.resize(self.table_img, (self.render_length, self.render_width))
        
        # Load paddle image
        paddle_path = assets_folder / 'paddle.png'
        self.paddle_img_orig = cv2.imread(str(paddle_path), cv2.IMREAD_UNCHANGED)
        if self.paddle_img_orig is None:
            raise FileNotFoundError(f"Could not load paddle image from {paddle_path}")
        
        # Pre-resize paddle image
        radius_px = int(self.paddle_radius * self.ppm)
        diameter = 2 * radius_px
        self.paddle_img = cv2.resize(self.paddle_img_orig, (diameter, diameter))
        
    def convert_to_render_coords_sys(self, pos):
        """
        Convert position to render coordinates (matching render.py line 74).
        
        This swaps Y and X, and negates X: (x, y) -> (y, -x)
        
        Args:
            pos: Position [x, y] in table frame (meters)
            
        Returns:
            numpy.ndarray: Position in render coordinates
        """
        return np.array((pos[1], -pos[0]))
    
    def robot_to_table_frame(self, pos_x, pos_y):
        """
        Transform robot frame coordinates to table frame coordinates.
        
        The robot base is offset from the table center by robot_x_offset in the X direction.
        
        Args:
            pos_x: X position in robot frame (meters)
            pos_y: Y position in robot frame (meters)
            
        Returns:
            tuple: (x, y) in table frame (meters)
        """
        # Apply X offset to transform from robot frame to table frame
        table_x = pos_x + self.robot_x_offset
        table_y = pos_y
        return table_x, table_y
    
    def position_to_pixel_coords(self, pos_x, pos_y):
        """
        Convert robot frame position to pixel coordinates.
        
        This follows the same logic as render.py draw_circle_with_image (lines 241-242).
        
        Args:
            pos_x: X position in robot frame (meters)
            pos_y: Y position in robot frame (meters)
            
        Returns:
            numpy.ndarray: Pixel coordinates [x, y]
        """
        # Transform to table frame
        table_x, table_y = self.robot_to_table_frame(pos_x, pos_y)
        
        # Convert to render coordinates
        pos_render = self.convert_to_render_coords_sys([table_x, table_y])
        
        # Convert to pixel coordinates (matching render.py lines 241-242)
        center = np.array(pos_render) + np.array((self.width / 2, self.length / 2))
        center = np.array((center[1], center[0])) * self.ppm
        
        return center.astype(int)
    
    def draw_paddle(self, frame, pos_x, pos_y):
        """
        Draw paddle at given position on the frame.
        
        Uses alpha blending to overlay paddle image (matching render.py draw_circle_with_image).
        
        Args:
            frame: Image to draw on (will be modified in place)
            pos_x: X position in robot frame (meters)
            pos_y: Y position in robot frame (meters)
        """
        center = self.position_to_pixel_coords(pos_x, pos_y)
        
        # Calculate paddle overlay position
        radius = int(self.paddle_radius * self.ppm)
        top_left = center - radius
        diameter = 2 * radius
        bottom_right = top_left + diameter
        
        # Calculate valid regions for overlay (handle edges)
        x_start = max(0, -top_left[0])
        y_start = max(0, -top_left[1])
        
        frame_top_left = np.array([max(0, top_left[0]), max(0, top_left[1])])
        frame_bottom_right = np.array([
            min(frame.shape[1], bottom_right[0]),
            min(frame.shape[0], bottom_right[1])
        ])
        
        y_end_offset = bottom_right[1] - frame_bottom_right[1]
        x_end_offset = bottom_right[0] - frame_bottom_right[0]
        y_end = self.paddle_img.shape[0] - y_end_offset
        x_end = self.paddle_img.shape[1] - x_end_offset
        
        # Check if paddle is within bounds
        if (y_start >= self.paddle_img.shape[0] or x_start >= self.paddle_img.shape[1] or 
            y_end <= 0 or x_end <= 0):
            return  # Paddle is out of bounds
        
        # Extract paddle region and apply alpha mask (matching render.py line 293)
        paddle_region = self.paddle_img[y_start:y_end, x_start:x_end]
        if paddle_region.shape[0] == 0 or paddle_region.shape[1] == 0:
            return
        
        # Apply alpha blending if paddle has alpha channel
        if paddle_region.shape[2] == 4:
            mask = paddle_region[:, :, 3] > 0
            frame[frame_top_left[1]:frame_bottom_right[1], 
                  frame_top_left[0]:frame_bottom_right[0]][mask] = paddle_region[:, :, :3][mask]
    
    def draw_velocity_arrow(self, frame, pos_x, pos_y, vel_x, vel_y):
        """
        Draw velocity vector as an arrow on the frame.
        
        Args:
            frame: Image to draw on (will be modified in place)
            pos_x: X position in robot frame (meters)
            pos_y: Y position in robot frame (meters)
            vel_x: X velocity in m/s
            vel_y: Y velocity in m/s
        """
        start = self.position_to_pixel_coords(pos_x, pos_y)
        
        # Calculate velocity magnitude
        vel_magnitude = np.sqrt(vel_x**2 + vel_y**2)
        
        if vel_magnitude < 0.01:  # Don't draw very small velocities
            return
        
        # Scale velocity to pixels for visualization
        vel_scale = 100.0  # Scale factor for visibility
        
        # Transform velocity vector to render coordinates (same as position differences)
        vel_render = self.convert_to_render_coords_sys([vel_x, vel_y])
        
        # Convert to pixel space
        vel_vec_px = np.array([vel_render[1], vel_render[0]]) * vel_scale
        
        end = start + vel_vec_px.astype(int)
        
        # Draw arrow (orange color for velocity)
        color = (0, 165, 255)  # Orange in BGR
        thickness = 2
        tip_length = 0.25
        
        cv2.arrowedLine(frame, tuple(start), tuple(end), color, thickness, 
                       tipLength=tip_length)
        
        # Add velocity magnitude label
        if vel_magnitude > 0.1:  # Only show text for significant velocities
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            text = f"{vel_magnitude:.2f}m/s"
            text_pos = (end[0] + 5, end[1])
            
            # Draw text background for readability
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, 1)
            cv2.rectangle(frame, 
                         (text_pos[0] - 2, text_pos[1] - text_height - 2),
                         (text_pos[0] + text_width + 2, text_pos[1] + baseline + 2),
                         (255, 255, 255), -1)
            
            cv2.putText(frame, text, text_pos, font, font_scale, color, 1)
    
    def render_frame(self, pos_x, pos_y, vel_x=None, vel_y=None, 
                    timestep=None, total_time=None):
        """
        Render a single frame with paddle at given position.
        
        Args:
            pos_x: X position in robot frame (meters)
            pos_y: Y position in robot frame (meters)
            vel_x: X velocity in m/s (optional)
            vel_y: Y velocity in m/s (optional)
            timestep: Optional timestep number to display
            total_time: Optional total elapsed time to display
            
        Returns:
            numpy.ndarray: BGR image array
        """
        frame = self.table_img.copy()
        
        # Draw paddle
        self.draw_paddle(frame, pos_x, pos_y)
        
        # Draw velocity arrow if provided
        if vel_x is not None and vel_y is not None:
            self.draw_velocity_arrow(frame, pos_x, pos_y, vel_x, vel_y)
        
        # Add text overlays
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (0, 0, 0)  # Black
        line_type = 2
        
        y_offset = 25
        
        if timestep is not None:
            text = f"Frame: {timestep}"
            cv2.putText(frame, text, (10, y_offset), font, font_scale, font_color, line_type)
            y_offset += 25
        
        if total_time is not None:
            text = f"Time: {total_time:.2f}s"
            cv2.putText(frame, text, (10, y_offset), font, font_scale, font_color, line_type)
            y_offset += 25
        
        # Add position text (robot frame)
        text = f"Pos: ({pos_x:.3f}, {pos_y:.3f})m"
        cv2.putText(frame, text, (10, y_offset), font, font_scale, font_color, line_type)
        y_offset += 25
        
        # Add velocity text if provided
        if vel_x is not None and vel_y is not None:
            vel_mag = np.sqrt(vel_x**2 + vel_y**2)
            text = f"Vel: ({vel_x:.2f}, {vel_y:.2f})m/s [{vel_mag:.2f}]"
            cv2.putText(frame, text, (10, y_offset), font, font_scale, (0, 165, 255), line_type)
        
        # Apply orientation rotation if vertical (matching render.py line 487)
        if self.orientation == 'vertical':
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        return frame


def create_trajectory_gif(paddle_data, renderer, output_path, 
                         max_frames=None, subsample=1, fps=20):
    """
    Create GIF from trajectory data.
    
    Args:
        paddle_data: Dictionary with paddle position, velocity, and timestamps
        renderer: RealTrajectoryRenderer instance
        output_path: Where to save the GIF
        max_frames: Optional maximum number of frames to render
        subsample: Subsample factor (1 = all frames, 2 = every other frame, etc.)
        fps: Frames per second for GIF playback
    """
    # Extract data
    pos_x = paddle_data['pos_x']
    pos_y = paddle_data['pos_y']
    vel_x = paddle_data['vel_x']
    vel_y = paddle_data['vel_y']
    timestamps = paddle_data['timestamps']
    
    # Calculate relative time
    relative_time = timestamps - timestamps[0]
    
    # Determine which frames to render
    n_frames = len(pos_x)
    if max_frames is not None:
        n_frames = min(n_frames, max_frames)
    
    # Apply subsampling
    indices = np.arange(0, n_frames, subsample)
    
    print(f"\nGenerating GIF:")
    print(f"  Total trajectory frames: {len(pos_x)}")
    print(f"  Frames to render: {len(indices)}")
    print(f"  Subsample factor: {subsample}")
    print(f"  Duration: {relative_time[-1]:.2f} seconds")
    print(f"  Playback FPS: {fps}")
    
    frames = []
    
    for idx, i in enumerate(indices):
        if idx % 50 == 0 and idx > 0:
            print(f"  Rendering frame {idx}/{len(indices)}...")
        
        # Render frame
        frame = renderer.render_frame(
            pos_x[i], 
            pos_y[i],
            vel_x=vel_x[i],
            vel_y=vel_y[i],
            timestep=i,
            total_time=relative_time[i]
        )
        
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to match Box2D simulator output (160px width, maintaining aspect ratio)
        aspect_ratio = frame_rgb.shape[1] / frame_rgb.shape[0]
        target_width = 160
        target_height = int(target_width / aspect_ratio)
        frame_rgb = cv2.resize(frame_rgb, (target_width, target_height))
        
        frames.append(frame_rgb)
    
    # Save as GIF
    duration = int(1000 / fps)  # milliseconds per frame
    
    print(f"\nSaving GIF...")
    print(f"  Output path: {output_path}")
    print(f"  Duration per frame: {duration}ms")
    
    imageio.mimsave(output_path, frames, format='GIF', loop=0, duration=duration)
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  File size: {file_size_mb:.2f} MB")


def print_trajectory_statistics(paddle_data):
    """Print statistics about the trajectory data."""
    pos_x = paddle_data['pos_x']
    pos_y = paddle_data['pos_y']
    vel_x = paddle_data['vel_x']
    vel_y = paddle_data['vel_y']
    timestamps = paddle_data['timestamps']
    
    relative_time = timestamps - timestamps[0]
    vel_magnitude = np.sqrt(vel_x**2 + vel_y**2)
    
    print("\nTrajectory Statistics:")
    print(f"  Number of frames: {len(pos_x)}")
    print(f"  Duration: {relative_time[-1]:.2f} seconds")
    print(f"  Average sample rate: {len(pos_x) / relative_time[-1]:.1f} Hz")
    print(f"\n  Position X (robot frame):")
    print(f"    Range: [{pos_x.min():.3f}, {pos_x.max():.3f}] meters")
    print(f"    Mean: {pos_x.mean():.3f} meters")
    print(f"  Position Y (robot frame):")
    print(f"    Range: [{pos_y.min():.3f}, {pos_y.max():.3f}] meters")
    print(f"    Mean: {pos_y.mean():.3f} meters")
    print(f"\n  Velocity magnitude:")
    print(f"    Range: [{vel_magnitude.min():.3f}, {vel_magnitude.max():.3f}] m/s")
    print(f"    Mean: {vel_magnitude.mean():.3f} m/s")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualize real robot trajectory data as GIF matching Box2D simulator style',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to HDF5 trajectory file (e.g., /nfs/data/airhockey/trajectory_data434.hdf5)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: creates directory based on input filename)'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Maximum number of frames to render'
    )
    parser.add_argument(
        '--subsample',
        type=int,
        default=1,
        help='Subsample factor (1=all frames, 2=every other frame, etc.)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=20,
        help='Frames per second for GIF playback'
    )
    parser.add_argument(
        '--table-length',
        type=float,
        default=1.9304,
        help='Table length in meters (simulation default: 1.9304)'
    )
    parser.add_argument(
        '--table-width',
        type=float,
        default=0.8636,
        help='Table width in meters (simulation default: 0.8636)'
    )
    parser.add_argument(
        '--robot-x-offset',
        type=float,
        default=1.2,
        help='Robot base X offset from table center in meters'
    )
    
    return parser.parse_args()


def main():
    """Main function to generate trajectory visualization."""
    args = parse_args()
    
    # Check if input file exists
    data_path = Path(args.input_file)
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        sys.exit(1)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Create output directory based on input filename
        input_stem = data_path.stem
        output_dir = Path(__file__).parent / f"{input_stem}_visualization"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'trajectory_visualization.gif'
    
    print("=" * 80)
    print("REAL ROBOT TRAJECTORY VISUALIZATION")
    print("=" * 80)
    print(f"\nInput file: {data_path}")
    print(f"Output directory: {output_dir}")
    print(f"Output file: {output_path}")
    
    # Load trajectory data
    train_vals = load_trajectory_data(data_path)
    
    # Extract paddle data
    paddle_data = extract_paddle_data(train_vals)
    
    # Print statistics
    print_trajectory_statistics(paddle_data)
    
    # Initialize renderer with simulation-matching parameters
    renderer = RealTrajectoryRenderer(
        table_length=args.table_length,
        table_width=args.table_width,
        paddle_radius=0.0508,
        render_size=360,
        robot_x_offset=args.robot_x_offset,
        orientation='vertical'
    )
    
    # Generate GIF
    create_trajectory_gif(
        paddle_data,
        renderer,
        output_path,
        max_frames=args.max_frames,
        subsample=args.subsample,
        fps=args.fps
    )
    
    print("\n" + "=" * 80)
    print("✓ Visualization complete!")
    print("=" * 80)
    print(f"\nTo view the GIF, open: {output_path}")


if __name__ == '__main__':
    main()

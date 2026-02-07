#!/usr/bin/env python3
"""
Visualize trajectory data by creating a GIF showing the paddle moving on the air hockey table.
This script reads HDF5 trajectory data and renders:
- Current paddle position (X, Y)
- Target paddle position (green crosshair)
- Force vector on the paddle (red arrow)

The trajectories are sampled at 20Hz.
"""

import h5py
import numpy as np
import cv2
import imageio
from pathlib import Path
import sys
import os
import argparse


def load_trajectory_data(filepath):
    """
    Load trajectory data from HDF5 file.
    
    Args:
        filepath: Path to HDF5 file
        
    Returns:
        numpy.ndarray: Train values array
    """
    print(f"Loading data from: {filepath}")
    with h5py.File(filepath, 'r') as f:
        train_vals = f['train_vals'][:]
        print(f"Data shape: {train_vals.shape}")
        print(f"Number of timesteps: {train_vals.shape[0]}")
        print(f"Number of fields per timestep: {train_vals.shape[1]}")
    return train_vals


def extract_positions(train_vals):
    """
    Extract X and Y positions, target positions, and forces from trajectory data.
    According to FIELD_DOCUMENTATION:
    - Field 0: cur_time (Unix timestamp)
    - Fields 5-10: pose (xyz rxryrz)
    - Fields 17-22: force (xyz rxryrz)
    - Fields 26-31: desired_pose (xyz rxryrz)
    
    Args:
        train_vals: Array of trajectory data
        
    Returns:
        tuple: (pos_x, pos_y, target_x, target_y, force_x, force_y, timestamps) arrays
    """
    pos_x = train_vals[:, 5]  # Field 5: X position
    pos_y = train_vals[:, 6]  # Field 6: Y position
    target_x = train_vals[:, 26]  # Field 26: Target X position
    target_y = train_vals[:, 27]  # Field 27: Target Y position
    force_x = train_vals[:, 17]  # Field 17: Force in X direction
    force_y = train_vals[:, 18]  # Field 18: Force in Y direction
    timestamps = train_vals[:, 0]  # Field 0: Unix timestamp
    
    return pos_x, pos_y, target_x, target_y, force_x, force_y, timestamps


class TrajectoryRenderer:
    """
    Renderer for trajectory visualization on air hockey table.
    """
    
    def __init__(self, table_width=1.0436, table_length=2.1104, paddle_radius=0.0508, ppm=400, robot_x_offset=1.2):
        """
        Initialize renderer for trajectory visualization.
        
        Args:
            table_width: Width of air hockey table in meters (real world: 1.0436m)
            table_length: Length of air hockey table in meters (real world: 2.1104m)
            paddle_radius: Radius of paddle in meters (real world: 0.0508m)
            ppm: Pixels per meter for rendering
            robot_x_offset: Robot base offset from table center in X (real world: 1.2m)
        """
        self.width = table_width
        self.length = table_length
        self.paddle_radius = paddle_radius
        self.ppm = ppm
        self.robot_x_offset = robot_x_offset  # Robot frame to table frame offset
        self.render_width = int(table_width * ppm)
        self.render_length = int(table_length * ppm)
        
        # Get path to assets folder (relative to this script)
        script_dir = Path(__file__).parent
        assets_folder = script_dir / '../../../assets'
        assets_folder = assets_folder.resolve()
        
        if not assets_folder.exists():
            raise FileNotFoundError(f"Assets folder not found at {assets_folder}")
        
        print(f"Loading assets from: {assets_folder}")
        
        # Load table image
        table_path = assets_folder / 'air_hockey_table.png'
        self.table_img = cv2.imread(str(table_path))
        if self.table_img is None:
            raise FileNotFoundError(f"Could not load table image from {table_path}")
        
        # Rotate and resize table image (matching render.py)
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
        
        print(f"Renderer initialized:")
        print(f"  Table: {self.width}m x {self.length}m")
        print(f"  Render size: {self.render_width}px x {self.render_length}px")
        print(f"  Paddle radius: {self.paddle_radius}m ({radius_px}px)")
        print(f"  Robot X offset: {self.robot_x_offset}m (robot frame → table frame)")
    
    def convert_to_render_coords(self, pos):
        """
        Convert robot coordinates to render coordinates.
        From render.py: return np.array((pos[1], -pos[0]))
        
        Args:
            pos: Position [x, y] in robot frame (meters)
            
        Returns:
            numpy.ndarray: Position in render coordinates
        """
        return np.array((pos[1], -pos[0]))
    
    def robot_to_pixel_coords(self, pos_x, pos_y):
        """
        Convert robot frame coordinates to pixel coordinates.
        
        Args:
            pos_x: X position in meters (robot frame)
            pos_y: Y position in meters (robot frame)
            
        Returns:
            numpy.ndarray: Pixel coordinates [x, y]
        """
        # Apply robot frame to table frame offset
        pos_x_table = pos_x + self.robot_x_offset
        pos_y_table = pos_y
        
        # Convert to render coordinates
        pos = self.convert_to_render_coords([pos_x_table, pos_y_table])
        
        # Convert to pixel coordinates (matching render.py logic)
        center = np.array(pos) + np.array((self.width / 2, self.length / 2))
        center = np.array((center[1], center[0])) * self.ppm
        
        return center.astype(int)
    
    def draw_paddle(self, frame, pos_x, pos_y):
        """
        Draw paddle at given position on the frame.
        
        Args:
            frame: Image to draw on (will be modified in place)
            pos_x: X position in meters (robot frame)
            pos_y: Y position in meters (robot frame)
        """
        center = self.robot_to_pixel_coords(pos_x, pos_y)
        
        # Calculate paddle overlay position
        radius = int(self.paddle_radius * self.ppm)
        top_left = (center - radius).astype(int)
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
        
        # Extract the region of the paddle image to overlay
        paddle_region = self.paddle_img[
            y_start:y_start + (frame_bottom_right[1] - frame_top_left[1]),
            x_start:x_start + (frame_bottom_right[0] - frame_top_left[0])
        ]
        
        if paddle_region.shape[0] == 0 or paddle_region.shape[1] == 0:
            # Paddle is completely off screen
            return
        
        # Alpha blend the paddle onto the frame (if paddle has alpha channel)
        if paddle_region.shape[2] == 4:  # Has alpha channel
            alpha = paddle_region[:, :, 3:4] / 255.0
            rgb = paddle_region[:, :, :3]
            
            # Get the frame region to blend with
            frame_region = frame[
                frame_top_left[1]:frame_bottom_right[1],
                frame_top_left[0]:frame_bottom_right[0]
            ]
            
            # Blend
            blended = (alpha * rgb + (1 - alpha) * frame_region).astype(np.uint8)
            
            # Write back to frame
            frame[
                frame_top_left[1]:frame_bottom_right[1],
                frame_top_left[0]:frame_bottom_right[0]
            ] = blended
        else:
            # No alpha channel, just copy
            frame[
                frame_top_left[1]:frame_bottom_right[1],
                frame_top_left[0]:frame_bottom_right[0]
            ] = paddle_region
    
    def draw_target(self, frame, target_x, target_y):
        """
        Draw target position marker on the frame.
        
        Args:
            frame: Image to draw on (will be modified in place)
            target_x: Target X position in meters (robot frame)
            target_y: Target Y position in meters (robot frame)
        """
        center = self.robot_to_pixel_coords(target_x, target_y)
        
        # Draw a green crosshair for the target
        color = (0, 255, 0)  # Green in BGR
        thickness = 2
        size = int(self.paddle_radius * self.ppm * 0.8)  # Slightly smaller than paddle
        
        # Draw crosshair lines
        cv2.line(frame, 
                (center[0] - size, center[1]), 
                (center[0] + size, center[1]), 
                color, thickness)
        cv2.line(frame, 
                (center[0], center[1] - size), 
                (center[0], center[1] + size), 
                color, thickness)
        
        # Draw circle around target
        cv2.circle(frame, tuple(center), size, color, thickness)
    
    def draw_force_arrow(self, frame, pos_x, pos_y, force_x, force_y):
        """
        Draw force vector as an arrow on the frame.
        
        Args:
            frame: Image to draw on (will be modified in place)
            pos_x: X position in meters (robot frame)
            pos_y: Y position in meters (robot frame)
            force_x: Force in X direction (Newtons)
            force_y: Force in Y direction (Newtons)
        """
        start = self.robot_to_pixel_coords(pos_x, pos_y)
        
        # Scale force to pixels (adjust scale factor for visibility)
        # Force is in Newtons, we'll scale it for visualization
        force_scale = 0.05  # Reduced from 5.0 to make arrows shorter
        
        # Force vector in robot frame
        force_vec = np.array([force_x, force_y])
        force_magnitude = np.linalg.norm(force_vec)
        
        if force_magnitude < 0.01:  # Don't draw very small forces
            return
        
        # Convert force direction to render coordinates
        # The force vector needs the same transformation as position differences
        # Apply render coordinate transformation: (y, -x)
        force_render = self.convert_to_render_coords([force_x, force_y])
        
        # Scale and convert to pixels
        # Note: we need to swap again when going to pixel space
        force_vec_px = np.array([force_render[1], force_render[0]]) * force_scale * self.ppm
        
        # Negate to fix direction (forces point opposite to the transformation)
        force_vec_px = -force_vec_px
        
        end = start + force_vec_px.astype(int)
        
        # Draw arrow
        color = (0, 0, 255)  # Red in BGR
        thickness = 2
        tip_length = 0.3
        
        cv2.arrowedLine(frame, tuple(start), tuple(end), color, thickness, 
                       tipLength=tip_length)
        
        # Add force magnitude label
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        text = f"{force_magnitude:.1f}N"
        text_pos = (end[0] + 5, end[1])
        cv2.putText(frame, text, text_pos, font, font_scale, color, 1)
    
    def render_frame(self, pos_x, pos_y, target_x=None, target_y=None, 
                    force_x=None, force_y=None, timestep=None, total_time=None):
        """
        Render a single frame with paddle at given position.
        
        Args:
            pos_x: X position in meters (robot frame)
            pos_y: Y position in meters (robot frame)
            target_x: Target X position in meters (robot frame)
            target_y: Target Y position in meters (robot frame)
            force_x: Force in X direction (Newtons)
            force_y: Force in Y direction (Newtons)
            timestep: Optional timestep number to display
            total_time: Optional total elapsed time to display
            
        Returns:
            numpy.ndarray: BGR image array
        """
        frame = self.table_img.copy()
        
        # Draw target position first (so it appears behind paddle)
        if target_x is not None and target_y is not None:
            self.draw_target(frame, target_x, target_y)
        
        # Draw paddle
        self.draw_paddle(frame, pos_x, pos_y)
        
        # Draw force arrow last (so it appears on top)
        if force_x is not None and force_y is not None:
            self.draw_force_arrow(frame, pos_x, pos_y, force_x, force_y)
        
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
        
        # Add target position text
        if target_x is not None and target_y is not None:
            text = f"Target: ({target_x:.3f}, {target_y:.3f})m"
            cv2.putText(frame, text, (10, y_offset), font, font_scale, (0, 255, 0), line_type)
            y_offset += 25
        
        # Add force text
        if force_x is not None and force_y is not None:
            force_mag = np.sqrt(force_x**2 + force_y**2)
            text = f"Force: ({force_x:.1f}, {force_y:.1f})N [{force_mag:.1f}N]"
            cv2.putText(frame, text, (10, y_offset), font, font_scale, (0, 0, 255), line_type)
        
        return frame


def create_trajectory_gif(pos_x, pos_y, target_x, target_y, force_x, force_y, 
                         timestamps, renderer, output_path, 
                         max_frames=None, subsample=1):
    """
    Create GIF from trajectory data.
    
    Args:
        pos_x: Array of X positions (meters)
        pos_y: Array of Y positions (meters)
        target_x: Array of target X positions (meters)
        target_y: Array of target Y positions (meters)
        force_x: Array of forces in X direction (Newtons)
        force_y: Array of forces in Y direction (Newtons)
        timestamps: Array of timestamps
        renderer: TrajectoryRenderer instance
        output_path: Where to save the GIF
        max_frames: Optional maximum number of frames to render
        subsample: Subsample factor (1 = all frames, 2 = every other frame, etc.)
    """
    # Calculate relative time
    relative_time = timestamps - timestamps[0]
    
    # Determine which frames to render
    n_frames = len(pos_x)
    if max_frames is not None:
        n_frames = min(n_frames, max_frames)
    
    # Apply subsampling
    indices = np.arange(0, n_frames, subsample)
    
    print(f"\nGenerating GIF with {len(indices)} frames...")
    print(f"  Total trajectory frames: {len(pos_x)}")
    print(f"  Subsample factor: {subsample}")
    print(f"  Duration: {relative_time[-1]:.2f} seconds")
    
    frames = []
    
    for idx, i in enumerate(indices):
        if idx % 50 == 0:
            print(f"  Rendering frame {idx}/{len(indices)}...")
        
        # Render frame
        frame = renderer.render_frame(
            pos_x[i], 
            pos_y[i],
            target_x=target_x[i],
            target_y=target_y[i],
            force_x=force_x[i],
            force_y=force_y[i],
            timestep=i,
            total_time=relative_time[i]
        )
        
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Optionally resize for smaller file size (keep aspect ratio)
        # Uncomment the following lines to resize:
        # target_width = 480
        # aspect_ratio = frame_rgb.shape[1] / frame_rgb.shape[0]
        # target_height = int(target_width / aspect_ratio)
        # frame_rgb = cv2.resize(frame_rgb, (target_width, target_height))
        
        frames.append(frame_rgb)
    
    # Save as GIF
    # Calculate FPS based on actual data sampling rate and subsampling
    actual_fps = len(pos_x) / relative_time[-1]  # Actual sampling rate
    playback_fps = actual_fps / subsample  # Adjust for subsampling
    playback_fps = min(playback_fps, 20)  # Cap at 20 fps for smooth playback
    
    duration = int(1000 / playback_fps)  # milliseconds per frame
    
    print(f"\nSaving GIF...")
    print(f"  Playback FPS: {playback_fps:.1f}")
    print(f"  Duration per frame: {duration}ms")
    
    imageio.mimsave(output_path, frames, format='GIF', loop=0, duration=duration)
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n✓ Saved GIF to: {output_path}")
    print(f"  File size: {file_size_mb:.2f} MB")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualize trajectory data as an animated GIF showing paddle position, target, and forces.'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to HDF5 trajectory file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: creates a directory based on input filename)'
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
    return parser.parse_args()


def main():
    """Main function to generate trajectory visualization."""
    
    # Parse arguments
    args = parse_args()
    
    # Configuration
    data_path = Path(args.input_file)
    
    # Check if data file exists
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        sys.exit(1)
    
    # Determine output directory based on input filename
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Create output directory based on input filename
        # e.g., trajectory_data434.hdf5 -> trajectory_data434_visualization
        input_stem = data_path.stem  # filename without extension
        output_dir = Path(__file__).parent / f"{input_stem}_visualization"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'trajectory_visualization.gif'
    
    print(f"Input file: {data_path}")
    print(f"Output directory: {output_dir}")
    print(f"Output file: {output_path}")
    
    # Load trajectory data
    print("=" * 80)
    print("TRAJECTORY VISUALIZATION")
    print("=" * 80)
    train_vals = load_trajectory_data(data_path)
    
    # Extract X and Y positions, targets, and forces
    pos_x, pos_y, target_x, target_y, force_x, force_y, timestamps = extract_positions(train_vals)
    
    # Calculate statistics
    relative_time = timestamps - timestamps[0]
    avg_sample_rate = len(pos_x) / relative_time[-1]
    force_magnitude = np.sqrt(force_x**2 + force_y**2)
    
    print(f"\nTrajectory Statistics:")
    print(f"  Number of frames: {len(pos_x)}")
    print(f"  Duration: {relative_time[-1]:.2f} seconds")
    print(f"  Average sample rate: {avg_sample_rate:.1f} Hz")
    print(f"  Position X range: [{pos_x.min():.3f}, {pos_x.max():.3f}] meters")
    print(f"  Position Y range: [{pos_y.min():.3f}, {pos_y.max():.3f}] meters")
    print(f"  Target X range: [{target_x.min():.3f}, {target_x.max():.3f}] meters")
    print(f"  Target Y range: [{target_y.min():.3f}, {target_y.max():.3f}] meters")
    print(f"  Force magnitude range: [{force_magnitude.min():.2f}, {force_magnitude.max():.2f}] N")
    print(f"  Force magnitude mean: {force_magnitude.mean():.2f} N")
    
    # Initialize renderer with real-world dimensions
    renderer = TrajectoryRenderer(
        table_width=1.0436,   # Real world table width
        table_length=2.1104,  # Real world table length
        paddle_radius=0.0508, # Real world paddle radius
        ppm=400,              # Pixels per meter (adjust for resolution vs file size)
        robot_x_offset=1.2    # Robot base offset from table center (real world)
    )
    
    # Generate GIF
    create_trajectory_gif(
        pos_x, 
        pos_y,
        target_x,
        target_y,
        force_x,
        force_y,
        timestamps, 
        renderer, 
        output_path,
        max_frames=args.max_frames,
        subsample=args.subsample
    )
    
    print("\n" + "=" * 80)
    print("✓ Visualization complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()


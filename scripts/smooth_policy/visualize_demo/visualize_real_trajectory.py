#!/usr/bin/env python3
"""
Visualize real robot trajectory data from HDF5 files as GIFs matching Box2D simulator style.

This script reads trajectory data collected from the real UR5e robot and renders it
using the same visual style as the Box2D simulator, ensuring exact visual alignment.

Key features:
- Loads HDF5 trajectory files from /nfs/data/airhockey/
- Transforms robot frame coordinates to simulation table frame
- Uses the same rendering assets and logic as Box2D simulator
- Generates GIFs with paddle position/velocity visualization
- Optionally visualizes puck position if present in train_vals (fields 32-34)

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
        numpy.ndarray: Train values array (N, D), where D is typically 32 or 35
    """
    print(f"Loading trajectory data from: {filepath}")
    with h5py.File(filepath, 'r') as f:
        train_vals = f['train_vals'][:]
        print(f"  Shape: {train_vals.shape}")
        print(f"  Timesteps: {train_vals.shape[0]}")
        print(f"  Features per timestep: {train_vals.shape[1]}")
    return train_vals


def extract_paddle_data(train_vals, require_puck=False):
    """
    Extract paddle position/velocity/timestamps and optional puck data.
    
    According to FIELD_DOCUMENTATION.md:
    - Field 0: cur_time (Unix timestamp)
    - Field 5: pose_x (X position in meters, robot frame)
    - Field 6: pose_y (Y position in meters, robot frame)
    - Field 11: speed_vx (X velocity in m/s)
    - Field 12: speed_vy (Y velocity in m/s)
    
    Additional optional puck fields:
    - Field 32: puck_x (table frame X)
    - Field 33: puck_y (table frame Y)
    - Field 34: puck_occlusion (1=occluded, 0=visible)

    Args:
        train_vals: Array of trajectory data (N, D)
        require_puck: If True, raise an error when puck fields are unavailable

    Returns:
        dict: Dictionary with paddle data and optional puck data
    """
    n_features = train_vals.shape[1]
    has_puck_xy = n_features >= 34
    has_puck_occlusion = n_features >= 35
    has_target = n_features >= 28

    if require_puck and not has_puck_xy:
        raise ValueError(
            f"--require-puck was set, but train_vals has only {n_features} columns "
            "(need at least 34 for puck_x/puck_y)."
        )

    data = {
        'pos_x': train_vals[:, 5],      # Robot frame X position
        'pos_y': train_vals[:, 6],      # Robot frame Y position
        'vel_x': train_vals[:, 11],     # X velocity
        'vel_y': train_vals[:, 12],     # Y velocity
        'timestamps': train_vals[:, 0],  # Unix timestamps
        'has_puck': has_puck_xy,
        'has_target': has_target,
    }

    if has_target:
        data['target_x'] = train_vals[:, 26]  # Table frame desired_pose X
        data['target_y'] = train_vals[:, 27]  # Table frame desired_pose Y
    else:
        data['target_x'] = None
        data['target_y'] = None

    if has_puck_xy:
        data['puck_x'] = train_vals[:, 32]  # Table frame X
        data['puck_y'] = train_vals[:, 33]  # Table frame Y
        data['puck_occluded'] = train_vals[:, 34] > 0.5 if has_puck_occlusion else None
        if has_puck_occlusion:
            print("  ✓ Puck fields detected: x, y, occlusion")
        else:
            print("  ✓ Puck fields detected: x, y (no occlusion column)")
    else:
        data['puck_x'] = None
        data['puck_y'] = None
        data['puck_occluded'] = None
        print("  ! No puck fields detected (expected columns 32-34)")

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
                 puck_radius=0.03175,
                 render_size=360,
                 robot_x_offset=1.2,
                 orientation='vertical',
                 paddle_input_frame='robot',
                 assets_dir=None,
                 quiet=False):
        """
        Initialize renderer with simulation-matching parameters.

        Args:
            table_length: Length of air hockey table in meters (simulation default: 1.9304m)
            table_width: Width of air hockey table in meters (simulation default: 0.8636m)
            paddle_radius: Radius of paddle in meters (0.0508m)
            puck_radius: Radius of puck in meters (simulation default: 0.03175m)
            render_size: Render size in pixels (default: 360, matching simulation)
            robot_x_offset: Robot base offset from table center in X (real world: ~1.2m)
            orientation: Render orientation ('vertical' or 'horizontal')
            paddle_input_frame: Coordinate frame for paddle x/y inputs.
                - 'robot': inputs are robot-frame and require robot_x_offset transform.
                - 'table': inputs are already in table/observation-centered frame.
            assets_dir: Optional explicit path to assets folder. If None, computed
                relative to this script file.
            quiet: If True, suppress print output during initialization.
        """
        if paddle_input_frame not in ('robot', 'table'):
            raise ValueError(
                f"Invalid paddle_input_frame='{paddle_input_frame}'. "
                "Expected one of: 'robot', 'table'."
            )
        self.length = table_length
        self.width = table_width
        self.paddle_radius = paddle_radius
        self.puck_radius = puck_radius
        self.robot_x_offset = robot_x_offset
        self.orientation = orientation
        self.paddle_input_frame = paddle_input_frame
        self.quiet = quiet
        self._assets_dir = assets_dir

        # Calculate pixels per meter and render dimensions (matching render.py logic)
        self.ppm = render_size / self.width
        self.render_width = int(render_size)
        self.render_length = int(self.ppm * self.length)

        if not self.quiet:
            print(f"Renderer initialized:")
            print(f"  Table dimensions: {self.length}m x {self.width}m")
            print(f"  Render size: {self.render_length}px x {self.render_width}px")
            print(f"  Pixels per meter: {self.ppm:.1f}")
            print(f"  Paddle radius: {self.paddle_radius}m ({int(self.paddle_radius * self.ppm)}px)")
            print(f"  Puck radius: {self.puck_radius}m ({int(self.puck_radius * self.ppm)}px)")
            print(f"  Robot X offset: {self.robot_x_offset}m")
            print(f"  Orientation: {self.orientation}")
            print(f"  Paddle input frame: {self.paddle_input_frame}")

        # Load assets (matching render.py)
        self._load_assets()
        
    def _load_assets(self):
        """Load table and paddle images from assets folder."""
        if self._assets_dir is not None:
            assets_folder = Path(self._assets_dir)
        else:
            # Find assets folder relative to this script
            script_dir = Path(__file__).parent
            assets_folder = script_dir / '../../../assets'
            assets_folder = assets_folder.resolve()

        if not assets_folder.exists():
            raise FileNotFoundError(f"Assets folder not found at {assets_folder}")

        if not self.quiet:
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

    def table_position_to_pixel_coords(self, table_x, table_y):
        """
        Convert table-frame position to pixel coordinates.

        Args:
            table_x: X position in table frame (meters)
            table_y: Y position in table frame (meters)

        Returns:
            numpy.ndarray: Pixel coordinates [x, y]
        """
        pos_render = self.convert_to_render_coords_sys([table_x, table_y])
        center = np.array(pos_render) + np.array((self.width / 2, self.length / 2))
        center = np.array((center[1], center[0])) * self.ppm
        return center.astype(int)
    
    def position_to_pixel_coords(self, pos_x, pos_y):
        """
        Convert paddle position to pixel coordinates.
        
        This follows the same logic as render.py draw_circle_with_image (lines 241-242)
        when using robot-frame input, and also supports direct table-frame input.
        
        Args:
            pos_x: X position in configured paddle_input_frame (meters)
            pos_y: Y position in configured paddle_input_frame (meters)
            
        Returns:
            numpy.ndarray: Pixel coordinates [x, y]
        """
        if self.paddle_input_frame == 'table':
            table_x, table_y = pos_x, pos_y
        else:
            # Transform to table frame
            table_x, table_y = self.robot_to_table_frame(pos_x, pos_y)
        
        # Convert to pixel coordinates
        return self.table_position_to_pixel_coords(table_x, table_y)
    
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

    def draw_target(self, frame, target_x, target_y):
        """
        Draw the target paddle position as a cross + circle marker.

        Mirrors ``AirHockeyRenderer.draw_target_marker`` (orange BGR (255,165,0)
        cross+circle with a black outline) so real-trajectory GIFs follow the
        same visual convention as the Box2D training renderer used by
        ``td3_training.py``.

        Args:
            frame: Image to draw on (will be modified in place)
            target_x: Target X position in the configured ``paddle_input_frame``
            target_y: Target Y position in the configured ``paddle_input_frame``
        """
        center = self.position_to_pixel_coords(target_x, target_y)
        center_int = (int(center[0]), int(center[1]))

        marker_size = 15
        thickness = 3
        color = (255, 165, 0)  # Matches AirHockeyRenderer.draw_target_marker
        outline = (0, 0, 0)

        # Outer circle (black outline + colored inner)
        cv2.circle(frame, center_int, marker_size, outline, thickness + 2)
        cv2.circle(frame, center_int, marker_size, color, thickness)

        # Cross lines (black outline + colored)
        cv2.line(
            frame,
            (center_int[0] - marker_size, center_int[1]),
            (center_int[0] + marker_size, center_int[1]),
            outline, thickness + 2,
        )
        cv2.line(
            frame,
            (center_int[0], center_int[1] - marker_size),
            (center_int[0], center_int[1] + marker_size),
            outline, thickness + 2,
        )
        cv2.line(
            frame,
            (center_int[0] - marker_size, center_int[1]),
            (center_int[0] + marker_size, center_int[1]),
            color, thickness,
        )
        cv2.line(
            frame,
            (center_int[0], center_int[1] - marker_size),
            (center_int[0], center_int[1] + marker_size),
            color, thickness,
        )

    def draw_puck(self, frame, puck_x, puck_y, puck_occluded=None):
        """
        Draw puck at table-frame coordinates.

        Args:
            frame: Image to draw on (will be modified in place)
            puck_x: Puck X position in table frame (meters)
            puck_y: Puck Y position in table frame (meters)
            puck_occluded: Optional bool occlusion flag
        """
        center = self.table_position_to_pixel_coords(puck_x, puck_y)
        radius = max(2, int(self.puck_radius * self.ppm))

        # Green when visible, red when occluded.
        if puck_occluded is None:
            color = (60, 180, 75)
        else:
            color = (30, 30, 220) if bool(puck_occluded) else (60, 180, 75)

        cv2.circle(frame, tuple(center), radius, color, -1)
        cv2.circle(frame, tuple(center), radius, (20, 20, 20), 1)

    def is_paddle_in_frame(self, pos_x, pos_y):
        """
        Check whether a paddle centered at the given position intersects the frame.

        Args:
            pos_x: X position in configured paddle_input_frame (meters)
            pos_y: Y position in configured paddle_input_frame (meters)

        Returns:
            bool: True if any part of the paddle would be visible.
        """
        center = self.position_to_pixel_coords(pos_x, pos_y)
        radius = int(self.paddle_radius * self.ppm)
        top_left = center - radius
        bottom_right = top_left + 2 * radius

        if bottom_right[0] <= 0 or bottom_right[1] <= 0:
            return False
        if top_left[0] >= self.render_length or top_left[1] >= self.render_width:
            return False
        return True
    
    def render_frame(self, pos_x, pos_y, vel_x=None, vel_y=None,
                    puck_x=None, puck_y=None, puck_occluded=None,
                    target_x=None, target_y=None,
                    timestep=None, total_time=None):
        """
        Render a single frame with paddle at given position.
        
        Args:
            pos_x: X position in robot frame (meters)
            pos_y: Y position in robot frame (meters)
            vel_x: X velocity in m/s (optional)
            vel_y: Y velocity in m/s (optional)
            puck_x: Puck X position in table frame (optional)
            puck_y: Puck Y position in table frame (optional)
            puck_occluded: Optional puck occlusion flag
            target_x: Target X position in table frame (optional)
            target_y: Target Y position in table frame (optional)
            timestep: Optional timestep number to display
            total_time: Optional total elapsed time to display
            
        Returns:
            numpy.ndarray: BGR image array
        """
        frame = self.table_img.copy()

        # Draw target first (behind everything else).
        if target_x is not None and target_y is not None:
            self.draw_target(frame, target_x, target_y)

        # Draw puck so paddle can appear on top.
        if puck_x is not None and puck_y is not None:
            self.draw_puck(frame, puck_x, puck_y, puck_occluded)
        
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
            y_offset += 25

        if puck_x is not None and puck_y is not None:
            if puck_occluded is None:
                text = f"Puck: ({puck_x:.3f}, {puck_y:.3f})m"
            else:
                occlusion_text = "occluded" if bool(puck_occluded) else "visible"
                text = f"Puck: ({puck_x:.3f}, {puck_y:.3f})m [{occlusion_text}]"
            cv2.putText(frame, text, (10, y_offset), font, font_scale, (60, 180, 75), line_type)
            y_offset += 25

        if target_x is not None and target_y is not None:
            text = f"Target: ({target_x:.3f}, {target_y:.3f})m"
            cv2.putText(frame, text, (10, y_offset), font, font_scale, (255, 165, 0), line_type)
        
        # Apply orientation rotation if vertical (matching render.py line 487)
        if self.orientation == 'vertical':
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        return frame


def create_trajectory_gif(paddle_data, renderer, output_path, 
                         max_frames=None, subsample=1, fps=20,
                         output_width=160):
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
    has_puck = paddle_data.get('has_puck', False)
    puck_x = paddle_data.get('puck_x')
    puck_y = paddle_data.get('puck_y')
    puck_occluded = paddle_data.get('puck_occluded')
    has_target = paddle_data.get('has_target', False)
    target_x = paddle_data.get('target_x')
    target_y = paddle_data.get('target_y')
    
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
    clipped_paddle_frames = 0
    
    for idx, i in enumerate(indices):
        if idx % 50 == 0 and idx > 0:
            print(f"  Rendering frame {idx}/{len(indices)}...")

        if not renderer.is_paddle_in_frame(pos_x[i], pos_y[i]):
            clipped_paddle_frames += 1
        
        # Render frame
        frame = renderer.render_frame(
            pos_x[i], 
            pos_y[i],
            vel_x=vel_x[i],
            vel_y=vel_y[i],
            puck_x=(puck_x[i] if has_puck else None),
            puck_y=(puck_y[i] if has_puck else None),
            puck_occluded=(puck_occluded[i] if (has_puck and puck_occluded is not None) else None),
            target_x=(target_x[i] if has_target else None),
            target_y=(target_y[i] if has_target else None),
            timestep=i,
            total_time=relative_time[i]
        )
        
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        aspect_ratio = frame_rgb.shape[1] / frame_rgb.shape[0]
        target_width = output_width
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
    if len(indices) > 0:
        clipped_ratio = clipped_paddle_frames / len(indices)
        if clipped_paddle_frames > 0:
            print(
                f"  ! Paddle out of frame in {clipped_paddle_frames}/{len(indices)} "
                f"frames ({clipped_ratio * 100:.1f}%)"
            )
        if clipped_ratio > 0.5:
            print(
                "  ! Warning: paddle is clipped in most frames; "
                "check coordinate frame / x-offset settings."
            )


def print_trajectory_statistics(paddle_data):
    """Print statistics about the trajectory data."""
    pos_x = paddle_data['pos_x']
    pos_y = paddle_data['pos_y']
    vel_x = paddle_data['vel_x']
    vel_y = paddle_data['vel_y']
    timestamps = paddle_data['timestamps']
    has_puck = paddle_data.get('has_puck', False)
    
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

    if has_puck:
        puck_x = paddle_data['puck_x']
        puck_y = paddle_data['puck_y']
        print(f"\n  Puck position (table frame):")
        print(f"    X range: [{puck_x.min():.3f}, {puck_x.max():.3f}] meters")
        print(f"    Y range: [{puck_y.min():.3f}, {puck_y.max():.3f}] meters")
        puck_occluded = paddle_data.get('puck_occluded')
        if puck_occluded is not None:
            occluded_ratio = 100.0 * puck_occluded.mean()
            print(f"    Occluded frames: {occluded_ratio:.1f}%")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualize real robot trajectory data as GIF matching Box2D simulator style',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'input_path',
        type=str,
        help='Path to HDF5 trajectory file or directory containing trajectory_data*.hdf5 files'
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
    parser.add_argument(
        '--puck-radius',
        type=float,
        default=0.03175,
        help='Puck radius in meters'
    )
    parser.add_argument(
        '--require-puck',
        action='store_true',
        help='Fail if puck fields (train_vals[:, 32:34]) are not present'
    )
    
    return parser.parse_args()


def visualize_single_file(data_path, output_dir, args):
    """
    Generate one visualization GIF for a single HDF5 trajectory file.

    Args:
        data_path: Path to HDF5 file
        output_dir: Output directory for this file's GIF
        args: Parsed CLI args
    """
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

    # Extract trajectory data
    paddle_data = extract_paddle_data(train_vals, require_puck=args.require_puck)

    # Print statistics
    print_trajectory_statistics(paddle_data)

    # Initialize renderer with simulation-matching parameters
    renderer = RealTrajectoryRenderer(
        table_length=args.table_length,
        table_width=args.table_width,
        paddle_radius=0.0508,
        puck_radius=args.puck_radius,
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


def main():
    """Main function to generate trajectory visualization."""
    args = parse_args()
    input_path = Path(args.input_path)

    if not input_path.exists():
        print(f"Error: Input path not found: {input_path}")
        sys.exit(1)

    if input_path.is_file():
        # Single-file mode
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = Path(__file__).parent / f"{input_path.stem}_visualization"
        visualize_single_file(input_path, output_dir, args)
        return

    # Directory mode
    trajectory_files = sorted(input_path.glob("trajectory_data*.hdf5"))
    if not trajectory_files:
        # Fallback: include all hdf5 files if strict naming pattern is absent.
        trajectory_files = sorted(input_path.glob("*.hdf5"))

    if not trajectory_files:
        print(f"Error: No .hdf5 trajectory files found in directory: {input_path}")
        sys.exit(1)

    if args.output_dir:
        batch_output_root = Path(args.output_dir)
    else:
        batch_output_root = Path(__file__).parent / f"{input_path.name}_visualization"
    batch_output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("REAL ROBOT TRAJECTORY VISUALIZATION (BATCH MODE)")
    print("=" * 80)
    print(f"Input directory: {input_path}")
    print(f"Found {len(trajectory_files)} trajectory files")
    print(f"Output root: {batch_output_root}")

    success = 0
    failures = 0
    for idx, data_path in enumerate(trajectory_files, start=1):
        print(f"\n[{idx}/{len(trajectory_files)}] Processing {data_path.name}")
        output_dir = batch_output_root / data_path.stem
        try:
            visualize_single_file(data_path, output_dir, args)
            success += 1
        except Exception as exc:
            failures += 1
            print(f"  ✗ Failed to process {data_path}: {exc}")

    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Successful: {success}")
    print(f"Failed: {failures}")
    print(f"Output root: {batch_output_root}")


if __name__ == '__main__':
    main()

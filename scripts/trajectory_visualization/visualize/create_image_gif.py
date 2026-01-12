#!/usr/bin/env python3
"""
Create a continuous GIF/video from the images stored in trajectory data.
This script extracts the RGB images from the HDF5 file and combines them
into a smooth animated GIF or video file.
"""

import h5py
import numpy as np
import imageio
from pathlib import Path
import argparse
import sys


def load_images_from_trajectory(file_path):
    """
    Load images from HDF5 trajectory file.
    
    Args:
        file_path: Path to the HDF5 file
        
    Returns:
        tuple: (images, train_vals) where images is a numpy array of RGB images
    """
    print(f"Loading images from: {file_path}")
    
    with h5py.File(file_path, 'r') as f:
        # Load images and associated data
        train_img = f['train_img'][:]
        train_vals = f['train_vals'][:]
    
    print(f"  Total frames: {len(train_img)}")
    print(f"  Image shape: {train_img.shape[1:]}")
    print(f"  Image dtype: {train_img.dtype}")
    print(f"  Value range: [{train_img.min()}, {train_img.max()}]")
    
    return train_img, train_vals


def add_frame_info(images, train_vals, add_text=True):
    """
    Optionally add frame information overlay to images.
    
    Args:
        images: Array of images (N, H, W, 3)
        train_vals: Array of trajectory values
        add_text: Whether to add text overlays
        
    Returns:
        numpy.ndarray: Images with optional text overlays
    """
    if not add_text:
        return images
    
    try:
        import cv2
    except ImportError:
        print("Warning: opencv-python not available, skipping text overlays")
        return images
    
    annotated_images = []
    timestamps = train_vals[:, 0]
    relative_time = timestamps - timestamps[0]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 255, 255)  # White
    bg_color = (0, 0, 0)  # Black background for text
    thickness = 1
    
    for idx, img in enumerate(images):
        # Make a copy to avoid modifying original
        img_copy = img.copy()
        
        # Add frame number
        frame_num = int(train_vals[idx, 2])
        text = f"Frame: {frame_num}"
        
        # Add background rectangle for better readability
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(img_copy, (5, 5), (15 + text_w, 20 + text_h), bg_color, -1)
        cv2.putText(img_copy, text, (10, 20), font, font_scale, font_color, thickness)
        
        # Add timestamp
        time_text = f"Time: {relative_time[idx]:.2f}s"
        (text_w, text_h), baseline = cv2.getTextSize(time_text, font, font_scale, thickness)
        cv2.rectangle(img_copy, (5, 30), (15 + text_w, 45 + text_h), bg_color, -1)
        cv2.putText(img_copy, time_text, (10, 45), font, font_scale, font_color, thickness)
        
        annotated_images.append(img_copy)
    
    return np.array(annotated_images)


def create_gif(images, output_path, fps=20, subsample=1, max_frames=None):
    """
    Create a GIF from a sequence of images.
    
    Args:
        images: Array of images (N, H, W, 3)
        output_path: Path to save the output GIF
        fps: Frames per second for playback
        subsample: Subsample factor (1 = all frames, 2 = every other frame, etc.)
        max_frames: Optional maximum number of frames to include
    """
    n_frames = len(images)
    
    # Apply max_frames limit
    if max_frames is not None:
        n_frames = min(n_frames, max_frames)
        images = images[:n_frames]
    
    # Apply subsampling
    if subsample > 1:
        images = images[::subsample]
        print(f"Subsampled to every {subsample} frame(s): {len(images)} frames")
    
    print(f"\nGenerating GIF...")
    print(f"  Total frames: {len(images)}")
    print(f"  FPS: {fps}")
    print(f"  Duration: {len(images) / fps:.2f} seconds")
    
    # Calculate duration per frame in milliseconds
    duration = int(1000 / fps)
    
    # Save as GIF
    print(f"\nSaving GIF to: {output_path}")
    imageio.mimsave(output_path, images, format='GIF', loop=0, duration=duration)
    
    # Print file size
    import os
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ Saved successfully!")
    print(f"  File size: {file_size_mb:.2f} MB")


def create_video(images, output_path, fps=20, subsample=1, max_frames=None, codec='mp4v'):
    """
    Create a video file from a sequence of images.
    
    Args:
        images: Array of images (N, H, W, 3)
        output_path: Path to save the output video
        fps: Frames per second for playback
        subsample: Subsample factor (1 = all frames, 2 = every other frame, etc.)
        max_frames: Optional maximum number of frames to include
        codec: Video codec to use ('mp4v', 'avc1', 'h264', etc.)
    """
    try:
        import cv2
    except ImportError:
        print("Error: opencv-python is required for video creation")
        print("Install with: pip install opencv-python")
        sys.exit(1)
    
    n_frames = len(images)
    
    # Apply max_frames limit
    if max_frames is not None:
        n_frames = min(n_frames, max_frames)
        images = images[:n_frames]
    
    # Apply subsampling
    if subsample > 1:
        images = images[::subsample]
        print(f"Subsampled to every {subsample} frame(s): {len(images)} frames")
    
    print(f"\nGenerating video...")
    print(f"  Total frames: {len(images)}")
    print(f"  FPS: {fps}")
    print(f"  Duration: {len(images) / fps:.2f} seconds")
    print(f"  Codec: {codec}")
    
    # Get frame dimensions
    height, width = images[0].shape[:2]
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Could not open video writer with codec '{codec}'")
        print("Trying alternative codec 'XVID'...")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_path = output_path.with_suffix('.avi')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        if not out.isOpened():
            print("Error: Could not initialize video writer")
            sys.exit(1)
    
    # Write frames (convert RGB to BGR for OpenCV)
    print(f"\nWriting frames...")
    for idx, img in enumerate(images):
        if idx % 100 == 0:
            print(f"  Writing frame {idx}/{len(images)}...")
        
        # Convert RGB to BGR
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.write(img_bgr)
    
    out.release()
    
    # Print file size
    import os
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n✓ Saved video to: {output_path}")
    print(f"  File size: {file_size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Create a GIF or video from trajectory images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create GIF with default settings
  %(prog)s /path/to/trajectory.hdf5
  
  # Create GIF at 30 fps with subsampling
  %(prog)s /path/to/trajectory.hdf5 --fps 30 --subsample 2
  
  # Create video instead of GIF
  %(prog)s /path/to/trajectory.hdf5 --format video --output trajectory.mp4
  
  # Limit to first 200 frames without text overlay
  %(prog)s /path/to/trajectory.hdf5 --max-frames 200 --no-text
        """
    )
    
    parser.add_argument(
        '--file_path',
        type=str,
        default='/nfs/data/airhockey/trajectory_data434.hdf5',
        help='Path to HDF5 trajectory file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='scripts/trajectory_visualization/visualize/trajectory_animation.gif',
        help='Output file path (default: trajectory_animation.gif or .mp4)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['gif', 'video'],
        default='gif',
        help='Output format (default: gif)'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=20,
        help='Frames per second for playback (default: 20)'
    )
    
    parser.add_argument(
        '--subsample',
        type=int,
        default=1,
        help='Subsample factor - use every Nth frame (default: 1 = all frames)'
    )
    
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Maximum number of frames to include (default: all frames)'
    )
    
    parser.add_argument(
        '--no-text',
        action='store_true',
        help='Do not add text overlays (frame number and timestamp)'
    )
    
    parser.add_argument(
        '--codec',
        type=str,
        default='mp4v',
        help='Video codec for video format (default: mp4v)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    file_path = Path(args.file_path)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    # Determine output path
    if args.output is None:
        if args.format == 'gif':
            output_path = Path('trajectory_animation.gif')
        else:
            output_path = Path('trajectory_animation.mp4')
    else:
        output_path = Path(args.output)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("TRAJECTORY IMAGE ANIMATION")
    print("=" * 80)
    print()
    
    # Load images
    images, train_vals = load_images_from_trajectory(file_path)
    
    # Add text overlays if requested
    if not args.no_text:
        print("\nAdding text overlays...")
        images = add_frame_info(images, train_vals, add_text=True)
    
    # Create animation
    if args.format == 'gif':
        create_gif(
            images,
            output_path,
            fps=args.fps,
            subsample=args.subsample,
            max_frames=args.max_frames
        )
    else:
        create_video(
            images,
            output_path,
            fps=args.fps,
            subsample=args.subsample,
            max_frames=args.max_frames,
            codec=args.codec
        )
    
    print("\n" + "=" * 80)
    print("✓ Animation creation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()


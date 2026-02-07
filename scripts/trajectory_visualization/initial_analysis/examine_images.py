#!/usr/bin/env python3
"""
Examine the images from the trajectory data to understand what they show.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def examine_images(file_path, output_dir):
    """
    Examine the images in detail.
    
    Args:
        file_path: Path to the HDF5 file
        output_dir: Directory to save analysis
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"IMAGE EXAMINATION")
    print(f"File: {file_path}")
    print(f"{'='*80}\n")
    
    with h5py.File(file_path, 'r') as f:
        train_img = f['train_img'][:]
        train_vals = f['train_vals'][:]
    
    print(f"Total frames: {len(train_img)}")
    print(f"Image shape: {train_img.shape[1:]}")
    print(f"Image dtype: {train_img.dtype}")
    print(f"Value range: [{train_img.min()}, {train_img.max()}]")
    
    # Create a grid of images showing the full trajectory
    n_samples = min(30, len(train_img))
    indices = np.linspace(0, len(train_img) - 1, n_samples, dtype=int)
    
    cols = 6
    rows = (n_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 3 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, (idx, ax) in enumerate(zip(indices, axes.flat)):
        ax.imshow(train_img[idx])
        # Show frame number from train_vals
        frame_num = int(train_vals[idx, 2])
        ax.set_title(f"Frame {frame_num} (idx {idx})", fontsize=8)
        ax.axis('off')
    
    # Hide unused subplots
    for ax in axes.flat[len(indices):]:
        ax.axis('off')
    
    plt.tight_layout()
    img_path = output_dir / "full_trajectory_sequence.png"
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {img_path}")
    plt.close()
    
    # Create a detailed view of a few key frames
    key_frames = [0, len(train_img)//4, len(train_img)//2, 3*len(train_img)//4, len(train_img)-1]
    
    fig, axes = plt.subplots(1, len(key_frames), figsize=(20, 4))
    
    for idx, ax in zip(key_frames, axes):
        ax.imshow(train_img[idx])
        frame_num = int(train_vals[idx, 2])
        timestamp = train_vals[idx, 0]
        ax.set_title(f"Frame {frame_num}\nTime: {timestamp:.2f}", fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    key_path = output_dir / "key_frames.png"
    plt.savefig(key_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {key_path}")
    plt.close()
    
    # Analyze image differences to detect motion
    print(f"\n{'='*80}")
    print("MOTION DETECTION")
    print("=" * 80)
    
    # Calculate frame differences
    diffs = np.abs(np.diff(train_img.astype(float), axis=0))
    motion_intensity = diffs.mean(axis=(1, 2, 3))
    
    print(f"Average motion intensity: {motion_intensity.mean():.2f}")
    print(f"Max motion intensity: {motion_intensity.max():.2f}")
    print(f"Frames with high motion (>mean): {np.sum(motion_intensity > motion_intensity.mean())}")
    
    # Plot motion over time
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(motion_intensity, linewidth=1)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Motion Intensity (mean pixel diff)")
    ax.set_title("Frame-to-Frame Motion Intensity")
    ax.grid(True, alpha=0.3)
    ax.axhline(motion_intensity.mean(), color='r', linestyle='--', label='Mean')
    ax.legend()
    
    plt.tight_layout()
    motion_path = output_dir / "motion_intensity.png"
    plt.savefig(motion_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {motion_path}")
    plt.close()
    
    # Create a difference visualization
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    
    diff_indices = np.linspace(0, len(diffs) - 1, 5, dtype=int)
    
    for i, idx in enumerate(diff_indices):
        # Show original frame
        axes[0, i].imshow(train_img[idx])
        axes[0, i].set_title(f"Frame {int(train_vals[idx, 2])}", fontsize=9)
        axes[0, i].axis('off')
        
        # Show difference (amplified for visibility)
        diff_img = (diffs[idx] * 5).clip(0, 255).astype(np.uint8)
        axes[1, i].imshow(diff_img)
        axes[1, i].set_title(f"Diff (x5)", fontsize=9)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    diff_path = output_dir / "frame_differences.png"
    plt.savefig(diff_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {diff_path}")
    plt.close()
    
    # Analyze RGB channels
    print(f"\n{'='*80}")
    print("RGB CHANNEL ANALYSIS")
    print("=" * 80)
    
    mean_r = train_img[:, :, :, 0].mean()
    mean_g = train_img[:, :, :, 1].mean()
    mean_b = train_img[:, :, :, 2].mean()
    
    print(f"Mean R channel: {mean_r:.2f}")
    print(f"Mean G channel: {mean_g:.2f}")
    print(f"Mean B channel: {mean_b:.2f}")
    
    # Create a single high-res image for detailed examination
    mid_idx = len(train_img) // 2
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Full color
    axes[0, 0].imshow(train_img[mid_idx])
    axes[0, 0].set_title("Full Color", fontsize=12)
    axes[0, 0].axis('off')
    
    # Grayscale
    gray = train_img[mid_idx].mean(axis=2).astype(np.uint8)
    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title("Grayscale", fontsize=12)
    axes[0, 1].axis('off')
    
    # Edge detection (simple)
    from scipy.ndimage import sobel
    edges_x = sobel(gray.astype(float), axis=0)
    edges_y = sobel(gray.astype(float), axis=1)
    edges = np.sqrt(edges_x**2 + edges_y**2)
    axes[1, 0].imshow(edges, cmap='hot')
    axes[1, 0].set_title("Edge Detection", fontsize=12)
    axes[1, 0].axis('off')
    
    # Thresholded
    threshold = gray.mean()
    binary = gray > threshold
    axes[1, 1].imshow(binary, cmap='gray')
    axes[1, 1].set_title(f"Threshold (>{threshold:.0f})", fontsize=12)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    detail_path = output_dir / "detailed_single_frame.png"
    plt.savefig(detail_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {detail_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Examine images from trajectory data"
    )
    
    parser.add_argument(
        'file_path',
        type=str,
        help='Path to HDF5 trajectory file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./trajectory_analysis_output',
        help='Directory to save plots'
    )
    
    args = parser.parse_args()
    
    examine_images(args.file_path, args.output_dir)
    
    print(f"\n{'='*80}")
    print("EXAMINATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()


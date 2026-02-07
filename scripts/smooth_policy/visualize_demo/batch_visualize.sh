#!/bin/bash
# Batch visualization script for multiple trajectory files
#
# Usage:
#   ./batch_visualize.sh [output_dir] [subsample] [max_frames]
#
# Examples:
#   ./batch_visualize.sh ./batch_output 2 100
#   ./batch_visualize.sh ./all_trajectories 1

set -e

# Default parameters
OUTPUT_DIR="${1:-./batch_visualizations}"
SUBSAMPLE="${2:-2}"
MAX_FRAMES="${3:-}"

# Data directory
DATA_DIR="/nfs/data/airhockey"

echo "=========================================="
echo "Batch Trajectory Visualization"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Subsample factor: $SUBSAMPLE"
echo "Max frames: ${MAX_FRAMES:-unlimited}"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Count total files
TOTAL_FILES=$(ls "$DATA_DIR"/trajectory_data*.hdf5 2>/dev/null | wc -l)
echo "Found $TOTAL_FILES trajectory files"
echo ""

# Process each file
COUNT=0
for file in "$DATA_DIR"/trajectory_data*.hdf5; do
    if [ ! -f "$file" ]; then
        echo "No trajectory files found in $DATA_DIR"
        exit 1
    fi
    
    COUNT=$((COUNT + 1))
    BASENAME=$(basename "$file" .hdf5)
    
    echo "[$COUNT/$TOTAL_FILES] Processing: $BASENAME"
    
    # Build command
    CMD="python scripts/smooth_policy/visualize_demo/visualize_real_trajectory.py"
    CMD="$CMD \"$file\""
    CMD="$CMD --output-dir \"$OUTPUT_DIR/$BASENAME\""
    CMD="$CMD --subsample $SUBSAMPLE"
    
    if [ -n "$MAX_FRAMES" ]; then
        CMD="$CMD --max-frames $MAX_FRAMES"
    fi
    
    # Run visualization
    eval $CMD
    
    echo ""
done

echo "=========================================="
echo "✓ Batch processing complete!"
echo "=========================================="
echo "Processed $COUNT trajectory files"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "To view all GIFs:"
echo "  ls $OUTPUT_DIR/*/trajectory_visualization.gif"

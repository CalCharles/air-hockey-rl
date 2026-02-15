#!/usr/bin/env bash
# Render inflection GIFs for all trajectories in the data directory.
# Usage: bash scripts/real_data_transforms/render_all_inflections.sh

DATA_DIR="/data2/air_hockey/air_hockey_state_data/datastor1/calebc/public/data/mouse/state_data_all_new/"

# Extract trajectory indices from filenames (trajectory_data<IDX>.hdf5)
indices=()
for f in "$DATA_DIR"/trajectory_data*.hdf5; do
    base=$(basename "$f")
    idx="${base#trajectory_data}"
    idx="${idx%.hdf5}"
    indices+=("$idx")
done

# Sort numerically
IFS=$'\n' sorted=($(sort -n <<<"${indices[*]}")); unset IFS

total=${#sorted[@]}
echo "Found $total trajectories"

for i in "${!sorted[@]}"; do
    idx="${sorted[$i]}"
    echo "=== [$((i+1))/$total] Rendering trajectory $idx ==="
    python scripts/real_data_transforms/trajectory_to_gif.py inflection --traj-idx "$idx"
    if [ $? -ne 0 ]; then
        echo "WARNING: Failed on trajectory $idx, continuing..."
    fi
done

echo "Done. Rendered $total trajectories."

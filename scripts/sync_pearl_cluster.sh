#!/bin/bash

# Number of machines
NUM_MACHINES=7

# Machine base name
MACHINE_BASE_NAME="pearl-cluster-"

# Source and destination details
SOURCE_DIR="/home/air_hockey/air-hockey-rl/"
REMOTE_USER="air_hockey"

# Remote directories
DEFAULT_REMOTE_DIR="/home/air_hockey/air-hockey-rl"
MACHINE_7_REMOTE_DIR="/home/air_hockey/sarthak/air-hockey-rl"

EXCLUDE_SUBFOLDER="baseline_models"

# Options:
# -a: archive mode (preserves permissions, timestamps, symlinks, etc.)
# -v: verbose mode
# -z: compress files during transfer
RSYNC_OPTIONS="-avz --exclude=${EXCLUDE_SUBFOLDER}"

# Loop through each machine in the cluster and perform synchronization
for i in $(seq 1 $NUM_MACHINES); do
  MACHINE="${MACHINE_BASE_NAME}${i}.local"
  echo "Synchronizing with $MACHINE..."
  if [ "$i" -eq 7 ]; then
    DESTINATION_DIR="$MACHINE_7_REMOTE_DIR"
  else
    DESTINATION_DIR="$DEFAULT_REMOTE_DIR"
  fi
  # Construct the remote destination path
  REMOTE_DESTINATION="${REMOTE_USER}@${MACHINE}:${DESTINATION_DIR}"
  
  # Perform the file transfer using rsync
  rsync $RSYNC_OPTIONS $SOURCE_DIR $REMOTE_DESTINATION
  
  # Check if rsync was successful
  if [ $? -eq 0 ]; then
    echo "Files synchronized successfully with $MACHINE."
  else
    echo "Error during file synchronization with $MACHINE."
  fi
done

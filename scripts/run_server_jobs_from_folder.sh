#!/bin/bash

# Directory containing the YAML files
CONFIG_DIR="/home/air_hockey/air-hockey-rl/configs/baseline_configs/box2d"

# List all YAML files in the directory
CONFIG_FILES=($CONFIG_DIR/*.yaml)

# Number of machines
NUM_MACHINES=7

# Machine base name
MACHINE_BASE_NAME="pearl-cluster-"

# Remote directories
DEFAULT_REMOTE_DIR="/home/air_hockey/air-hockey-rl"
MACHINE_7_REMOTE_DIR="/home/air_hockey/sarthak/air-hockey-rl"

# TMUX session name
TMUX_SESSION_NAME="training_session"

# Loop through the YAML files and distribute them across the machines
for i in "${!CONFIG_FILES[@]}"; do
  CONFIG_FILE="${CONFIG_FILES[$i]}"
  MACHINE_NUM=$(( (i % NUM_MACHINES) + 1 ))
  MACHINE="${MACHINE_BASE_NAME}${MACHINE_NUM}.local"

  if [ "$MACHINE_NUM" -eq 7 ]; then
    REMOTE_DIR="$MACHINE_7_REMOTE_DIR"
  else
    REMOTE_DIR="$DEFAULT_REMOTE_DIR"
  fi


  # Construct the command to create a tmux session and window
  REMOTE_CMD="
if ! tmux has-session -t $TMUX_SESSION_NAME 2>/dev/null; then
  tmux new-session -d -s $TMUX_SESSION_NAME
fi
tmux new-window -t $TMUX_SESSION_NAME -n training_${i}
"

  # Execute the command on the remote machine
  echo "Executing on $MACHINE:"
  echo "$REMOTE_CMD"
  ssh -t "$MACHINE" "$REMOTE_CMD"
  # Send the training command to the tmux window
  TRAIN_CMD="cd $REMOTE_DIR && conda activate sarthak_rl_35 && python scripts/train.py --cfg $CONFIG_FILE --device cuda"
  ssh -t "$MACHINE" "tmux send-keys -t $TMUX_SESSION_NAME:training_${i} \"$TRAIN_CMD\" C-m"

done

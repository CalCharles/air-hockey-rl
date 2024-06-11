#!/bin/bash

# Directory containing the YAML files
CONFIG_DIR="/path/to/yaml/folder"

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

  ssh -t "$MACHINE" << EOF
if ! tmux has-session -t $TMUX_SESSION_NAME 2>/dev/null; then
  tmux new-session -d -s $TMUX_SESSION_NAME
fi
tmux new-window -t $TMUX_SESSION_NAME -n "training_${i}" "cd $REMOTE_DIR && git pull && source activate sarthak_rl_35 && python train.py --cfg $CONFIG_FILE --device cuda"
EOF
done

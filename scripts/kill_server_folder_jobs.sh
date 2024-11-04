#!/bin/bash

# Number of machines
NUM_MACHINES=7

# Machine base name
MACHINE_BASE_NAME="pearl-cluster-"

# Loop through the machines and kill all tmux sessions
for i in $(seq 1 $NUM_MACHINES); do
  MACHINE="${MACHINE_BASE_NAME}${i}.local"

  # Construct the command to kill all tmux sessions
  REMOTE_CMD="tmux kill-session -t robosuite_training_session"

  # Execute the command on the remote machine
  echo "Killing all tmux sessions on $MACHINE"
  ssh -t "$MACHINE" "$REMOTE_CMD"
done

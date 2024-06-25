#!/bin/bash

# Directory containing the YAML files
CONFIG_DIR="/home/air_hockey/air-hockey-rl/configs/baseline_configs/boox2d"

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

# Source machine details
SOURCE_MACHINE="air_hockey@pearl-cluster.local"
SOURCE_DIR="/home/air_hockey/air-hockey-rl/baseline_models/batch_runs"

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

  REMOTE_CONFIG_FILE="$REMOTE_DIR/$(basename $CONFIG_FILE)"
  MONITOR_SCRIPT="/tmp/monitor_and_cleanup_${i}.sh"

  # Copy the configuration file to the remote machine
  scp "$CONFIG_FILE" "$MACHINE:$REMOTE_CONFIG_FILE"

  # Extract task name from YAML file
  TASK_NAME=$(yq '.air_hockey.task' "$CONFIG_FILE")

  # Create monitoring and cleanup script
  cat <<EOF > $MONITOR_SCRIPT
#!/bin/bash
mkdir -p $REMOTE_DIR/baseline_models/$TASK_NAME
BEFORE=\$(ls $REMOTE_DIR/baseline_models/$TASK_NAME)
cd $REMOTE_DIR
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
  source "/opt/conda/etc/profile.d/conda.sh"
else
  echo "Conda setup script not found. Exiting."
  exit 1
fi
conda activate sarthak_rl_35
python scripts/train.py --cfg $REMOTE_CONFIG_FILE --device cuda
AFTER=\$(ls $REMOTE_DIR/baseline_models/$TASK_NAME)
NEW_FOLDERS=\$(comm -13 <(echo "\$BEFORE") <(echo "\$AFTER"))
echo "Transferring folders"
# Transfer the new folders back to the source machine
for folder in \$NEW_FOLDERS; do
  scp -r "$REMOTE_DIR/baseline_models/$TASK_NAME/\$folder" "$SOURCE_MACHINE:$SOURCE_DIR"
  if [ $? -eq 0 ]; then
    # Remove the folder if it was transferred successfully
    rm -r "$REMOTE_DIR/baseline_models/$TASK_NAME/$folder"
  else
    echo "Error transferring $folder. It has not been removed from the machine."
  fi
done

# Remove the remote config file
rm $REMOTE_CONFIG_FILE

# Remove this script
rm -- "\$0"
EOF

  # Copy the monitoring script to the remote machine
  scp "$MONITOR_SCRIPT" "$MACHINE:$REMOTE_DIR/monitor_and_cleanup_${i}.sh"
  rm "$MONITOR_SCRIPT"

  # Construct the command to create a tmux session and window
  REMOTE_CMD="
if ! tmux has-session -t $TMUX_SESSION_NAME 2>/dev/null; then
  tmux new-session -d -s $TMUX_SESSION_NAME
fi
tmux new-window -t $TMUX_SESSION_NAME -n training_${i}
"

  # Execute the command on the remote machine to create the tmux window
  echo "Executing on $MACHINE:"
  echo "$REMOTE_CMD"
  ssh -t "$MACHINE" "$REMOTE_CMD"

  # Send the training command to the new tmux window
  TRAIN_CMD="bash $REMOTE_DIR/monitor_and_cleanup_${i}.sh"
  ssh -t "$MACHINE" "tmux send-keys -t $TMUX_SESSION_NAME:training_${i} \"$TRAIN_CMD\" C-m"

done

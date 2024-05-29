#!/bin/bash

# List of session names
wandb_project_name="test_2"
cfg="box2d.yaml"
conda_env="airhockey"
# checkpoints_path_base="halfcheetah_percent"


semi_vdice_lambda=(0.5)
true_vdice_lambda=(0.99)


seed=(100)

ed_name="or_ration"

# specify GPUs
# GPUS=(0 1 2 3)
GPUS=(0)

# Initialize an experiment counter
experiment_counter=0

# Loop through each parameter set
for current_seed in "${seed[@]}"; do
  for current_semi_vdice_lambda in "${semi_vdice_lambda[@]}"; do
    for current_true_vdice_lambda in "${true_vdice_lambda[@]}"; do
            # Calculate the device number for the current session
            device_index=$(( experiment_counter % ${#GPUS[@]} ))
            device=${GPUS[$device_index]}

            # Construct the session name based on parameters and ed_name
            run_name="${wandb_project_name}_seed_${current_seed}_${ed_name}"

            run_name="${run_name//./_}" # Replace dots with underscores

            # # Append session name to the checkpoints path
            # checkpoints_path="${checkpoints_path_base}/${run_name}"
            # checkpoints_path="${checkpoints_path//./_}" # Replace dots with underscores

            # Create a new tmux session with the session name
            tmux new-session -d -s $run_name

            # Activate the conda environment
            tmux send-keys -t $run_name "conda activate $conda_env" C-m

            # tmux send-keys -t $run_name "python3 /home/shuozhe/anaconda3/envs/airhockey/lib/python3.9/site-packages/robosuite/scripts/setup_macros.py" C-m

            # Start the experiment with the specified parameters
            tmux send-keys -t $run_name "CUDA_VISIBLE_DEVICES=$device \
                                            python3 rma.py \
                                            --cfg $cfg \
                                            --seed $current_seed \
                                            --wandb_project_name $wandb_project_name \
                                            --wandb True \
                                            --save_model True \
                                            --run_name $run_name" C-m

            # Increment the experiment counter
            experiment_counter=$((experiment_counter + 1))

            # Delay to avoid potential race conditions
            sleep 5
        done
    done
done

                                            # --semi_vdice_lambda $current_semi_vdice_lambda \
                                            # --true_vdice_lambda $current_true_vdice_lambda \
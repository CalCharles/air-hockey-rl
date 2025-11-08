#!/usr/bin/env python3
"""
Hyperparameter sweep script for iterative smoothing.
Varies norm_adv, reward_scaling, and reward_normalization with 5 seeds each.
"""

import os
from pickle import FALSE
import subprocess
import itertools
from datetime import datetime
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='Run hyperparameter sweep for iterative smoothing')
    parser.add_argument('--config', type=str, 
                       default='scripts/smooth_policy/configs/puck_juggle/velocity.yaml',
                       help='Path to config file')
    parser.add_argument('--num_iterations', type=int, default=150,
                       help='Number of training iterations')
    parser.add_argument('--dry_run', action='store_true',
                       help='Print commands without executing')
    parser.add_argument('--parallel', action='store_true',
                       help='Run jobs in parallel (background processes)')
    parser.add_argument('--num_gpus', type=int, default=4,
                       help='Number of GPUs to use (cuda:0 to cuda:N-1)')
    parser.add_argument('--base_dir', type=str, default='runs/scaling',
                       help='Base directory for storing results')
    parser.add_argument('--args_file', type=str, default=None,
                       help='Path to args file') # this overrides 
    parser.add_argument('--finetune_directory', type=str, default=None,
                       help='Path to finetune directory') # this is where we will build up
    
    args = parser.parse_args()
    

    # TODO: this is where hyperparameters are specified
    # Define hyperparameter combinations
    hyperparams = {
        "dynamic_reward_scaling": [True, False],
        "caps_coef_consecutive": [0.001, 0.0003, 0.003, 0.01],
        "caps_coef_nearby": [0.001, 0.0003, 0.003, 0.01]
    }

    exclude_combinations = [
        ("dynamic_reward_scaling", False),
        ("dynamic_freeze_policy", True),
        ("vf_coef", 0.1)
    ]
    
    # Generate all combinations
    combinations = list(itertools.product(*hyperparams.values()))
    combinations = [comb for comb in combinations if comb not in exclude_combinations]
    
    param_names = list(hyperparams.keys())
    
    # Seeds to run
    # seeds = [0, 1, 2, 3]
    seeds = [0]
    
    # GPU devices
    gpus = [f"cuda:{i}" for i in range(args.num_gpus)]
    
    # Base directory for results
    base_dir = args.base_dir
    
    if os.path.exists(base_dir):
        raise FileExistsError(f"Base directory '{base_dir}' already exists. Please remove it or choose a different directory.")
    os.makedirs(base_dir)
    
    job_count = 0
    total_jobs = len(combinations) * len(seeds)
    
    print(f"Starting hyperparameter sweep with {len(combinations)} combinations and {len(seeds)} seeds each")
    print(f"Total jobs: {total_jobs}")
    print(f"Using GPUs: {', '.join(gpus)}")
    
    # Store all commands for potential batch execution
    all_commands = []
    
    for combo_idx, combo in enumerate(combinations):
        # Create combination name
        combo_name = "_".join([f"{param}-{value}" for param, value in zip(param_names, combo)])
        combo_dir = os.path.join(base_dir, combo_name)
        os.makedirs(combo_dir, exist_ok=True)
        
        print(f"\nProcessing combination {combo_idx + 1}/{len(combinations)}: {combo_name}")
        
        for seed in seeds:
            # Select GPU in round-robin fashion
            gpu = gpus[job_count % len(gpus)]
            
            # Create seed directory
            seed_dir = os.path.join(combo_dir, f"seed_{seed}")
            
            # not finetuning, just running
            if args.finetune_directory is None:
                # Build command - run as module
                cmd = [
                    sys.executable, "-m", "scripts.smooth_policy.iterative_smoothing",
                    "--seed", str(seed),
                    "--device", gpu,
                    "--log_parent_dir", seed_dir,
                    "--run_name", f"sweep_{combo_name}_seed_{seed}"
                ]
                if args.args_file is not None:
                    cmd.append(f"--args_file")
                    cmd.append(str(args.args_file))
                    cmd.append(f"--num_iterations")
                    cmd.append(str(args.num_iterations))
                else:
                    cmd.append(f"--config")
                    cmd.append(str(args.config))

                # Add hyperparameter flags
                for param, value in zip(param_names, combo):
                    # Handle flag addition based on value type:
                    if isinstance(value, bool):
                        if value:
                            cmd.append(f"--{param}")
                    else:
                        cmd.append(f"--{param}")
                        cmd.append(str(value))
            else:
                # finetuning, so disregard all others
                args_file = os.path.join(args.finetune_directory, combo_name, f"seed_{seed}", "args.yaml")
                model_path = os.path.join(args.finetune_directory, combo_name, f"seed_{seed}", "iterative_smoothing_model.pth")
                cmd = [
                    sys.executable, "-m", "scripts.smooth_policy.iterative_smoothing",
                    "--seed", str(seed),
                    "--device", gpu,
                    "--log_parent_dir", seed_dir,
                    "--run_name", f"sweep_{combo_name}_seed_{seed}",
                    "--args_file", args_file, # all hyperparameters are in the args file
                    "--model_path", model_path,
                    "--finetune",
                ]

            cmd_str = ' '.join(cmd)
            print(f"  Seed {seed} on {gpu}: {cmd_str}")
            
            all_commands.append((cmd, cmd_str, combo_name, seed, gpu))
            
            if not args.dry_run:
                if args.parallel:
                    # Run in background
                    print(f"    Starting job in background...")
                    subprocess.Popen(cmd, cwd=os.getcwd())
                else:
                    # Run sequentially
                    print(f"    Running job...")
                    result = subprocess.run(cmd, cwd=os.getcwd())
                    if result.returncode != 0:
                        print(f"    ERROR: Job failed with return code {result.returncode}")
                    else:
                        print(f"    SUCCESS: Job completed")
            
            job_count += 1
    
    # Save command list for reference
    with open(os.path.join(base_dir, "sweep_commands.txt"), "w") as f:
        f.write(f"# Hyperparameter sweep commands generated on {datetime.now()}\n")
        f.write(f"# Total jobs: {total_jobs}\n\n")
        for cmd, cmd_str, combo_name, seed, gpu in all_commands:
            f.write(f"# Combination: {combo_name}, Seed: {seed}, GPU: {gpu}\n")
            f.write(f"{cmd_str}\n\n")
    
    if args.dry_run:
        print(f"\nDry run complete. Would have executed {total_jobs} jobs.")
        print(f"Commands saved to {os.path.join(base_dir, 'sweep_commands.txt')}")
    elif args.parallel:
        print(f"\nAll {total_jobs} jobs started in parallel.")
        print("Monitor GPU usage with 'nvidia-smi' to track progress.")
        print(f"Use 'python scripts/smooth_policy/monitor_sweep.py' to check completion status.")
    else:
        print(f"\nAll {total_jobs} jobs completed sequentially.")

if __name__ == "__main__":
    main()

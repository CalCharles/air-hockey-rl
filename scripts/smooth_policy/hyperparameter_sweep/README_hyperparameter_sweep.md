# Hyperparameter Sweep for Iterative Smoothing

This directory contains scripts for running systematic hyperparameter sweeps on the iterative smoothing algorithm.

## Files

- `hyperparameter_sweep.py`: Main script that runs the hyperparameter sweep
- `monitor_sweep.py`: Script to monitor progress of running sweeps
- `kill_sweep.py`: Script to kill all running sweep processes
- `README_hyperparameter_sweep.md`: This documentation file

## Quick Start

### 1. Run a Dry Run (Recommended First)
```bash
python3 scripts/smooth_policy/hyperparameter_sweep.py --dry_run
```

### 2. Run the Full Sweep
```bash
# Sequential execution (safer, slower)
python3 scripts/smooth_policy/hyperparameter_sweep.py

# Parallel execution (faster, uses more resources)
python3 scripts/smooth_policy/hyperparameter_sweep.py --parallel
```

### 3. Monitor Progress
```bash
# Check progress once
python3 scripts/smooth_policy/monitor_sweep.py

# Watch mode (refresh every 30 seconds)
python3 scripts/smooth_policy/monitor_sweep.py --watch 30

# Verbose output (shows individual seed status)
python3 scripts/smooth_policy/monitor_sweep.py --verbose
```

### 4. Kill Running Processes (if needed)
```bash
# Check what would be killed (dry run)
python3 scripts/smooth_policy/kill_sweep.py --dry_run

# Kill all sweep processes gracefully
python3 scripts/smooth_policy/kill_sweep.py

# Force kill immediately
python3 scripts/smooth_policy/kill_sweep.py --force

# Kill processes on specific GPU
python3 scripts/smooth_policy/kill_sweep.py --gpu 0
```

## Hyperparameters Being Swept

The script varies these 3 boolean hyperparameters:
- `norm_adv`: Normalize advantages
- `reward_scaling`: Scale rewards by 1/10
- `reward_normalization`: Normalize rewards using running statistics

This creates 2³ = 8 combinations, each run with 5 different seeds (0-4).

## Directory Structure

Results are organized in `runs/scaling/` with the following structure:
```
runs/scaling/
├── norm_adv_False_reward_scaling_False_reward_normalization_False/
│   ├── seed_0/
│   ├── seed_1/
│   ├── seed_2/
│   ├── seed_3/
│   └── seed_4/
├── norm_adv_False_reward_scaling_False_reward_normalization_True/
│   └── ... (5 seeds)
└── ... (8 total combinations)
```

## GPU Distribution

Jobs are distributed across GPUs in round-robin fashion:
- Total jobs: 40 (8 combinations × 5 seeds)
- GPUs used: cuda:0 through cuda:7
- Each GPU gets 5 jobs

## Command Line Options

### hyperparameter_sweep.py
```bash
python3 scripts/smooth_policy/hyperparameter_sweep.py [OPTIONS]

Options:
  --config_path PATH          Config file path (default: scripts/smooth_policy/configs/puck_juggle/default_config.yaml)
  --num_iterations INT        Number of training iterations (default: 150)
  --dry_run                   Print commands without executing
  --parallel                  Run jobs in parallel (background processes)
  --num_gpus INT             Number of GPUs to use (default: 8)
```

### monitor_sweep.py
```bash
python3 scripts/smooth_policy/monitor_sweep.py [OPTIONS]

Options:
  --base_dir PATH            Base directory for sweep results (default: runs/scaling)
  --verbose, -v              Show detailed status for each seed
  --watch SECONDS, -w        Watch mode: refresh every N seconds
```

### kill_sweep.py
```bash
python3 scripts/smooth_policy/kill_sweep.py [OPTIONS]

Options:
  --force, -f                Force kill processes immediately (SIGKILL instead of SIGTERM)
  --dry_run                  Show what would be killed without actually killing
  --watch SECONDS, -w        Watch mode: check for processes every N seconds
  --gpu GPU_ID               Kill processes running on specific GPU only
```

## Examples

### Custom Configuration
```bash
python3 scripts/smooth_policy/hyperparameter_sweep.py \
    --config_path scripts/smooth_policy/configs/puck_touch/default_config.yaml \
    --num_iterations 100
```

### Limited GPU Usage
```bash
python3 scripts/smooth_policy/hyperparameter_sweep.py \
    --num_gpus 4 \
    --parallel
```

### Monitor with Auto-refresh
```bash
python3 scripts/smooth_policy/monitor_sweep.py --watch 60 --verbose
```

### Emergency Stop
```bash
# Kill all sweep processes immediately
python3 scripts/smooth_policy/kill_sweep.py --force

# Kill only processes on GPU 3
python3 scripts/smooth_policy/kill_sweep.py --gpu 3
```

## Output Files

Each seed directory contains:
- `config.yaml`: Environment configuration
- `args.yaml`: Training arguments
- `iterative_smoothing_model.pth`: Final trained model
- `checkpoint_*/`: Intermediate checkpoints (every 10 iterations)
- TensorBoard logs and evaluation results

## Tips

1. **Start with a dry run** to verify commands are correct
2. **Use parallel execution** for faster completion, but monitor GPU memory usage
3. **Check progress regularly** using the monitor script
4. **Sequential execution** is safer if you're unsure about resource limits
5. **Commands are saved** in `runs/scaling/sweep_commands.txt` for reference

## Troubleshooting

- If jobs fail, check individual log files in the seed directories
- Use `nvidia-smi` to monitor GPU usage and memory
- The monitor script shows which jobs are stuck or failed
- Commands can be re-run individually from the saved command file

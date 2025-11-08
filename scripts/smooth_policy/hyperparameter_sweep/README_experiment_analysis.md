# Experiment Analysis Scripts

This directory contains scripts for extracting and visualizing final metric values from hyperparameter sweep experiments.

## Overview

The workflow consists of two main scripts:

1. **`extract_final_returns.py`** - Extracts final metric values from tensorboard logs and saves them to YAML files
2. **`plot_hyperparameter_comparison.py`** - Creates 1D dot plot visualizations comparing hyperparameter settings

## Script 1: extract_final_returns.py

### Purpose
Extracts final metric values from experiment directories containing multiple hyperparameter settings and seeds, saving results to structured YAML files.

### Features
- Extracts multiple metrics simultaneously (default: `charts/avg_episodic_return` and `charts/max_episodic_return`)
- Handles missing data gracefully
- Calculates summary statistics (mean, std, min, max) across seeds
- Saves results in a structured YAML format for easy analysis

### Usage

#### Basic Usage
```bash
# Extract default metrics from pos_reward_scaling experiment
python scripts/smooth_policy/extract_final_returns.py --experiment_dir runs/pos_reward_scaling

# Extract from iterative_smoothing experiment
python scripts/smooth_policy/extract_final_returns.py --experiment_dir runs/iterative_smoothing
```

#### Advanced Usage
```bash
# Extract specific metrics
python scripts/smooth_policy/extract_final_returns.py \
    --experiment_dir runs/pos_reward_scaling \
    --metrics charts/avg_episodic_return eval/success_rate

# Extract single metric (backward compatibility)
python scripts/smooth_policy/extract_final_returns.py \
    --experiment_dir runs/pos_reward_scaling \
    --metric charts/avg_episodic_return

# Custom output filename
python scripts/smooth_policy/extract_final_returns.py \
    --experiment_dir runs/pos_reward_scaling \
    --output_filename results_summary.yaml

# List available metrics first
python scripts/smooth_policy/extract_final_returns.py \
    --experiment_dir runs/pos_reward_scaling \
    --list_metrics
```

### Output Format

The script creates a `final_returns.yaml` file in each hyperparameter directory with the following structure:

```yaml
hyperparameter_setting: "no_scaling"
metrics:
- charts/avg_episodic_return
- charts/max_episodic_return
seeds:
  seed_0:
    seed_number: 0
    log_directory: "runs/pos_reward_scaling/no_scaling/seed_0"
    metrics:
      charts/avg_episodic_return: 126.56
      charts/max_episodic_return: 849.57
  seed_1:
    seed_number: 1
    log_directory: "runs/pos_reward_scaling/no_scaling/seed_1"
    metrics:
      charts/avg_episodic_return: 93.94
      charts/max_episodic_return: 576.05
summary:
  charts/avg_episodic_return:
    num_successful_seeds: 4
    mean: 127.36
    std: 20.72
    min: 93.94
    max: 147.19
    values: [126.56, 93.94, 141.74, 147.19]
  charts/max_episodic_return:
    num_successful_seeds: 4
    mean: 910.38
    std: 249.13
    min: 576.05
    max: 1273.36
    values: [849.57, 576.05, 942.56, 1273.36]
```

## Script 2: plot_hyperparameter_comparison.py

### Purpose
Creates 1D dot plot visualizations comparing final metric values across different hyperparameter settings.

### Features
- Side-by-side comparison of hyperparameter settings
- Individual seed values shown as dots with jitter for visibility
- Mean values displayed as red horizontal lines with numerical labels
- Consistent scaling across all hyperparameter settings
- Support for multiple metrics (individual plots or side-by-side)
- High-resolution PNG output (300 DPI)

### Usage

#### Basic Usage
```bash
# Create comparison plot for avg_episodic_return
python scripts/plot_hyperparameter_comparison.py --experiment_dir runs/pos_reward_scaling

# Plot multiple metrics individually
python scripts/plot_hyperparameter_comparison.py \
    --experiment_dir runs/pos_reward_scaling \
    --metrics charts/avg_episodic_return charts/max_episodic_return
```

#### Advanced Usage
```bash
# Create side-by-side comparison of multiple metrics
python scripts/plot_hyperparameter_comparison.py \
    --experiment_dir runs/pos_reward_scaling \
    --metrics charts/avg_episodic_return charts/max_episodic_return \
    --side_by_side

# Custom output directory
python scripts/plot_hyperparameter_comparison.py \
    --experiment_dir runs/pos_reward_scaling \
    --output_dir plots/

# Only create side-by-side plots (no individual plots)
python scripts/plot_hyperparameter_comparison.py \
    --experiment_dir runs/pos_reward_scaling \
    --metrics charts/avg_episodic_return charts/max_episodic_return \
    --side_by_side \
    --no-individual
```

### Output Files

The script generates PNG files with descriptive names:

- `pos_reward_scaling_charts_avg_episodic_return_comparison.png` - Individual metric plot
- `pos_reward_scaling_charts_max_episodic_return_comparison.png` - Individual metric plot
- `pos_reward_scaling_multi_metric_comparison.png` - Side-by-side comparison (if `--side_by_side` used)

### Plot Features

Each plot includes:
- **X-axis**: Hyperparameter setting names (rotated 45° for readability)
- **Y-axis**: Metric values (consistent scale across all settings)
- **Dots**: Individual seed results (with small jitter for visibility)
- **Red lines**: Mean values for each hyperparameter setting
- **Text labels**: Numerical mean values next to red lines
- **Statistics box**: Global min, max, mean, and standard deviation
- **Grid**: Horizontal grid lines for easier reading

## Complete Workflow Example

Here's a complete example of analyzing a hyperparameter sweep:

```bash
# Step 1: Extract final metric values from all hyperparameter settings
python scripts/smooth_policy/extract_final_returns.py \
    --experiment_dir runs/pos_reward_scaling \
    --metrics charts/avg_episodic_return charts/max_episodic_return

# Step 2: Create visualization comparing hyperparameter settings
python scripts/plot_hyperparameter_comparison.py \
    --experiment_dir runs/pos_reward_scaling \
    --metrics charts/avg_episodic_return charts/max_episodic_return \
    --side_by_side

# Step 3: Check the generated files
ls runs/pos_reward_scaling/*/final_returns.yaml  # YAML files in each hyperparam dir
ls runs/pos_reward_scaling_*.png                 # Generated plots
```

## Directory Structure

The scripts expect the following directory structure:

```
runs/
└── experiment_name/
    ├── hyperparameter_setting_1/
    │   ├── seed_0/
    │   │   └── events.out.tfevents.*
    │   ├── seed_1/
    │   │   └── events.out.tfevents.*
    │   └── final_returns.yaml  # Generated by extract_final_returns.py
    ├── hyperparameter_setting_2/
    │   ├── seed_0/
    │   └── seed_1/
    └── experiment_name_*.png   # Generated by plot_hyperparameter_comparison.py
```

## Dependencies

Both scripts require:
- `numpy`
- `matplotlib`
- `pyyaml`
- `tensorboard` (for extract_final_returns.py)

## Error Handling

Both scripts handle common issues gracefully:
- Missing tensorboard files
- Corrupted or incomplete logs
- Missing metrics
- Empty directories
- Invalid YAML files

Error messages are informative and help identify specific issues with individual seeds or hyperparameter settings.

## Integration with Existing Workflow

These scripts integrate seamlessly with the existing air-hockey-rl codebase:
- Use the same matplotlib backend (`Agg`) as existing plotting utilities
- Follow the same file naming conventions
- Compatible with the existing tensorboard logging format
- Reuse patterns from `scripts/utils.py` and `scripts/smooth_policy/plot_comparison.py`

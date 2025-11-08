#!/usr/bin/env python3
"""
Create 1D dot plots comparing final metric values across hyperparameter settings.

This script reads final_returns.yaml files from each hyperparameter directory and creates
side-by-side 1D dot plots showing the distribution of final metric values across seeds.

Usage:
    python plot_hyperparameter_comparison.py --experiment_dir runs/pos_reward_scaling
    python plot_hyperparameter_comparison.py --experiment_dir runs/pos_reward_scaling --metric charts/max_episodic_return
"""

import os
import yaml
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Set style for better-looking plots
plt.style.use('default')


def load_hyperparameter_results(experiment_dir: str) -> Dict[str, Dict]:
    """
    Load final_returns.yaml files from all hyperparameter directories.
    
    Args:
        experiment_dir: Path to experiment directory containing hyperparameter subdirectories
        
    Returns:
        Dictionary mapping hyperparameter names to their results data
    """
    experiment_path = Path(experiment_dir)
    results = {}
    
    # Find all hyperparameter directories with final_returns.yaml files
    for item in experiment_path.iterdir():
        if item.is_dir():
            yaml_file = item / 'final_returns.yaml'
            if yaml_file.exists():
                try:
                    with open(yaml_file, 'r') as f:
                        data = yaml.safe_load(f)
                    results[item.name] = data
                    print(f"Loaded results for: {item.name}")
                except Exception as e:
                    print(f"Error loading {yaml_file}: {e}")
    
    return results


def extract_metric_values(results: Dict[str, Dict], metric: str) -> Dict[str, List[float]]:
    """
    Extract metric values for all seeds across all hyperparameter settings.
    
    Args:
        results: Dictionary of hyperparameter results
        metric: Metric name to extract
        
    Returns:
        Dictionary mapping hyperparameter names to lists of metric values
    """
    metric_data = {}
    
    for hyperparam_name, data in results.items():
        values = []
        
        # Extract values from individual seeds
        if 'seeds' in data:
            for seed_name, seed_data in data['seeds'].items():
                if 'metrics' in seed_data and metric in seed_data['metrics']:
                    value = seed_data['metrics'][metric]
                    if value is not None:
                        values.append(float(value))
        
        metric_data[hyperparam_name] = values
        print(f"{hyperparam_name}: {len(values)} valid seeds for {metric}")
    
    return metric_data


def create_1d_dot_plot(metric_data: Dict[str, List[float]], 
                       metric_name: str,
                       output_path: str = None,
                       figsize: Tuple[int, int] = (15, 8)) -> None:
    """
    Create a 1D dot plot comparing metric values across hyperparameter settings.
    
    Args:
        metric_data: Dictionary mapping hyperparameter names to metric values
        metric_name: Name of the metric being plotted
        output_path: Path to save the plot
        figsize: Figure size (width, height)
    """
    # Filter out hyperparameter settings with no data
    filtered_data = {k: v for k, v in metric_data.items() if len(v) > 0}
    
    if not filtered_data:
        print(f"No data found for metric {metric_name}")
        return
    
    # Sort hyperparameter names for consistent ordering
    hyperparam_names = sorted(filtered_data.keys())
    n_hyperparams = len(hyperparam_names)
    
    # Calculate global min/max for consistent scaling
    all_values = []
    for values in filtered_data.values():
        all_values.extend(values)
    
    if not all_values:
        print(f"No valid values found for metric {metric_name}")
        return
    
    global_min = min(all_values)
    global_max = max(all_values)
    value_range = global_max - global_min
    y_margin = value_range * 0.1  # 10% margin
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Colors for different hyperparameter settings
    colors = plt.cm.Set3(np.linspace(0, 1, n_hyperparams))
    
    # Plot data for each hyperparameter setting
    x_positions = []
    labels = []
    
    for i, hyperparam_name in enumerate(hyperparam_names):
        values = filtered_data[hyperparam_name]
        
        if not values:
            continue
            
        # Create x positions with some jitter for better visibility
        x_base = i + 1
        x_jitter = np.random.normal(0, 0.05, len(values))  # Small jitter
        x_coords = [x_base + jitter for jitter in x_jitter]
        
        # Plot individual points
        ax.scatter(x_coords, values, 
                  color=colors[i], 
                  alpha=0.7, 
                  s=60,  # Point size
                  edgecolors='black',
                  linewidth=0.5,
                  label=f'{hyperparam_name} (n={len(values)})')
        
        # Add mean line
        mean_value = np.mean(values)
        ax.hlines(mean_value, x_base - 0.3, x_base + 0.3, 
                 colors='red', linestyles='solid', linewidth=2)
        
        # Add mean value text
        ax.text(x_base + 0.35, mean_value, f'{mean_value:.1f}', 
               verticalalignment='center', fontsize=9, fontweight='bold')
        
        # Store for x-axis labels
        x_positions.append(x_base)
        labels.append(hyperparam_name)
    
    # Customize the plot
    ax.set_xlim(0.5, n_hyperparams + 0.5)
    ax.set_ylim(global_min - y_margin, global_max + y_margin)
    
    # Set x-axis
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_xlabel('Hyperparameter Settings')
    
    # Set y-axis
    ax.set_ylabel(f'{metric_name}')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Title
    ax.set_title(f'Final {metric_name} Values Across Hyperparameter Settings\n'
                f'(Red lines show means, dots show individual seeds)', 
                fontsize=14, pad=20)
    
    # Add statistics text box
    stats_text = f"Global Statistics:\n"
    stats_text += f"Min: {global_min:.2f}\n"
    stats_text += f"Max: {global_max:.2f}\n"
    stats_text += f"Overall Mean: {np.mean(all_values):.2f}\n"
    stats_text += f"Overall Std: {np.std(all_values):.2f}"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
           fontsize=9)
    
    # Tight layout
    plt.tight_layout()
    
    # Save the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
    
    plt.close()


def create_multi_metric_comparison(results: Dict[str, Dict], 
                                 metrics: List[str],
                                 experiment_dir: str,
                                 output_dir: str = None) -> None:
    """
    Create comparison plots for multiple metrics.
    
    Args:
        results: Dictionary of hyperparameter results
        metrics: List of metrics to plot
        experiment_dir: Name of experiment directory (for naming)
        output_dir: Directory to save plots
    """
    if output_dir is None:
        output_dir = Path(experiment_dir).parent
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    experiment_name = Path(experiment_dir).name
    
    for metric in metrics:
        print(f"\nCreating plot for {metric}...")
        
        # Extract metric data
        metric_data = extract_metric_values(results, metric)
        
        # Create safe filename
        safe_metric_name = metric.replace('/', '_').replace('\\', '_')
        output_path = output_dir / f"{experiment_name}_{safe_metric_name}_comparison.png"
        
        # Create plot
        create_1d_dot_plot(metric_data, metric, str(output_path))


def create_side_by_side_comparison(results: Dict[str, Dict],
                                 metrics: List[str],
                                 experiment_dir: str,
                                 output_dir: str = None) -> None:
    """
    Create a side-by-side comparison plot for multiple metrics.
    
    Args:
        results: Dictionary of hyperparameter results
        metrics: List of metrics to plot side by side
        experiment_dir: Name of experiment directory
        output_dir: Directory to save plots
    """
    if output_dir is None:
        output_dir = Path(experiment_dir).parent
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    experiment_name = Path(experiment_dir).name
    n_metrics = len(metrics)
    
    # Create subplots
    fig, axes = plt.subplots(1, n_metrics, figsize=(7 * n_metrics, 8))
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Extract metric data
        metric_data = extract_metric_values(results, metric)
        
        # Filter out empty data
        filtered_data = {k: v for k, v in metric_data.items() if len(v) > 0}
        
        if not filtered_data:
            ax.text(0.5, 0.5, f'No data for\n{metric}', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(metric)
            continue
        
        # Sort hyperparameter names
        hyperparam_names = sorted(filtered_data.keys())
        n_hyperparams = len(hyperparam_names)
        
        # Calculate scaling
        all_values = []
        for values in filtered_data.values():
            all_values.extend(values)
        
        global_min = min(all_values)
        global_max = max(all_values)
        value_range = global_max - global_min
        y_margin = value_range * 0.1
        
        # Colors
        colors = plt.cm.Set3(np.linspace(0, 1, n_hyperparams))
        
        # Plot data
        x_positions = []
        labels = []
        
        for j, hyperparam_name in enumerate(hyperparam_names):
            values = filtered_data[hyperparam_name]
            
            if not values:
                continue
            
            # X positions with jitter
            x_base = j + 1
            x_jitter = np.random.normal(0, 0.05, len(values))
            x_coords = [x_base + jitter for jitter in x_jitter]
            
            # Plot points
            ax.scatter(x_coords, values, 
                      color=colors[j], 
                      alpha=0.7, 
                      s=60,
                      edgecolors='black',
                      linewidth=0.5)
            
            # Mean line
            mean_value = np.mean(values)
            ax.hlines(mean_value, x_base - 0.3, x_base + 0.3, 
                     colors='red', linestyles='solid', linewidth=2)
            
            # Mean text
            ax.text(x_base + 0.35, mean_value, f'{mean_value:.1f}', 
                   verticalalignment='center', fontsize=8, fontweight='bold')
            
            x_positions.append(x_base)
            labels.append(hyperparam_name)
        
        # Customize subplot
        ax.set_xlim(0.5, n_hyperparams + 0.5)
        ax.set_ylim(global_min - y_margin, global_max + y_margin)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_xlabel('Hyperparameter Settings')
        ax.set_ylabel(metric)
        ax.set_title(metric, fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Overall title
    fig.suptitle(f'Hyperparameter Comparison: {experiment_name}', fontsize=16, y=0.98)
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    output_path = output_dir / f"{experiment_name}_multi_metric_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved multi-metric plot to: {output_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Create 1D dot plots comparing hyperparameter settings')
    parser.add_argument('--experiment_dir', type=str, required=True,
                       help='Path to experiment directory containing hyperparameter subdirectories')
    parser.add_argument('--metrics', type=str, nargs='+', 
                       default=['charts/avg_episodic_return'],
                       help='Metrics to plot (default: charts/avg_episodic_return)')
    parser.add_argument('--output_dir', type=str, 
                       help='Directory to save plots (default: parent of experiment_dir)')
    parser.add_argument('--side_by_side', action='store_true',
                       help='Create side-by-side comparison of multiple metrics')
    parser.add_argument('--individual', action='store_true', default=True,
                       help='Create individual plots for each metric (default: True)')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from: {args.experiment_dir}")
    results = load_hyperparameter_results(args.experiment_dir)
    
    if not results:
        print("No hyperparameter results found!")
        return
    
    print(f"Found {len(results)} hyperparameter settings:")
    for name in sorted(results.keys()):
        print(f"  - {name}")
    
    # Determine available metrics
    available_metrics = set()
    for data in results.values():
        if 'metrics' in data:
            available_metrics.update(data['metrics'])
    
    print(f"\nAvailable metrics: {sorted(available_metrics)}")
    
    # Filter requested metrics to only available ones
    valid_metrics = [m for m in args.metrics if m in available_metrics]
    if not valid_metrics:
        print(f"None of the requested metrics {args.metrics} are available!")
        return
    
    print(f"Plotting metrics: {valid_metrics}")
    
    # Create plots
    if args.individual:
        create_multi_metric_comparison(results, valid_metrics, args.experiment_dir, args.output_dir)
    
    if args.side_by_side and len(valid_metrics) > 1:
        create_side_by_side_comparison(results, valid_metrics, args.experiment_dir, args.output_dir)


if __name__ == '__main__':
    main()

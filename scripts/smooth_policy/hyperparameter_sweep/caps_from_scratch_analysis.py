import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import yaml
import re
from pathlib import Path
from collections import defaultdict
import argparse


def parse_hyperparameters_from_dirname(dirname):
    """
    Parse hyperparameters from directory name.
    
    Args:
        dirname (str): Directory name like 'dynamic_reward_scaling-False_caps_coef_consecutive-0.001_caps_coef_nearby-0.0003'
        
    Returns:
        dict: Parsed hyperparameters
    """
    # Extract hyperparameters using regex
    pattern = r'dynamic_reward_scaling-(\w+)_caps_coef_consecutive-([\d.]+)_caps_coef_nearby-([\d.]+)'
    match = re.match(pattern, dirname)
    
    if not match:
        return None
    
    return {
        'dynamic_reward_scaling': match.group(1) == 'True',
        'caps_coef_consecutive': float(match.group(2)),
        'caps_coef_nearby': float(match.group(3))
    }


def load_experiment_data(base_dir):
    """
    Load all experiment data from final_returns.yaml files.
    
    Args:
        base_dir (str): Base directory containing experiment subdirectories
        
    Returns:
        list: List of experiment data dictionaries
    """
    experiments = []
    
    # Find all final_returns.yaml files
    pattern = os.path.join(base_dir, "**", "final_returns.yaml")
    yaml_files = glob.glob(pattern, recursive=True)
    
    for yaml_file in yaml_files:
        try:
            with open(yaml_file, 'r') as f:
                data = yaml.safe_load(f)
            
            # Extract directory name to parse hyperparameters
            exp_dir = os.path.dirname(yaml_file)
            dirname = os.path.basename(exp_dir)
            hyperparams = parse_hyperparameters_from_dirname(dirname)
            
            if hyperparams is None:
                print(f"Could not parse hyperparameters from {dirname}")
                continue
            
            # Extract metrics
            avg_return = data['summary']['charts/avg_episodic_return']['mean']
            max_return = data['summary']['charts/max_episodic_return']['mean']
            
            experiments.append({
                'hyperparams': hyperparams,
                'avg_return': avg_return,
                'max_return': max_return,
                'dirname': dirname,
                'yaml_file': yaml_file
            })
            
        except Exception as e:
            print(f"Error loading {yaml_file}: {e}")
    
    return experiments


def plot_dynamic_scaling_comparison(experiments, output_dir=None):
    """
    Plot comparison between dynamic_reward_scaling True vs False.
    """
    # Group by dynamic_reward_scaling
    true_experiments = [exp for exp in experiments if exp['hyperparams']['dynamic_reward_scaling']]
    false_experiments = [exp for exp in experiments if not exp['hyperparams']['dynamic_reward_scaling']]
    
    # Extract returns
    true_avg_returns = [exp['avg_return'] for exp in true_experiments]
    false_avg_returns = [exp['avg_return'] for exp in false_experiments]
    true_max_returns = [exp['max_return'] for exp in true_experiments]
    false_max_returns = [exp['max_return'] for exp in false_experiments]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot average returns
    ax1.scatter([0] * len(false_avg_returns), false_avg_returns, alpha=0.7, s=60, label=f'False (n={len(false_avg_returns)})')
    ax1.scatter([1] * len(true_avg_returns), true_avg_returns, alpha=0.7, s=60, label=f'True (n={len(true_avg_returns)})')
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['False', 'True'])
    ax1.set_xlabel('Dynamic Reward Scaling')
    ax1.set_ylabel('Average Episodic Return')
    ax1.set_title('Average Episodic Return by Dynamic Scaling')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add mean lines
    if false_avg_returns:
        ax1.axhline(y=np.mean(false_avg_returns), xmin=0, xmax=0.4, color='blue', linestyle='--', alpha=0.8)
        ax1.text(-0.3, np.mean(false_avg_returns), f'{np.mean(false_avg_returns):.1f}', ha='center', va='bottom')
    if true_avg_returns:
        ax1.axhline(y=np.mean(true_avg_returns), xmin=0.6, xmax=1, color='orange', linestyle='--', alpha=0.8)
        ax1.text(1.3, np.mean(true_avg_returns), f'{np.mean(true_avg_returns):.1f}', ha='center', va='bottom')
    
    # Plot max returns
    ax2.scatter([0] * len(false_max_returns), false_max_returns, alpha=0.7, s=60, label=f'False (n={len(false_max_returns)})')
    ax2.scatter([1] * len(true_max_returns), true_max_returns, alpha=0.7, s=60, label=f'True (n={len(true_max_returns)})')
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['False', 'True'])
    ax2.set_xlabel('Dynamic Reward Scaling')
    ax2.set_ylabel('Max Episodic Return')
    ax2.set_title('Max Episodic Return by Dynamic Scaling')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add mean lines
    if false_max_returns:
        ax2.axhline(y=np.mean(false_max_returns), xmin=0, xmax=0.4, color='blue', linestyle='--', alpha=0.8)
        ax2.text(-0.3, np.mean(false_max_returns), f'{np.mean(false_max_returns):.1f}', ha='center', va='bottom')
    if true_max_returns:
        ax2.axhline(y=np.mean(true_max_returns), xmin=0.6, xmax=1, color='orange', linestyle='--', alpha=0.8)
        ax2.text(1.3, np.mean(true_max_returns), f'{np.mean(true_max_returns):.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if output_dir:
        save_path = os.path.join(output_dir, 'dynamic_scaling_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()


def plot_consecutive_coef_comparison(experiments, output_dir=None):
    """
    Plot comparison across different values of caps_coef_consecutive.
    """
    # Group by caps_coef_consecutive
    consecutive_groups = defaultdict(list)
    for exp in experiments:
        consecutive_coef = exp['hyperparams']['caps_coef_consecutive']
        consecutive_groups[consecutive_coef].append(exp)
    
    # Sort by coefficient value
    sorted_coefs = sorted(consecutive_groups.keys())
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot average returns
    for i, coef in enumerate(sorted_coefs):
        exps = consecutive_groups[coef]
        avg_returns = [exp['avg_return'] for exp in exps]
        ax1.scatter([i] * len(avg_returns), avg_returns, alpha=0.7, s=60, label=f'{coef} (n={len(avg_returns)})')
        
        # Add mean line
        if avg_returns:
            mean_val = np.mean(avg_returns)
            ax1.plot([i-0.3, i+0.3], [mean_val, mean_val], 'k-', alpha=0.8, linewidth=2)
            ax1.text(i, mean_val + 20, f'{mean_val:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax1.set_xticks(range(len(sorted_coefs)))
    ax1.set_xticklabels([str(coef) for coef in sorted_coefs])
    ax1.set_xlabel('Caps Coefficient Consecutive')
    ax1.set_ylabel('Average Episodic Return')
    ax1.set_title('Average Episodic Return by Consecutive Coefficient')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot max returns
    for i, coef in enumerate(sorted_coefs):
        exps = consecutive_groups[coef]
        max_returns = [exp['max_return'] for exp in exps]
        ax2.scatter([i] * len(max_returns), max_returns, alpha=0.7, s=60, label=f'{coef} (n={len(max_returns)})')
        
        # Add mean line
        if max_returns:
            mean_val = np.mean(max_returns)
            ax2.plot([i-0.3, i+0.3], [mean_val, mean_val], 'k-', alpha=0.8, linewidth=2)
            ax2.text(i, mean_val + 30, f'{mean_val:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax2.set_xticks(range(len(sorted_coefs)))
    ax2.set_xticklabels([str(coef) for coef in sorted_coefs])
    ax2.set_xlabel('Caps Coefficient Consecutive')
    ax2.set_ylabel('Max Episodic Return')
    ax2.set_title('Max Episodic Return by Consecutive Coefficient')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        save_path = os.path.join(output_dir, 'consecutive_coef_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()


def plot_nearby_coef_comparison(experiments, output_dir=None):
    """
    Plot comparison across different values of caps_coef_nearby.
    """
    # Group by caps_coef_nearby
    nearby_groups = defaultdict(list)
    for exp in experiments:
        nearby_coef = exp['hyperparams']['caps_coef_nearby']
        nearby_groups[nearby_coef].append(exp)
    
    # Sort by coefficient value
    sorted_coefs = sorted(nearby_groups.keys())
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot average returns
    for i, coef in enumerate(sorted_coefs):
        exps = nearby_groups[coef]
        avg_returns = [exp['avg_return'] for exp in exps]
        ax1.scatter([i] * len(avg_returns), avg_returns, alpha=0.7, s=60, label=f'{coef} (n={len(avg_returns)})')
        
        # Add mean line
        if avg_returns:
            mean_val = np.mean(avg_returns)
            ax1.plot([i-0.3, i+0.3], [mean_val, mean_val], 'k-', alpha=0.8, linewidth=2)
            ax1.text(i, mean_val + 20, f'{mean_val:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax1.set_xticks(range(len(sorted_coefs)))
    ax1.set_xticklabels([str(coef) for coef in sorted_coefs])
    ax1.set_xlabel('Caps Coefficient Nearby')
    ax1.set_ylabel('Average Episodic Return')
    ax1.set_title('Average Episodic Return by Nearby Coefficient')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot max returns
    for i, coef in enumerate(sorted_coefs):
        exps = nearby_groups[coef]
        max_returns = [exp['max_return'] for exp in exps]
        ax2.scatter([i] * len(max_returns), max_returns, alpha=0.7, s=60, label=f'{coef} (n={len(max_returns)})')
        
        # Add mean line
        if max_returns:
            mean_val = np.mean(max_returns)
            ax2.plot([i-0.3, i+0.3], [mean_val, mean_val], 'k-', alpha=0.8, linewidth=2)
            ax2.text(i, mean_val + 30, f'{mean_val:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax2.set_xticks(range(len(sorted_coefs)))
    ax2.set_xticklabels([str(coef) for coef in sorted_coefs])
    ax2.set_xlabel('Caps Coefficient Nearby')
    ax2.set_ylabel('Max Episodic Return')
    ax2.set_title('Max Episodic Return by Nearby Coefficient')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        save_path = os.path.join(output_dir, 'nearby_coef_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()


def print_summary_statistics(experiments):
    """
    Print summary statistics for the experiments.
    """
    print(f"\n=== EXPERIMENT SUMMARY ===")
    print(f"Total experiments: {len(experiments)}")
    
    # Dynamic scaling breakdown
    true_count = sum(1 for exp in experiments if exp['hyperparams']['dynamic_reward_scaling'])
    false_count = len(experiments) - true_count
    print(f"Dynamic scaling True: {true_count}, False: {false_count}")
    
    # Consecutive coefficient values
    consecutive_values = sorted(set(exp['hyperparams']['caps_coef_consecutive'] for exp in experiments))
    print(f"Consecutive coefficient values: {consecutive_values}")
    
    # Nearby coefficient values
    nearby_values = sorted(set(exp['hyperparams']['caps_coef_nearby'] for exp in experiments))
    print(f"Nearby coefficient values: {nearby_values}")
    
    # Overall performance stats
    avg_returns = [exp['avg_return'] for exp in experiments]
    max_returns = [exp['max_return'] for exp in experiments]
    
    print(f"\nOverall Average Return: {np.mean(avg_returns):.2f} ± {np.std(avg_returns):.2f}")
    print(f"Overall Max Return: {np.mean(max_returns):.2f} ± {np.std(max_returns):.2f}")
    
    # Best performing experiments
    best_avg = max(experiments, key=lambda x: x['avg_return'])
    best_max = max(experiments, key=lambda x: x['max_return'])
    
    print(f"\nBest Average Return: {best_avg['avg_return']:.2f}")
    print(f"  Hyperparams: {best_avg['hyperparams']}")
    print(f"Best Max Return: {best_max['max_return']:.2f}")
    print(f"  Hyperparams: {best_max['hyperparams']}")


def main():
    parser = argparse.ArgumentParser(description='Analyze caps_from_scratch experiments')
    parser.add_argument('base_dir', help='Base directory containing caps_from_scratch experiments')
    parser.add_argument('--output-dir', help='Directory to save plots (optional)')
    parser.add_argument('--analysis', choices=['all', 'dynamic', 'consecutive', 'nearby'], 
                       default='all', help='Which analysis to run')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.base_dir):
        print(f"Error: Directory {args.base_dir} does not exist")
        return
    
    # Create output directory if specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load experiment data
    print("Loading experiment data...")
    experiments = load_experiment_data(args.base_dir)
    
    if not experiments:
        print("No experiment data found!")
        return
    
    # Print summary statistics
    print_summary_statistics(experiments)
    
    # Run analyses
    if args.analysis in ['all', 'dynamic']:
        print("\nGenerating dynamic scaling comparison...")
        plot_dynamic_scaling_comparison(experiments, args.output_dir)
    
    if args.analysis in ['all', 'consecutive']:
        print("\nGenerating consecutive coefficient comparison...")
        plot_consecutive_coef_comparison(experiments, args.output_dir)
    
    if args.analysis in ['all', 'nearby']:
        print("\nGenerating nearby coefficient comparison...")
        plot_nearby_coef_comparison(experiments, args.output_dir)


if __name__ == "__main__":
    main()

import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import argparse
from pathlib import Path


def find_log_directories(base_dir):
    """
    Find all subdirectories that contain tensorboard log files (.tfevents files).
    
    Args:
        base_dir (str): Base directory to search for log files
        
    Returns:
        list: List of directories containing log files
    """
    log_dirs = []
    
    # Search for all .tfevents files recursively
    tfevents_pattern = os.path.join(base_dir, "**", "*.tfevents.*")
    tfevents_files = glob.glob(tfevents_pattern, recursive=True)
    
    # Get unique directories containing these files
    for tfevents_file in tfevents_files:
        log_dir = os.path.dirname(tfevents_file)
        if log_dir not in log_dirs:
            log_dirs.append(log_dir)
    
    return sorted(log_dirs)


def extract_metric_data(log_dir, metric='eval/ep_return'):
    """
    Extract metric data from a tensorboard log directory.
    
    Args:
        log_dir (str): Path to directory containing tensorboard logs
        metric (str): Metric to extract (default: 'eval/ep_return')
        
    Returns:
        tuple: (steps, values) or (None, None) if metric not found
    """
    try:
        # Initialize event accumulator
        ea = event_accumulator.EventAccumulator(
            log_dir,
            size_guidance={
                event_accumulator.COMPRESSED_HISTOGRAMS: 0,
                event_accumulator.IMAGES: 0,
                event_accumulator.AUDIO: 0,
                event_accumulator.SCALARS: 0,
                event_accumulator.HISTOGRAMS: 0,
            }
        )
        
        # Load events
        ea.Reload()
        
        # Check if metric exists
        if metric not in ea.Tags()['scalars']:
            print(f"Metric '{metric}' not found in {log_dir}")
            return None, None
        
        # Extract scalar events
        scalar_events = ea.Scalars(metric)
        
        # Handle case where scalar_events might be a single event
        if hasattr(event_accumulator, "ScalarEvent") and isinstance(scalar_events, event_accumulator.ScalarEvent):
            scalar_events = [scalar_events]
        
        # Extract steps and values
        steps = [event.step for event in scalar_events]
        values = [event.value for event in scalar_events]
        
        return steps, values
        
    except Exception as e:
        print(f"Error processing {log_dir}: {e}")
        return None, None


def create_comparison_plot(base_dir, metric='eval/ep_return', output_path=None, max_plots_per_figure=12):
    """
    Create side-by-side comparison plots of a metric across all log directories.
    
    Args:
        base_dir (str): Base directory containing log subdirectories
        metric (str): Metric to plot (default: 'eval/ep_return')
        output_path (str): Path to save the plot (optional)
        max_plots_per_figure (int): Maximum number of subplots per figure
    """
    # Find all log directories
    log_dirs = find_log_directories(base_dir)
    
    if not log_dirs:
        print(f"No log directories found in {base_dir}")
        return
    
    print(f"Found {len(log_dirs)} log directories")
    
    # Extract data from all directories
    plot_data = []
    for log_dir in log_dirs:
        steps, values = extract_metric_data(log_dir, metric)
        if steps is not None and values is not None:
            # Create a readable label from the directory path
            rel_path = os.path.relpath(log_dir, base_dir)
            label = rel_path.replace(os.sep, '/')
            plot_data.append((label, steps, values))
    
    if not plot_data:
        print("No valid data found to plot")
        return
    
    # Calculate number of figures needed
    num_plots = len(plot_data)
    num_figures = (num_plots + max_plots_per_figure - 1) // max_plots_per_figure
    
    for fig_idx in range(num_figures):
        start_idx = fig_idx * max_plots_per_figure
        end_idx = min((fig_idx + 1) * max_plots_per_figure, num_plots)
        current_plots = plot_data[start_idx:end_idx]
        
        # Calculate subplot grid
        n_plots = len(current_plots)
        if n_plots <= 4:
            rows, cols = 2, 2
        elif n_plots <= 6:
            rows, cols = 2, 3
        elif n_plots <= 9:
            rows, cols = 3, 3
        else:
            rows, cols = 3, 4
        
        # Create figure and subplots
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if num_figures > 1:
            fig.suptitle(f'{metric} Comparison (Part {fig_idx+1}/{num_figures})', fontsize=16)
        else:
            fig.suptitle(f'{metric} Comparison', fontsize=16)
        
        # Flatten axes array for easier indexing
        if rows * cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Plot each experiment
        for i, (label, steps, values) in enumerate(current_plots):
            ax = axes[i]
            ax.plot(steps, values, linewidth=2)
            ax.set_title(label, fontsize=10, wrap=True)
            ax.set_xlabel('Steps')
            ax.set_ylabel('Average Return')
            ax.grid(True, alpha=0.3)
            
            # Add some statistics to the plot
            if values:
                final_value = values[-1]
                max_value = max(values)
                ax.text(0.02, 0.98, f'Final: {final_value:.2f}\nMax: {max_value:.2f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontsize=8)
        
        # Hide unused subplots
        for j in range(len(current_plots), len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        
        # Save the figure
        if output_path:
            if num_figures == 1:
                save_path = output_path
            else:
                base, ext = os.path.splitext(output_path)
                save_path = f"{base}_part{fig_idx+1}{ext}"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.show()


def plot_all_experiments_overlay(base_dir, metrics, output_path=None, group_by_parent=True):
    """
    Create an overlay plot showing all experiments on the same axes, optionally grouped by parent directory.
    
    Args:
        base_dir (str): Base directory containing log subdirectories
        metric (str): Metric to plot (default: 'eval/ep_return')
        output_path (str): Path to save the plot (optional)
        group_by_parent (bool): Whether to group experiments by parent directory
    """
    # Find all log directories
    log_dirs = find_log_directories(base_dir)
    
    if not log_dirs:
        print(f"No log directories found in {base_dir}")
        return
    
    print(f"Found {len(log_dirs)} log directories")
    
    # Ensure `metrics` is a list of metric names
    if isinstance(metrics, str):
        metrics = [metrics]
    n_metrics = len(metrics)

    if group_by_parent:
        grouped_data = {m: {} for m in metrics}
        for log_dir in log_dirs:
            rel_path = os.path.relpath(log_dir, base_dir)
            path_parts = rel_path.split(os.sep)
            if len(path_parts) >= 2:
                parent_dir = path_parts[0]
                experiment = path_parts[1]
                seed = path_parts[2] if len(path_parts) > 2 else 'seed_0'
                for metric in metrics:
                    steps, values = extract_metric_data(log_dir, metric)
                    if steps is None or values is None:
                        continue
                    if parent_dir not in grouped_data[metric]:
                        grouped_data[metric][parent_dir] = {}
                    if experiment not in grouped_data[metric][parent_dir]:
                        grouped_data[metric][parent_dir][experiment] = []
                    grouped_data[metric][parent_dir][experiment].append((seed, steps, values))

        # For each parent directory, plot all metrics as subplots
        for parent_dir in set.union(*(set(g.keys()) for g in grouped_data.values())):
            fig, axes = plt.subplots(1, n_metrics, figsize=(7*n_metrics, 6))
            if n_metrics == 1:
                axes = [axes]
            fig.suptitle(f"Overlay Metrics - {parent_dir}", fontsize=18)
            for ax_idx, metric in enumerate(metrics):
                ax = axes[ax_idx]
                exp_group = grouped_data.get(metric, {}).get(parent_dir, {})
                # Collect all seeds for color mapping
                all_seeds = []
                for seeds_data in exp_group.values():
                    for (seed, _, _) in seeds_data:
                        if seed not in all_seeds:
                            all_seeds.append(seed)
                seed_to_color = {}
                color_map = plt.cm.get_cmap('tab10', max(len(all_seeds), 1))
                for idx, seed in enumerate(all_seeds):
                    seed_to_color[seed] = color_map(idx)
                for i, (experiment, seeds_data) in enumerate(exp_group.items()):
                    for j, (seed, steps, values) in enumerate(seeds_data):
                        line_styles = ['-', '--', '-.', ':', '-']
                        line_style = line_styles[j % len(line_styles)]
                        label = f"{experiment}_{seed}"
                        color = seed_to_color.get(seed, "black")
                        ax.plot(steps, values, color=color, linestyle=line_style,
                                linewidth=1.5, label=label, alpha=0.8)

                ax.set_title(f"{metric}", fontsize=15)
                ax.set_xlabel('Steps')
                ax.set_ylabel('Value')
                ax.grid(True, alpha=0.3)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            if output_path:
                base, ext = os.path.splitext(output_path)
                save_path = f"{base}_{parent_dir}_overlay{ext}"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved overlay plot to {save_path}")
            plt.show()

    else:
        # Simple overlay of all experiments for each metric as a subplot
        # Aggregate data for each metric
        experiment_data_by_metric = {m: {} for m in metrics}
        for log_dir in log_dirs:
            rel_path = os.path.relpath(log_dir, base_dir)
            path_parts = rel_path.split(os.sep)
            for metric in metrics:
                steps, values = extract_metric_data(log_dir, metric)
                if steps is None or values is None:
                    continue
                if len(path_parts) >= 2:
                    experiment = path_parts[0] if len(path_parts) == 2 else path_parts[1]
                    seed = path_parts[-1] if 'seed' in path_parts[-1] else 'seed_0'
                    if experiment not in experiment_data_by_metric[metric]:
                        experiment_data_by_metric[metric][experiment] = []
                    experiment_data_by_metric[metric][experiment].append((seed, steps, values))
                else:
                    label = rel_path.replace(os.sep, '/')
                    if "misc" not in experiment_data_by_metric[metric]:
                        experiment_data_by_metric[metric]["misc"] = []
                    experiment_data_by_metric[metric]["misc"].append((label, steps, values))

        fig, axes = plt.subplots(1, n_metrics, figsize=(7*n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]
        fig.suptitle(f"Overlay Metrics - All Experiments", fontsize=18)

        for ax_idx, metric in enumerate(metrics):
            ax = axes[ax_idx]
            experiment_data = experiment_data_by_metric[metric]
            # Collect all seeds for color mapping
            all_seeds = []
            for exp_seeds_data in experiment_data.values():
                for (seed, _, _) in exp_seeds_data:
                    if seed not in all_seeds:
                        all_seeds.append(seed)
            seed_to_color = {}
            color_map = plt.cm.get_cmap('tab10', max(len(all_seeds), 1))
            for idx, seed in enumerate(all_seeds):
                seed_to_color[seed] = color_map(idx)
            for i, (experiment, seeds_data) in enumerate(experiment_data.items()):
                for j, (seed, steps, values) in enumerate(seeds_data):
                    line_styles = ['-', '--', '-.', ':', '-']
                    line_style = line_styles[j % len(line_styles)]
                    label = f"{experiment}_{seed}"
                    color = seed_to_color.get(seed, "black")
                    ax.plot(steps, values, color=color, linestyle=line_style,
                            linewidth=1.5, label=label, alpha=0.8)
            ax.set_title(metric, fontsize=15)
            ax.set_xlabel('Steps')
            ax.set_ylabel('Value')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            ax.grid(True, alpha=0.3)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved overlay plot to {output_path}")
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot comparison of training metrics across experiments')
    parser.add_argument('base_dir', help='Base directory containing experiment subdirectories')
    parser.add_argument('--metric', default='charts/avg_episodic_return', 
                       help='Metric to plot (default: charts/avg_episodic_return)')
    parser.add_argument('--output', help='Output path for saving plots')
    parser.add_argument('--mode', choices=['subplots', 'overlay'], default='subplots',
                       help='Plot mode: subplots (side-by-side) or overlay (all on same axes)')
    parser.add_argument('--max-plots', type=int, default=12,
                       help='Maximum number of subplots per figure (for subplot mode)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.base_dir):
        print(f"Error: Directory {args.base_dir} does not exist")
        return
    
    if args.mode == 'subplots':
        create_comparison_plot(args.base_dir, args.metric, args.output, args.max_plots)
    else:
        metrics = ['charts/avg_episodic_return', 'charts/max_episodic_return', 'charts/min_episodic_return']
        plot_all_experiments_overlay(args.base_dir, metrics, args.output, group_by_parent=False)


if __name__ == "__main__":
    main()

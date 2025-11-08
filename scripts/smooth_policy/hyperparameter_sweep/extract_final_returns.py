#!/usr/bin/env python3
"""
Extract final metric values from experiment directories and save to YAML files.

For each hyperparameter setting directory, this script:
1. Finds all seed subdirectories
2. Extracts the final values of specified metrics from each seed's tensorboard logs
3. Saves the results to a YAML file in the hyperparameter directory

By default, extracts both charts/avg_episodic_return and charts/max_episodic_return.

Usage:
    python extract_final_returns.py --experiment_dir runs/pos_reward_scaling
    python extract_final_returns.py --experiment_dir runs/iterative_smoothing --metrics eval/ep_return eval/success_rate
    python extract_final_returns.py --experiment_dir runs/pos_reward_scaling --metric charts/avg_episodic_return
"""

import os
import glob
import yaml
import argparse
import numpy as np
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator
from typing import Dict, List, Optional, Tuple


def find_tfevents_file(directory: str) -> Optional[str]:
    """Find the tensorboard events file in a directory."""
    pattern = os.path.join(directory, "events.out.tfevents.*")
    tfevents_files = glob.glob(pattern)
    
    if not tfevents_files:
        return None
    
    # Return the most recent file if multiple exist
    return max(tfevents_files, key=os.path.getmtime)


def extract_final_metrics_values(log_dir: str, metrics: List[str]) -> Dict[str, Optional[float]]:
    """
    Extract the final values of multiple metrics from tensorboard logs.
    
    Args:
        log_dir: Directory containing tensorboard event files
        metrics: List of metric names to extract
        
    Returns:
        Dictionary mapping metric names to their final values (or None if not found)
    """
    tfevents_file = find_tfevents_file(log_dir)
    if not tfevents_file:
        print(f"No tensorboard events file found in {log_dir}")
        return {metric: None for metric in metrics}
    
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
        
        results = {}
        available_metrics = ea.Tags()['scalars']
        
        for metric in metrics:
            # Check if metric exists
            if metric not in available_metrics:
                print(f"Metric '{metric}' not found in {log_dir}")
                results[metric] = None
                continue
            
            # Extract scalar events
            scalar_events = ea.Scalars(metric)
            
            # Handle case where scalar_events might be a single event
            if hasattr(event_accumulator, "ScalarEvent") and isinstance(scalar_events, event_accumulator.ScalarEvent):
                scalar_events = [scalar_events]
            
            if not scalar_events:
                print(f"No data found for metric '{metric}' in {log_dir}")
                results[metric] = None
            else:
                # Return the average of the last 10 values (or fewer if less data)
                last_n = 10
                values = [event.value for event in scalar_events[-last_n:]]
                avg_final = float(sum(values) / len(values))
                results[metric] = avg_final
        
        return results
        
    except Exception as e:
        print(f"Error processing {log_dir}: {e}")
        return {metric: None for metric in metrics}


def extract_final_metric_value(log_dir: str, metric: str = 'charts/avg_episodic_return') -> Optional[float]:
    """
    Extract the final value of a metric from tensorboard logs.
    
    Args:
        log_dir: Directory containing tensorboard event files
        metric: Metric name to extract
        
    Returns:
        Final metric value or None if not found
    """
    tfevents_file = find_tfevents_file(log_dir)
    if not tfevents_file:
        print(f"No tensorboard events file found in {log_dir}")
        return None
    
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
            available_metrics = ea.Tags()['scalars']
            print(f"Available metrics: {available_metrics}")
            return None
        
        # Extract scalar events
        scalar_events = ea.Scalars(metric)
        
        # Handle case where scalar_events might be a single event
        if hasattr(event_accumulator, "ScalarEvent") and isinstance(scalar_events, event_accumulator.ScalarEvent):
            scalar_events = [scalar_events]
        
        if not scalar_events:
            print(f"No data found for metric '{metric}' in {log_dir}")
            return None
        
        # Return the final (last) value
        final_value = scalar_events[-1].value
        return float(final_value)
        
    except Exception as e:
        print(f"Error processing {log_dir}: {e}")
        return None


def process_hyperparameter_directory(hyperparam_dir: str, metrics: List[str] = None) -> Dict:
    """
    Process a hyperparameter directory to extract final metric values from all seeds.
    
    Args:
        hyperparam_dir: Path to hyperparameter directory containing seed subdirectories
        metrics: List of metric names to extract
        
    Returns:
        Dictionary with seed results and metadata
    """
    if metrics is None:
        metrics = ['charts/avg_episodic_return', 'charts/max_episodic_return']
    hyperparam_path = Path(hyperparam_dir)
    
    # Find all seed directories
    seed_dirs = []
    for item in hyperparam_path.iterdir():
        if item.is_dir() and item.name.startswith('seed_'):
            seed_dirs.append(item)
    
    if not seed_dirs:
        print(f"No seed directories found in {hyperparam_dir}")
        return {}
    
    # Sort seed directories by seed number
    seed_dirs.sort(key=lambda x: int(x.name.split('_')[1]))
    
    results = {
        'hyperparameter_setting': hyperparam_path.name,
        'metrics': metrics,
        'seeds': {},
        'summary': {}
    }
    
    # Dictionary to store final values for each metric across seeds
    final_values_by_metric = {metric: [] for metric in metrics}
    
    for seed_dir in seed_dirs:
        seed_name = seed_dir.name
        seed_number = int(seed_name.split('_')[1])
        
        print(f"Processing {seed_dir}")
        metric_values = extract_final_metrics_values(str(seed_dir), metrics)
        
        # Store results for this seed
        seed_result = {
            'seed_number': seed_number,
            'log_directory': str(seed_dir),
            'metrics': {}
        }
        
        # Process each metric
        has_valid_data = False
        for metric in metrics:
            final_value = metric_values.get(metric)
            if final_value is not None:
                seed_result['metrics'][metric] = final_value
                final_values_by_metric[metric].append(final_value)
                has_valid_data = True
            else:
                seed_result['metrics'][metric] = None
        
        if not has_valid_data:
            seed_result['error'] = 'Could not extract any metric values'
        
        results['seeds'][seed_name] = seed_result
    
    # Calculate summary statistics for each metric
    results['summary'] = {}
    for metric in metrics:
        final_values = final_values_by_metric[metric]
        if final_values:
            results['summary'][metric] = {
                'num_successful_seeds': len(final_values),
                'mean': float(np.mean(final_values)),
                'std': float(np.std(final_values)),
                'min': float(np.min(final_values)),
                'max': float(np.max(final_values)),
                'values': final_values
            }
        else:
            results['summary'][metric] = {
                'num_successful_seeds': 0,
                'error': 'No successful extractions for this metric'
            }
    
    return results


def process_experiment_directory(experiment_dir: str, metrics: List[str] = None, 
                                output_filename: str = 'final_returns.yaml') -> None:
    """
    Process an entire experiment directory, extracting final metric values for each hyperparameter setting.
    
    Args:
        experiment_dir: Path to experiment directory containing hyperparameter subdirectories
        metrics: List of metric names to extract
        output_filename: Name of output YAML file to create in each hyperparameter directory
    """
    if metrics is None:
        metrics = ['charts/avg_episodic_return', 'charts/max_episodic_return']
    experiment_path = Path(experiment_dir)
    
    if not experiment_path.exists():
        print(f"Experiment directory does not exist: {experiment_dir}")
        return
    
    print(f"Processing experiment directory: {experiment_dir}")
    print(f"Looking for metrics: {metrics}")
    print(f"Output filename: {output_filename}")
    print("-" * 50)
    
    # Find all hyperparameter directories (directories containing seed subdirectories)
    hyperparam_dirs = []
    for item in experiment_path.iterdir():
        if item.is_dir():
            # Check if this directory contains seed subdirectories
            seed_dirs = [subitem for subitem in item.iterdir() 
                        if subitem.is_dir() and subitem.name.startswith('seed_')]
            if seed_dirs:
                hyperparam_dirs.append(item)
    
    if not hyperparam_dirs:
        print(f"No hyperparameter directories with seeds found in {experiment_dir}")
        return
    
    print(f"Found {len(hyperparam_dirs)} hyperparameter directories:")
    for hdir in hyperparam_dirs:
        print(f"  - {hdir.name}")
    print()
    
    # Process each hyperparameter directory
    for hyperparam_dir in hyperparam_dirs:
        print(f"\nProcessing hyperparameter setting: {hyperparam_dir.name}")
        
        results = process_hyperparameter_directory(str(hyperparam_dir), metrics)
        
        if results:
            # Save results to YAML file in the hyperparameter directory
            output_path = hyperparam_dir / output_filename
            
            try:
                with open(output_path, 'w') as f:
                    yaml.dump(results, f, default_flow_style=False, indent=2)
                print(f"Saved results to: {output_path}")
                
                # Print summary
                if 'summary' in results:
                    for metric in metrics:
                        if metric in results['summary'] and 'mean' in results['summary'][metric]:
                            summary = results['summary'][metric]
                            print(f"  {metric}: {summary['num_successful_seeds']} seeds, "
                                  f"mean={summary['mean']:.3f} ± {summary['std']:.3f}")
                
            except Exception as e:
                print(f"Error saving results to {output_path}: {e}")
        else:
            print(f"No results to save for {hyperparam_dir.name}")


def main():
    parser = argparse.ArgumentParser(description='Extract final metric values from experiment directories')
    parser.add_argument('--experiment_dir', type=str, required=True,
                       help='Path to experiment directory containing hyperparameter subdirectories')
    parser.add_argument('--metrics', type=str, nargs='+', 
                       default=['charts/avg_episodic_return', 'charts/max_episodic_return'],
                       help='Metrics to extract (default: charts/avg_episodic_return charts/max_episodic_return)')
    parser.add_argument('--metric', type=str, 
                       help='Single metric to extract (for backward compatibility)')
    parser.add_argument('--output_filename', type=str, default='final_returns.yaml',
                       help='Output YAML filename (default: final_returns.yaml)')
    parser.add_argument('--list_metrics', action='store_true',
                       help='List available metrics in the first seed directory found')
    
    args = parser.parse_args()
    
    # Handle backward compatibility for single metric
    if args.metric:
        metrics = [args.metric]
    else:
        metrics = args.metrics
    
    if args.list_metrics:
        # Find first seed directory to list available metrics
        experiment_path = Path(args.experiment_dir)
        for item in experiment_path.rglob('seed_*'):
            if item.is_dir():
                tfevents_file = find_tfevents_file(str(item))
                if tfevents_file:
                    try:
                        ea = event_accumulator.EventAccumulator(str(item))
                        ea.Reload()
                        metrics = ea.Tags()['scalars']
                        print(f"Available metrics in {item}:")
                        for metric in sorted(metrics):
                            print(f"  - {metric}")
                        return
                    except Exception as e:
                        print(f"Error reading metrics from {item}: {e}")
                        continue
        print("No tensorboard logs found to list metrics")
        return
    
    process_experiment_directory(args.experiment_dir, metrics, args.output_filename)


if __name__ == '__main__':
    main()

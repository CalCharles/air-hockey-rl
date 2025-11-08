#!/usr/bin/env python3
"""Monitor hyperparameter sweep progress"""

import os
import glob
import argparse
from datetime import datetime

def check_progress(base_dir="runs/scaling", verbose=False):
    """Check progress of hyperparameter sweep"""
    
    if not os.path.exists(base_dir):
        print(f"No scaling runs found in {base_dir}")
        return
    
    combinations = [d for d in os.listdir(base_dir) 
                   if os.path.isdir(os.path.join(base_dir, d)) and not d.startswith('.')]
    
    if not combinations:
        print(f"No combination directories found in {base_dir}")
        return
    
    total_expected = len(combinations) * 5  # 5 seeds per combination
    completed = 0
    running = 0
    failed = 0
    
    print(f"Hyperparameter Sweep Progress Report")
    print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Base directory: {base_dir}")
    print("=" * 60)
    
    combination_details = []
    
    for combo in sorted(combinations):
        combo_dir = os.path.join(base_dir, combo)
        seeds = [d for d in os.listdir(combo_dir) 
                if d.startswith("seed_") and os.path.isdir(os.path.join(combo_dir, d))]
        
        combo_completed = 0
        combo_running = 0
        combo_failed = 0
        
        seed_details = []
        
        for seed in sorted(seeds):
            seed_dir = os.path.join(combo_dir, seed)
            
            # Check if final model exists (completed)
            final_model = os.path.join(seed_dir, "iterative_smoothing_model.pth")
            
            # Check if any checkpoint exists (running)
            checkpoint_pattern = os.path.join(seed_dir, "checkpoint_*", "iterative_smoothing_model.pth")
            checkpoints = glob.glob(checkpoint_pattern)
            
            # Check if config exists but no progress (failed/not started)
            config_file = os.path.join(seed_dir, "config.yaml")
            
            if os.path.exists(final_model):
                status = "COMPLETED"
                combo_completed += 1
            elif checkpoints:
                # Find latest checkpoint
                latest_checkpoint = max(checkpoints, key=os.path.getctime)
                checkpoint_num = latest_checkpoint.split('checkpoint_')[1].split('/')[0]
                status = f"RUNNING (checkpoint {checkpoint_num})"
                combo_running += 1
            elif os.path.exists(config_file):
                status = "FAILED/STUCK"
                combo_failed += 1
            else:
                status = "NOT STARTED"
            
            seed_details.append((seed, status))
        
        completed += combo_completed
        running += combo_running
        failed += combo_failed
        
        combination_details.append((combo, combo_completed, combo_running, combo_failed, seed_details))
    
    # Print summary
    print(f"Overall Progress: {completed}/{total_expected} completed ({completed/total_expected*100:.1f}%)")
    print(f"Running: {running}, Failed/Stuck: {failed}")
    print()
    
    # Print detailed breakdown
    for combo, combo_completed, combo_running, combo_failed, seed_details in combination_details:
        status_summary = f"{combo_completed}/5 completed"
        if combo_running > 0:
            status_summary += f", {combo_running} running"
        if combo_failed > 0:
            status_summary += f", {combo_failed} failed"
        
        print(f"{combo}: {status_summary}")
        
        if verbose:
            for seed, status in seed_details:
                print(f"  {seed}: {status}")
            print()
    
    # GPU usage summary
    if running > 0:
        print(f"\nNote: {running} jobs appear to be running. Check GPU usage with 'nvidia-smi'")
    
    return {
        'total_expected': total_expected,
        'completed': completed,
        'running': running,
        'failed': failed,
        'combinations': combination_details
    }

def main():
    parser = argparse.ArgumentParser(description='Monitor hyperparameter sweep progress')
    parser.add_argument('--base_dir', type=str, default='runs/scaling',
                       help='Base directory for sweep results')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed status for each seed')
    parser.add_argument('--watch', '-w', type=int, metavar='SECONDS',
                       help='Watch mode: refresh every N seconds')
    
    args = parser.parse_args()
    
    if args.watch:
        import time
        try:
            while True:
                os.system('clear' if os.name == 'posix' else 'cls')  # Clear screen
                check_progress(args.base_dir, args.verbose)
                print(f"\nRefreshing in {args.watch} seconds... (Ctrl+C to exit)")
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
    else:
        check_progress(args.base_dir, args.verbose)

if __name__ == "__main__":
    main()

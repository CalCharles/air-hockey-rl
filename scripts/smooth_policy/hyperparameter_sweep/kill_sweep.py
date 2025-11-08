#!/usr/bin/env python3
"""
Kill all running hyperparameter sweep processes.
This script finds and terminates all processes running iterative_smoothing.py.
"""

import subprocess
import argparse
import sys
import time

def find_sweep_processes():
    """Find all processes running iterative_smoothing"""
    try:
        # Use ps to find processes containing iterative_smoothing
        result = subprocess.run(
            ['ps', 'aux'], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        processes = []
        lines = result.stdout.strip().split('\n')
        
        for line in lines[1:]:  # Skip header
            if 'iterative_smoothing' in line and 'python' in line:
                # Parse ps output: USER PID %CPU %MEM VSZ RSS TTY STAT START TIME COMMAND
                parts = line.split(None, 10)  # Split into max 11 parts
                if len(parts) >= 11:
                    pid = parts[1]
                    command = parts[10]
                    processes.append((pid, command))
        
        return processes
    
    except subprocess.CalledProcessError as e:
        print(f"Error finding processes: {e}")
        return []

def kill_processes(processes, force=False):
    """Kill the specified processes"""
    if not processes:
        print("No iterative_smoothing processes found.")
        return
    
    signal = 'SIGKILL' if force else 'SIGTERM'
    kill_flag = '-9' if force else '-15'
    
    print(f"Found {len(processes)} iterative_smoothing processes:")
    for pid, command in processes:
        print(f"  PID {pid}: {command[:100]}...")
    
    if not force:
        print(f"\nSending {signal} to processes...")
    else:
        print(f"\nForce killing processes with {signal}...")
    
    killed_count = 0
    failed_count = 0
    
    for pid, command in processes:
        try:
            subprocess.run(['kill', kill_flag, pid], check=True)
            print(f"  ✓ Killed PID {pid}")
            killed_count += 1
        except subprocess.CalledProcessError:
            print(f"  ✗ Failed to kill PID {pid} (may already be dead)")
            failed_count += 1
    
    print(f"\nSummary: {killed_count} killed, {failed_count} failed")
    
    if not force and killed_count > 0:
        print("\nWaiting 5 seconds for graceful shutdown...")
        time.sleep(5)
        
        # Check if any processes are still running
        remaining = find_sweep_processes()
        if remaining:
            print(f"\n{len(remaining)} processes still running. Use --force to kill them immediately.")
            return remaining
        else:
            print("All processes terminated successfully.")
    
    return []

def main():
    parser = argparse.ArgumentParser(description='Kill all running hyperparameter sweep processes')
    parser.add_argument('--force', '-f', action='store_true',
                       help='Force kill processes immediately (SIGKILL instead of SIGTERM)')
    parser.add_argument('--dry_run', action='store_true',
                       help='Show what would be killed without actually killing')
    parser.add_argument('--watch', '-w', type=int, metavar='SECONDS',
                       help='Watch mode: check for processes every N seconds')
    
    args = parser.parse_args()
    
    if args.watch:
        print(f"Watching for iterative_smoothing processes every {args.watch} seconds...")
        print("Press Ctrl+C to stop watching")
        
        try:
            while True:
                processes = find_sweep_processes()
                if processes:
                    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Found {len(processes)} processes:")
                    for pid, command in processes:
                        print(f"  PID {pid}: {command[:80]}...")
                else:
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] No processes found")
                
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nStopped watching.")
            return
    
    # Find processes
    processes = find_sweep_processes()
    
    if args.dry_run:
        if processes:
            print(f"Would kill {len(processes)} processes:")
            for pid, command in processes:
                signal = 'SIGKILL' if args.force else 'SIGTERM'
                print(f"  kill {'--force' if args.force else ''} PID {pid}: {command[:80]}...")
        else:
            print("No iterative_smoothing processes found to kill.")
        return
    
    # Kill processes
    remaining = kill_processes(processes, args.force)
    
    # If graceful kill failed and there are remaining processes, offer force kill
    if remaining and not args.force:
        response = input(f"\nForce kill the remaining {len(remaining)} processes? (y/N): ")
        if response.lower() in ['y', 'yes']:
            kill_processes(remaining, force=True)

def kill_by_gpu(gpu_id):
    """Kill processes running on a specific GPU"""
    processes = find_sweep_processes()
    gpu_processes = []
    
    for pid, command in processes:
        if f'cuda:{gpu_id}' in command:
            gpu_processes.append((pid, command))
    
    if gpu_processes:
        print(f"Found {len(gpu_processes)} processes on GPU {gpu_id}")
        kill_processes(gpu_processes, force=False)
    else:
        print(f"No processes found running on GPU {gpu_id}")

if __name__ == "__main__":
    # Check if specific GPU kill is requested
    if len(sys.argv) == 3 and sys.argv[1] == '--gpu':
        try:
            gpu_id = int(sys.argv[2])
            kill_by_gpu(gpu_id)
        except ValueError:
            print("Error: GPU ID must be an integer")
            sys.exit(1)
    else:
        main()

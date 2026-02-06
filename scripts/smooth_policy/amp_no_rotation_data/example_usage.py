#!/usr/bin/env python3
"""
Example script demonstrating how to use the AMP dataset.

This shows how to:
1. Load the dataset
2. Access state pairs
3. Create a PyTorch DataLoader
4. Iterate through the data
"""

import torch
from torch.utils.data import TensorDataset, DataLoader
import argparse


def load_amp_dataset(dataset_path):
    """
    Load the AMP dataset and print basic information.
    
    Args:
        dataset_path: Path to the .pt dataset file
        
    Returns:
        dict: Dataset dictionary containing state_pairs and metadata
    """
    print(f"Loading dataset from: {dataset_path}")
    data = torch.load(dataset_path)
    
    # Print basic info
    print(f"\nDataset contents:")
    print(f"  Keys: {list(data.keys())}")
    print(f"  State pairs shape: {data['state_pairs'].shape}")
    print(f"  Data type: {data['state_pairs'].dtype}")
    
    # Print statistics if available
    if 'stats' in data:
        stats = data['stats']
        print(f"\nDataset statistics:")
        print(f"  Valid trajectories: {stats['valid_trajectories']}")
        print(f"  Total state pairs: {stats['total_pairs']:,}")
        print(f"  Total timesteps: {stats['total_timesteps']:,}")
    
    return data


def create_dataloader(state_pairs, batch_size=256, shuffle=True, num_workers=0):
    """
    Create a PyTorch DataLoader from state pairs.
    
    Args:
        state_pairs: Tensor of shape (N, 2, 4)
        batch_size: Batch size for the dataloader
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        
    Returns:
        DataLoader: PyTorch DataLoader object
    """
    # Split into current and next states
    current_states = state_pairs[:, 0, :]  # (N, 4)
    next_states = state_pairs[:, 1, :]     # (N, 4)
    
    # Create TensorDataset
    dataset = TensorDataset(current_states, next_states)
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"\nCreated DataLoader:")
    print(f"  Dataset size: {len(dataset):,}")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of batches: {len(dataloader)}")
    print(f"  Shuffle: {shuffle}")
    
    return dataloader


def example_iteration(dataloader, num_batches=5):
    """
    Demonstrate iterating through the dataloader.
    
    Args:
        dataloader: PyTorch DataLoader
        num_batches: Number of batches to print
    """
    print(f"\n{'='*80}")
    print("EXAMPLE ITERATION")
    print('='*80)
    
    for batch_idx, (current_states, next_states) in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        
        print(f"\nBatch {batch_idx}:")
        print(f"  Current states shape: {current_states.shape}")
        print(f"  Next states shape: {next_states.shape}")
        
        # Show first sample in batch
        print(f"  First sample - Current state:")
        print(f"    X pos: {current_states[0, 0]:.4f} m")
        print(f"    Y pos: {current_states[0, 1]:.4f} m")
        print(f"    X vel: {current_states[0, 2]:.4f} m/s")
        print(f"    Y vel: {current_states[0, 3]:.4f} m/s")
        print(f"  First sample - Next state:")
        print(f"    X pos: {next_states[0, 0]:.4f} m")
        print(f"    Y pos: {next_states[0, 1]:.4f} m")
        print(f"    X vel: {next_states[0, 2]:.4f} m/s")
        print(f"    Y vel: {next_states[0, 3]:.4f} m/s")
    
    print(f"\n... (remaining {len(dataloader) - num_batches} batches not shown)")


def compute_state_statistics(state_pairs):
    """
    Compute and display statistics about the states.
    
    Args:
        state_pairs: Tensor of shape (N, 2, 4)
    """
    print(f"\n{'='*80}")
    print("STATE STATISTICS")
    print('='*80)
    
    # Flatten to get all states
    all_states = state_pairs.reshape(-1, 4)
    
    dim_names = ['X Position (m)', 'Y Position (m)', 
                 'X Velocity (m/s)', 'Y Velocity (m/s)']
    
    for i, name in enumerate(dim_names):
        values = all_states[:, i]
        print(f"\n{name}:")
        print(f"  Min: {values.min():.4f}")
        print(f"  Max: {values.max():.4f}")
        print(f"  Mean: {values.mean():.4f}")
        print(f"  Std: {values.std():.4f}")
        print(f"  Median: {values.median():.4f}")


def main():
    """Main function demonstrating dataset usage."""
    parser = argparse.ArgumentParser(
        description='Example usage of AMP dataset'
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default='amp_dataset.pt',
        help='Path to the AMP dataset file'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Batch size for DataLoader'
    )
    parser.add_argument(
        '--num-batches',
        type=int,
        default=5,
        help='Number of batches to display'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("AMP DATASET USAGE EXAMPLE")
    print("="*80)
    
    # Load dataset
    data = load_amp_dataset(args.dataset_path)
    state_pairs = data['state_pairs']
    
    # Compute statistics
    compute_state_statistics(state_pairs)
    
    # Create dataloader
    dataloader = create_dataloader(
        state_pairs,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    # Example iteration
    example_iteration(dataloader, num_batches=args.num_batches)
    
    print(f"\n{'='*80}")
    print("✓ Example complete!")
    print('='*80)
    
    # Print code snippet
    print("\nTo use this dataset in your training code:")
    print("```python")
    print("import torch")
    print("from torch.utils.data import TensorDataset, DataLoader")
    print("")
    print("# Load dataset")
    print(f"data = torch.load('{args.dataset_path}')")
    print("state_pairs = data['state_pairs']")
    print("")
    print("# Split into current and next states")
    print("current_states = state_pairs[:, 0, :]")
    print("next_states = state_pairs[:, 1, :]")
    print("")
    print("# Create dataloader")
    print("dataset = TensorDataset(current_states, next_states)")
    print(f"dataloader = DataLoader(dataset, batch_size={args.batch_size}, shuffle=True)")
    print("")
    print("# Training loop")
    print("for current, next_state in dataloader:")
    print("    # Your AMP training logic here")
    print("    pass")
    print("```")


if __name__ == '__main__':
    main()

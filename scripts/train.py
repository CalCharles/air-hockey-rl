"""
Refactored training script using modular trainer architecture.

This script uses the new modular training framework with algorithm-specific
trainers and a factory pattern for clean, extensible code.
"""

from scripts.trainers.trainer_factory import TrainerFactory
import argparse
import yaml
import os


def main():
    """Main training function using modular trainer architecture."""
    parser = argparse.ArgumentParser(description='Train air hockey RL agent using modular architecture.')
    parser.add_argument('--cfg', type=str, default=None, help='Path to the configuration file.')
    parser.add_argument('--wandb', action='store_true', help='Use wandb for logging.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training.')
    parser.add_argument('--clear', action='store_true', help='Removes prior folders for the task.')
    args = parser.parse_args()
    
    # Load configuration
    if args.cfg is None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        air_hockey_cfg_fp = os.path.join(dir_path, '../configs/baseline_configs/random_configs/default_train_puck_vel.yaml')
    else:
        air_hockey_cfg_fp = args.cfg
    
    with open(air_hockey_cfg_fp, 'r') as f:
        air_hockey_cfg = yaml.safe_load(f)
        
    # Validate configuration
    assert 'n_threads' in air_hockey_cfg, "Please specify the number of threads to use for training."
    assert 'algorithm' in air_hockey_cfg, "Please specify the algorithm to use for training."
    
    # Create trainer using factory pattern
    try:
        trainer = TrainerFactory.create_trainer(
            air_hockey_cfg['algorithm'],
            air_hockey_cfg,
            use_wandb=args.wandb,
            device=args.device,
            clear_prior_task_results=args.clear,
            progress_bar=False  # Set to False to match original behavior
        )
    except ValueError as e:
        print(f"Error: {e}")
        supported_algs = TrainerFactory.get_supported_algorithms()
        print(f"Supported algorithms: {', '.join(supported_algs)}")
        return 1
    
    # Execute training
    try:
        trainer.train()
        print("Training completed successfully!")
        return 0
    except Exception as e:
        print(f"Training failed with error: {e}")
        return 1


if __name__ == "__main__":
    main()
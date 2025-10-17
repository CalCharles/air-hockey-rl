from stable_baselines3 import SAC, HerReplayBuffer
from .base_trainer import BaseTrainer


class SACTrainer(BaseTrainer):
    """
    SAC (Soft Actor-Critic) trainer implementation.
    
    This class handles SAC-specific model creation and configuration,
    including support for goal-conditioned tasks with HER replay buffer.
    """
    
    def _create_model(self, env, log_parent_dir, seed):
        """
        Create SAC model with algorithm-specific parameters.
        
        Args:
            env: Training environment
            log_parent_dir (str): Parent directory for logging
            seed (int): Random seed
            
        Returns:
            SAC model instance
        """
        # Handle goal-conditioned tasks with HER (currently commented out in original)
        # Future implementation for HER replay buffer:
        if self.config['air_hockey'].get('return_goal_obs', False):
            # This is the commented-out HER implementation from original code
            # Uncomment and modify as needed for goal-conditioned tasks
            # n_sampled_goal = 4
            # return SAC(
            #     "MultiInputPolicy",
            #     env,
            #     replay_buffer_class=HerReplayBuffer,
            #     replay_buffer_kwargs=dict(
            #         n_sampled_goal=n_sampled_goal,
            #         goal_selection_strategy="future",
            #     ),
            #     learning_starts=10000,
            #     verbose=1,
            #     buffer_size=int(1e6),
            #     learning_rate=1e-3,
            #     gamma=self.config['gamma'],
            #     batch_size=512,
            #     tensorboard_log=log_parent_dir,
            #     seed=seed,
            #     device=self.device,
            # )
            pass
        
        # Standard SAC model (current implementation)
        return SAC(
            self._get_policy_type(),
            env,
            verbose=1,
            buffer_size=int(1e6),
            learning_rate=1e-3,
            gamma=self.config['gamma'],
            tensorboard_log=log_parent_dir,
            seed=seed,
            device=self.device,
        )
        
    def _get_policy_type(self):
        """
        Return appropriate policy type for SAC.
        
        Returns:
            str: Policy type ('MultiInputPolicy' for goal-conditioned, 'MlpPolicy' otherwise)
        """
        if self.config['air_hockey'].get('return_goal_obs', False):
            return "MultiInputPolicy"
        return "MlpPolicy"
        
    def _setup_algorithm_specific_config(self):
        """
        Setup SAC-specific configuration parameters.
        
        Sets the goal observation flag for SAC goal-conditioned tasks.
        """
        # Set goal observation flag for SAC goal-conditioned tasks
        if 'goal' in self.config['air_hockey']['task']:
            self.config['air_hockey']['return_goal_obs'] = True
        else:
            self.config['air_hockey']['return_goal_obs'] = False
            
    def _load_model_for_evaluation(self, model_filepath, env):
        """
        Load SAC model for evaluation.
        
        Args:
            model_filepath (str): Path to the saved model
            env: Environment for evaluation
            
        Returns:
            Loaded SAC model instance
        """
        return SAC.load(model_filepath, env=env)

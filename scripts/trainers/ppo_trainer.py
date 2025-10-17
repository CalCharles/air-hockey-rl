from stable_baselines3 import PPO
from .base_trainer import BaseTrainer


class PPOTrainer(BaseTrainer):
    """
    PPO (Proximal Policy Optimization) trainer implementation.
    
    This class handles PPO-specific model creation and configuration.
    """
    
    def _create_model(self, env, log_parent_dir, seed):
        """
        Create PPO model with algorithm-specific parameters.
        
        Args:
            env: Training environment
            log_parent_dir (str): Parent directory for logging
            seed (int): Random seed
            
        Returns:
            PPO model instance
        """
        return PPO(
            self._get_policy_type(), 
            env, 
            verbose=1, 
            tensorboard_log=log_parent_dir, 
            device=self.device, 
            seed=seed,
            gamma=self.config['gamma']
        )
        
    def _get_policy_type(self):
        """
        Return policy type for PPO.
        
        Returns:
            str: Policy type ('MlpPolicy')
        """
        return "MlpPolicy"
        
    def _setup_algorithm_specific_config(self):
        """
        Setup PPO-specific configuration parameters.
        
        PPO doesn't use goal observations, so we ensure it's disabled.
        """
        # PPO doesn't support goal-conditioned tasks in this implementation
        self.config['air_hockey']['return_goal_obs'] = False
        
    def _load_model_for_evaluation(self, model_filepath, env):
        """
        Load PPO model for evaluation.
        
        Args:
            model_filepath (str): Path to the saved model
            env: Environment for evaluation (not used for PPO loading)
            
        Returns:
            Loaded PPO model instance
        """
        from stable_baselines3 import PPO
        return PPO.load(model_filepath)
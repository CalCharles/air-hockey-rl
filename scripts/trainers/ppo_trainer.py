from stable_baselines3 import PPO
from .base_trainer import BaseTrainer


class PPOTrainer(BaseTrainer):
    def _create_model(self, env, log_parent_dir, seed):
        return PPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            tensorboard_log=log_parent_dir, 
            device=self.device, 
            seed=seed,
            gamma=self.config['gamma']
        )
        
    def _setup_algorithm_specific_config(self):
        # PPO doesn't support goal-conditioned tasks in this implementation
        self.config['air_hockey']['return_goal_obs'] = False
        
    def _load_model_for_evaluation(self, model_filepath, env):
        from stable_baselines3 import PPO
        return PPO.load(model_filepath)
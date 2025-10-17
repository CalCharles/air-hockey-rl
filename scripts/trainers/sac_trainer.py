from stable_baselines3 import SAC, HerReplayBuffer
from .base_trainer import BaseTrainer


class SACTrainer(BaseTrainer):
    """
    SAC (Soft Actor-Critic) trainer implementation.
    """
    
    def _create_model(self, env, log_parent_dir, seed):
        if self.config['air_hockey'].get('return_goal_obs', False):
            n_sampled_goal = 4
            return SAC(
                "MultiInputPolicy",
                env,
                replay_buffer_class=HerReplayBuffer,
                replay_buffer_kwargs=dict(
                    n_sampled_goal=n_sampled_goal,
                    goal_selection_strategy="future",
                ),
                learning_starts=10000,
                verbose=1,
                buffer_size=int(1e6),
                learning_rate=1e-3,
                gamma=self.config['gamma'],
                batch_size=512,
                tensorboard_log=log_parent_dir,
                seed=seed,
                device=self.device,
            )
        
        # Standard SAC model (current implementation)
        return SAC(
            "MultiInputPolicy",
            env,
            verbose=1,
            tensorboard_log=log_parent_dir,
            seed=seed,
            device=self.device,

            train_freq=(25, "step"),
            gradient_steps=5, # 5 gradient updates per 25 time steps
        )
        
    def _setup_algorithm_specific_config(self):
        # Set goal observation flag for SAC goal-conditioned tasks
        if 'goal' in self.config['air_hockey']['task']:
            self.config['air_hockey']['return_goal_obs'] = True
        else:
            self.config['air_hockey']['return_goal_obs'] = False
            
    def _load_model_for_evaluation(self, model_filepath, env):
        return SAC.load(model_filepath, env=env)

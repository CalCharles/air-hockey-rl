import sys
sys.path.append('..')
from utils import EvalCallback
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class CurriculumCallback(EvalCallback):
    def __init__(self, eval_env, curriculum_config=None, eval_freq=5000, n_eval_eps=30, verbose: int = 0):
        super().__init__(eval_env, eval_freq, n_eval_eps, verbose)
        self.curriculum_config = curriculum_config
        self.traj_num = -1
        self.traj_id = [] # np.zeros((self.model.replay_buffer_size, )) - 1
        self.traj_start = []
        self.tot_success_per_traj = []
        self.tot_return_per_traj = []
        self.successes = np.array
        self.classifier = LogisticRegression()
        self.ego_goals = []
        

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        super()._on_rollout_start()
        if self.training_env.envs[0].unwrapped.current_timestep == 0:
            self.traj_num += 1
            self.traj_start.append(self.training_env.envs[0].unwrapped.n_timesteps_so_far)
            
        if self.traj_num > 100:
            # Create a logistic regression classifier
            X_train, X_test, y_train, y_test = train_test_split(np.array(self.ego_goals), np.array(self.successes), test_size=0.2, random_state=42)
            import pdb; pdb.set_trace()
            self.classifier.fit(X_train, y_train)
            # Predict on the test set
            predictions = self.classifier.predict(X_test)

            # Evaluate the classifier
            accuracy = accuracy_score(y_test, predictions)
            print(f"Accuracy: {accuracy}")

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        # self.training_env.envs[0].unwrapped.n_timesteps_so_far
        # print(self.training_env.envs[0].unwrapped.n_timesteps_so_far)
        # print(self.training_env.envs[0].unwrapped.ego_goal_pos)
        
        return True

    def _on_rollout_end(self):
        """
        This event is triggered before updating the policy.
        """
        self.traj_id.append(self.traj_num)
        obs = self.model.replay_buffer.observations
        total_timesteps = self.training_env.envs[0].unwrapped.n_timesteps_so_far
        
        
        if self.training_env.envs[0].unwrapped.current_timestep == 0 and len(self.traj_start) > 1:
            ag = obs['achieved_goal'][self.traj_start[-1]:total_timesteps].squeeze()
            dg = obs['desired_goal'][self.traj_start[-1]:total_timesteps].squeeze()
            rewards = self.training_env.envs[0].unwrapped.compute_reward(ag, dg, {})
            self.tot_success_per_traj.append(np.sum(rewards > 0.0))
            self.successes.concatenate( (self.successes, rewards.unsqueeze(0) > 0.0), axis=0)
            self.tot_return_per_traj.append(np.sum(rewards))
            self.ego_goals.append(self.training_env.envs[0].unwrapped.ego_goal_pos)


        # print(self.tot_success_per_traj)
        # print(rewards.shape)
        # success_weighted_reward = 
        # obs = obs['desired_goal']
        
        
        # print(obs[:60])
        # print(self.traj_id[:60])
        # print(self.training_env.envs[0].unwrapped.ego_goal_pos)
        # print(self.training_env.envs[0].unwrapped.current_timestep)
        # print(np.where(obs['desired_goal'] == 0.0)[0].shape)
        
        
        
        # self.training_env.envs[0].unwrapped.n_timesteps_so_far
        # import pdb; pdb.set_trace()
        # print(obs['desired_goal'][:100])
        

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        import pdb; pdb.set_trace()
        
        
        super()._on_training_end()
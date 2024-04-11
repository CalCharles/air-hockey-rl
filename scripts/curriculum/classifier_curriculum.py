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
        self.successes = np.array(())
        self.classifier = LogisticRegression()
        self.ego_goals = []
        

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        super()._on_rollout_start()
        # import pdb; pdb.set_trace()
        
        
        if self.training_env.envs[0].unwrapped.current_timestep == 0:
            self.traj_num += 1
            self.traj_start.append(self.training_env.envs[0].unwrapped.n_timesteps_so_far)            
            self.on_trajectory_start()
            
    def on_trajectory_start(self) -> None:
        """
        This event is triggered before the start of a new trajectory.
        """
        if self.traj_num > 200:
            # Create a logistic regression classifier
            self.successes = np.array(self.tot_success_per_traj) > 0
            
            N = 200
            N_most_recent_successes = self.successes[-N:]
            N_most_recent_ego_goals = self.ego_goals[-N:]
            X_train, X_test, y_train, y_test = train_test_split(N_most_recent_ego_goals, N_most_recent_successes, test_size=0.2, random_state=42)
            # import pdb; pdb.set_trace()
            self.classifier.fit(X_train, y_train)
            # Predict on the test set
            predictions = self.classifier.predict(X_test)

            # Evaluate the classifier
            accuracy = accuracy_score(y_test, predictions)
            print(f"Accuracy: {accuracy}")
            self._goal_selector()
            
    def _goal_selector(self):
        """
        Selects the goal based on the current goal success rate
        """

        test_goal_set = np.array([[-0.69, 0.0]])
        
        for env in self.training_env.envs:
            env.set_goals('home', goal_set = test_goal_set)


    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
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
            rewards = np.random.random(size=(3,)) - 0.5 # self.training_env.envs[0].get_wrapper_attr('compute_reward')(ag, dg, {})
            self.tot_success_per_traj.append(np.sum(rewards > 0.0))
            self.tot_return_per_traj.append(np.sum(rewards))
            self.ego_goals.append(self.training_env.envs[0].unwrapped.ego_goal_pos)
        

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        # import pdb; pdb.set_trace()
        
        
        super()._on_training_end()
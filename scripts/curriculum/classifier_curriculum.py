import sys
sys.path.append('..')
from utils import EvalCallback
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scripts.curriculum.mlp import MLP, init_weights
import torch
import matplotlib.pyplot as plt
class CurriculumCallback(EvalCallback):
    def __init__(self, eval_env, curriculum_config=None, log_dir=None, eval_freq=5000, n_eval_eps=30, verbose: int = 0):
        super().__init__(eval_env=eval_env, 
                         log_dir=log_dir, 
                         eval_freq=eval_freq, 
                         n_eval_eps=n_eval_eps, 
                         verbose=verbose)
        self.curriculum_config = curriculum_config
        self.traj_num = -1
        self.traj_id = [] # np.zeros((self.model.replay_buffer_size, )) - 1
        self.traj_start = []
        self.tot_success_per_traj = []
        self.tot_return_per_traj = []
        self.successes = [] #np.array(())
        self.ego_goals = []
        self.cur_buffer_ego_goals = []
        self.cur_buffer_successes = []
        self.start_goal_gen = 0 # number of trajectories after which we start generating new goals
        self.eval_env = eval_env
        self.update_goal_gen_freq = 10
        self.classifier_type = 'mlp'
        if self.classifier_type == 'logistic_regression':
            self.classifier = LogisticRegression()
        elif self.classifier_type == 'mlp':
            self.classifier = MLP(input_size=2, hidden_sizes=[64, 64], output_size=1)
            self.opimizer = torch.optim.Adam(self.classifier.parameters(), lr=1e-3)
            self.classifier_optimizer_steps = 1000
            self.loss_fn = torch.nn.BCELoss()
        self.use_balanced_dataset = False
        
        
    def _preprocesses_ppo_buffer(self):
        """
        Preprocesses the PPO buffer to include the achieved goal
        """
        obs = self.model.rollout_buffer.observations.reshape(-1, 10, order='F')
        rewards = self.model.rollout_buffer.rewards.flatten(order='F')
        episode_starts = self.model.rollout_buffer.episode_starts.flatten(order='F')
        episode_starts_idx = np.where(episode_starts == 1)[0]
        
        # obs = self.model.rollout_buffer.observations[:, 0, :]
        # rewards = self.model.rollout_buffer.rewards[:, 0]
        # episode_starts = self.model.rollout_buffer.episode_starts[:, 0]
        # episode_starts_idx = np.where(episode_starts == 1)[0]
        
        plt.clf()
        xy = obs[:, :2]

        
        self.cur_buffer_ego_goals = []
        self.cur_buffer_successes = []
        for i in range(episode_starts_idx.shape[0]-1):
            cur_episode_rewards = rewards[episode_starts_idx[i]:episode_starts_idx[i+1]]
            cur_goal = obs[episode_starts_idx[i], -2:]
            success = np.any(cur_episode_rewards > 0.0)
            goal = cur_goal #obs[episode_starts_idx[i], -2:]
            self.successes.append(success)
            self.ego_goals.append(goal)
            self.cur_buffer_successes.append(success)
            self.cur_buffer_ego_goals.append(goal)
            cur_traj = obs[episode_starts_idx[i]:episode_starts_idx[i+1], 4:6]
            # if i < 25:
            #     if success:
            #         # plt.plot(cur_traj[:,0], cur_traj[:,1], '-g')
                    
            #         plt.plot(goal[0], goal[1], 'bo', markersize=5)
            #         indixies = np.arange(cur_traj.shape[0])
            #         plt.scatter(cur_traj[:,0], cur_traj[:,1], c=indixies, )
            #         plt.savefig('trajectories.png')
            #         import pdb; pdb.set_trace()
                    
                    
            #     else:
            #         continue
            #         plt.plot(cur_traj[:,0], cur_traj[:,1], '-r')
                    
                
        
    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        super()._on_rollout_start()
    
    def _train_goal_predictor(self):
        # Create a logistic regression classifier
        N = 200
        N_most_recent_successes = np.array(self.successes[-N:])
        N_most_recent_ego_goals = np.array(self.ego_goals[-N:])
        sr = np.sum(N_most_recent_successes) / N
        min_success_type = min(sr * N, (1 - sr) * N)
        succ_idx = np.where(N_most_recent_successes == 1)[0]
        fail_idx = np.where(N_most_recent_successes == 0)[0]
        num_succ = succ_idx.shape[0]
        num_fail = fail_idx.shape[0]
        
        if num_succ > num_fail:
            succ_idx_choices = np.arange(0, succ_idx.shape[0])
            balanced_succ_idx = np.random.choice(succ_idx_choices, num_fail, replace=False)
            balanced_succ_goals = N_most_recent_ego_goals[succ_idx[balanced_succ_idx]]
            balanced_features = np.concatenate((balanced_succ_goals, N_most_recent_ego_goals[fail_idx]), axis=0)
            balanced_labels = np.concatenate((np.ones(num_fail), np.zeros(num_fail)))
        else:
            fail_idx_choices = np.arange(0, fail_idx.shape[0])
            balanced_fail_idx = np.random.choice(fail_idx_choices, num_succ, replace=False)
            balanced_fail_goals = N_most_recent_ego_goals[fail_idx[balanced_fail_idx]]
            balanced_features = np.concatenate((balanced_fail_goals, N_most_recent_ego_goals[succ_idx]), axis=0)
            balanced_labels = np.concatenate((np.zeros(num_succ), np.ones(num_succ)))
        
        if self.classifier_type == 'logistic_regression':
            

                
            X_train, X_test, y_train, y_test = train_test_split(balanced_features, balanced_labels.astype(np.uint8), test_size=0.2, random_state=42)
            self.classifier.fit(X_train, y_train)
            # Predict on the test set
            predictions = self.classifier.predict_proba(X_test)
            # Evaluate the classifier
            accuracy = accuracy_score(y_test, predictions.argmax(axis=1))
            self.classifier_acc = accuracy
            print(f"Accuracy: {accuracy}")
            
        elif self.classifier_type == 'mlp':
            if self.use_balanced_dataset:
                ego_goals = balanced_features
                successes = balanced_labels
            else:
                ego_goals = np.array(self.cur_buffer_ego_goals)
                successes = np.array(self.cur_buffer_successes)
            X_train, X_test, y_train, y_test = train_test_split(ego_goals, successes, test_size=0.2, random_state=42)
            
            input = torch.tensor(X_train, dtype=torch.float32)
            target = torch.tensor(y_train, dtype=torch.float32)

            self.classifier.apply(init_weights)
            for i in range(self.classifier_optimizer_steps):
                self.opimizer.zero_grad()
                output = self.classifier(input)
                loss = self.loss_fn(output.squeeze(), target)
                loss.backward()
                self.opimizer.step()
            y_pred = torch.round(self.classifier(torch.tensor(X_test))).detach().numpy()
            accuracy = accuracy_score(y_test, y_pred)
            self.classifier_acc = accuracy


            min_y = self.eval_env.table_y_left + self.eval_env.ego_goal_radius
            max_y = self.eval_env.table_y_right - self.eval_env.ego_goal_radius
            max_x = 0 - self.eval_env.ego_goal_radius
            min_x = self.eval_env.table_x_top + self.eval_env.ego_goal_radius
            x_range = np.abs(max_x - min_x)
            y_range = np.abs(max_y - min_y)
            scale = 150
            x = torch.linspace(min_x,max_x, int(x_range * scale))  # include x_right
            y = torch.linspace(min_y, max_y,  int(y_range * scale))  # include y_top

            # Create a meshgrid
            xx, yy = torch.meshgrid(x, y, indexing='ij')

            # Flatten the grid to list of points
            points = torch.stack([xx.flatten(), yy.flatten()], dim=1)            
            self.goal_predictions = self.classifier(points)[:,0].detach().numpy().reshape(len(x), len(y))
            plt.clf()
            plt.imshow(self.goal_predictions, extent=(min_x, max_x, min_y, max_y), origin='lower')
            # plt.plot(N_most_recent_ego_goals[N_most_recent_successes == 1][:,0], N_most_recent_ego_goals[N_most_recent_successes == 1][:,1], 'ro', label='success')
            # plt.plot(N_most_recent_ego_goals[N_most_recent_successes == 0][:,0], N_most_recent_ego_goals[N_most_recent_successes == 0][:,1], 'bo', label='fail')    
            plt.plot(ego_goals[successes == 1][:,0], ego_goals[successes == 1][:,1], 'ro', label='success')
            plt.plot(ego_goals[successes == 0][:,0], ego_goals[successes == 0][:,1], 'bo', label='fail')            
            plt.legend(title='legend')
            plt.savefig('goal_predictions.png')

            # plt.imsave('goal_predictions.png', self.goal_predictions, origin='lower')
            
            
    def _goal_selector(self):
        """
        Selects the goal based on the current goal success rate
        """
        
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
        self._preprocesses_ppo_buffer()
        self._train_goal_predictor()

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        super()._on_training_end()
        
    @property
    def _current_timesteps(self):
        return self.training_env.get_attr('current_timestep')
    
    @property
    def _num_time_steps(self):
        return self.training_env.get_attr('n_timesteps_so_far')
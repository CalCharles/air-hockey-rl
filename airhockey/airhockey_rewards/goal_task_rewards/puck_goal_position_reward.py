import numpy as np
from airhockey.airhockey_rewards import AirHockeyRewardBase

class AirHockeyPuckGoalPositionReward(AirHockeyRewardBase):
    def __init__(self, task_env):
        super().__init__(task_env)

    def compute_reward(self, achieved_goal, desired_goal, info):
        # if not vectorized, convert to vector
        single = len(achieved_goal.shape) == 1
        if single:
            achieved_goal = achieved_goal.reshape(1, -1)
            desired_goal = desired_goal.reshape(1, -1)
        # return euclidean distance between the two points
        dist = np.linalg.norm(achieved_goal[:, :2] - desired_goal[:, :2], axis=1)
        sigmoid_scale = 2
        radius = self.task_env.goal_radius
        # reward_raw = 1 - (dist / radius)
        # reward_mask = dist >= radius
        # reward_raw[reward_mask] = 0
        # reward = 1 / (1 + np.exp(-reward_raw * sigmoid_scale))
        # reward[reward_mask] = 0


        bonus = 10.0 if self.task_env.current_timestep > self.task_env.falling_time else 0.0 # this prevents the falling initiliazwed puck from triggering a success
        reward = np.array([-dist if dist > radius else bonus])
        # print(dist, radius, reward)
        if single:
            # print(reward)
            
            reward = reward.squeeze()
        # print(reward)
        return reward

    def get_base_reward(self, state_info):
        reward = self.compute_reward(self.task_env.get_achieved_goal(state_info), self.task_env.get_desired_goal(), {})
        success = reward > 0.0
        success = success.item()
        return reward, success

import copy
import numpy as np
from gymnasium.spaces import Box
from airhockey.airhockey_tasks.utils import RewardRegion
from airhockey.airhockey_rewards import AirHockeyRewardBase

class AirHockeyPaddleReachPositionNegRegionsReward(AirHockeyRewardBase):
    def __init__(self, task_env):
        super().__init__(task_env)

    def compute_reward(self, achieved_goal, desired_goal, info):
        single = len(achieved_goal.shape) == 1
        if single:
            achieved_goal = achieved_goal.reshape(1, -1)
            desired_goal = desired_goal.reshape(1, -1)
        dist = np.linalg.norm(achieved_goal[:, :2] - desired_goal[:, :2], axis=1)
        max_euclidean_distance = np.linalg.norm(np.array([self.task_env.table_x_bot, self.task_env.table_y_right]) - np.array([self.task_env.table_x_top, self.task_env.table_y_left]))
        reward = - (dist / max_euclidean_distance)

        for nrr in self.task_env.reward_regions:
            reward += nrr.check_reward(achieved_goal)

        if single:
            reward = reward[0]
        return reward
    
    def get_base_reward(self, state_info):
        reward = self.compute_reward(self.task_env.get_achieved_goal(state_info), self.task_env.get_desired_goal(), {})
        success = reward > 0.9
        success = success.item()
        return reward, success

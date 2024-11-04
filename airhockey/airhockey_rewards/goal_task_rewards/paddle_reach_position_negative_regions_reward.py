import copy
import numpy as np
from gymnasium.spaces import Box
from airhockey.airhockey_tasks.utils import RewardRegion
from airhockey.airhockey_rewards import AirHockeyRewardBase

class AirHockeyPaddleReachPositionNegRegionsReward(AirHockeyRewardBase):
    def __init__(self, task_env):
        super().__init__(task_env)

    def compute_reward(self, achieved_goal, desired_goal, info):
        radius = self.task_env.goal_radius
        single = len(achieved_goal.shape) == 1
        if single:
            achieved_goal = achieved_goal.reshape(1, -1)
            desired_goal = desired_goal.reshape(1, -1)
        dist = np.linalg.norm(achieved_goal[:, :2] - desired_goal[:, :2], axis=1)
        max_euclidean_distance = np.linalg.norm(np.array([self.task_env.table_x_bot, self.task_env.table_y_right]) - np.array([self.task_env.table_x_top, self.task_env.table_y_left]))
        bonus = 10 if self.task_env.current_timestep > self.task_env.falling_time else 0 # this prevents the falling initiliazed puck from triggering a success
        reward = - (dist / max_euclidean_distance) if dist > radius else bonus

        for nrr in self.task_env.reward_regions:
            reward += nrr.check_reward(achieved_goal)

        if single:
            reward = reward[0]
        return reward
    
    def get_base_reward(self, state_info):
        ag = self.task_env.get_achieved_goal(state_info)
        dg = self.task_env.get_desired_goal()
        reward = self.compute_reward(ag, dg, {})
        dist = np.linalg.norm(ag - dg, axis=0)

        success = dist < self.task_env.goal_radius
        success = success.item()
        self.success = success
        return reward, success

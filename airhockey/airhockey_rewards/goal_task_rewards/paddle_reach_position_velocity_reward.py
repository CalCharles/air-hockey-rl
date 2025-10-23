import numpy as np
from airhockey.airhockey_rewards import AirHockeyRewardBase

class AirHockeyPaddleReachPositionVelocityReward(AirHockeyRewardBase):
    def __init__(self, task_env):
        super().__init__(task_env)

    def compute_reward(self, achieved_goal, desired_goal, info):
        single = len(achieved_goal.shape) == 1
        if single:
            achieved_goal = achieved_goal.reshape(1, -1)
            desired_goal = desired_goal.reshape(1, -1)
        dist = np.linalg.norm(achieved_goal[:, :2] - desired_goal[:, :2], axis=1)
        max_euclidean_distance = np.linalg.norm(np.array([self.task_env.table_x_bot, self.task_env.table_y_right]) - np.array([self.task_env.table_x_top, self.task_env.table_y_left]))
        pos_reward = 1 - (dist / max_euclidean_distance)
        
        current_vel = achieved_goal[:, 2:]
        goal_vel = desired_goal[:, 2:]
        mag_current = np.linalg.norm(current_vel, axis=1)
        mag_goal = np.linalg.norm(goal_vel, axis=1)
        mag_diff = np.abs(mag_current - mag_goal)
        maximum_mag_diff = np.abs(np.linalg.norm(np.array([self.task_env.max_paddle_vel, self.task_env.max_paddle_vel]) - np.array([0, 0])))
        vel_mag_reward = 1 - mag_diff / maximum_mag_diff

        dist = np.linalg.norm(current_vel - goal_vel)
        denom = mag_current * mag_goal + 1e-8
        vel_cos = np.sum(current_vel * goal_vel, axis=1) / denom
        vel_cos = np.clip(vel_cos, -1, 1)
        norm_cos_sim = (vel_cos + 1) / 2
        vel_angle_reward = norm_cos_sim
        vel_reward = (vel_angle_reward + vel_mag_reward) / 2
        
        self.task_env.latest_pos_reward = pos_reward
        self.task_env.latest_vel_reward = vel_reward
        reward = 0.5 * pos_reward + 0.5 * vel_reward

        if single:
            reward = reward[0]
        return reward

    def get_base_reward(self, state_info):
        reward = self.compute_reward(self.task_env.get_achieved_goal(state_info), self.task_env.get_desired_goal(), {})
        success = self.task_env.latest_pos_reward >= 0.9 and self.task_env.latest_vel_reward >= 0.8
        success = success.item()
        return reward, success

import numpy as np
from airhockey.airhockey_rewards import AirHockeyRewardBase

class AirHockeyPuckGoalPositionVelocityReward(AirHockeyRewardBase):
    def __init__(self, task_env):
        super().__init__(task_env)

    def compute_reward(self, achieved_goal, desired_goal, info):
        single = len(achieved_goal.shape) == 1
        if single:
            achieved_goal = achieved_goal.reshape(1, -1)
            desired_goal = desired_goal.reshape(1, -1)
        dist = np.linalg.norm(achieved_goal[:, :2] - desired_goal[:, :2], axis=1)
        denom = np.linalg.norm(achieved_goal[:, 2:], axis=1) * np.linalg.norm(desired_goal[:, 2:], axis=1) + 1e-8
        vel_cos = np.sum(achieved_goal[:, 2:] * desired_goal[:, 2:], axis=1) / denom
        vel_cos = np.clip(vel_cos, -1, 1)
        vel_angle = np.arccos(vel_cos)
        mag_achieved = np.linalg.norm(achieved_goal[:, 2:], axis=1)
        mag_desired = np.linalg.norm(desired_goal[:, 2:], axis=1)
        mag_diff = np.abs(mag_achieved - mag_desired)
        sigmoid_scale = 2
        radius = self.task_env.goal_radius
        reward_raw = 1 - (dist / radius)
        mask = dist >= radius
        reward_raw[mask] = 0
        reward = 1 / (1 + np.exp(-reward_raw * sigmoid_scale))
        reward_mask = dist >= radius
        reward[reward_mask] = 0
        position_reward = reward

        vel_mag_reward = 1 - mag_diff / self.task_env.max_paddle_vel

        reward_mask = position_reward == 0
        norm_cos_sim = (vel_cos + 1) / 2
        vel_angle_reward = norm_cos_sim
        vel_angle_reward[reward_mask] = 0
        vel_mag_reward[reward_mask] = 0
        vel_reward = (vel_angle_reward + vel_mag_reward) / 2

        reward = 0.5 * position_reward + 0.5 * vel_reward

        if single:
            reward = reward[0]
        return reward

    def get_base_reward(self, state_info):
        reward = self.compute_reward(self.task_env.get_achieved_goal(state_info), self.task_env.get_desired_goal(), {})
        success = reward > 0.0
        success = success.item()
        return reward, success

import numpy as np
from airhockey.airhockey_rewards import AirHockeyRewardBase

class AirHockeyPuckGoalPositionReward(AirHockeyRewardBase):
    def __init__(self, task_env, puck_touch_reward=None):
        super().__init__(task_env)
        self.puck_touch_reward = puck_touch_reward

    def compute_reward(self, achieved_goal, desired_goal):
        # if not vectorized, convert to vector
        single = len(achieved_goal.shape) == 1
        if single:
            achieved_goal = achieved_goal.reshape(1, -1)
            desired_goal = desired_goal.reshape(1, -1)
            
        # return euclidean distance between the two points
        dist = np.linalg.norm(achieved_goal[:, :2] - desired_goal[:, :2], axis=1)

        radius = self.task_env.goal_radius
        # bonus = 10 if self.task_env.current_timestep > self.task_env.falling_time else 0 # this prevents the falling initiliazwed puck from triggering a success
        reward = -dist if dist > radius else self.task_env.puck_goal_success_bonus
        
        if single and isinstance(reward, list):
            reward = reward[0]
            
        return reward

    def get_base_reward(self, state_info):
        ag = self.task_env.get_achieved_goal(state_info)
        dg = self.task_env.get_desired_goal()
        reward = self.compute_reward(self.task_env.get_achieved_goal(state_info), self.task_env.get_desired_goal())
        puck_touch_reward = self.puck_touch_reward.compute_reward(np.array(state_info['paddles']['paddle_ego']['position']),
                                                                  np.array(state_info['pucks'][0]['position']))
        
        reward = max(reward, puck_touch_reward)
        
        
        dist = np.linalg.norm(ag - dg, axis=0)
        success = dist < self.task_env.goal_radius
        success = success.item()
        return reward, success

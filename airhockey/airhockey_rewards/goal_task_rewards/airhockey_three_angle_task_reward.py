from ..airhockey_reward_base import AirHockeyRewardBase
import numpy as np

class AirHockeyPuckThreeAngleReward(AirHockeyRewardBase):
    def __init__(self, task_env):
        super().__init__(task_env)

    def compute_reward(self, achieved_goal, desired_goal, info):
        dg = np.dot(self.task_env.angles, desired_goal)
        if np.linalg.norm(np.array(info['puck_vel'])) > 0.1 and np.linalg.norm(np.array(info['puck_pos']) - np.array(info['paddle_pos'])) < 0.1:
            ag_vector = np.array([np.cos(achieved_goal), np.sin(achieved_goal)])
            dg_vector = np.array([np.cos(dg), np.sin(dg)])
            cosine_similarity = np.dot(ag_vector.squeeze(axis=1), dg_vector)

            reward = cosine_similarity
        
        else:
            reward = 0

        return reward


    def get_base_reward(self, state_info):
        ag = self.task_env.get_achieved_goal(state_info)
        dg = self.task_env.get_desired_goal()
        reward = self.compute_reward(ag, dg, {"paddle_pos": state_info['paddles']['paddle_ego']['position'], "puck_pos": state_info['pucks'][0]['position'], "puck_vel": state_info['pucks'][0]['velocity']})
        dist = np.linalg.norm(ag - dg, axis=0)

        success = dist < np.deg2rad(self.task_env.success_threshold_deg)
        self.success = success
        return reward, success
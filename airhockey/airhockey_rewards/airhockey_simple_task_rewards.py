from .airhockey_reward_base import AirHockeyRewardBase
import numpy as np

class AirHockeyPuckVelReward(AirHockeyRewardBase):
    def __init__(self, task_env):
        super().__init__(task_env)

    def get_base_reward(self, state_info):
        puck_pos = state_info['pucks'][0]['position']
        paddle_pos = state_info['paddles']['paddle_ego']['position']
        min_dist = self.task_env.paddle_radius + self.task_env.puck_radius
        dist = np.linalg.norm(np.array(puck_pos) - np.array(paddle_pos))

        # reward for positive velocity towards the top of the board
        puck_vel = -state_info['pucks'][0]['velocity'][0]
        puck_height = -puck_pos[0]

        reward = max(puck_vel * 5, 0) + 0.5 / dist
        success = puck_height > 0.5 and self.task_env.current_timestep > 25
        return reward, success


class AirHockeyPuckHeightReward(AirHockeyRewardBase):
    def __init__(self, task_env):
        super().__init__(task_env)
        
    def get_base_reward(self, state_info):
        puck_height = -state_info['pucks'][0]['position'][0]
        puck_vel = -state_info['pucks'][0]['velocity'][0]
        puck_pos = state_info['pucks'][0]['position']

        paddle_pos = state_info['paddles']['paddle_ego']['position']
        min_dist = self.task_env.paddle_radius + self.task_env.puck_radius
        dist = np.linalg.norm(np.array(puck_pos) - np.array(paddle_pos))

        reward = max(puck_vel, 0) * 5 if puck_height < 0 else max(puck_vel, 0) * -10
        success = puck_height > 0 and self.task_env.current_timestep > 25

        if dist - min_dist < 0.05:
            if not self.task_env.touching:
                reward += 20
                self.task_env.num_touches += 1
            self.task_env.touching = True
        else:
            self.task_env.touching = False

        if success:
            reward = 60
        return reward, success


class AirHockeyPuckCatchReward(AirHockeyRewardBase):
    def __init__(self, task_env):
        super().__init__(task_env)

    def get_base_reward(self, state_info):
        # reward for getting close to the puck, but make sure not to displace it
        puck_pos = state_info['pucks'][0]['position']
        paddle_pos = state_info['paddles']['paddle_ego']['position']
        dist = np.linalg.norm(np.array(puck_pos) - np.array(paddle_pos))
        max_dist = 0.16 * self.task_env.width
        min_dist = self.task_env.paddle_radius + self.task_env.puck_radius
        reward = 1 - ((dist - min_dist) / (max_dist - min_dist))
        reward = max(reward, 0)
        success = reward >= 0.9 and self.task_env.current_timestep > 75
        return reward, success


class AirHockeyPuckJuggleReward(AirHockeyRewardBase):
    def __init__(self, task_env):
        super().__init__(task_env)

    def get_base_reward(self, state_info):
        reward = 0
        x_pos = state_info['pucks'][0]['position'][0]
        x_higher = self.task_env.table_x_top
        x_lower = self.task_env.table_x_bot
        if x_higher / 4 < x_pos < 0:
            reward += 15
        elif x_pos < x_higher / 4:
            reward -= 1
        success = reward > 0 and self.task_env.current_timestep > 50
        return reward, success


class AirHockeyPuckStrikeReward(AirHockeyRewardBase):
    def __init__(self, task_env):
        super().__init__(task_env)
        
    def get_base_reward(self, state_info):
        x_vel = state_info['pucks'][0]['velocity'][0]
        y_vel = state_info['pucks'][0]['velocity'][1]
        vel_mag = np.linalg.norm(np.array([x_vel, y_vel]))
        reward = vel_mag
        max_rew = 2  # estimated max vel
        min_rew = 0  # min acceptable good velocity

        initial_pos = self.task_env.puck_initial_position
        current_pos = state_info['pucks'][0]['position']
        dist = np.linalg.norm(np.array(initial_pos) - np.array(current_pos))
        has_moved = dist > 0.01

        if reward <= min_rew and not has_moved:
            return -5, False  # negative rew for standing still and hasn't moved
        reward = min(reward, max_rew)
        reward = (reward - min_rew) / (max_rew - min_rew)
        success = reward > (0.1)  # means the puck is moving
        if reward > 0:
            reward *= 10
        return reward, success


class AirHockeyPuckTouchReward(AirHockeyRewardBase):
    def __init__(self, task_env):
        super().__init__(task_env)

    def get_base_reward(self, state_info):
        # reward for getting close to the puck, but make sure not to displace it
        puck_pos = state_info['pucks'][0]['position']
        paddle_pos = state_info['paddles']['paddle_ego']['position']
        min_dist = self.task_env.paddle_radius + self.task_env.puck_radius
        dist = np.linalg.norm(np.array(puck_pos) - np.array(paddle_pos))
        max_dist = 0.16 * self.task_env.width
        reward = 1 - ((dist - min_dist) / (max_dist - min_dist))
        reward = max(reward, 0)
        # let's also make sure puck does not deviate from initial position
        puck_initial_position = self.task_env.puck_initial_position
        puck_current_position = state_info['pucks'][0]['position']
        delta = np.linalg.norm(np.array(puck_initial_position) - np.array(puck_current_position))
        epsilon = 0.01 + min_dist

        success = dist < (self.task_env.paddle_radius + self.task_env.puck_radius + 0.02)

        if reward > 0:
            reward *= 20  # make it more significant

        return reward, success

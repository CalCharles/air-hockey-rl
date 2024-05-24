import numpy as np
from gymnasium.spaces import Box
from .airhockey_base import AirHockeyBaseEnv


class AirHockeyPuckVelEnv(AirHockeyBaseEnv):
    def initialize_spaces(self):
        # setup observation / action / reward spaces
        paddle_obs_low = [self.table_x_top, self.table_y_left, -self.max_paddle_vel, -self.max_paddle_vel]
        paddle_obs_high = [self.table_x_bot, self.table_y_right, self.max_paddle_vel, self.max_paddle_vel]
        
        puck_obs_low = [self.table_x_top, self.table_y_left, -self.max_puck_vel, -self.max_puck_vel]
        puck_obs_high = [self.table_x_bot, self.table_y_right, self.max_puck_vel, self.max_puck_vel]

        low = paddle_obs_low + puck_obs_low
        high = paddle_obs_high + puck_obs_high

        self.observation_space = self.get_obs_space(low, high)
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32) # 2D action space
        self.reward_range = Box(low=-1, high=1) # need to make sure rewards are between 0 and 1
        
    @staticmethod
    def from_dict(state_dict):
        return AirHockeyPuckVelEnv(**state_dict)

    def create_world_objects(self):
        name = 'puck_{}'.format(0)
        pos, vel = self.get_puck_configuration()
        self.simulator.spawn_puck(pos, vel, name)
        
        name = 'paddle_ego'
        pos, vel = self.get_paddle_configuration(name)
        self.simulator.spawn_paddle(pos, vel, name)
    
    def validate_configuration(self):
        assert self.num_pucks == 1
        assert self.num_blocks == 0
        assert self.num_obstacles == 0
        assert self.num_targets == 0
        assert self.num_paddles == 1

    def get_observation(self, state_info):
        ego_paddle_x_pos = state_info['paddles']['paddle_ego']['position'][0]
        ego_paddle_y_pos = state_info['paddles']['paddle_ego']['position'][1]
        ego_paddle_x_vel = state_info['paddles']['paddle_ego']['velocity'][0]
        ego_paddle_y_vel = state_info['paddles']['paddle_ego']['velocity'][1]
        
        puck_x_pos = state_info['pucks'][0]['position'][0]
        puck_y_pos = state_info['pucks'][0]['position'][1]
        puck_x_vel = state_info['pucks'][0]['velocity'][0]
        puck_y_vel = state_info['pucks'][0]['velocity'][1]       

        obs = np.array([ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel, puck_x_pos, puck_y_pos, puck_x_vel, puck_y_vel])
        return obs

    def get_base_reward(self, state_info):
        # reward for positive velocity towards the top of the board
        reward = -state_info['pucks'][0]['velocity'][0]
        max_rew = 2 # estimated max vel
        min_rew = 0  # min acceptable good velocity
        if reward < min_rew:
            return 0, False
        reward = min(reward, max_rew)
        reward = (reward - min_rew) / (max_rew - min_rew)
        success = reward > 0.5 and self.current_timestep > 25
        return reward, success


class AirHockeyPuckHeightEnv(AirHockeyBaseEnv):
    def initialize_spaces(self):
        # setup observation / action / reward spaces
        paddle_obs_low = [self.table_x_top, self.table_y_left, -self.max_paddle_vel, -self.max_paddle_vel]
        paddle_obs_high = [self.table_x_bot, self.table_y_right, self.max_paddle_vel, self.max_paddle_vel]
        
        puck_obs_low = [self.table_x_top, self.table_y_left, -self.max_puck_vel, -self.max_puck_vel]
        puck_obs_high = [self.table_x_bot, self.table_y_right, self.max_puck_vel, self.max_puck_vel]

        low = paddle_obs_low + puck_obs_low
        high = paddle_obs_high + puck_obs_high

        self.observation_space = self.get_obs_space(low, high)
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32) # 2D action space
        self.reward_range = Box(low=-1, high=1) # need to make sure rewards are between 0 and 1
        
    @staticmethod
    def from_dict(state_dict):
        return AirHockeyPuckHeightEnv(**state_dict)

    def create_world_objects(self):
        name = 'puck_{}'.format(0)
        pos, vel = self.get_puck_configuration()
        self.simulator.spawn_puck(pos, vel, name)
        
        name = 'paddle_ego'
        pos, vel = self.get_paddle_configuration(name)
        self.simulator.spawn_paddle(pos, vel, name)
    
    def validate_configuration(self):
        assert self.num_pucks == 1
        assert self.num_blocks == 0
        assert self.num_obstacles == 0
        assert self.num_targets == 0
        assert self.num_paddles == 1
    
    def get_observation(self, state_info):
        ego_paddle_x_pos = state_info['paddles']['paddle_ego']['position'][0]
        ego_paddle_y_pos = state_info['paddles']['paddle_ego']['position'][1]
        ego_paddle_x_vel = state_info['paddles']['paddle_ego']['velocity'][0]
        ego_paddle_y_vel = state_info['paddles']['paddle_ego']['velocity'][1]
        
        puck_x_pos = state_info['pucks'][0]['position'][0]
        puck_y_pos = state_info['pucks'][0]['position'][1]
        puck_x_vel = state_info['pucks'][0]['velocity'][0]
        puck_y_vel = state_info['pucks'][0]['velocity'][1]       

        obs = np.array([ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel, puck_x_pos, puck_y_pos, puck_x_vel, puck_y_vel])
        return obs

    def get_base_reward(self, state_info):
        reward = -state_info['pucks'][0]['position'][0]
        # min acceptable reward is 0 height and above
        reward = max(reward, 0)
        # let's normalize reward w.r.t. the top half length of the table
        # aka within the range [0, self.length / 2]
        max_rew = self.length / 2
        min_rew = 0
        reward = (reward - min_rew) / (max_rew - min_rew)
        success = reward > 0.5 and self.current_timestep > 25
        return reward, success


class AirHockeyPuckCatchEnv(AirHockeyBaseEnv):
    def initialize_spaces(self):
        # setup observation / action / reward spaces
        paddle_obs_low = [self.table_x_top, self.table_y_left, -self.max_paddle_vel, -self.max_paddle_vel]
        paddle_obs_high = [self.table_x_bot, self.table_y_right, self.max_paddle_vel, self.max_paddle_vel]
        
        puck_obs_low = [self.table_x_top, self.table_y_left, -self.max_puck_vel, -self.max_puck_vel]
        puck_obs_high = [self.table_x_bot, self.table_y_right, self.max_puck_vel, self.max_puck_vel]

        low = paddle_obs_low + puck_obs_low
        high = paddle_obs_high + puck_obs_high

        self.observation_space = self.get_obs_space(low, high)
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32) # 2D action space
        self.reward_range = Box(low=-1, high=1) # need to make sure rewards are between 0 and 1
        
    @staticmethod
    def from_dict(state_dict):
        return AirHockeyPuckCatchEnv(**state_dict)

    def create_world_objects(self):
        name = 'puck_{}'.format(0)
        pos, vel = self.get_puck_configuration()
        self.simulator.spawn_puck(pos, vel, name)
        
        name = 'paddle_ego'
        pos, vel = self.get_paddle_configuration(name)
        self.simulator.spawn_paddle(pos, vel, name)
    
    def validate_configuration(self):
        assert self.num_pucks == 1
        assert self.num_blocks == 0
        assert self.num_obstacles == 0
        assert self.num_targets == 0
        assert self.num_paddles == 1
    
    def get_observation(self, state_info):
        ego_paddle_x_pos = state_info['paddles']['paddle_ego']['position'][0]
        ego_paddle_y_pos = state_info['paddles']['paddle_ego']['position'][1]
        ego_paddle_x_vel = state_info['paddles']['paddle_ego']['velocity'][0]
        ego_paddle_y_vel = state_info['paddles']['paddle_ego']['velocity'][1]
        
        puck_x_pos = state_info['pucks'][0]['position'][0]
        puck_y_pos = state_info['pucks'][0]['position'][1]
        puck_x_vel = state_info['pucks'][0]['velocity'][0]
        puck_y_vel = state_info['pucks'][0]['velocity'][1]       

        obs = np.array([ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel, puck_x_pos, puck_y_pos, puck_x_vel, puck_y_vel])
        return obs

    def get_base_reward(self, state_info):
        # reward for getting close to the puck, but make sure not to displace it
        puck_pos = state_info['pucks'][0]['position']
        paddle_pos = state_info['paddles']['paddle_ego']['position']
        dist = np.linalg.norm(np.array(puck_pos) - np.array(paddle_pos))
        max_dist = 0.16 * self.width
        min_dist = self.paddle_radius + self.puck_radius
        reward = 1 - ((dist - min_dist) / (max_dist - min_dist))
        reward = max(reward, 0)
        success = reward >= 0.9 and self.current_timestep > 75
        return reward, success
    
    
class AirHockeyPuckJuggleEnv(AirHockeyBaseEnv):
    def initialize_spaces(self):
        # setup observation / action / reward spaces
        paddle_obs_low = [self.table_x_top, self.table_y_left, -self.max_paddle_vel, -self.max_paddle_vel]
        paddle_obs_high = [self.table_x_bot, self.table_y_right, self.max_paddle_vel, self.max_paddle_vel]
        
        puck_obs_low = [self.table_x_top, self.table_y_left, -self.max_puck_vel, -self.max_puck_vel]
        puck_obs_high = [self.table_x_bot, self.table_y_right, self.max_puck_vel, self.max_puck_vel]

        low = paddle_obs_low + puck_obs_low
        high = paddle_obs_high + puck_obs_high

        self.observation_space = self.get_obs_space(low, high)
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32) # 2D action space
        self.reward_range = Box(low=-1, high=1) # need to make sure rewards are between 0 and 1
        
    @staticmethod
    def from_dict(state_dict):
        return AirHockeyPuckJuggleEnv(**state_dict)

    def create_world_objects(self):
        name = 'puck_{}'.format(0)
        pos, vel = self.get_puck_configuration()
        self.simulator.spawn_puck(pos, vel, name)
        
        name = 'paddle_ego'
        pos, vel = self.get_paddle_configuration(name)
        self.simulator.spawn_paddle(pos, vel, name)
    
    def validate_configuration(self):
        assert self.num_pucks == 1
        assert self.num_blocks == 0
        assert self.num_obstacles == 0
        assert self.num_targets == 0
        assert self.num_paddles == 1
    
    def get_observation(self, state_info):
        ego_paddle_x_pos = state_info['paddles']['paddle_ego']['position'][0]
        ego_paddle_y_pos = state_info['paddles']['paddle_ego']['position'][1]
        ego_paddle_x_vel = state_info['paddles']['paddle_ego']['velocity'][0]
        ego_paddle_y_vel = state_info['paddles']['paddle_ego']['velocity'][1]
        
        puck_x_pos = state_info['pucks'][0]['position'][0]
        puck_y_pos = state_info['pucks'][0]['position'][1]
        puck_x_vel = state_info['pucks'][0]['velocity'][0]
        puck_y_vel = state_info['pucks'][0]['velocity'][1]       

        obs = np.array([ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel, puck_x_pos, puck_y_pos, puck_x_vel, puck_y_vel])
        return obs

    def get_base_reward(self, state_info):
        reward = 0
        x_pos = state_info['pucks'][0]['position'][0]
        x_higher = self.table_x_top
        x_lower = self.table_x_bot
        if x_higher / 4 < x_pos < 0:
            reward += 15
        elif x_pos < x_higher / 4:
            reward -= 1
        success = reward > 0 and self.current_timestep > 50
        return reward, success


class AirHockeyPuckStrikeEnv(AirHockeyBaseEnv):
    def initialize_spaces(self):
        # setup observation / action / reward spaces
        paddle_obs_low = [self.table_x_top, self.table_y_left, -self.max_paddle_vel, -self.max_paddle_vel]
        paddle_obs_high = [self.table_x_bot, self.table_y_right, self.max_paddle_vel, self.max_paddle_vel]
        
        puck_obs_low = [self.table_x_top, self.table_y_left, -self.max_puck_vel, -self.max_puck_vel]
        puck_obs_high = [self.table_x_bot, self.table_y_right, self.max_puck_vel, self.max_puck_vel]

        low = paddle_obs_low + puck_obs_low
        high = paddle_obs_high + puck_obs_high

        self.observation_space = self.get_obs_space(low, high)
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32) # 2D action space
        self.reward_range = Box(low=-1, high=1) # need to make sure rewards are between 0 and 1
        
    @staticmethod
    def from_dict(state_dict):
        return AirHockeyPuckStrikeEnv(**state_dict)

    def create_world_objects(self):
        puck_x_low = self.length / 5
        puck_x_high = self.length / 3
        # puck_y_low = -self.width / 2 + self.puck_radius
        # puck_y_high = self.width / 2 - self.puck_radius
        puck_y_low = -self.width / 2 + self.simulator.table_y_offset + self.simulator.puck_radius
        puck_y_high = self.width / 2 - self.simulator.table_y_offset - self.simulator.puck_radius
        puck_x = self.rng.uniform(low=puck_x_low, high=puck_x_high)
        puck_y = self.rng.uniform(low=puck_y_low, high=puck_y_high)
        name = 'puck_{}'.format(0)
        pos = (puck_x, puck_y)
        vel = (0, 0)
        self.simulator.spawn_puck(pos, vel, name, affected_by_gravity=False)
        
        name = 'paddle_ego'
        pos, vel = self.get_paddle_configuration(name)
        self.simulator.spawn_paddle(pos, vel, name)
    
    def validate_configuration(self):
        assert self.num_pucks == 1
        assert self.num_blocks == 0
        assert self.num_obstacles == 0
        assert self.num_targets == 0
        assert self.num_paddles == 1
    
    def get_observation(self, state_info):
        ego_paddle_x_pos = state_info['paddles']['paddle_ego']['position'][0]
        ego_paddle_y_pos = state_info['paddles']['paddle_ego']['position'][1]
        ego_paddle_x_vel = state_info['paddles']['paddle_ego']['velocity'][0]
        ego_paddle_y_vel = state_info['paddles']['paddle_ego']['velocity'][1]
        
        puck_x_pos = state_info['pucks'][0]['position'][0]
        puck_y_pos = state_info['pucks'][0]['position'][1]
        puck_x_vel = state_info['pucks'][0]['velocity'][0]
        puck_y_vel = state_info['pucks'][0]['velocity'][1]       

        obs = np.array([ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel, puck_x_pos, puck_y_pos, puck_x_vel, puck_y_vel])
        return obs

    def get_base_reward(self, state_info):
        x_vel = state_info['pucks'][0]['velocity'][0]
        y_vel = state_info['pucks'][0]['velocity'][1]
        vel_mag = np.linalg.norm(np.array([x_vel, y_vel]))
        reward = vel_mag
        max_rew = 2 # estimated max vel
        min_rew = 0  # min acceptable good velocity
        
        initial_pos = self.puck_initial_position
        current_pos = state_info['pucks'][0]['position']
        dist = np.linalg.norm(np.array(initial_pos) - np.array(current_pos))
        has_moved = dist > 0.01
        
        if reward <= min_rew and not has_moved:
            return -5, False # negative rew for standing still and hasn't moved
        reward = min(reward, max_rew)
        reward = (reward - min_rew) / (max_rew - min_rew)
        success = reward > (0.1) # means the puck is moving
        if reward > 0:
            reward *= 10
        return reward, success


class AirHockeyPuckTouchEnv(AirHockeyBaseEnv):
    def initialize_spaces(self):
        # setup observation / action / reward spaces
        paddle_obs_low = [self.table_x_top, self.table_y_left, -self.max_paddle_vel, -self.max_paddle_vel]
        paddle_obs_high = [self.table_x_bot, self.table_y_right, self.max_paddle_vel, self.max_paddle_vel]
        puck_obs_low = [self.table_x_top, self.table_y_left, -self.max_puck_vel, -self.max_puck_vel]
        puck_obs_high = [self.table_x_bot, self.table_y_right, self.max_puck_vel, self.max_puck_vel]
        low = paddle_obs_low + puck_obs_low
        high = paddle_obs_high + puck_obs_high
        self.observation_space = self.get_obs_space(low, high)
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32) # 2D action space
        self.reward_range = Box(low=-1, high=1) # need to make sure rewards are between 0 and 1
    @staticmethod
    def from_dict(state_dict):
        return AirHockeyPuckTouchEnv(**state_dict)
    def create_world_objects(self):
        puck_x_low = self.length / 5
        puck_x_high = self.length / 3
        # puck_y_low = -self.width / 2 + self.puck_radius
        # puck_y_high = self.width / 2 - self.puck_radius
        # puck_y_low = -self.width / 2 + self.simulator.table_y_offset + self.simulator.puck_radius
        # puck_y_high = self.width / 2 - self.simulator.table_y_offset - self.simulator.puck_radius
        # puck_x = self.rng.uniform(low=puck_x_low, high=puck_x_high)
        # puck_y = self.rng.uniform(low=puck_y_low, high=puck_y_high)
        name = 'puck_{}'.format(0)
        pos, vel = self.get_puck_configuration()
        self.simulator.spawn_puck(pos, vel, name)

        name = 'paddle_ego'
        pos, vel = self.get_paddle_configuration(name)
        self.simulator.spawn_paddle(pos, vel, name)
    
    def create_world_objects_from_state(self, state_vector):
        name = 'puck_{}'.format(0)
        puck_pos, puck_vel = state_vector[:2], state_vector[2:4]
        self.simulator.spawn_puck(puck_pos, puck_vel, name)

        name = 'paddle_ego'
        paddle_pos, paddle_vel = state_vector[4:6], state_vector[6:]
        self.simulator.spawn_paddle(paddle_pos, paddle_vel, name)

    def validate_configuration(self):
        assert self.num_pucks == 1
        assert self.num_blocks == 0
        assert self.num_obstacles == 0
        assert self.num_targets == 0
        assert self.num_paddles == 1
    def get_observation(self, state_info):
        ego_paddle_x_pos = state_info['paddles']['paddle_ego']['position'][0]
        ego_paddle_y_pos = state_info['paddles']['paddle_ego']['position'][1]
        ego_paddle_x_vel = state_info['paddles']['paddle_ego']['velocity'][0]
        ego_paddle_y_vel = state_info['paddles']['paddle_ego']['velocity'][1]
        puck_x_pos = state_info['pucks'][0]['position'][0]
        puck_y_pos = state_info['pucks'][0]['position'][1]
        puck_x_vel = state_info['pucks'][0]['velocity'][0]
        puck_y_vel = state_info['pucks'][0]['velocity'][1]
        puck_y_vel = state_info['pucks'][0]['velocity'][1]
        obs = np.array([ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel, puck_x_pos, puck_y_pos, puck_x_vel, puck_y_vel])
        return obs
    def get_base_reward(self, state_info):
        # reward for getting close to the puck, but make sure not to displace it
        puck_pos = state_info['pucks'][0]['position']
        paddle_pos = state_info['paddles']['paddle_ego']['position']
        min_dist = self.paddle_radius + self.puck_radius
        dist = np.linalg.norm(np.array(puck_pos) - np.array(paddle_pos))
        max_dist = 0.16 * self.width
        reward = 1 - ((dist - min_dist) / (max_dist - min_dist))
        reward = max(reward, 0)
        # let's also make sure puck does not deviate from initial position
        puck_initial_position = self.puck_initial_position
        puck_current_position = state_info['pucks'][0]['position']
        delta = np.linalg.norm(np.array(puck_initial_position) - np.array(puck_current_position))
        epsilon = 0.01 + min_dist
        # if delta >= epsilon:
        #     reward -= 1
        # success = reward >= 0.9 and dist < epsilon
        success = dist < epsilon
        # print("dist: ", dist)
        # print("dist < epsilon: ", dist < epsilon)
        # print("reward: ", reward)
        # print("===========================================")
        epsilon = 0.01 + min_dist
        # if delta >= epsilon:
        #     reward -= 1
        # success = reward >= 0.9 and dist < epsilon
        success = dist < epsilon
        # print("dist: ", dist)
        # print("dist < epsilon: ", dist < epsilon)
        # print("reward: ", reward)
        # print("===========================================")
        if reward > 0:
            reward *= 20 # make it more significant
        return reward, success
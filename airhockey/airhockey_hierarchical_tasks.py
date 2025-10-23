import numpy as np
from gymnasium.spaces import Box
from .airhockey_base import AirHockeyBaseEnv
from .airhockey_rewards import AirHockeyMoveBlockReward, AirHockeyStrikeCrowdReward

class AirHockeyMoveBlockEnv(AirHockeyBaseEnv):
    def initialize_spaces(self, obs_type):
        # setup observation / action / reward spaces
        low, high = self.init_observation(obs_type)
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32) # 2D action space
        self.reward_range = Box(low=-1, high=1) # need to make sure rewards are between 0 and 1
        self.reward = AirHockeyMoveBlockReward(self)
        
    @staticmethod
    def from_dict(state_dict):
        return AirHockeyMoveBlockEnv(**state_dict)

    def create_world_objects(self):
        pucks_positions = []
        name = 'puck_{}'.format(0)
        pos, vel = self.get_puck_configuration()
        self.simulator.spawn_puck(pos, vel, name)
        pucks_positions = [pos]

        # need to take into account pucks so far since we do not want to spawn anything directly below them
        puck_y_positions = [pos[1] for pos in pucks_positions]
        bad_regions = [(y - self.puck_radius, y + self.puck_radius) for y in puck_y_positions]
        
        for i in range(self.num_blocks):
            block_name = 'block_{}'.format(i)
            pos, vel = self.get_block_configuration(bad_regions)
            self.simulator.spawn_block(pos, vel, block_name, affected_by_gravity=False)
        
        name = 'paddle_ego'
        pos, vel = self.get_paddle_configuration(name)
        self.simulator.spawn_paddle(pos, vel, name)
    
    def validate_configuration(self):
        assert self.num_pucks == 1
        assert self.num_blocks == 1
        assert self.num_obstacles == 0
        assert self.num_targets == 0
        assert self.num_paddles == 1

    def get_observation(self, state_info, **kwargs):
        return self.get_observation_by_type(state_info, obs_type=self.obs_type, **kwargs)

    def get_observation(self, state_info, obs_type='single_block', **kwargs):
        return self.get_observation_by_type(state_info, obs_type=obs_type, **kwargs)

    # def get_observation(self, state_info):
    #     ego_paddle_x_pos = state_info['paddles']['paddle_ego']['position'][0]
    #     ego_paddle_y_pos = state_info['paddles']['paddle_ego']['position'][1]
    #     ego_paddle_x_vel = state_info['paddles']['paddle_ego']['velocity'][0]
    #     ego_paddle_y_vel = state_info['paddles']['paddle_ego']['velocity'][1]
        
    #     puck_x_pos = state_info['pucks'][0]['position'][0]
    #     puck_y_pos = state_info['pucks'][0]['position'][1]
    #     puck_x_vel = state_info['pucks'][0]['velocity'][0]
    #     puck_y_vel = state_info['pucks'][0]['velocity'][1]       

    #     block_x_pos = state_info['blocks'][0]['current_position'][0]
    #     block_y_pos = state_info['blocks'][0]['current_position'][1]
    #     block_initial_x_pos = state_info['blocks'][0]['initial_position'][0]
    #     block_initial_y_pos = state_info['blocks'][0]['initial_position'][1]
    #     obs = np.array([ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel, puck_x_pos, puck_y_pos, puck_x_vel, puck_y_vel, block_x_pos, block_y_pos, block_initial_x_pos, block_initial_y_pos])
    #     return obs

class AirHockeyStrikeCrowdEnv(AirHockeyBaseEnv):
    def initialize_spaces(self, obs_type):
        # setup observation / action / reward spaces
        paddle_obs_low = [self.table_x_top, self.table_y_left, -self.max_paddle_vel, -self.max_paddle_vel]
        paddle_obs_high = [self.table_x_bot, self.table_y_right, self.max_paddle_vel, self.max_paddle_vel]
        
        puck_obs_low = [self.table_x_top, self.table_y_left, -self.max_puck_vel, -self.max_puck_vel]
        puck_obs_high = [self.table_x_bot, self.table_y_right, self.max_puck_vel, self.max_puck_vel]
        
        block_obs_low = [self.table_x_top, self.table_y_left, self.table_x_top, self.table_y_left]
        block_obs_high = [self.table_x_bot, self.table_y_right, self.table_x_bot, self.table_y_right]

        puck_hist_low = [self.table_x_top, self.table_y_left, 0] * 5
        puck_hist_high = [self.table_x_bot, self.table_y_right, 0] * 5

        if obs_type == "paddle":
            low = paddle_obs_low
            high = paddle_obs_high
        elif obs_type == "vel":
            low = paddle_obs_low + puck_obs_low
            high = paddle_obs_high + puck_obs_high
        elif obs_type == "history":
            low = paddle_obs_low + puck_hist_low
            high = paddle_obs_high + puck_hist_high
        elif obs_type == "many_blocks_vel":
            low = paddle_obs_low + puck_obs_low + [block_obs_low[0], block_obs_low[1]] * self.num_blocks
            high = paddle_obs_high + puck_obs_high + [block_obs_high[0], block_obs_high[1]] * self.num_blocks
        elif obs_type == "many_blocks_history":
            low = paddle_obs_low + [block_obs_low[0], block_obs_low[1]] * self.num_blocks + puck_hist_low
            high = paddle_obs_high + [block_obs_high[0], block_obs_high[1]] * self.num_blocks + puck_hist_high

        self.observation_space = self.get_obs_space(low, high)
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32) # 2D action space
        self.reward_range = Box(low=-1, high=1) # need to make sure rewards are between 0 and 1
        self.reward = AirHockeyStrikeCrowdReward(self)
        
    @staticmethod
    def from_dict(state_dict):
        return AirHockeyStrikeCrowdEnv(**state_dict)

    def create_world_objects(self):
        self.block_initial_positions = {}
        center_y = self.rng.uniform(-0.15, 0.15)  # todo: determine dynamically
        
        # pucks moving downwards that we want to hit directly
        for i in range(self.num_pucks):
            # compute the region for the blocks
            margin = 0.008 #self.block_width / 10
            # starts at center
            prev_row_y_min = None
            prev_row_y_max = None
            
            # let's spawn all the blocks
            # start with 5 blocks
            x = self.table_x_top / 2
            block_space = self.block_width + margin
            n_rows = 5
            row_x_positions = [x + block_space * i for i in range(5)]
            # 0: center_x
            # 1: center_x - block_space
            # 2: center_x + block_space
            # 3: center_x - 2 * block_space
            # 4: center_x + 2 * block_space
            col_y_positions = [center_y, center_y - block_space, center_y + block_space, 
                            center_y - 2 * block_space, center_y + 2 * block_space]  
            
            # for row 0, shift none
            # for row 1, shift to the right by ((prev_x_max - prev_x_min) - (curr_x_max - curr_x_min)) /2
            
            y_min = center_y - 2 * block_space - self.block_width / 2
            y_max = center_y + 2 * block_space + self.block_width / 2
            
            for row_idx in range(n_rows):
                x = row_x_positions[row_idx]
                row_size = 5 - row_idx
                curr_row_y_min = float('inf')
                curr_row_y_max = float('-inf')
                block_positions = []
                for col_idx in range(row_size):
                    y = col_y_positions[col_idx]
                    curr_row_y_min = min(curr_row_y_min, y - self.block_width / 2)
                    curr_row_y_max = max(curr_row_y_max, y + self.block_width / 2)
                    block_positions.append((x, y))
                if row_idx % 2 == 0:
                    shift_amount = 0
                else:
                    shift_amount = ((prev_row_y_max - prev_row_y_min) - (curr_row_y_max - curr_row_y_min)) / 2
                for col_idx, pos in enumerate(block_positions):
                    block_name = f'block_{row_idx}_{col_idx}'
                    pos = (pos[0], pos[1] + shift_amount)
                    self.block_initial_positions[block_name] = pos
                    vel = (0, 0)
                    self.simulator.spawn_block(pos, vel, block_name, affected_by_gravity=False)
                prev_row_y_min = curr_row_y_min
                prev_row_y_max = curr_row_y_max
                    
            assert y_min > self.table_y_left
            assert y_max < self.table_y_right
            # clearance space
            y_min -= self.block_width
            y_max += self.block_width
            
            puck_x = self.length / 5
            puck_y = 0
            pos = (puck_x, puck_y)
            vel = (0, 0)
            name = 'puck_{}'.format(0)
            self.simulator.spawn_puck(pos, vel, name, affected_by_gravity=False)

        name = 'paddle_ego'
        pos, vel = self.get_paddle_configuration(name)
        self.simulator.spawn_paddle(pos, vel, name)
    
    def validate_configuration(self):
        assert self.num_pucks == 1
        assert self.num_blocks == 15
        assert self.num_obstacles == 0
        assert self.num_targets == 0
        assert self.num_paddles == 1

    def get_observation(self, state_info, obs_type='many_blocks', **kwargs):
        return self.get_observation_by_type(state_info, obs_type=obs_type, **kwargs)

    # def get_observation(self, state_info):
    #     ego_paddle_x_pos = state_info['paddles']['paddle_ego']['position'][0]
    #     ego_paddle_y_pos = state_info['paddles']['paddle_ego']['position'][1]
    #     ego_paddle_x_vel = state_info['paddles']['paddle_ego']['velocity'][0]
    #     ego_paddle_y_vel = state_info['paddles']['paddle_ego']['velocity'][1]
        
    #     puck_x_pos = state_info['pucks'][0]['position'][0]
    #     puck_y_pos = state_info['pucks'][0]['position'][1]
    #     puck_x_vel = state_info['pucks'][0]['velocity'][0]
    #     puck_y_vel = state_info['pucks'][0]['velocity'][1]       

    #     blocks = state_info['blocks']
    #     block_initial_positions = []
    #     for block in blocks:
    #         block_initial_positions.append(block['initial_position'])
    #     block_initial_positions = np.array(block_initial_positions).flatten()
    #     obs = np.array([ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel, puck_x_pos, puck_y_pos, puck_x_vel, puck_y_vel] + block_initial_positions.tolist())
    #     return obs

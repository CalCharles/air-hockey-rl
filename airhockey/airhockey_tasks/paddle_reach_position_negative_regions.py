import copy
import numpy as np
from gymnasium.spaces import Box
from gymnasium import spaces
from .abstract_airhockey_goal_task import AirHockeyGoalEnv
from airhockey.airhockey_tasks.utils import RewardRegion
from airhockey.airhockey_rewards import AirHockeyPaddleReachPositionNegRegionsReward
import math
from types import SimpleNamespace

class AirHockeyPaddleReachPositionNegRegionsEnv(AirHockeyGoalEnv):
    def __init__(self, **kwargs):
        
        defaults = {
            'dense_goal': True,
            'goal_selector': 'stationary',
            'max_timesteps': 1000,
            'num_positive_reward_regions': 0,
            'positive_reward_range': [1, 1],
            'num_negative_reward_regions': 0,
            'negative_reward_range': [-1, -1],
            'reward_region_shapes': [],
            'reward_region_scale_range': [0, 0],
            'reward_normalized_radius_min': 0.1,
            'reward_normalized_radius_max': 0.1,
            'reward_velocity_limits_min': [0, 0],
            'reward_velocity_limits_max': [0, 0],
            'reward_movement_types': [],
            'initialization_description_pth': "",
            'paddle_offsets': [0, 0, 0, 0],
            'paddle_clipping': [1, 0, -0.1, -0.15],
            'obs_type': "negative_regions_paddle",
            'goal_radius_type': "fixed",
            'base_goal_radius': 0.05,
        }
        
        kwargs = {**defaults, **kwargs}
        config = SimpleNamespace(**kwargs)

        self.init_dict = self.load_initialization(config.initialization_description_pth)
        self.num_negative_reward_regions = config.num_negative_reward_regions
        self.negative_reward_range = config.negative_reward_range
        self.reward_region_shapes = config.reward_region_shapes
        self.reward_region_scale_range = config.reward_region_scale_range
        self.reward_normalized_radius_min = config.reward_normalized_radius_min
        self.reward_normalized_radius_max = config.reward_normalized_radius_max
        self.reward_velocity_limits_min = config.reward_velocity_limits_min
        self.reward_velocity_limits_max = config.reward_velocity_limits_max
        self.goal_radius_type = config.goal_radius_type
        self.base_goal_radius = config.base_goal_radius
        self.reward = AirHockeyPaddleReachPositionNegRegionsReward(self)
        super().__init__(**kwargs)
        
    @staticmethod
    def from_dict(state_dict):
        return AirHockeyPaddleReachPositionNegRegionsEnv(**state_dict)

    def start_callbacks(self):
        # starts callbacks for the real robot, should be overwritten for most methods
        # but the default logic should suffice
        region_info = [r.state.tolist() + [r.radius] for r in self.reward_regions]
        goal_info = self.goal_pos.tolist() + [self.goal_radius]
        self.simulator.start_callbacks(region_info=region_info, goal_info=goal_info)


    def load_initialization(self, pth):
        '''
        We can initialize preset arrangements as a n x m grid with the hashes:
        s: paddle
        x: negative reward region
        a: allowed to initialize a negative region here
        o: nothing
        g: goal
        TODO: we could also look at all the files in the path and use all of them as initializations
        '''
        lines = list()
        # TODO: we only specify a single paddle/goal pos for now, but we could do multiple
        self.init_paddle_pos = None 
        self.init_goal_pos = None
        self.init_negative_pos = list()
        self.init_allowed_negative_pos = list()
        if len(pth):
            with open(pth, 'r') as file:
                line = file.readline()
                line_counter = 0
                while line:
                    for col, v in enumerate(line):
                        if v == 's':
                            self.init_paddle_pos = [line_counter, col]
                        elif v == 'x':
                            self.init_negative_pos.append([line_counter, col])
                        elif v == 'a':
                            self.init_allowed_negative_pos.append([line_counter, col])
                        elif v == 'g':
                            self.init_goal_pos = [line_counter, col]
                    line_counter += 1
                    lines.append(line)
                    line = file.readline()
            self.grid_dims = [len(lines), len(lines[0])]
        else: # initialize these values to -1, but probably unnecessary
            self.grid_dims = [-1,-1]

    def initialize_spaces(self, obs_type):
        # setup observation / action / reward spaces
        low, high = self.init_observation(obs_type.replace("negative_regions_", ""))

        nrr_obs_low = [-math.inf] * 12 * self.num_negative_reward_regions
        nrr_obs_high = [-math.inf] * 12 * self.num_negative_reward_regions

        goal_low = [0, self.table_y_left]
        goal_high = [self.table_x_bot, self.table_y_right]
                    
        
        self.min_goal_radius = self.width / 8
        self.max_goal_radius = self.width / 4
        self.goal_radius = self.base_goal_radius

        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32) # 2D action space
        self.reward_range = Box(low=-1, high=1) # need to make sure rewards are between 0 and 1

        if self.grid_dims is not None:
            self.grid_lengths = np.array([(self.table_x_bot - 0) / self.grid_dims[0], 
                                 (self.table_y_right - self.table_y_left) / self.grid_dims[1]])
            self.grid_midpoints = list()
            for i in np.arange(self.grid_dims[0]):
                midpoint_row = list()
                for j in np.arange(self.grid_dims[1]):
                    midpoint_row.append(np.array([0, self.table_y_left]) + np.array([i + 0.5,j + 1.0]) * self.grid_lengths)
                self.grid_midpoints.append(midpoint_row)
            self.grid_midpoints = np.array(self.grid_midpoints)
        
        self.set_goals(self.goal_radius_type)

        if len(self.init_negative_pos):
            # TODO: ignores the num_negative_reward_region, could have those initialized randomly
            for pos in self.init_negative_pos:
                radius_range = [np.array(self.grid_lengths) / 2, np.array(self.grid_lengths) / 2]
                pos = copy.deepcopy(self.grid_midpoints[pos[0], pos[1]])
                self.reward_regions.append(RewardRegion(self.negative_reward_range, 
                                                            self.reward_region_scale_range, 
                                                            [np.array(pos), np.array(pos)],
                                                            radius_range, shapes=self.reward_region_shapes, object_radius = self.paddle_radius))
        # TODO: implement the allowed regions code
        # elif len(self.init_allowed_negative_pos.keys()):
        else:
            radius_range = np.array((self.table_x_bot/ 2 * np.array(self.reward_normalized_radius_min), (self.table_y_right - self.table_y_left) * np.array(self.reward_normalized_radius_max) / 2))
            self.reward_regions = [RewardRegion(self.negative_reward_range, 
                                                     self.reward_region_scale_range, 
                                                     [np.array([0, self.table_y_left]), np.array([self.table_x_bot,self.table_y_right])],
                                                     radius_range, shapes=self.reward_region_shapes) for _ in range(self.num_negative_reward_regions)]
        
        reward_regions_states_shape = np.concatenate([nrr.get_state() for nrr in self.reward_regions]).shape
        # these are misspecified but you can check later
        reward_region_states_low = [-100] * reward_regions_states_shape[0]
        reward_region_states_high = [100] * reward_regions_states_shape[0]

        if self.return_goal_obs:
            low = low + reward_region_states_low
            high = high + reward_region_states_high
            self.observation_space = self.get_goal_obs_space(low, high, goal_low, goal_high)
        else:
            low = low + reward_region_states_low + goal_low 
            high = high + reward_region_states_high + goal_high 
            self.observation_space = self.get_obs_space(low, high)
            
    def create_world_objects(self):
        name = 'paddle_ego'
        if self.init_paddle_pos is None: pos, vel = self.get_paddle_configuration(name)
        else: 
            pos = copy.deepcopy(self.grid_midpoints[self.init_paddle_pos[0], self.init_paddle_pos[1]])
            vel = (0,0)
            print("init paddle", self.init_paddle_pos, pos)
        self.simulator.spawn_paddle(pos, vel, name)
        
    def validate_configuration(self):
        assert self.num_pucks == 0
        assert self.num_blocks == 0
        assert self.num_obstacles == 0
        assert self.num_targets == 0
        assert self.num_paddles == 1
        assert self.num_negative_reward_regions > 0
    
    def get_achieved_goal(self, state_info):
        position = state_info['paddles']['paddle_ego']['position']
        return np.array([position[0], position[1]])
    
    def get_desired_goal(self):
        position = self.goal_pos
        return np.array([position[0], position[1]])
    
    def get_observation(self, state_info, obs_type="negative_regions_paddle", **kwargs):
        state_info["negative_regions"] = [nrr.get_state() for nrr in self.reward_regions]
        return self.get_observation_by_type(state_info, obs_type=obs_type, **kwargs)

    # def get_observation(self, state_info):
    #     ego_paddle_x_pos = state_info['paddles']['paddle_ego']['position'][0]
    #     ego_paddle_y_pos = state_info['paddles']['paddle_ego']['position'][1]
    #     ego_paddle_x_vel = state_info['paddles']['paddle_ego']['velocity'][0]
    #     ego_paddle_y_vel = state_info['paddles']['paddle_ego']['velocity'][1]
    #     reward_regions_states = [nrr.get_state() for nrr in self.reward_regions]

    #     obs = np.array([ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel])
    #     return np.concatenate([obs] + reward_regions_states)
    
    def set_goals(self, goal_radius_type, goal_pos=None, alt_goal_pos=None, goal_set=None):
        self.goal_set = goal_set
        # sample goal position
        min_y = self.table_y_left
        max_y = self.table_y_right
        min_x = 0
        max_x = self.table_x_bot / 2 # set goal positions to be in the bottom half of the table.
        if self.init_goal_pos is None: goal_position = self.rng.uniform(low=(min_x, min_y), high=(max_x, max_y))
        else: 
            goal_position = copy.deepcopy(self.grid_midpoints[self.init_goal_pos[0], self.init_goal_pos[1]])
            print("init_goal", self.grid_midpoints,self.table_x_bot, self.init_goal_pos, goal_position)
        self.goal_radius = self.min_goal_radius # not too important
        self.goal_pos = goal_position if self.goal_set is None else self.goal_set[0, :2]

    def reset(self, seed=None, **kwargs):
        for nrr in self.reward_regions:
            nrr.reset()
        return super().reset(seed, **kwargs)
import math

import numpy as np
from gymnasium.spaces import Box
from .airhockey_base import AirHockeyBaseEnv
from .airhockey_rewards import AirHockeyPuckCatchReward, AirHockeyPuckVelReward, AirHockeyPuckTouchReward, AirHockeyPuckHeightReward, AirHockeyPuckJuggleReward, AirHockeyPuckJuggleLinearTopReward, AirHockeyPuckJuggleNoBaseReward, AirHockeyPuckJuggleUpperHalfReward, AirHockeyPuckJuggleUpperHalfMidBandReward, AirHockeyPuckStrikeReward, AirHockeyStrikeCrowdReward, AirHockeyPaddleFreeMovementReward

class AirHockeyPuckVelEnv(AirHockeyBaseEnv):
    def initialize_spaces(self, obs_type):
        # setup observation / action / reward spaces
        low, high = self.init_observation(obs_type)
        self.action_space = self.single_action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32) # 2D action space
        self.reward_range = Box(low=-1, high=1) # need to make sure rewards are between 0 and 1
        self.reward = AirHockeyPuckVelReward(self)
        
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

    def get_observation(self, state_info, obs_type ="vel", **kwargs):
        return self.get_observation_by_type(state_info, obs_type=obs_type, **kwargs)
        # ego_paddle_x_pos = state_info['paddles']['paddle_ego']['position'][0]
        # ego_paddle_y_pos = state_info['paddles']['paddle_ego']['position'][1]
        # ego_paddle_x_vel = state_info['paddles']['paddle_ego']['velocity'][0]
        # ego_paddle_y_vel = state_info['paddles']['paddle_ego']['velocity'][1]
        
        # puck_x_pos = state_info['pucks'][0]['position'][0]
        # puck_y_pos = state_info['pucks'][0]['position'][1]
        # puck_x_vel = state_info['pucks'][0]['velocity'][0]
        # puck_y_vel = state_info['pucks'][0]['velocity'][1]       

        # obs = np.array([ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel, puck_x_pos, puck_y_pos, puck_x_vel, puck_y_vel])
        # return obs

class AirHockeyPuckHeightEnv(AirHockeyBaseEnv):

    def __init__(self, **kwargs):
        super(AirHockeyPuckHeightEnv, self).__init__(**kwargs)
        self.num_touches = 0
        self.touching = False

    def initialize_spaces(self, obs_type):
        # setup observation / action / reward spaces
        low, high = self.init_observation(obs_type)
        self.action_space = self.single_action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32) # 2D action space
        self.reward_range = Box(low=-1, high=1) # need to make sure rewards are between 0 and 1
        self.reward = AirHockeyPuckHeightReward(self)

    @staticmethod
    def from_dict(state_dict):
        # print("state_dict", state_dict)
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

    def get_observation(self, state_info, obs_type ="vel", **kwargs):
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

    #     obs = np.array([ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel, puck_x_pos, puck_y_pos, puck_x_vel, puck_y_vel])
    #     return obs

    def has_finished(self, state_info, multiagent=False):
        terminated, truncated, puck_within_home, puck_within_alt_home, puck_within_ego_goal, puck_within_alt_goal = super().has_finished(state_info, multiagent)
        terminated = terminated or self.success_in_ep
        return terminated, truncated, puck_within_home, puck_within_alt_home, puck_within_ego_goal, puck_within_alt_goal

class AirHockeyPuckCatchEnv(AirHockeyBaseEnv):
    def initialize_spaces(self, obs_type):
        # setup observation / action / reward spaces
        low, high = self.init_observation(obs_type)
        self.action_space = self.single_action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32) # 2D action space
        self.reward_range = Box(low=-1, high=1) # need to make sure rewards are between 0 and 1
        self.reward = AirHockeyPuckCatchReward(self)
        
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

    def get_observation(self, state_info, obs_type ="vel", **kwargs):
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

    #     obs = np.array([ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel, puck_x_pos, puck_y_pos, puck_x_vel, puck_y_vel])
    #     return obs
    
class AirHockeyPuckJuggleEnv(AirHockeyBaseEnv):
    def initialize_spaces(self, obs_type):
        # setup observation / action / reward spaces
        low, high = self.init_observation(obs_type)
        self.action_space = self.single_action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32) # 2D action space
        self.reward_range = Box(low=-1, high=1) # need to make sure rewards are between 0 and 1
        self.count_hit = False
        self.hits = 0
        self.reward = AirHockeyPuckJuggleReward(self)
        
    @staticmethod
    def from_dict(state_dict):
        return AirHockeyPuckJuggleEnv(**state_dict)

    def create_world_objects(self):
        for i in range(self.num_pucks):
            name = 'puck_{}'.format(i)
            pos, vel = self.get_puck_configuration()
            self.simulator.spawn_puck(pos, vel, name)
        
        name = 'paddle_ego'
        pos, vel = self.get_paddle_configuration(name)
        self.simulator.spawn_paddle(pos, vel, name)
    
    def validate_configuration(self):
        assert self.num_pucks > 0
        assert self.num_blocks == 0
        assert self.num_obstacles == 0
        assert self.num_targets == 0
        assert self.num_paddles == 1

    def get_observation(self, state_info, obs_type ="vel", **kwargs):
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

    #     obs = np.array([ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel, puck_x_pos, puck_y_pos, puck_x_vel, puck_y_vel])
    #     return obs

class AirHockeyPuckJuggleLinearTopEnv(AirHockeyPuckJuggleEnv):
    def __init__(self, **kwargs):
        spawn_prob = float(kwargs.get("puck_spawn_near_paddle_prob", 0.0))
        offset_min_m = float(kwargs.get("puck_near_paddle_offset_min_m", 0.025))
        offset_max_m = float(kwargs.get("puck_near_paddle_offset_max_m", 0.05))
        horizontal_std_m = float(kwargs.get("puck_near_paddle_horizontal_std_m", 0.015))
        speed_max_m_s = float(kwargs.get("puck_near_paddle_speed_max_m_s", 0.2))

        self.puck_spawn_near_paddle_prob = float(np.clip(spawn_prob, 0.0, 1.0))
        self.puck_near_paddle_offset_min_m = float(max(0.0, offset_min_m))
        self.puck_near_paddle_offset_max_m = float(max(0.0, offset_max_m))
        if self.puck_near_paddle_offset_max_m < self.puck_near_paddle_offset_min_m:
            self.puck_near_paddle_offset_min_m, self.puck_near_paddle_offset_max_m = (
                self.puck_near_paddle_offset_max_m,
                self.puck_near_paddle_offset_min_m,
            )
        self.puck_near_paddle_horizontal_std_m = float(max(0.0, horizontal_std_m))
        self.puck_near_paddle_speed_max_m_s = float(max(0.0, speed_max_m_s))

        super().__init__(**kwargs)

    def initialize_spaces(self, obs_type):
        low, high = self.init_observation(obs_type)
        self.action_space = self.single_action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.reward_range = Box(low=-1, high=1)
        self.count_hit = False
        self.hits = 0
        self.reward = AirHockeyPuckJuggleLinearTopReward(self)
    
    @staticmethod
    def from_dict(state_dict):
        return AirHockeyPuckJuggleLinearTopEnv(**state_dict)

    def get_paddle_configuration(self, name):
        if name == 'paddle_ego':
            x_low = self.paddle_x_min + self.paddle_radius
            x_high = self.paddle_x_max - self.paddle_radius
            y_low = self.paddle_y_min + self.paddle_radius
            y_high = self.paddle_y_max - self.paddle_radius
            if x_low >= x_high:
                x_low, x_high = self.table_x_top + self.paddle_radius, self.table_x_bot - self.paddle_radius
            if y_low >= y_high:
                y_low, y_high = self.table_y_left + self.paddle_radius, self.table_y_right - self.paddle_radius
            x_pos = self.rng.uniform(low=x_low, high=x_high)
            y_pos = self.rng.uniform(low=y_low, high=y_high)
            return (x_pos, y_pos), (0, 0)
        elif name == 'paddle_alt':
            x_pos = self.table_x_top + self.paddle_radius
            return (x_pos, 0), (0, 0)
        else:
            raise ValueError("Invalid paddle name")

    def create_world_objects(self):
        if self.num_pucks != 1 or self.puck_spawn_near_paddle_prob <= 0.0:
            return super().create_world_objects()

        spawn_near_paddle = (
            self.rng.uniform(low=0.0, high=1.0) < self.puck_spawn_near_paddle_prob
        )
        if not spawn_near_paddle:
            return super().create_world_objects()

        paddle_name = "paddle_ego"
        paddle_pos, paddle_vel = self.get_paddle_configuration(paddle_name)
        puck_name = "puck_0"
        puck_pos, puck_vel = self.get_puck_configuration(
            paddle_pos=paddle_pos,
            spawn_near_paddle=True,
        )
        self.simulator.spawn_puck(puck_pos, puck_vel, puck_name)
        self.simulator.spawn_paddle(paddle_pos, paddle_vel, paddle_name)

    def _sample_puck_speed_velocity(self, max_speed):
        speed = self.rng.uniform(low=0.0, high=max_speed)
        heading = self.rng.uniform(low=0.0, high=2 * math.pi)
        return (speed * math.cos(heading), speed * math.sin(heading))

    def _sample_puck_upper_half_linear_top(self, bad_regions=None, max_speed=1.2):
        # Use base-frame coordinates here; Box2D conversion happens in spawn_puck.
        # "Upper half" is the top side of the table (x from table_x_top to centerline).
        x_low = self.table_x_top + self.puck_radius
        x_high = 0.0 - self.puck_radius
        y_low = self.table_y_left + self.puck_radius
        y_high = self.table_y_right - self.puck_radius

        y_pos = None
        if bad_regions is not None:
            while y_pos is None:
                proposed_y_pos = self.rng.uniform(low=y_low, high=y_high)
                if all(not (region[0] < proposed_y_pos < region[1]) for region in bad_regions):
                    y_pos = proposed_y_pos
        else:
            y_pos = self.rng.uniform(low=y_low, high=y_high)

        x_pos = self.rng.uniform(low=x_low, high=x_high)
        vel = self._sample_puck_speed_velocity(max_speed=max_speed)
        return (x_pos, y_pos), vel

    def _sample_puck_near_paddle(self, paddle_pos):
        paddle_x, paddle_y = paddle_pos
        offset = self.rng.uniform(
            low=self.puck_near_paddle_offset_min_m,
            high=self.puck_near_paddle_offset_max_m,
        )
        x_pos = paddle_x - offset
        y_pos = paddle_y + self.rng.normal(
            loc=0.0,
            scale=self.puck_near_paddle_horizontal_std_m,
        )
        x_pos = float(
            np.clip(
                x_pos,
                self.table_x_top + self.puck_radius,
                self.table_x_bot - self.puck_radius,
            )
        )
        y_pos = float(
            np.clip(
                y_pos,
                self.table_y_left + self.puck_radius,
                self.table_y_right - self.puck_radius,
            )
        )
        vel = self._sample_puck_speed_velocity(
            max_speed=self.puck_near_paddle_speed_max_m_s
        )
        return (x_pos, y_pos), vel

    def get_puck_configuration(
        self,
        bad_regions=None,
        paddle_pos=None,
        spawn_near_paddle=False,
    ):
        if spawn_near_paddle and paddle_pos is not None:
            return self._sample_puck_near_paddle(paddle_pos=paddle_pos)
        return self._sample_puck_upper_half_linear_top(bad_regions=bad_regions)


class AirHockeyPuckJuggleNoBaseRewardEnv(AirHockeyPuckJuggleLinearTopEnv):
    def initialize_spaces(self, obs_type):
        low, high = self.init_observation(obs_type)
        self.action_space = self.single_action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.reward_range = Box(low=-1, high=1)
        self.count_hit = False
        self.hits = 0
        self.reward = AirHockeyPuckJuggleNoBaseReward(self)

    @staticmethod
    def from_dict(state_dict):
        return AirHockeyPuckJuggleNoBaseRewardEnv(**state_dict)


class AirHockeyPuckJuggleUpperHalfRewardEnv(AirHockeyPuckJuggleLinearTopEnv):
    def initialize_spaces(self, obs_type):
        low, high = self.init_observation(obs_type)
        self.action_space = self.single_action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.reward_range = Box(low=-1, high=1)
        self.count_hit = False
        self.hits = 0
        self.reward = AirHockeyPuckJuggleUpperHalfReward(self)

    @staticmethod
    def from_dict(state_dict):
        return AirHockeyPuckJuggleUpperHalfRewardEnv(**state_dict)


class AirHockeyPuckJuggleUpperHalfMidBandRewardEnv(AirHockeyPuckJuggleLinearTopEnv):
    def initialize_spaces(self, obs_type):
        low, high = self.init_observation(obs_type)
        self.action_space = self.single_action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.reward_range = Box(low=-1, high=1)
        self.count_hit = False
        self.hits = 0
        self.reward = AirHockeyPuckJuggleUpperHalfMidBandReward(self)

    @staticmethod
    def from_dict(state_dict):
        return AirHockeyPuckJuggleUpperHalfMidBandRewardEnv(**state_dict)

    def get_paddle_configuration(self, name):
        if name == 'paddle_ego':
            x_pos = (self.paddle_x_min + self.paddle_x_max) / 2.0
            y_pos = (self.paddle_y_min + self.paddle_y_max) / 2.0
            x_pos = np.clip(
                x_pos,
                self.table_x_top + self.paddle_radius,
                self.table_x_bot - self.paddle_radius,
            )
            y_pos = np.clip(
                y_pos,
                self.table_y_left + self.paddle_radius,
                self.table_y_right - self.paddle_radius,
            )
            return (float(x_pos), float(y_pos)), (0, 0)
        if name == 'paddle_alt':
            x_pos = self.table_x_top + self.paddle_radius
            return (x_pos, 0), (0, 0)
        raise ValueError("Invalid paddle name")

    def get_puck_configuration(
        self,
        bad_regions=None,
        paddle_pos=None,
        spawn_near_paddle=False,
    ):
        if spawn_near_paddle and paddle_pos is not None:
            return super().get_puck_configuration(
                bad_regions=bad_regions,
                paddle_pos=paddle_pos,
                spawn_near_paddle=True,
            )
        del bad_regions
        return self._sample_puck_upper_half_linear_top(bad_regions=None, max_speed=0.5)

class AirHockeyPuckStrikeEnv(AirHockeyBaseEnv):
    def initialize_spaces(self, obs_type):
        # setup observation / action / reward spaces
        low, high = self.init_observation(obs_type)
        self.action_space = self.single_action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32) # 2D action space
        self.reward_range = Box(low=-1, high=1) # need to make sure rewards are between 0 and 1
        self.reward = AirHockeyPuckStrikeReward(self)
        
    @staticmethod
    def from_dict(state_dict):
        return AirHockeyPuckStrikeEnv(**state_dict)

    def create_world_objects(self):
        puck_x_low = self.length / 5
        puck_x_high = self.length / 3
        puck_y_low = -self.width / 2 + self.puck_radius
        puck_y_high = self.width / 2 - self.puck_radius
        # puck_y_low = -self.width / 2 + self.simulator.table_y_offset + self.simulator.puck_radius
        # puck_y_high = self.width / 2 - self.simulator.table_y_offset - self.simulator.puck_radius
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

    def get_observation(self, state_info, obs_type ="vel", **kwargs):
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

    #     obs = np.array([ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel, puck_x_pos, puck_y_pos, puck_x_vel, puck_y_vel])
    #     return obs

class AirHockeyPuckTouchEnv(AirHockeyBaseEnv):
    def initialize_spaces(self, obs_type):
        # setup observation / action / reward spaces
        low, high = self.init_observation(obs_type)
        self.action_space = self.single_action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32) # 2D action space
        self.reward_range = Box(low=-1, high=1) # need to make sure rewards are between 0 and 1
        self.reward = AirHockeyPuckTouchReward(self)

    @staticmethod
    def from_dict(state_dict):
        return AirHockeyPuckTouchEnv(**state_dict)

    def create_world_objects(self):
        name = 'puck_{}'.format(0)
        # pos, vel = self.get_puck_configuration()
        y_pos = self.rng.uniform(low=-self.width / 3, high=self.width / 3)
        pos = (self.table_x_top + 1.1, y_pos)
        vel = (1, 0)
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

    def get_observation(self, state_info, obs_type='vel', **kwargs):
        return self.get_observation_by_type(state_info, obs_type=obs_type, **kwargs)
        # ego_paddle_x_pos = state_info['paddles']['paddle_ego']['position'][0]
        # ego_paddle_y_pos = state_info['paddles']['paddle_ego']['position'][1]
        # ego_paddle_x_vel = state_info['paddles']['paddle_ego']['velocity'][0]
        # ego_paddle_y_vel = state_info['paddles']['paddle_ego']['velocity'][1]
        # puck_x_pos = state_info['pucks'][0]['position'][0]
        # puck_y_pos = state_info['pucks'][0]['position'][1]
        # puck_x_vel = state_info['pucks'][0]['velocity'][0]
        # puck_y_vel = state_info['pucks'][0]['velocity'][1]

        # obs = np.array([ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel, puck_x_pos, puck_y_pos, puck_x_vel, puck_y_vel])
        # return obs


class AirHockeyPaddleFreeMovementEnv(AirHockeyBaseEnv):
    def initialize_spaces(self, obs_type):
        # setup observation / action / reward spaces
        low, high = self.init_observation(obs_type)
        self.action_space = self.single_action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32) # 2D action space
        self.reward_range = Box(low=-1, high=1) # need to make sure rewards are between 0 and 1
        self.reward = AirHockeyPaddleFreeMovementReward(self)
        
    @staticmethod
    def from_dict(state_dict):
        return AirHockeyPaddleFreeMovementEnv(**state_dict)

    def get_paddle_configuration(self, name):
        """
        Initialize paddle with random position in lower half of table and random velocity.
        
        Position ranges:
        - x: From center (0) to bottom of table (table_x_bot), staying away from edges
        - y: Across the width of the table, staying away from edges
        
        Velocity ranges:
        - Reasonable velocities up to 50% of max_paddle_vel
        """
        if name == 'paddle_ego':
            # Position: lower half of table (x from 0 to table_x_bot)
            # Leave some margin from edges (2 * paddle_radius)
            x_min = 0 + 2 * self.paddle_radius
            x_max = self.table_x_bot - 2 * self.paddle_radius
            y_min = self.table_y_left + 2 * self.paddle_radius
            y_max = self.table_y_right - 2 * self.paddle_radius
            
            x_pos = self.rng.uniform(low=x_min, high=x_max)
            y_pos = self.rng.uniform(low=y_min, high=y_max)
            pos = (x_pos, y_pos)
            
            # Velocity: reasonable range (up to 50% of max velocity)
            max_init_vel = 0.5 * self.max_paddle_vel
            vx = self.rng.uniform(low=-max_init_vel, high=max_init_vel)
            vy = self.rng.uniform(low=-max_init_vel, high=max_init_vel)
            vel = (vx, vy)
            
            return pos, vel
        elif name == 'paddle_alt':
            x_pos = self.table_x_top + self.paddle_radius
            vel = (0, 0)
            return (x_pos, 0), vel
        else:
            raise ValueError("Invalid paddle name")

    def create_world_objects(self):
        # Only spawn a paddle, no puck or other objects
        name = 'paddle_ego'
        pos, vel = self.get_paddle_configuration(name)
        self.simulator.spawn_paddle(pos, vel, name)
    
    def validate_configuration(self):
        assert self.num_pucks == 0
        assert self.num_blocks == 0
        assert self.num_obstacles == 0
        assert self.num_targets == 0
        assert self.num_paddles == 1

    def get_observation(self, state_info, obs_type="vel", **kwargs):
        return self.get_observation_by_type(state_info, obs_type=obs_type, **kwargs)

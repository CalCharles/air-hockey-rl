import math

import numpy as np
from gymnasium.spaces import Box
from .airhockey_base import AirHockeyBaseEnv
from .airhockey_rewards import AirHockeyPuckCatchReward, AirHockeyPuckVelReward, AirHockeyPuckTouchReward, AirHockeyPuckHeightReward, AirHockeyPuckJuggleReward, AirHockeyPuckJuggleLinearTopReward, AirHockeyPuckJuggleNoBaseReward, AirHockeyPuckJuggleUpperHalfReward, AirHockeyPuckJuggleUpperHalfMidBandReward, AirHockeyPuckStrikeReward, AirHockeyStrikeCrowdReward, AirHockeyPaddleFreeMovementReward, AirHockeyPinballTriangleSideReward

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
        self._spawn_triangle_obstacles()

    def _spawn_triangle_obstacles(self):
        """Static triangle bodies in Box2D only (fixture restitution; no custom contact impulses)."""
        if self.num_obstacles <= 0:
            return
        if self.simulator_name != "box2d":
            raise NotImplementedError("Triangle obstacles are only implemented for the box2d simulator.")
        shape = str(getattr(self.simulator, "obstacle_shape", "triangle")).lower()
        if shape != "triangle":
            raise ValueError(f"Unsupported obstacle_shape={shape!r}; expected 'triangle'.")

        tri_size = float(getattr(self.simulator, "triangle_obstacle_size", 0.08))
        # Must match airhockey_box2d.spawn_obstacle: half_base = 0.75 * tri_size (base along y).
        half_base_y = 0.75 * tri_size
        h_tri = (np.sqrt(3.0) / 2.0) * tri_size
        margin = max(half_base_y, 2.0 * h_tri / 3.0) + float(self.puck_radius) + 0.01
        length = float(self.length)
        # Along x: "top" = goal end (table_x_top). Sample from 10% of table length below that edge
        # down to 40% of table length from the top (farther toward center than the old top-third cap).
        x_lo = self.table_x_top + 0.10 * length + margin
        x_hi = self.table_x_top + 0.40 * length - margin
        y_lo = self.table_y_left + margin
        y_hi = self.table_y_right - margin
        if x_lo >= x_hi or y_lo >= y_hi:
            raise ValueError("Table geometry too tight for triangle obstacles with current margins.")

        y_gap = max(0.02, float(self.puck_radius) * 0.25)
        min_center_sep_y = 2.0 * half_base_y + y_gap
        n = self.num_obstacles
        if n > 1:
            min_y_span = 2.0 * half_base_y + (n - 1) * min_center_sep_y
            if (y_hi - y_lo) + 1e-9 < min_y_span:
                raise ValueError(
                    f"Table y-range too small for {n} triangles with separated y "
                    f"(need >= {min_y_span:.4f} m, have {y_hi - y_lo:.4f} m)."
                )

        def _y_separated(y, ys):
            return all(abs(float(y) - float(py)) >= min_center_sep_y for py in ys)

        positions = [None] * n
        placed_y = []

        for i in range(n):
            if i < len(self.obstacle_positions):
                raw = self.obstacle_positions[i]
                x = float(np.clip(raw[0], x_lo, x_hi))
                y = float(np.clip(raw[1], y_lo, y_hi))
                if not _y_separated(y, placed_y):
                    raise ValueError(
                        f"obstacle_positions[{i}] y={y:.4f} overlaps another obstacle in y "
                        f"(need |Δy| >= {min_center_sep_y:.4f} m between triangle centers)."
                    )
                placed_y.append(y)
                positions[i] = (x, y)

        for i in range(n):
            if positions[i] is not None:
                continue
            for _ in range(8000):
                x = float(self.rng.uniform(x_lo, x_hi))
                y = float(self.rng.uniform(y_lo, y_hi))
                if _y_separated(y, placed_y):
                    placed_y.append(y)
                    positions[i] = (x, y)
                    break
            else:
                raise ValueError(
                    f"Could not sample a non-overlapping y for triangle {i} "
                    f"(try fewer obstacles, smaller triangle_obstacle_size, or set obstacle_positions)."
                )

        for i in range(n):
            x, y = positions[i]
            self.simulator.spawn_obstacle((x, y), f"triangle_obstacle_{i}")
    
    def validate_configuration(self):
        assert self.num_pucks > 0
        assert self.num_blocks == 0
        assert self.num_obstacles >= 0
        if self.num_obstacles > 0:
            assert self.simulator_name == "box2d"
            if len(self.obstacle_positions) > self.num_obstacles:
                raise ValueError("obstacle_positions has more entries than num_obstacles.")
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
        linear_center_cutoff_x = float(kwargs.get("puck_linear_top_spawn_center_cutoff_x", 0.0))
        linear_goal_cutoff_x = kwargs.get("puck_linear_top_spawn_goal_cutoff_x", None)
        linear_speed_min = float(kwargs.get("puck_linear_top_spawn_speed_min", 0.0))
        linear_speed_max = float(kwargs.get("puck_linear_top_spawn_speed_max", 1.2))

        spawn_prob = float(kwargs.get("puck_spawn_near_paddle_prob", 0.0))
        offset_min_m = float(kwargs.get("puck_near_paddle_offset_min_m", 0.025))
        offset_max_m = float(kwargs.get("puck_near_paddle_offset_max_m", 0.05))
        horizontal_std_m = float(kwargs.get("puck_near_paddle_horizontal_std_m", 0.015))
        speed_min_m_s = float(kwargs.get("puck_near_paddle_speed_min_m_s", 0.0))
        speed_max_m_s = float(kwargs.get("puck_near_paddle_speed_max_m_s", 0.2))

        self.puck_linear_top_spawn_center_cutoff_x = linear_center_cutoff_x
        self.puck_linear_top_spawn_goal_cutoff_x = (
            None if linear_goal_cutoff_x is None else float(linear_goal_cutoff_x)
        )
        self.puck_linear_top_spawn_speed_min = float(max(0.0, linear_speed_min))
        self.puck_linear_top_spawn_speed_max = float(max(0.0, linear_speed_max))
        if self.puck_linear_top_spawn_speed_max < self.puck_linear_top_spawn_speed_min:
            raise ValueError(
                "puck_linear_top_spawn_speed_max must be >= puck_linear_top_spawn_speed_min"
            )

        self.puck_spawn_near_paddle_prob = float(np.clip(spawn_prob, 0.0, 1.0))
        self.puck_near_paddle_offset_min_m = float(max(0.0, offset_min_m))
        self.puck_near_paddle_offset_max_m = float(max(0.0, offset_max_m))
        if self.puck_near_paddle_offset_max_m < self.puck_near_paddle_offset_min_m:
            self.puck_near_paddle_offset_min_m, self.puck_near_paddle_offset_max_m = (
                self.puck_near_paddle_offset_max_m,
                self.puck_near_paddle_offset_min_m,
            )
        self.puck_near_paddle_horizontal_std_m = float(max(0.0, horizontal_std_m))
        self.puck_near_paddle_speed_min_m_s = float(max(0.0, speed_min_m_s))
        self.puck_near_paddle_speed_max_m_s = float(max(0.0, speed_max_m_s))
        if self.puck_near_paddle_speed_max_m_s < self.puck_near_paddle_speed_min_m_s:
            raise ValueError(
                "puck_near_paddle_speed_max_m_s must be >= puck_near_paddle_speed_min_m_s"
            )

        frac = kwargs.get("puck_spawn_fixed_x_from_goal_frac", None)
        self.puck_spawn_fixed_x_from_goal_frac = (
            None if frac is None else float(np.clip(float(frac), 0.0, 1.0))
        )

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
            super().create_world_objects()
            return

        spawn_near_paddle = (
            self.rng.uniform(low=0.0, high=1.0) < self.puck_spawn_near_paddle_prob
        )
        if not spawn_near_paddle:
            super().create_world_objects()
            return

        paddle_name = "paddle_ego"
        paddle_pos, paddle_vel = self.get_paddle_configuration(paddle_name)
        puck_name = "puck_0"
        puck_pos, puck_vel = self.get_puck_configuration(
            paddle_pos=paddle_pos,
            spawn_near_paddle=True,
        )
        self.simulator.spawn_puck(puck_pos, puck_vel, puck_name)
        self.simulator.spawn_paddle(paddle_pos, paddle_vel, paddle_name)
        self._spawn_triangle_obstacles()

    def _sample_puck_speed_velocity(self, min_speed, max_speed):
        speed = self.rng.uniform(low=min_speed, high=max_speed)
        heading = self.rng.uniform(low=0.0, high=2 * math.pi)
        return (speed * math.cos(heading), speed * math.sin(heading))

    def _sample_puck_upper_half_linear_top(self, bad_regions=None):
        # Use base-frame coordinates here; Box2D conversion happens in spawn_puck.
        # "Upper half" is the top side of the table (x from table_x_top to centerline).
        x_table_low = self.table_x_top + self.puck_radius
        x_table_high = self.table_x_bot - self.puck_radius
        x_low = x_table_low
        if self.puck_linear_top_spawn_goal_cutoff_x is not None:
            x_low = max(x_low, self.puck_linear_top_spawn_goal_cutoff_x)
        x_high = self.puck_linear_top_spawn_center_cutoff_x - self.puck_radius
        x_low = float(np.clip(x_low, x_table_low, x_table_high))
        x_high = float(np.clip(x_high, x_table_low, x_table_high))
        if x_low >= x_high:
            raise ValueError(
                "Invalid linear-top puck spawn x-range after cutoffs: "
                f"x_low={x_low}, x_high={x_high}, "
                f"goal_cutoff={self.puck_linear_top_spawn_goal_cutoff_x}, "
                f"center_cutoff={self.puck_linear_top_spawn_center_cutoff_x}"
            )
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
        vel = self._sample_puck_speed_velocity(
            min_speed=self.puck_linear_top_spawn_speed_min,
            max_speed=self.puck_linear_top_spawn_speed_max,
        )
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
            min_speed=self.puck_near_paddle_speed_min_m_s,
            max_speed=self.puck_near_paddle_speed_max_m_s,
        )
        return (x_pos, y_pos), vel

    def _sample_puck_fixed_x_from_goal_frac_zero_vel(self):
        """x at ``table_x_top + frac * length`` (e.g. 0.5 = centerline); random y; rest speed 0."""
        frac = float(self.puck_spawn_fixed_x_from_goal_frac)
        x_pos = self.table_x_top + frac * float(self.length)
        x_pos = float(
            np.clip(
                x_pos,
                self.table_x_top + self.puck_radius,
                self.table_x_bot - self.puck_radius,
            )
        )
        y_low = self.table_y_left + self.puck_radius
        y_high = self.table_y_right - self.puck_radius
        y_pos = float(self.rng.uniform(low=y_low, high=y_high))
        return (x_pos, y_pos), (0.0, 0.0)

    def get_puck_configuration(
        self,
        bad_regions=None,
        paddle_pos=None,
        spawn_near_paddle=False,
    ):
        if self.puck_spawn_fixed_x_from_goal_frac is not None:
            return self._sample_puck_fixed_x_from_goal_frac_zero_vel()
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


class AirHockeyPuckJugglePinballTriangleSidesEnv(AirHockeyPuckJuggleLinearTopEnv):
    """Pinball: reward only for puck contacts on triangle sloped edges (not the flat base)."""

    def initialize_spaces(self, obs_type):
        low, high = self.init_observation(obs_type)
        self.action_space = self.single_action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.reward_range = Box(low=-1, high=1)
        self.count_hit = False
        self.hits = 0
        self.reward = AirHockeyPinballTriangleSideReward(self)

    @staticmethod
    def from_dict(state_dict):
        return AirHockeyPuckJugglePinballTriangleSidesEnv(**state_dict)


class AirHockeyPuckJuggleUpperHalfMidBandRewardEnv(AirHockeyPuckJuggleLinearTopEnv):
    def __init__(self, **kwargs):
        kwargs.setdefault("puck_linear_top_spawn_speed_max", 0.5)
        super().__init__(**kwargs)

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
        return self._sample_puck_upper_half_linear_top(bad_regions=None)

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

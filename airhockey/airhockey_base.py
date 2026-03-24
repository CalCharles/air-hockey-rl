from gymnasium import Env
import numpy as np
from gymnasium.spaces import Box
from gymnasium import spaces
from abc import ABC, abstractmethod
import math
from .sims.real.coordinate_transform import get_clip_limits
from .utils import get_observation_by_type, dict_to_namespace

from typing import Tuple
from types import SimpleNamespace
import copy

def get_box2d_simulator_fn():
    from airhockey.sims import AirHockeyBox2D
    return AirHockeyBox2D
    
def get_robosuite_simulator_fn():
    from airhockey.sims import AirHockeyRobosuite
    return AirHockeyRobosuite

def get_real_simulator_fn():
    from airhockey.sims import AirHockeyReal
    return AirHockeyReal


class AirHockeyBaseEnv(ABC, Env):
    def __init__(self, **kwargs):
        
        self.defaults = {
            'ignore_done': False,
            'hard_reset': True,
            'camera_names': ["birdview", "sideview"],
            'camera_heights': 512,
            'camera_widths': 512,
            'camera_depths': False,
            'camera_segmentations': None,
            'renderer': "mujoco",
            'renderer_config': None,
            'table_xml': "arenas/air_hockey_table.xml",
            'paddle_bounds': [],
            'paddle_edge_bounds': [],
            'center_offset_constant': 1.2,
            'action_x_ratio': 0.26,
            'action_y_ratio': 0.12,
            'num_positive_reward_regions': 0,
            'positive_reward_range': [1, 1],
            'num_negative_reward_regions': 0,
            'negative_reward_range': [-1, -1],
            'reward_region_shapes': [],
            'compute_online_rewards': True,
            'reward_region_scale_range': [0, 0],
            'reward_normalized_radius_min': 0.1,
            'reward_normalized_radius_max': 0.1,
            'reward_velocity_limits_min': [0, 0],
            'reward_velocity_limits_max': [0, 0],
            'reward_movement_types': [],
            'terminate_on_puck_hit_bottom': False,  # TODO Specify this parameter in the yaml config
            'terminate_on_puck_hit_paddle': False,
            'terminate_on_puck_pass_paddle': False,
            'terminate_on_puck_pass_paddle_consecutive_steps': 5, # magic number
            'puck_pass_paddle_enter_margin_m': 0.01,
            'puck_pass_paddle_exit_margin_m': -0.01,
            'puck_pass_paddle_score_threshold': 3,
            'dense_goal': True,
            'goal_selector': 'stationary',
            'max_timesteps': 1000,
            'domain_random': False,
            'random_variables': [],
            'random_variable_ranges': {},
            'initialization_description_pth': "",
            'solrefs': [None, None, None, None],
            'obs_type': "vel",
            'base_goal_radius': 0.15,
            'puck_goal_success_bonus': 0.0,
            'paddle_puck_success_bonus': 0.0,

            'use_smooth_penalty': False,
            'use_reward_shaping': True,
            'base_reward_scaling': 1.0,
            'jerk_penalty_coeff': 0.0,
            'velocity_penalty_coeff': 0.0,
            'enable_survival_bonus': False,
            'survival_bonus_per_step': 0.25,
        }
        
        # handle defaults, keeps values for duplicate keys from right side!
        kwargs = {**self.defaults, **kwargs}

        config = dict_to_namespace(kwargs)

        # domain randomization
        self.domain_random = config.domain_random
        self.random_variables = config.random_variables
        self.random_variable_ranges = config.random_variable_ranges

        if config.simulator == 'box2d':
            simulator_fn = get_box2d_simulator_fn()
        elif config.simulator == 'robosuite':
            simulator_fn = get_robosuite_simulator_fn()
        elif config.simulator == 'real':
            simulator_fn = get_real_simulator_fn()
        else:
            raise ValueError("Invalid simulator type. Must be 'box2d' or 'robosuite'.")

        simulator_params = config.simulator_params
        simulator_params.seed = config.seed
        simulator_params.paddle_bounds = config.paddle_bounds
        simulator_params.paddle_edge_bounds = config.paddle_edge_bounds
        simulator_params.center_offset_constant = config.center_offset_constant
        self.simulator_name = config.simulator
        self.simulator = simulator_fn.from_dict(vars(simulator_params))
        self.render_length = self.simulator.render_length
        self.render_width = self.simulator.render_width
        self.render_masks = self.simulator.render_masks
        self.ppm = self.simulator.ppm
        
        self.simulator_params = simulator_params

        self.max_timesteps = config.max_timesteps
        self.current_timestep = 0
        self.n_training_steps = config.n_training_steps
        self.n_timesteps_so_far = 0
        self.rng = np.random.RandomState(config.seed)
        self.dynamic_virtual_objects = list() # if the environment has these, put them in at subclass initialization
        self.reward_regions = list()
        
        # termination conditions
        self.terminate_on_out_of_bounds = config.terminate_on_out_of_bounds
        self.terminate_on_enemy_goal = config.terminate_on_enemy_goal
        self.terminate_on_puck_stop = config.terminate_on_puck_stop
        self.terminate_on_puck_hit_bottom = config.terminate_on_puck_hit_bottom
        # Optional simulator parameter: additional boundary margin (meters)
        # around table_x_bot for puck-hit-bottom termination.
        self.puck_hit_bottom_boundary_m = float(
            getattr(simulator_params, "puck_hit_bottom_boundary_m", 0.03)
        )
        self.terminate_on_puck_hit_paddle = config.terminate_on_puck_hit_paddle
        self.terminate_on_puck_pass_paddle = config.terminate_on_puck_pass_paddle
        self.terminate_on_puck_pass_paddle_consecutive_steps = max(
            1, int(config.terminate_on_puck_pass_paddle_consecutive_steps)
        )
        self.puck_pass_paddle_enter_margin_m = float(
            config.puck_pass_paddle_enter_margin_m
        )
        self.puck_pass_paddle_exit_margin_m = float(
            config.puck_pass_paddle_exit_margin_m
        )
        self.puck_pass_paddle_score_threshold = max(
            1, int(config.puck_pass_paddle_score_threshold)
        )
        self._puck_pass_paddle_score = 0
        self.puck_low_motion_radius_m = 0.03
        self.puck_low_motion_window_clean = 20
        self.puck_low_motion_window_occluded = 20
        
        # reward function
        self.compute_online_rewards = config.compute_online_rewards
        self.goal_conditioned = True if 'goal' in config.task or 'reach' in config.task else False
        self.goal_min_x_velocity = -config.goal_max_x_velocity
        self.goal_max_x_velocity = config.goal_max_x_velocity
        self.goal_min_y_velocity = config.goal_min_y_velocity
        self.goal_max_y_velocity = config.goal_max_y_velocity
        self.return_goal_obs = config.return_goal_obs
        self.dense_goal = config.dense_goal
        self.task = config.task
        self.multiagent = config.num_paddles == 2
        self.truncate_rew = config.truncate_rew
        self.wall_bumping_rew = config.wall_bumping_rew
        self.direction_change_rew = config.direction_change_rew
        self.horizontal_vel_rew = config.horizontal_vel_rew
        self.diagonal_motion_rew = config.diagonal_motion_rew
        self.stand_still_rew = config.stand_still_rew
        self.use_reward_shaping = config.use_reward_shaping
        self.base_reward_scaling = config.base_reward_scaling
        self.jerk_penalty_coeff = config.jerk_penalty_coeff
        self.velocity_penalty_coeff = config.velocity_penalty_coeff
        self.enable_survival_bonus = bool(config.enable_survival_bonus)
        self.survival_bonus_per_step = float(config.survival_bonus_per_step)
        self.simulator_params = simulator_params
        self.width = simulator_params.width
        self.length = simulator_params.length
        self.paddle_radius = simulator_params.paddle_radius
        self.puck_radius = simulator_params.puck_radius
        self.puck_damping = getattr(simulator_params, 'puck_damping', None)
        if config.simulator == "robosuite":
            self.solrefs = [
                getattr(simulator_params, 'top_solref', None),
                getattr(simulator_params, 'bot_solref', None),
                getattr(simulator_params, 'left_solref', None),
                getattr(simulator_params, 'right_solref', None)
            ]

        self.paddle_radius = simulator_params.paddle_radius
        self.puck_radius = simulator_params.puck_radius
        self.block_width = simulator_params.block_width
        
        self.table_x_top = -self.length / 2
        self.table_x_bot = self.length / 2
        self.table_y_right = self.width / 2
        self.table_y_left = -self.width / 2
        self.center_offset_constant = config.center_offset_constant
        self.action_x_ratio = config.action_x_ratio
        self.action_y_ratio = config.action_y_ratio
        # import pdb; pdb.set_trace()
        if len(config.paddle_bounds) == 0: # use preset values
            self.paddle_x_min = 0 - 2 * self.paddle_radius # self.table_x_top / 2 + 2 * self.paddle_radius
            self.paddle_x_max = self.table_x_bot + 2 * self.paddle_radius
            self.paddle_y_min = self.table_y_left - 2 * self.paddle_radius
            self.paddle_y_max = self.table_y_right + 2 * self.paddle_radius
        else:
            self.paddle_x_min, self.paddle_x_max, self.paddle_y_min, self.paddle_y_max = config.paddle_bounds
            self.move_lims = [-1,-1]
            # real world bounds: x_min_lim = -0.8, x_max_lim = -0.33, y_min = -0.3582, y_max = 0.350
        self.boundary_lims = [self.paddle_x_min, self.paddle_x_max, self.paddle_y_min, self.paddle_y_max]
        self.move_lims = [-1,-1]

        if len(config.paddle_edge_bounds):
            self.edge_lims = config.paddle_edge_bounds
        else:
            self.edge_lims = [0,0,100,100]

        self.max_paddle_vel = self.simulator.max_paddle_vel
        self.max_puck_vel = self.simulator.max_puck_vel
        self.goal_set = None

        self._base_get_observation_by_type = get_observation_by_type
        self.get_observation_by_type = self._get_observation_by_type_with_position_homography
        self.obs_type = config.obs_type
        
        self.num_pucks = config.num_pucks
        self.multiagent = config.num_paddles > 1
        self.num_blocks = config.num_blocks
        self.num_obstacles = config.num_obstacles
        self.num_targets = config.num_targets
        self.num_paddles = config.num_paddles
        
        self.validate_configuration()

        self.goal_selector = config.goal_selector
        self.initialize_spaces(self.obs_type)
        self.falling_time = 25
        self.metadata = {}
        self._last_done_reasons = {"terminated": [], "truncated": []}
        self.start_callbacks()
        self.domain_random = config.domain_random
        self.reset()


    @abstractmethod
    def from_dict(state_dict):
        pass

    @abstractmethod
    def initialize_spaces(self, obs_type):
        pass
    
    def init_observation(self, obs_type):
        paddle_obs_low = [self.table_x_top, self.table_y_left, -self.max_paddle_vel, -self.max_paddle_vel]
        paddle_obs_high = [self.table_x_bot, self.table_y_right, self.max_paddle_vel, self.max_paddle_vel]
        
        puck_obs_low = [self.table_x_top, self.table_y_left, -self.max_puck_vel, -self.max_puck_vel]
        puck_obs_high = [self.table_x_bot, self.table_y_right, self.max_puck_vel, self.max_puck_vel]
        
        puck_hist_low = [self.table_x_top, self.table_y_left, 0] * 5
        puck_hist_high = [self.table_x_bot, self.table_y_right, 0] * 5

        # history: deltas
        paddle_hist_low = [self.table_x_top, self.table_y_left, 0] * 5
        paddle_hist_high = [self.table_x_bot, self.table_y_right, 0] * 5


        paddle_accel_low = [-1000, -1000]
        paddle_accel_high = [1000, 1000]
        
        paddle_force_low = [-1000, -1000]
        paddle_force_high = [1000, 1000]

        block_obs_low = [self.table_x_top, self.table_y_left, self.table_x_top, self.table_y_left]
        block_obs_high = [self.table_x_bot, self.table_y_right, self.table_x_bot, self.table_y_right]

        if obs_type == "paddle":
            low = paddle_obs_low
            high = paddle_obs_high
        elif obs_type == "pos":
            low = paddle_obs_low[:2] + puck_obs_low[:2]
            high = paddle_obs_high[:2] + puck_obs_high[:2]
        elif obs_type == "vel":
            low = paddle_obs_low + puck_obs_low
            high = paddle_obs_high + puck_obs_high
        elif obs_type == "history":
            low = paddle_hist_low + puck_hist_low
            high = paddle_hist_high + puck_hist_high
        elif obs_type == "paddle_acceleration_vel":
            low = paddle_obs_low + paddle_accel_low + paddle_force_low + puck_obs_low
            high = paddle_obs_high + paddle_accel_high + paddle_force_high + puck_obs_high
        elif obs_type == "paddle_acceleration_history":
            low = paddle_obs_low + paddle_accel_low + paddle_force_low + puck_hist_low
            high = paddle_obs_high + paddle_accel_high + paddle_force_high + puck_hist_high
        elif obs_type == "single_block_vel":
            low = paddle_obs_low + puck_obs_low + block_obs_low
            high = paddle_obs_high + puck_obs_high + block_obs_high
        elif obs_type == "single_block_history":
            low = paddle_obs_low + block_obs_low + puck_hist_low
            high = paddle_obs_high + block_obs_high + puck_hist_high
        elif obs_type == "many_blocks_vel":
            low = paddle_obs_low + puck_obs_low + [block_obs_low[0], block_obs_low[1]] * self.num_blocks
            high = paddle_obs_high + puck_obs_high + [block_obs_high[0], block_obs_high[1]] * self.num_blocks
        elif obs_type == "many_blocks_history":
            low = paddle_obs_low + [block_obs_low[0], block_obs_low[1]] * self.num_blocks + puck_hist_low
            high = paddle_obs_high + [block_obs_high[0], block_obs_high[1]] * self.num_blocks + puck_hist_high

        self.observation_space = self.single_observation_space = self.get_obs_space(low, high)
        return low, high
    
    @abstractmethod
    def create_world_objects(self):
        pass
    
    @abstractmethod
    def validate_configuration(self):
        pass

    @abstractmethod
    def get_observation(self, state_info):
        pass

    def get_base_reward(self, state_info):
        return self.reward.get_base_reward(state_info)

    def get_current_state(self): 
        # gets the current state and info
        state_info = self.simulator.get_current_state()
        obs = self.get_observation(state_info, obs_type=self.obs_type, puck_history=self.simulator.puck_history, paddle_history=self.simulator.paddle_history)
        return obs, state_info

    def _get_observation_by_type_with_position_homography(self, state_info, obs_type='vel', **kwargs):
        return self._base_get_observation_by_type(
            state_info,
            obs_type=obs_type,
            position_homography=getattr(self.simulator, "obs_position_homography", None),
            **kwargs,
        )

    def define_get_observation(self, getter, obs_type=""):
        if len(obs_type) > 0: self.obs_type = obs_type
        self.get_observation_by_type = getter

    def start_callbacks(self):
        # starts callbacks for the real robot, should be overwritten for most methods
        # but the default logic should suffice
        self.simulator.start_callbacks()

    def get_obs_space(self, low: list, high: list):
        return Box(low=np.array(low), high=np.array(high), dtype=float)

    def reset(self, seed=None, **kwargs):

        if self.domain_random:
            if self.simulator_name == 'box2d':
                simulator_fn = get_box2d_simulator_fn()
            elif self.simulator_name == 'robosuite':
                simulator_fn = get_robosuite_simulator_fn()
            else:
                raise ValueError("Invalid simulator type. Must be 'box2d' or 'robosuite'.")


            if self.domain_random:
                for counter, var in enumerate(self.random_variables):
                    setattr(self.simulator_params, var, np.random.uniform(*getattr(self.random_variable_ranges, var)))


            self.simulator = simulator_fn.from_dict(vars(self.simulator_params))
            self.render_length = self.simulator.render_length
            self.render_width = self.simulator.render_width
            self.render_masks = self.simulator.render_masks
            self.ppm = self.simulator.ppm
        
        # print("Resetting environment")

        if seed is None: # determine next seed, in a deterministic manner
            seed = self.rng.randint(0, int(1e8))

        self.rng = np.random.RandomState(seed)
        sim_seed = self.rng.randint(0, int(1e8))
        self.simulator.reset(sim_seed, **kwargs) # no point in getting state since no spawning
        self.create_world_objects()
        if self.simulator_name == "robosuite":
            self.simulator.update_table(*self.solrefs)
        self.simulator.instantiate_objects()
        state_info = self.simulator.get_current_state()
        self.simulator.set_object_links()
        self.current_state = state_info
        obs = self.get_observation(state_info, obs_type=self.obs_type, puck_history=self.simulator.puck_history, paddle_history=self.simulator.paddle_history)
        self.n_timesteps_so_far += self.current_timestep
        self.current_timestep = 0
        self.success_in_ep = False
        self.max_reward_in_single_step = -np.inf
        self.min_reward_in_single_step = np.inf
        self.episode_return = 0.0
        self.episode_length = 0
        self.episode_motion_data = {'velocity_mags': [], 'acceleration_mags': [], 'jerk_mags': []}
        self._last_done_reasons = {"terminated": [], "truncated": []}
        self._puck_pass_paddle_score = 0

        if 'pucks' in state_info and len(state_info['pucks']) > 0:
            self.puck_initial_position = state_info['pucks'][0]['position']
        
        return obs, {**{'success': False}, **vars(self.simulator_params)}

    def soft_reset(self):
        """Reset episode counters without physical robot movement.

        Calls simulator.soft_reset() if available, then resets env-level
        episode tracking state and returns the current observation.
        """
        if hasattr(self.simulator, "soft_reset"):
            state_info = self.simulator.soft_reset()
        else:
            state_info = self.simulator.get_current_state()
        self.current_state = state_info
        obs = self.get_observation(
            state_info,
            obs_type=self.obs_type,
            puck_history=self.simulator.puck_history,
            paddle_history=self.simulator.paddle_history,
        )
        self.n_timesteps_so_far += self.current_timestep
        self.current_timestep = 0
        self.success_in_ep = False
        self.max_reward_in_single_step = -np.inf
        self.min_reward_in_single_step = np.inf
        self.episode_return = 0.0
        self.episode_length = 0
        self.episode_motion_data = {'velocity_mags': [], 'acceleration_mags': [], 'jerk_mags': []}
        self._last_done_reasons = {"terminated": [], "truncated": []}
        self._puck_pass_paddle_score = 0
        return obs, {**{'success': False}, **vars(self.simulator_params)}

    def reset_from_state(self, state_vector, seed=None):
        if seed is None: # determine next seed, in a deterministic manner
            seed = self.rng.randint(0, int(1e8))

        self.rng = np.random.RandomState(seed)
        sim_seed = self.rng.randint(0, int(1e8))
        self.simulator.reset(sim_seed) # no point in getting state since no spawning
        self.create_world_objects_from_state(state_vector)
        self.simulator.instantiate_objects()
        state_info = self.simulator.get_current_state()
        self.current_state = state_info
        self._puck_pass_paddle_score = 0
        obs = self.get_observation(state_info, obs_type=self.obs_type, puck_history=self.simulator.puck_history, paddle_history=self.simulator.paddle_history)
        return obs, {'success': False}

    def get_puck_configuration(self, bad_regions=None):
        y_pos = None
        if bad_regions is not None:
            while y_pos is None:
                for region in bad_regions:
                    proposed_y_pos = self.rng.uniform(low=-self.width / 3, high=self.width / 3)  # doesnt spawn at edges
                    if not (proposed_y_pos > region[0] and proposed_y_pos < region[1]):
                        y_pos = proposed_y_pos
        else:
            y_pos = self.rng.uniform(low=-self.width / 3, high=self.width / 3)
        pos = (self.table_x_top + 0.01, y_pos)
        vel = (1, 0)
        return pos, vel
    
    def get_block_configuration(self, bad_regions=None):
        y_pos = None
        if bad_regions is not None:
            while y_pos is None:
                for region in bad_regions:
                    proposed_y_pos = self.rng.uniform(low=self.table_y_left + 2 * self.block_width, high=self.table_y_right - 2 * self.block_width)
                    region_with_margin = (region[0] - self.block_width, region[1] + self.block_width)
                    if not (proposed_y_pos > region_with_margin[0] and proposed_y_pos < region_with_margin[1]):
                        y_pos = proposed_y_pos
        else:
            y_pos = self.rng.uniform(low=-self.width / 3, high=self.width / 3)
        x_pos = self.rng.uniform(low=self.table_x_top + 2 * self.block_width, high=0 - self.block_width)
        pos = (x_pos, y_pos)
        vel = (0, 0)
        return pos, vel
    
    def get_paddle_configuration(self, name):
        # Check for reference state (set by ReferenceStateWrapper)
        if name == 'paddle_ego' and hasattr(self, '_ref_paddle_state') and self._ref_paddle_state is not None:
            pos, vel = self._ref_paddle_state
            self._ref_paddle_state = None  # Clear after use
            return pos, vel
        
        # Default behavior
        if name == 'paddle_ego':
            x_pos = self.table_x_bot * 3/4
        elif name == 'paddle_alt':
            x_pos = self.table_x_top + self.paddle_radius
        else:
            raise ValueError("Invalid paddle name")
        vel = (0, 0)
        return (x_pos, 0), vel

    def _puck_low_motion_cluster_window(self, state_info):
        """Return (is_low_motion_cluster, active_window_size) for puck history."""
        if "pucks" not in state_info or len(state_info["pucks"]) == 0:
            return False, 0
        puck_history = state_info["pucks"][0].get("history", [])
        if not isinstance(puck_history, list):
            return False, 0

        clean_window = int(self.puck_low_motion_window_clean)
        occluded_window = int(self.puck_low_motion_window_occluded)
        # Real-world simulator states may intentionally expose a short history
        # for observation compatibility. For low-motion termination only, fallback
        # to the simulator's full internal puck history when available.
        if len(puck_history) < clean_window and getattr(self, "simulator_name", None) == "real":
            simulator_puck_history = getattr(getattr(self, "simulator", None), "puck_history", None)
            if isinstance(simulator_puck_history, list):
                puck_history = simulator_puck_history

        if len(puck_history) < clean_window:
            return False, 0

        recent_clean = puck_history[-clean_window:]
        clean_has_occlusion = any(
            len(entry) >= 3 and int(np.asarray(entry[2]).reshape(-1)[0]) > 0
            for entry in recent_clean
        )
        active_window = occluded_window if clean_has_occlusion else clean_window
        if len(puck_history) < active_window:
            return False, active_window

        recent_window = puck_history[-active_window:]
        positions = []
        for entry in recent_window:
            if entry is None or len(entry) < 2:
                return False, active_window
            positions.append([float(entry[0]), float(entry[1])])
        positions_arr = np.asarray(positions, dtype=np.float64)
        if positions_arr.shape != (active_window, 2):
            return False, active_window

        radius = float(self.puck_low_motion_radius_m)
        pairwise_dist = np.linalg.norm(
            positions_arr[:, None, :] - positions_arr[None, :, :],
            axis=-1,
        )
        within_anchor = np.all(pairwise_dist <= radius, axis=1)
        return bool(np.any(within_anchor)), active_window

    def _is_puck_observation_occluded(self, puck_state):
        occluded_raw = puck_state.get("occluded", 0)
        try:
            return int(np.asarray(occluded_raw).reshape(-1)[0]) > 0
        except (TypeError, ValueError, IndexError):
            return False

    def _update_puck_pass_paddle_score(self, puck_state, paddle_state):
        puck_x = float(puck_state["position"][0])
        paddle_x = float(paddle_state["position"][0])
        rel_x = puck_x - (paddle_x + self.paddle_radius)
        is_occluded = self._is_puck_observation_occluded(puck_state)

        if is_occluded:
            if rel_x < self.puck_pass_paddle_exit_margin_m:
                self._puck_pass_paddle_score = 0
        elif rel_x > self.puck_pass_paddle_enter_margin_m:
            self._puck_pass_paddle_score = min(
                self.puck_pass_paddle_score_threshold,
                self._puck_pass_paddle_score + 1,
            )
        elif rel_x < self.puck_pass_paddle_exit_margin_m:
            self._puck_pass_paddle_score = 0
        else:
            self._puck_pass_paddle_score = max(
                self._puck_pass_paddle_score - 1,
                0,
            )

        return self._puck_pass_paddle_score >= self.puck_pass_paddle_score_threshold

    def has_finished(self, state_info, multiagent=False):
        truncated = False
        terminated = False
        termination_reasons = []
        truncation_reasons = []
        puck_within_alt_home = False
        puck_within_home = False

        if bool(state_info.get("protective_stop", False)):
            terminated = True
            termination_reasons.append("protective_stop")

        if self.current_timestep > self.max_timesteps:
            truncated = True
            truncation_reasons.append("max_timesteps_exceeded")
        else:
            if self.terminate_on_out_of_bounds:
                # check if we hit any walls or are above the middle of the board
                if state_info['paddles']['paddle_ego']['position'][0] < self.paddle_x_min or \
                    state_info['paddles']['paddle_ego']['position'][0] > self.paddle_x_max or \
                    state_info['paddles']['paddle_ego']['position'][1] >  self.paddle_y_max or \
                    state_info['paddles']['paddle_ego']['position'][1] < self.paddle_y_min:
                    truncated = True
                    truncation_reasons.append("paddle_out_of_bounds")
                    print("paddle out of bounds with position: ", state_info['paddles']['paddle_ego']['position'])
                    print("X_min, X_max, Y_min, Y_max: ", self.paddle_x_min + self.paddle_radius, self.paddle_x_max - self.paddle_radius, self.table_y_left + self.paddle_radius, self.table_y_right - self.paddle_radius)

        bottom_center_point = np.array([self.table_x_bot, 0])
        top_center_point = np.array([self.table_x_top, 0])
        
        puck_within_home = False
        puck_within_alt_home = False

        if self.terminate_on_puck_hit_bottom:
            puck_pos = state_info['pucks'][0]['position']
            if abs(puck_pos[0] - self.table_x_bot) < (
                self.puck_radius + self.puck_hit_bottom_boundary_m
            ):
                terminated = True
                termination_reasons.append("puck_hit_bottom")

        if self.terminate_on_enemy_goal:
            if not terminated and puck_within_home:
                truncated = True
                truncation_reasons.append("enemy_goal")

        if multiagent:
            if truncated and not terminated:
                termination_reasons.extend(truncation_reasons)
                truncation_reasons = []
                termination_reasons.append("multiagent_truncation_promoted_to_termination")
            terminated = terminated or truncated or puck_within_alt_home or puck_within_home
            truncated = False
            
        if self.terminate_on_puck_stop:
            low_motion_cluster, low_motion_window = self._puck_low_motion_cluster_window(state_info)
            if not truncated and low_motion_cluster:
                truncated = True
                truncation_reasons.append(f"puck_low_motion_window_{low_motion_window}")


        if "pucks" in state_info.keys():
            # puck paddle distance
            puck_paddle_distance = np.linalg.norm(np.array(state_info['pucks'][0]['position']) - np.array(state_info['paddles']['paddle_ego']['position']))

            if self.terminate_on_puck_pass_paddle:
                puck_passed_now = self._update_puck_pass_paddle_score(
                    state_info['pucks'][0],
                    state_info['paddles']['paddle_ego'],
                )
                if puck_passed_now:
                    truncated = True
                    truncation_reasons.append("puck_passed_paddle")
                    # print("Puck pass paddle")
            else:
                self._puck_pass_paddle_score = 0

            if self.terminate_on_puck_hit_paddle:
                if puck_paddle_distance <= (self.paddle_radius + self.puck_radius + 0.02):
                    puck_within_home = True
                    terminated = True
                    termination_reasons.append("puck_hit_paddle")
                    # print("Puck hit paddle")
        else:
            self._puck_pass_paddle_score = 0
        
        puck_within_ego_goal = False
        puck_within_alt_goal = False
        self._last_done_reasons = {
            "terminated": list(dict.fromkeys(termination_reasons)),
            "truncated": list(dict.fromkeys(truncation_reasons)),
        }
                    
        return terminated, truncated, puck_within_home, puck_within_alt_home, puck_within_ego_goal, puck_within_alt_goal


    def get_smooth_penalty(self, state_info, action=None):
        pass

    def get_reward_shaping(self, state_info, action=None):
        additional_rew = 0.0
        
        # small negative reward for changing direction
        if self.current_timestep > 0:
            old_vel = self.old_state['paddles']['paddle_ego']['velocity']
            new_vel = state_info['paddles']['paddle_ego']['velocity']
            vel_unit = old_vel / (np.linalg.norm(old_vel) + 1e-8)
            new_vel_unit = new_vel / (np.linalg.norm(new_vel) + 1e-8)
            cosine_sim = np.dot(vel_unit, new_vel_unit) / (np.linalg.norm(vel_unit) * np.linalg.norm(new_vel_unit) + 1e-8)
            norm_cosine_sim = (cosine_sim + 1) / 2
            max_change_dir_rew = self.direction_change_rew
            direction_rew = max_change_dir_rew * (1 - norm_cosine_sim)

            additional_rew += direction_rew


        # small negative reward for moving too fast in horizontal direction
        max_vel = self.max_paddle_vel
        max_vel_rew = self.horizontal_vel_rew
        normalized_y_vel = np.abs(state_info['paddles']['paddle_ego']['velocity'][1]) / max_vel
        additional_rew += max_vel_rew * normalized_y_vel
        
        # negative penalty for diagonal motion
        # angle of vector will be close to % 45 degrees if moving diagonally
        angle = np.arctan2(state_info['paddles']['paddle_ego']['velocity'][1], state_info['paddles']['paddle_ego']['velocity'][0])
        angle = np.abs(angle)
        # check if sufficiently close to pi/4, 3pi/4, 5pi/4, 7pi/4
        threshold = np.pi / 12
        # check if between (pi/4 - pi/12, pi/4 + pi/12), ...
        if np.abs(angle - -np.pi / 4) < threshold or np.abs(angle - 3 * -np.pi / 4) < threshold or \
            np.abs(angle - np.pi / 4) < threshold or np.abs(angle - 3 * np.pi / 4) < threshold:
            additional_rew += self.diagonal_motion_rew
        
        # small positive reward for keeping still
        if np.linalg.norm(state_info['paddles']['paddle_ego']['velocity']) < 0.01:
            additional_rew += self.stand_still_rew

        # TODO: small negative reward for acceleration with sign change significantly
        # sign_change = (state_info['paddles']['paddle_ego']['velocity'] * action)
        # sign_change[sign_change > 0] = 0
        # sign_change[sign_change < 0] = 1
        # velocity_change = np.abs(state_info['paddles']['paddle_ego']['velocity'] - action) * sign_change
        # additional_rew += self.aceleration_penalty * velocity_change

        # determine if close to walls
        if self.wall_bumping_rew != 0:
            bump_right = state_info['paddles']['paddle_ego']['position'][1] > self.table_y_right - 2 * self.paddle_radius
            bump_left = state_info['paddles']['paddle_ego']['position'][1] < self.table_y_left + 2 * self.paddle_radius
            bump_top = state_info['paddles']['paddle_ego']['position'][0] < 0 + 4 * self.paddle_radius
            bump_bottom = state_info['paddles']['paddle_ego']['position'][0] > self.table_x_bot - 4 * self.paddle_radius
            if bump_left or bump_right or bump_top or bump_bottom:
                additional_rew += self.wall_bumping_rew
        
        # jerk penalty - negative reward for high jerk (if jerk data is available)
        if self.jerk_penalty_coeff != 0.0 and 'jerk' in state_info['paddles']['paddle_ego']:
            jerk_magnitude = np.linalg.norm(state_info['paddles']['paddle_ego']['jerk'])
            jerk_penalty = -self.jerk_penalty_coeff * jerk_magnitude
            additional_rew += jerk_penalty
        
        # velocity penalty - negative reward for high velocity
        if self.velocity_penalty_coeff != 0.0 and 'velocity' in state_info['paddles']['paddle_ego']:
            velocity_magnitude = np.linalg.norm(state_info['paddles']['paddle_ego']['velocity'])
            velocity_penalty = -self.velocity_penalty_coeff * velocity_magnitude
            additional_rew += velocity_penalty
        
        # TODO: require simulators to send contact info in state
        return additional_rew

    def step(self, action):
        if not self.multiagent:
            obs, reward, is_finished, truncated, info = self.single_agent_step(action)
            is_finished = is_finished or truncated
            return obs, reward, is_finished, truncated, info
        else:
            return self.multi_step(action)
    
    def single_step_dynamic_virtual(self, action):
        # step any dynamic virtual objects to update their state
        for dvo in self.dynamic_virtual_objects:
            dvo.step(self.current_state, action)
        

    def single_agent_step(self, inp_action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action = copy.deepcopy(inp_action)
        paddle_x_pos = self.current_state['paddles']['paddle_ego']['position'][0]
        paddle_y_pos = self.current_state['paddles']['paddle_ego']['position'][1]
        min_max_limits = get_clip_limits(paddle_x_pos,paddle_y_pos,self.boundary_lims, self.edge_lims)
        paddle_x_min, paddle_x_max,paddle_y_min, paddle_y_max = min_max_limits
        if paddle_x_pos < paddle_x_min + self.paddle_radius:
            action[0] = max(action[0], 0)
        if paddle_x_pos > paddle_x_max - self.paddle_radius:
            action[0] = min(action[0], 0)
        if paddle_y_pos < paddle_y_min + self.paddle_radius:
            action[1] = max(action[1], 0)
        if paddle_y_pos > paddle_y_max - self.paddle_radius:
            action[1] = min(action[1], 0)

        self.last_action = action

        next_state = self.simulator.get_transition(action)
        # print(action, min_max_limits, next_state['paddles']['paddle_ego']['position'], self.paddle_radius)
        if self.current_timestep > 0:
            self.old_state = self.current_state
        self.current_state = next_state
        
        vel_mag = 0.0
        acc_mag = 0.0
        jerk_mag = 0.0

        # Collect motion data
        if 'paddles' in next_state and 'paddle_ego' in next_state['paddles']:
            paddle_data = next_state['paddles']['paddle_ego']
            
            vel_mag = np.linalg.norm(paddle_data['velocity'])
            acc_mag = np.linalg.norm(paddle_data['acceleration']) if 'acceleration' in paddle_data else 0
            jerk_mag = np.linalg.norm(paddle_data['jerk']) if 'jerk' in paddle_data else 0
            
            self.episode_motion_data['velocity_mags'].append(vel_mag)
            self.episode_motion_data['acceleration_mags'].append(acc_mag)
            self.episode_motion_data['jerk_mags'].append(jerk_mag)
        success = self.success_in_ep 

        info = {}
        info['success'] = success
        info['paddle_velocity_mag'] = float(vel_mag)
        info['paddle_acceleration_mag'] = float(acc_mag)
        info['paddle_jerk_mag'] = float(jerk_mag)
        info['paddle_puck_collision_count'] = int(next_state.get('paddle_puck_collision_count', 0))
        info['protective_stop'] = bool(next_state.get('protective_stop', False))
        info['transition_hold_active'] = bool(next_state.get('transition_hold_active', False))
        info['transition_hold_reason'] = str(next_state.get('transition_hold_reason', "none"))
        info['transition_hold_steps_remaining'] = int(next_state.get('transition_hold_steps_remaining', 0))
        info['command_rearm_event'] = bool(next_state.get('command_rearm_event', False))
        info['controller_connected'] = bool(next_state.get('controller_connected', True))
        info['control_program_running'] = bool(next_state.get('control_program_running', True))
        info['robot_step_ready'] = bool(next_state.get('robot_step_ready', True))
        info['robot_command_ready'] = bool(next_state.get('robot_command_ready', True))
        info['command_block_reason'] = str(next_state.get('command_block_reason', "none"))

        hit_a_puck = False
        is_finished, truncated, puck_within_home, puck_within_alt_home, puck_within_goal, _ = self.has_finished(next_state)
        termination_reasons = list(self._last_done_reasons.get("terminated", []))
        truncation_reasons = list(self._last_done_reasons.get("truncated", []))
        episode_end_type = None
        episode_end_reasons = []
        if is_finished:
            episode_end_type = "terminated"
            episode_end_reasons = termination_reasons
        elif truncated:
            episode_end_type = "truncated"
            episode_end_reasons = truncation_reasons
        info['termination_reasons'] = termination_reasons
        info['truncation_reasons'] = truncation_reasons
        info['episode_end_type'] = episode_end_type
        info['episode_end_reasons'] = episode_end_reasons
        info['episode_end_reason'] = episode_end_reasons[0] if len(episode_end_reasons) > 0 else None
        if not truncated:
            reward, success = self.get_base_reward(next_state)
            # scale reward
            reward = reward * self.base_reward_scaling

            # import pdb; pdb.set_trace()
            if not info['success'] and success:
                info['success'] = success
                self.success_in_ep = success
        else:
            reward = self.truncate_rew
        
        if self.use_reward_shaping:
            reward += self.get_reward_shaping(next_state)
        survival_bonus = 0.0
        if self.enable_survival_bonus and (not is_finished) and (not truncated):
            survival_bonus = self.survival_bonus_per_step
            reward += survival_bonus
        info['survival_bonus'] = float(survival_bonus)
        
        self.max_reward_in_single_step = max(self.max_reward_in_single_step, reward)
        self.min_reward_in_single_step = min(self.min_reward_in_single_step, reward)        
        self.episode_return += reward
        self.episode_length += 1

        info['max_reward'] = self.max_reward_in_single_step
        info['min_reward'] = self.min_reward_in_single_step

        if truncated or is_finished:
            info['episode_return'] = self.episode_return
            info['episode_length'] = self.episode_length
            info['motion_data'] = self.episode_motion_data.copy()
            # Reset for next episode
            self.episode_motion_data = {'velocity_mags': [], 'acceleration_mags': [], 'jerk_mags': []}

        self.current_timestep += 1
        
        # # DEBUG STATEMENETS 4 LINES BELOW!
        # is_finished = False
        # truncated = False
        # # only end if timesteps
        # if self.current_timestep >= self.max_timesteps:
        #     is_finished = True

        obs_state = getattr(self.simulator, "observation_state_info", None)
        if obs_state is None:
            obs_state = next_state
            obs_puck_history = self.simulator.puck_history
            obs_paddle_history = self.simulator.paddle_history
        else:
            obs_puck_history = getattr(self.simulator, "observation_puck_history", self.simulator.puck_history)
            obs_paddle_history = getattr(self.simulator, "observation_paddle_history", self.simulator.paddle_history)
        obs = self.get_observation(
            obs_state,
            obs_type=self.obs_type,
            puck_history=obs_puck_history,
            paddle_history=obs_paddle_history,
        )
        info.update(vars(self.simulator_params))
        return obs, reward, is_finished, truncated, info
    
    def multi_step(self, joint_action):
        raise NotImplementedError("Multi-agent step function not implemented yet. But shouldn't take much work, it is mostly copy-pasting. But need to do specific rewards per player")

    def get_joint_reward(self, ego_hit_a_puck, alt_hit_a_puck, 
                         puck_within_ego_home, puck_within_alt_home,
                         puck_within_ego_goal, puck_within_alt_goal):
        NotImplementedError("Joint reward function not implemented yet.")

    def create_world_objects_from_state(self, state_vector):
        # assigns positions to the state components
        # WARNING: in domains with more objects this should be defined differentlym assumes data is from "vel" data type
        name = 'puck_{}'.format(0)
        # puck_pos, puck_vel = state_vector[:2], state_vector[2:4]
        puck_pos, puck_vel = state_vector[4:6], state_vector[6:8]
        self.simulator.spawn_puck(puck_pos, puck_vel, name)

        name = 'paddle_ego'
        # paddle_pos, paddle_vel = state_vector[4:6], state_vector[6:]
        paddle_pos, paddle_vel = state_vector[:2], state_vector[2:4]
        self.simulator.spawn_paddle(paddle_pos, paddle_vel, name)
    

    # functions to adjust base reward scaling
    def set_base_reward_scaling(self, scaling_factor):
        self.base_reward_scaling = scaling_factor

    def multiplicative_scale_base_reward(self, factor):
        self.base_reward_scaling = self.base_reward_scaling * factor

    def get_base_reward_scaling(self):
        return self.base_reward_scaling



def populate_state_info(paddles, pucks, blocks):
        # populates a state infor dictionary based on the components
        # takes in paddles, pucks and blocks as lists
        # TODO: might need to handle NRRs
        state_info = {}
        
        if len(paddles) > 0:
            ego_paddle_x_pos = paddles[0][0]
            ego_paddle_y_pos = paddles[0][1]
            ego_paddle_x_vel = paddles[0][0]
            ego_paddle_y_vel = paddles[0][1]
            
            state_info['paddles'] = {'paddle_ego': {'position': (ego_paddle_x_pos, ego_paddle_y_pos),
                                                    'velocity': (ego_paddle_x_vel, ego_paddle_y_vel)}}
            
        if len(paddles) > 1:
            alt_paddle_x_pos = paddles[1][0]
            alt_paddle_y_pos = paddles[1][1]
            alt_paddle_x_vel = paddles[1][0]
            alt_paddle_y_vel = paddles[1][1]
            
            state_info['paddles']["paddle_alt"] = {'position': (alt_paddle_x_pos, alt_paddle_y_pos),
                                                   'velocity': (alt_paddle_x_vel, alt_paddle_y_vel)}

        if len(blocks) > 0:
            state_info['blocks'] = []
            for b in blocks:
                # block initial positions come as index 2,3
                block_x_pos = b[0]
                block_y_pos = b[1]
                initial_x_pos = b[2]
                initial_y_pos = b[3]

                state_info['blocks'].append({'current_position': (block_x_pos, block_y_pos),
                                        'initial_position': (initial_x_pos, initial_y_pos)})

        if len(pucks) > 0:
            state_info['pucks'] = []
            for p in pucks:
                puck_x_pos = p[0]
                puck_y_pos = p[1]
                puck_x_vel = p[0]
                puck_y_vel = p[1]
                state_info['pucks'].append({'position': (puck_x_pos, puck_y_pos), 
                                'velocity': (puck_x_vel, puck_y_vel)})
        return state_info

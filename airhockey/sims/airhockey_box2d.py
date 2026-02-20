from Box2D.b2 import world, contactListener
from Box2D import (b2CircleShape, b2FixtureDef, b2LoopShape, b2PolygonShape,
                   b2_dynamicBody, b2_staticBody, b2Filter, b2Vec2)
import numpy as np
from collections import deque
import yaml
import inspect
from types import SimpleNamespace
from ..utils import dict_to_namespace

from matplotlib import pyplot as plt

class PIDController:
    """
    PID controller for paddle position control.
    
    The controller computes a force based on position error (P), accumulated error (I),
    and error derivative (D) to smoothly reach target positions.
    """
    def __init__(self, Kp=1000.0, Ki=50.0, Kd=100.0, dt=0.05):
        """
        Initialize PID controller.
        
        Args:
            Kp: Proportional gain (default: 1000.0)
            Ki: Integral gain (default: 50.0)
            Kd: Derivative gain (default: 100.0)
            dt: Time step in seconds (default: 0.05)
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        
        # State variables for integral and derivative terms
        self.integral_error = np.zeros(2)
        self.previous_error = np.zeros(2)
        
        # Anti-windup limits for integral term
        self.integral_limit = 1.0  # Maximum accumulated error (meters * seconds)
    
    def reset(self):
        """Reset the controller state (integral and derivative terms)."""
        self.integral_error = np.zeros(2)
        self.previous_error = np.zeros(2)
    
    def compute(self, target_pos, current_pos, current_vel=None):
        """
        Compute control force based on PID control law.
        
        Args:
            target_pos: Target position (2D numpy array or tuple)
            current_pos: Current position (2D numpy array or tuple)
            current_vel: Current velocity (2D numpy array or tuple, optional)
                        If provided, used for derivative term instead of error derivative
        
        Returns:
            force: 2D numpy array of forces to apply
        """
        # Convert to numpy arrays
        target_pos = np.array(target_pos, dtype=float)
        current_pos = np.array(current_pos, dtype=float)
        
        # Compute position error
        error = target_pos - current_pos
        
        # Proportional term
        P_term = self.Kp * error
        
        # Integral term with anti-windup
        self.integral_error += error * self.dt
        # Clamp integral error to prevent windup
        self.integral_error = np.clip(self.integral_error, 
                                     -self.integral_limit, 
                                     self.integral_limit)
        I_term = self.Ki * self.integral_error
        
        # Derivative term
        if current_vel is not None:
            # Use velocity directly (derivative of position is velocity)
            # We want to dampen velocity, so negative sign
            current_vel = np.array(current_vel, dtype=float)
            D_term = -self.Kd * current_vel
        else:
            # Use error derivative
            error_derivative = (error - self.previous_error) / self.dt
            D_term = self.Kd * error_derivative
        
        # Update previous error
        self.previous_error = error.copy()
        
        # Compute total force
        force = P_term + I_term + D_term
        
        return force

class CollisionForceListener(contactListener):
    def __init__(self, wall_bounce_scale=0.01, wall_tag="table_wall"):
        contactListener.__init__(self)
        self.collision_forces = list()
        self.wall_bounce_scale = wall_bounce_scale
        self.wall_tag = wall_tag
    
    def reset(self):
        del self.collision_forces
        self.collision_forces = list()

    @staticmethod
    def _is_puck(body):
        return body.userData is not None and "puck" in str(body.userData)

    def _is_wall(self, body):
        return body.userData == self.wall_tag

    def PostSolve(self, contact, impulse):
        fixtureA = contact.fixtureA
        fixtureB = contact.fixtureB
        bodyA = fixtureA.body
        bodyB = fixtureB.body
        world_manifold = contact.worldManifold

        # Calculate the forces for each contact point
        for i in range(contact.manifold.pointCount):
            if i < len(impulse.normalImpulses):
                normal_impulse = impulse.normalImpulses[i]
                normal = world_manifold.normal

                self.collision_forces.append({
                    'bodyA': bodyA.userData,
                    'bodyB': bodyB.userData,
                    'normal_force': normal_impulse / 60.0,
                    'contact_normal': (normal.x, normal.y)
                })

                # Apply legacy wall-bounce impulse ONLY for puck-wall collisions.
                is_puck_wall = (self._is_puck(bodyA) and self._is_wall(bodyB)) or (self._is_puck(bodyB) and self._is_wall(bodyA))
                if is_puck_wall:
                    puck_body = bodyA if self._is_puck(bodyA) else bodyB
                    pre_vel = np.array(
                        puck_body.GetLinearVelocityFromWorldPoint(contact.worldManifold.points[i]),
                        dtype=float,
                    )
                    pre_speed = float(np.linalg.norm(pre_vel))
                    normal_np = np.array([normal.x, normal.y], dtype=float)
                    normal_norm = float(np.linalg.norm(normal_np))
                    if normal_norm > 1e-8:
                        projected_vel = float(np.dot(pre_vel, normal_np) / normal_norm)
                        puck_body.ApplyLinearImpulse(
                            normal * self.wall_bounce_scale * projected_vel,
                            puck_body.worldCenter,
                            True,
                        )
                        puck_body.ApplyLinearImpulse(
                            normal * self.wall_bounce_scale / 4,
                            puck_body.worldCenter,
                            True,
                        )

                        # Hard cap: post-processing must not increase puck speed.
                        post_vel = np.array([puck_body.linearVelocity[0], puck_body.linearVelocity[1]], dtype=float)
                        post_speed = float(np.linalg.norm(post_vel))
                        if post_speed > pre_speed + 1e-8 and pre_speed > 1e-8:
                            clipped = post_vel * (pre_speed / post_speed)
                            puck_body.linearVelocity = b2Vec2(float(clipped[0]), float(clipped[1]))

class AirHockeyBox2D:
    def __init__(self, **kwargs):
        defaults = {
            'action_x_scaling': 1.0,
            'action_y_scaling': 1.0,
            'rmax_x': 0.26,
            'rmax_y': 0.12,
            # Real-equivalent workspace and edge shaping limits (base frame).
            'x_min_lim': -0.8,
            'x_max_lim': -0.33,
            'y_min': -0.3582,
            'y_max': 0.350,
            'top_abs': 0.8,
            'bot_abs': 0.1,
            'max_bias_p': -0.15,
            'max_bias_m': -0.15,
            'hist_len': 2,
            'render_masks': False,
            'gravity': -5,
            'paddle_density': 1000,
            'puck_density': 250,
            'block_density': 1000,
            'max_paddle_vel': 2,
            'time_frequency': 20,
            'step_frequency': 20,
            'action_step_lag': 0,
            'paddle_bounds': [],
            'paddle_edge_bounds': [],
            'center_offset_constant': 1.2,
            'puck_restitution': 1.0,
            'action_lag': 0.0,
            'puck_noise': False,
            'puck_noise_std': 0.005,
            # Optional stochastic puck occlusions in observations.
            'enable_random_occlusions': False,
            # Target occluded-frame rate approximation in bad regions:
            # expected_run_length * start_probability ~= target_rate.
            'random_occlusion_target_rate': 0.05,
            # Start probability in non-bad regions.
            'random_occlusion_other_region_rate': 0.01,
            # Run-length weights for lengths 1..N (N also sets max consecutive occlusions).
            'random_occlusion_length_weights': [75, 39, 18, 9, 4, 2, 1],
            # Bad-region bounds in base-centered coordinates (same frame as analysis outputs).
            'random_occlusion_bad_region_box_x': [-0.4, -0.35],
            'random_occlusion_bad_region_box_y': [-0.3, 0.0],
            'random_occlusion_bad_region_right_x': [0.35, 0.8],
        }

        kwargs = {**defaults, **kwargs}
        config = dict_to_namespace(kwargs)
        

        # physics / world params
        self.length, self.width = config.length, config.width
        self.paddle_radius = config.paddle_radius
        self.puck_radius = config.puck_radius
        self.block_width = config.block_width
        self.max_force_timestep = config.max_force_timestep
        self.step_frequency = config.step_frequency # number of action steps per second
        self.time_frequency = config.time_frequency # number of simulation steps per second
        self.time_per_step = 1 / self.time_frequency
        self.force_scaling = config.force_scaling
        self.absorb_target = config.absorb_target
        self.paddle_damping = config.paddle_damping
        self.puck_damping = config.puck_damping
        self.gravity = config.gravity
        self.puck_restitution = config.puck_restitution
        self.puck_min_height = (-config.length / 2) + (config.length / 3)
        self.paddle_max_height = 0
        self.block_min_height = 0
        self.max_speed_start = config.width
        self.min_speed_start = -config.width
        self.paddle_density = config.paddle_density
        self.puck_density = config.puck_density
        self.block_density = config.block_density
        self.action_x_scaling = config.action_x_scaling
        self.action_y_scaling = config.action_y_scaling
        self.rmax_x = config.rmax_x
        self.rmax_y = config.rmax_y
        self.move_lims = np.array([self.rmax_x, self.rmax_y], dtype=float)
        self.x_min_lim = config.x_min_lim
        self.x_max_lim = config.x_max_lim
        self.y_min = config.y_min
        self.y_max = config.y_max
        self.top_abs = config.top_abs
        self.bot_abs = config.bot_abs
        self.max_bias_p = config.max_bias_p
        self.max_bias_m = config.max_bias_m
        self.lims = (self.x_min_lim, self.x_max_lim, self.y_min, self.y_max)
        # Keep tuple ordering consistent with real environment construction.
        self.edge_lims = (self.top_abs, self.bot_abs, self.max_bias_p, self.max_bias_m)
        self.hist_len = config.hist_len
        self.center_offset_constant = config.center_offset_constant
        self.wall_bounce_scale = config.wall_bounce_scale
        self.action_lag = config.action_lag 
        assert self.action_lag >= 0 and self.action_lag <= 1, "Action lag must be between 0 and 1"
        self.puck_noise = config.puck_noise
        self.puck_noise_std = float(config.puck_noise_std)
        # random occlusion simulation
        self.enable_random_occlusions = bool(config.enable_random_occlusions)
        self.random_occlusion_target_rate = float(config.random_occlusion_target_rate)
        self.random_occlusion_other_region_rate = float(config.random_occlusion_other_region_rate)
        self.random_occlusion_length_weights = np.array(config.random_occlusion_length_weights, dtype=float).reshape(-1)
        self.random_occlusion_bad_region_box_x = tuple(sorted([float(v) for v in config.random_occlusion_bad_region_box_x]))
        self.random_occlusion_bad_region_box_y = tuple(sorted([float(v) for v in config.random_occlusion_bad_region_box_y]))
        self.random_occlusion_bad_region_right_x = tuple(sorted([float(v) for v in config.random_occlusion_bad_region_right_x]))

        if self.random_occlusion_length_weights.size == 0:
            self.random_occlusion_length_weights = np.array([1.0], dtype=float)
        self._occlusion_max_run = int(self.random_occlusion_length_weights.size)
        run_lengths = np.arange(1, self._occlusion_max_run + 1, dtype=float)
        weight_sum = float(np.sum(self.random_occlusion_length_weights))
        if weight_sum <= 0:
            weight_sum = 1.0
        self._occlusion_expected_run = float(np.sum(run_lengths * self.random_occlusion_length_weights) / weight_sum)
        if self._occlusion_expected_run <= 0:
            self._occlusion_expected_run = 1.0
        self.random_occlusion_bad_region_rate = float(
            np.clip(self.random_occlusion_target_rate / self._occlusion_expected_run, 0.0, 1.0)
        )
        self._occlusion_run_remaining = {}
        self._occlusion_last_visible_base = {}
        self._occlusion_prev_occluded = {}

        self.last_action = np.zeros(2) # keep the last action taken, used for action lag
        self.last_target_position = None  # base-frame target used for visualization/debugging
        self.previous_acceleration = np.zeros(2)  # for jerk calculation
        self.jerk = np.zeros(2)
        self.pose_hist = deque(maxlen=self.hist_len)
        self.dpose_hist = deque(maxlen=self.hist_len)
        
        # Initialize PID controller with configurable gains
        # Default values are tuned for the air hockey environment
        pid_kp = kwargs.get('pid_kp', 1000.0)
        pid_ki = kwargs.get('pid_ki', 50.0)
        pid_kd = kwargs.get('pid_kd', 100.0)
        self.use_pid = kwargs.get('use_pid', False)  # Flag to enable/disable PID control
        self.pid_controller = PIDController(Kp=pid_kp, Ki=pid_ki, Kd=pid_kd, dt=self.time_per_step)

        # these assume 2d, in 3d since we have height it would be higher mass
        self.paddle_mass = self.paddle_density * np.pi * self.paddle_radius ** 2
        self.puck_mass = self.puck_density * np.pi * self.puck_radius ** 2
        self.chump_dict = {}
        # these 2 will depend on the other parameters
        self.max_paddle_vel = config.max_paddle_vel # m/s. This will be dependent on the robot arm
        # compute maximum force based on max paddle velocity
        max_a = self.max_paddle_vel / self.time_per_step
        max_f = self.paddle_mass * max_a
        # assume maximum force transfer
        puck_max_a = max_f / self.puck_mass
        self.max_puck_vel = puck_max_a * self.time_per_step
        self.world = world(gravity=(0, self.gravity), doSleep=True) # gravity is negative usually

        # box2d visualization params (but the visualization is done in the Render file)
        self.ppm = config.render_size / self.width
        self.render_width = int(config.render_size)
        self.render_length = int(self.ppm * self.length)
        self.render_masks = config.render_masks

        self.table_x_min = -self.width / 2
        self.table_x_max = self.width / 2
        self.table_y_min = -self.length / 2
        self.table_y_max = self.length / 2

        self.min_goal_radius = self.width / 16
        self.max_goal_radius = self.width / 4

        self.metadata = {}

        # creating the ground -- need to only call once! otherwise it can be laggy
        self.ground_body = self.world.CreateBody(
            shapes=b2LoopShape(vertices=[(self.table_x_min, self.table_y_min),
                                         (self.table_x_min, self.table_y_max),
                                         (self.table_x_max, self.table_y_max),
                                         (self.table_x_max, self.table_y_min)]),
            userData="table_wall",
        )
        # self.ground_body.fixtures[0].friction = 0.0
        self.reset(config.seed)

        # Initialize the contact listener
        self.collision_listener = CollisionForceListener(
            wall_bounce_scale=self.wall_bounce_scale,
            wall_tag="table_wall",
        )
        self.world.contactListener = self.collision_listener
        self.total_timesteps = 0
        from cProfile import Profile
        from pstats import SortKey, Stats
        self.profiler = Profile()

    def start_callbacks(self, **kwargs):
        return

    @staticmethod
    def from_dict(state_dict):
        # create a dictionary of only the relevant parameters
        return AirHockeyBox2D(**state_dict)

    def reset(self, seed, **kwargs):
        self.rng = np.random.RandomState(seed)
        self.timestep = 0

        if hasattr(self, "object_dict"):
            for body in self.object_dict.values():
                self.world.DestroyBody(body)

        if type(self.gravity) == list:
            self.world.gravity = (0, self.rng.uniform(low=self.gravity[0], high=self.gravity[1]))
        
        if hasattr(self, "collision_listener"): self.collision_listener.reset()

        self.paddles = dict()
        self.pucks = dict()
        self.blocks = dict()
        self.block_initial_positions = dict()
        self.obstacles = dict()
        self.targets = dict()
        
        self.multiagent = False

        self.puck_history = list()
        self.paddle_history = list()
        self.paddle_attrs = None
        self.target_attrs = None

        self.object_dict = dict()
        self.last_action = np.zeros(2) # keep the last action taken
        self.last_target_position = None
        self.previous_acceleration = np.zeros(2)  # reset for jerk calculation
        self.jerk = np.zeros(2)
        self.pose_hist = deque(maxlen=self.hist_len)
        self.dpose_hist = deque(maxlen=self.hist_len)
        self._occlusion_run_remaining = {}
        self._occlusion_last_visible_base = {}
        self._occlusion_prev_occluded = {}
        
        # Reset PID controller
        if hasattr(self, 'pid_controller'):
            self.pid_controller.reset()
        
        state_info = self.get_current_state()
        return state_info
    
    def set_object_links(self):
        # set up object names
        self.paddle_names = list(self.paddles.keys())
        if "paddle_ego_acceleration" in self.paddle_names: self.paddle_names.pop(self.paddle_names.index("paddle_ego_acceleration"))
        if "paddle_ego_force" in self.paddle_names: self.paddle_names.pop(self.paddle_names.index("paddle_ego_force"))
        if "paddle_ego_jerk" in self.paddle_names: self.paddle_names.pop(self.paddle_names.index("paddle_ego_jerk"))

        
        self.puck_names = list(self.pucks.keys())
        self.block_names = list(self.blocks.keys())

        self.paddle_names.sort()
        self.puck_names.sort()
        self.block_names.sort()

        
        # TODO: obstacles and targets not implemented
        # self.obstacle_names = [self.obstacles.keys()]
        # self.target_names = [self.targets.keys()]

    
    def convert_from_box2d_coords(self, state_info):
        # traverse through state_info until we find tuple, then correct
        for key, value in state_info.items():
            if type(value) == list:
                for i in range(len(value)):
                    for key2, value2 in value[i].items():
                        if type(value2) == tuple:
                            state_info[key][i][key2] = (-value2[1], value2[0])
            else:
                for key2, value2 in value.items():
                    for key3, value3 in value2.items():
                        state_info[key][key2][key3] = (-value3[1], value3[0])
        return state_info
    
    def base_coord_to_box2d(self, coord):
        return (coord[1], -coord[0])
    
    def get_current_state(self):

        state_info = {}
        
        if 'paddle_ego' in self.paddles:
            ego_paddle_x_pos = self.paddles['paddle_ego'].position[0]
            ego_paddle_y_pos = self.paddles['paddle_ego'].position[1]
            ego_paddle_x_vel = self.paddles['paddle_ego'].linearVelocity[0]
            ego_paddle_y_vel = self.paddles['paddle_ego'].linearVelocity[1]
            ego_paddle_x_acc = self.paddles['paddle_ego_acceleration'][0]
            ego_paddle_y_acc = self.paddles['paddle_ego_acceleration'][1]
            ego_paddle_x_force = self.paddles['paddle_ego_force'][0]
            ego_paddle_y_force = self.paddles['paddle_ego_force'][1]
            ego_paddle_x_jerk = self.paddles['paddle_ego_jerk'][0]
            ego_paddle_y_jerk = self.paddles['paddle_ego_jerk'][1]
            
            state_info['paddles'] = {'paddle_ego': {'position': (ego_paddle_x_pos, ego_paddle_y_pos),
                                                    'velocity': (ego_paddle_x_vel, ego_paddle_y_vel),
                                                    'acceleration': (ego_paddle_x_acc, ego_paddle_y_acc),
                                                    'force': (ego_paddle_x_force, ego_paddle_y_force),
                                                    'jerk': (ego_paddle_x_jerk, ego_paddle_y_jerk)
                                                    }}

        if 'paddle_alt' in self.paddles:
            alt_paddle_x_pos = self.paddles['paddle_alt'].position[0]
            alt_paddle_y_pos = self.paddles['paddle_alt'].position[1]
            alt_paddle_x_vel = self.paddles['paddle_alt'].linearVelocity[0]
            alt_paddle_y_vel = self.paddles['paddle_alt'].linearVelocity[1]
            
            state_info['paddles']['paddle_alt'] = {'position': (alt_paddle_x_pos, alt_paddle_y_pos),
                                                   'velocity': (alt_paddle_x_vel, alt_paddle_y_vel)}

        if len(self.blocks) > 0:
            state_info['blocks'] = []
            for block_name in self.blocks:
                block_x_pos = self.blocks[block_name].position[0]
                block_y_pos = self.blocks[block_name].position[1]
                initial_x_pos = self.block_initial_positions[block_name][0]
                initial_y_pos = self.block_initial_positions[block_name][1]

                state_info['blocks'].append({'current_position': (block_x_pos, block_y_pos),
                                        'initial_position': (initial_x_pos, initial_y_pos)})

        if len(self.pucks) > 0:
            state_info['pucks'] = []
            for puck_name in self.pucks:
                puck_x_pos_true = self.pucks[puck_name].position[0]
                puck_y_pos_true = self.pucks[puck_name].position[1]
                puck_x_pos_true, puck_y_pos_true = self._get_noisy_puck_position((puck_x_pos_true, puck_y_pos_true))
                puck_base_xy_true = self._box2d_to_base_coords((puck_x_pos_true, puck_y_pos_true))
                occluded, puck_base_xy_observed = self._update_random_occlusion(puck_name, puck_base_xy_true)
                puck_box2d_observed = self._base_to_box2d_coords(puck_base_xy_observed)
                puck_x_vel = self.pucks[puck_name].linearVelocity[0]
                puck_y_vel = self.pucks[puck_name].linearVelocity[1]
                state_info['pucks'].append({'position': (float(puck_box2d_observed[0]), float(puck_box2d_observed[1])),
                                'velocity': (puck_x_vel, puck_y_vel),
                                'occluded': int(occluded)})

        return self.convert_from_box2d_coords(state_info)

    def _get_noisy_puck_position(self, position):
        if not self.puck_noise:
            return position
        noise = self.rng.normal(loc=0.0, scale=self.puck_noise_std, size=2)
        noisy_position = np.array(position, dtype=float) + noise
        return float(noisy_position[0]), float(noisy_position[1])

    def _is_in_bad_occlusion_region_base(self, puck_base_xy):
        x = float(puck_base_xy[0])
        y = float(puck_base_xy[1])
        box_x_min, box_x_max = self.random_occlusion_bad_region_box_x
        box_y_min, box_y_max = self.random_occlusion_bad_region_box_y
        right_x_min, right_x_max = self.random_occlusion_bad_region_right_x

        in_box_region = (box_x_min <= x <= box_x_max) and (box_y_min <= y <= box_y_max)
        in_right_strip = (right_x_min <= x <= right_x_max)
        return in_box_region or in_right_strip

    def _sample_occlusion_run_length(self):
        lengths = np.arange(1, self._occlusion_max_run + 1, dtype=int)
        weights = np.maximum(self.random_occlusion_length_weights, 0.0)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0:
            return 1
        probs = weights / weight_sum
        return int(self.rng.choice(lengths, p=probs))

    def _update_random_occlusion(self, puck_name, true_puck_base_xy):
        if not self.enable_random_occlusions:
            self._occlusion_last_visible_base[puck_name] = (
                float(true_puck_base_xy[0]),
                float(true_puck_base_xy[1]),
            )
            self._occlusion_prev_occluded[puck_name] = False
            return False, np.array(true_puck_base_xy, dtype=float)

        remaining = int(self._occlusion_run_remaining.get(puck_name, 0))
        prev_occluded = bool(self._occlusion_prev_occluded.get(puck_name, False))
        if remaining > 0:
            self._occlusion_run_remaining[puck_name] = remaining - 1
            observed = self._occlusion_last_visible_base.get(
                puck_name, (-2.0 + self.center_offset_constant, 0.0)
            )
            self._occlusion_prev_occluded[puck_name] = True
            return True, np.array(observed, dtype=float)

        # Force at least one visible frame between occlusion runs so max consecutive
        # occluded frames is strictly capped by the sampled run length (<= 7 by default).
        if prev_occluded:
            self._occlusion_last_visible_base[puck_name] = (
                float(true_puck_base_xy[0]),
                float(true_puck_base_xy[1]),
            )
            self._occlusion_prev_occluded[puck_name] = False
            return False, np.array(true_puck_base_xy, dtype=float)

        in_bad_region = self._is_in_bad_occlusion_region_base(true_puck_base_xy)
        start_prob = self.random_occlusion_bad_region_rate if in_bad_region else self.random_occlusion_other_region_rate
        if self.rng.uniform(0.0, 1.0) < start_prob:
            run_len = self._sample_occlusion_run_length()
            self._occlusion_run_remaining[puck_name] = max(run_len - 1, 0)
            observed = self._occlusion_last_visible_base.get(
                puck_name, (-2.0 + self.center_offset_constant, 0.0)
            )
            self._occlusion_prev_occluded[puck_name] = True
            return True, np.array(observed, dtype=float)

        self._occlusion_last_visible_base[puck_name] = (
            float(true_puck_base_xy[0]),
            float(true_puck_base_xy[1]),
        )
        self._occlusion_prev_occluded[puck_name] = False
        return False, np.array(true_puck_base_xy, dtype=float)
    
    def instantiate_objects(self):
        pass # we don't need to do anything here
    
    def spawn_paddle(self, pos, vel, name, affected_by_gravity=False, movable=True):
        assert name == 'paddle_ego' or name == 'paddle_alt'
        pos = self.base_coord_to_box2d(pos)
        vel = self.base_coord_to_box2d(vel)
        radius = self.paddle_radius
        paddle = self.world.CreateDynamicBody(
            fixtures=b2FixtureDef(
                shape=b2CircleShape(radius=radius),
                density=self.paddle_density,
                restitution = 1.0,
                filter=b2Filter (maskBits=1,
                                 categoryBits=1)),
            bullet=True,
            position=pos,
            linearVelocity=vel,
            linearDamping=self.paddle_damping
        )
        if not affected_by_gravity:
            paddle.gravityScale = 0
        
        self.paddles[name] = paddle
        if name == "paddle_ego":
            self.paddles['paddle_ego_acceleration'] = (0, 0)
            self.paddles['paddle_ego_force'] = (0, 0)
            self.paddles['paddle_ego_jerk'] = (0, 0)
        self.object_dict[name] = paddle
        self.paddle_history += [(-2 + self.center_offset_constant,0,1) for i in range(5)]
        
        if 'paddle_ego' in self.paddles and 'paddle_alt' in self.paddles:
            self.multiagent = True
    
    def spawn_puck(self, pos, vel, name, affected_by_gravity=True, movable=True):
        pos = self.base_coord_to_box2d(pos)
        vel = self.base_coord_to_box2d(vel)
        radius = self.puck_radius
        puck = self.world.CreateDynamicBody(
            fixtures=b2FixtureDef(
                shape=b2CircleShape(radius=radius),
                density=self.puck_density,
                restitution = self.puck_restitution,
                filter=b2Filter (maskBits=1,
                                 categoryBits=1),
                friction=0.0),
            bullet=True,
            position=pos,
            linearVelocity=vel,
            linearDamping=self.puck_damping,
            angularDamping=100000,
            userData=name
        )
        if not affected_by_gravity:
            puck.gravityScale = 0
        self.pucks[name] = puck
        self.object_dict[name] = puck
        self.puck_history += [(-2 + self.center_offset_constant,0,1) for i in range(5)]
        
    def spawn_block(self, pos, vel, name, affected_by_gravity=False, movable=True):
        pos = self.base_coord_to_box2d(pos)
        vel = self.base_coord_to_box2d(vel)
        vertices = [([-self.block_width / 2, -self.block_width / 2]), ([self.block_width / 2, -self.block_width / 2]), ([self.block_width / 2, self.block_width / 2]), ([-self.block_width / 2, self.block_width / 2])]
        block = self.world.CreateDynamicBody(
            fixtures=b2FixtureDef(
                shape=b2PolygonShape(vertices=vertices),
                density=self.block_density,
                restitution=1.0,
                filter=b2Filter(maskBits=1, categoryBits=1)),
            bullet=True,
            position=pos,
            linearVelocity=vel,
            linearDamping=self.puck_damping
        )
        if not affected_by_gravity:
            block.gravityScale = 0
        self.blocks[name] = block
        self.block_initial_positions[name] = pos
        self.object_dict[name] = block

    def convert_to_box2d_coords(self, action):
        action = np.array((action[1], -action[0]))
        return action

    def _box2d_to_base_coords(self, coord):
        return np.array((-coord[1], coord[0]), dtype=float)

    def _base_to_box2d_coords(self, coord):
        return np.array((coord[1], -coord[0]), dtype=float)

    def _get_edge(self, x, y, w, h):
        # Mirror the real environment's rectangular projection with numerical guards.
        if np.abs(x) <= w and np.abs(y) <= h:
            return np.array([x, y], dtype=float)

        eps = 1e-8
        if np.abs(x) < eps:
            return np.array([0.0, np.sign(y) * h], dtype=float)
        if np.abs(y) < eps:
            return np.array([np.sign(x) * w, 0.0], dtype=float)

        s = y / x
        if -h / 2 <= s * w / 2 <= h / 2:
            if x > 0:
                return np.array([w, s * w], dtype=float)
            return np.array([-w, -s * w], dtype=float)

        s_r = x / y
        if y > 0:
            return np.array([h * s_r, h], dtype=float)
        return np.array([-h * s_r, -h], dtype=float)

    def _clip_limits(self, x, y):
        # Mirror real coordinate_transform.clip_limits behavior.
        # Box2D/base observations use centered x, while real clip limits are in raw robot x.
        # Convert to raw-x for clipping, then shift back to centered frame.
        x_min_lim, x_max_lim, y_min, y_max = self.lims
        top_abs, bot_abs, max_bias_m, max_bias_p = self.edge_lims
        x_raw = x - self.center_offset_constant
        y = np.clip(y, y_min, y_max)
        x_min = x_min_lim
        x_max = min(x_max_lim, max_bias_m - top_abs * y, max_bias_p + top_abs * y)
        x_raw = np.clip(x_raw, x_min, x_max)
        x_centered = x_raw + self.center_offset_constant
        return np.array([x_centered, y], dtype=float)

    def _clip_pid_target_to_workspace(self, target_pos):
        """Clip PID target with real-equivalent workspace + edge limits."""
        target_base = self._box2d_to_base_coords(target_pos)
        clipped_base = self._clip_limits(target_base[0], target_base[1])
        return self._base_to_box2d_coords(clipped_base)

    def _filter_update(self):
        """
        Real-equivalent low-pass update on commanded pose deltas.
        desired = current_pose + mean(desired_i - current_i) over history window.
        """
        pose_vel = np.array(self.dpose_hist[-1], dtype=float) - np.array(self.pose_hist[-1], dtype=float)
        transform_vel = pose_vel
        if len(self.dpose_hist) > 1:
            pose_vels = [
                np.array(self.dpose_hist[i], dtype=float) - np.array(self.pose_hist[i], dtype=float)
                for i in range(len(self.dpose_hist))
            ]
            transform_vel = np.mean(pose_vels, axis=0)
        return np.array(self.pose_hist[-1], dtype=float) + transform_vel

    def _compute_pid_target_pos(self, pos, act):
        """
        Mirror real-env target logic for PID in base frame:
        - scale normalized action by move limits
        - construct raw target from current pose + move vector
        - project with a rectangular per-step movement bound
        - clip to workspace + edge bounds
        """
        pos_base = self._box2d_to_base_coords(pos)
        act_base = self._box2d_to_base_coords(act)
        move_vector_base = np.array(act_base, dtype=float) * self.move_lims
        target_raw_base = pos_base + move_vector_base
        rel_base = target_raw_base - pos_base
        edge_rel = self._get_edge(rel_base[0], rel_base[1], self.move_lims[0], self.move_lims[1])
        target_rect_base = pos_base + edge_rel
        target_clipped_base = self._clip_limits(target_rect_base[0], target_rect_base[1])
        return self._base_to_box2d_coords(target_clipped_base)

    # s, a -> s'
    def get_transition(self, action, other_action=None):
        if self.multiagent:
            return self.get_multiagent_transition(action, other_action)
        else:
            action = self.convert_to_box2d_coords(action)
            return self.get_singleagent_transition(action)
    
    def get_singleagent_transition(self, action):

        action_list = [np.copy(self.last_action), np.copy(action)]
        time_to_sim = [self.time_per_step * self.action_lag, self.time_per_step * (1 - self.action_lag)]
        
        # Store initial velocity for acceleration calculation over the entire step
        initial_vel = np.array([self.paddles['paddle_ego'].linearVelocity[0], self.paddles['paddle_ego'].linearVelocity[1]])

        for act, sim_time in zip(action_list, time_to_sim):
            pos = np.array([self.paddles['paddle_ego'].position[0], self.paddles['paddle_ego'].position[1]])
            
            # Boundary constraint: prevent paddle from going into opponent's side
            if pos[1] > 0 - 3 * self.paddle_radius:
                act[1] = min(act[1], 0)
            
            # Compute force using either PID controller or legacy controller
            if self.use_pid:
                # PID controller target uses real-like scaled delta + rect projection + clipping.
                target_pos = self._compute_pid_target_pos(pos, act)
                self.pose_hist.append(np.array(pos, dtype=float))
                self.dpose_hist.append(np.array(target_pos, dtype=float))
                target_pos = self._filter_update()
                self.last_target_position = self._box2d_to_base_coords(target_pos)
                current_vel = np.array([self.paddles['paddle_ego'].linearVelocity[0], 
                                       self.paddles['paddle_ego'].linearVelocity[1]])
                
                # Compute force using PID controller
                force = self.pid_controller.compute(target_pos, pos, current_vel)
                
            else:
                self.last_target_position = None
                # Legacy controller: action is delta position
                # let's use simple time-optimal control to figure out the force to apply
                delta_pos = np.array([act[0], act[1]])
                current_vel = np.array([self.paddles['paddle_ego'].linearVelocity[0], 
                                       self.paddles['paddle_ego'].linearVelocity[1]])
                
                # first let's determine velocity
                vel = delta_pos / sim_time
                vel_mag = np.linalg.norm(vel)
                vel_unit = vel / (vel_mag + 1e-8)

                # first clipping
                if vel_mag > self.max_paddle_vel:
                    vel = vel_unit * self.max_paddle_vel

                force = self.paddles['paddle_ego'].mass * vel / sim_time

            # clipping/normalization applies to both controllers
            force_mag = np.linalg.norm(force)
            force_unit = force / (force_mag + 1e-8)

            # second clipping
            if force_mag > self.max_force_timestep:
                force = force_unit * self.max_force_timestep

            if self.force_scaling > 0:
                force = force * self.force_scaling
            force = force.astype(float)
            if self.paddles['paddle_ego'].position[1] > 0: 
                new_force = self.force_scaling * self.paddles['paddle_ego'].mass * act[1]
                if new_force < -self.max_force_timestep:
                    new_force = -self.max_force_timestep
                force[1] = min(new_force, 0)
            else:
                force = force * np.array([self.action_x_scaling, self.action_y_scaling])
            
            # Clip force to maximum allowed
            force_mag = np.linalg.norm(force)
            if force_mag > self.max_force_timestep:
                force = force / force_mag * self.max_force_timestep
            
            # Apply force to paddle
            if 'paddle_ego' in self.paddles:
                self.paddles['paddle_ego'].ApplyForceToCenter(force.astype(float), True)

            self.world.Step(sim_time, 100, 100)
            
            # correct blocks for t=0
            if self.timestep == 0 and len(self.blocks) > 0:
                for block_name in self.blocks:
                    block = self.blocks[block_name]
                    x, y = self.block_initial_positions[block_name]
                    block.position = (x, y)
            
            vel = np.array([self.paddles['paddle_ego'].linearVelocity[0], self.paddles['paddle_ego'].linearVelocity[1]])
            vel_mag = np.linalg.norm(vel)

            # keep velocity at a maximum value
            if vel_mag > self.max_paddle_vel:
                self.paddles['paddle_ego'].linearVelocity = b2Vec2(vel[0] / vel_mag * self.max_paddle_vel, vel[1] / vel_mag * self.max_paddle_vel)
                
            # check if out of bounds and correct
            pos = np.array([self.paddles['paddle_ego'].position[0], self.paddles['paddle_ego'].position[1]], dtype=float)
            if self.use_pid:
                pos = self._clip_pid_target_to_workspace(pos)
            else:
                if pos[0] < self.table_x_min:
                    pos[0] = self.table_x_min
                if pos[0] > self.table_x_max:
                    pos[0] = self.table_x_max
                if pos[1] > 0:
                    pos[1] = 0
                if pos[1] > self.table_y_max:
                    pos[1] = self.table_y_max
            self.paddles['paddle_ego'].position = (pos[0], pos[1])
            
            state_info = self.get_current_state()
            if 'pucks' in state_info:
                for puck in state_info['pucks']:
                    self.puck_history.append(list(puck["position"]) + [int(puck.get("occluded", 0))])
            else:
                for i in range(len(self.pucks.keys())):
                    self.puck_history.append([-2 + self.center_offset_constant,0,1])
            
            if 'paddles' in state_info:
                for paddle_name, paddle_data in state_info['paddles'].items():
                    self.paddle_history.append(list(paddle_data["position"]) + [0])
            else:
                for i in range(len(self.paddles.keys())):
                    if 'paddle_ego_acceleration' not in self.paddles or 'paddle_ego_force' not in self.paddles or 'paddle_ego_jerk' not in self.paddles:
                        self.paddle_history.append([-2 + self.center_offset_constant,0,1])
            
            total_force = np.array(force)

            collision_forces = self.get_collision_forces()
            for collision in collision_forces:
                if collision['bodyA'] == 'paddle_ego':
                    total_force[0] += collision['normal_force'] * collision['contact_normal'][0]
                    total_force[1] += collision['normal_force'] * collision['contact_normal'][1]
                elif collision['bodyB'] == 'paddle_ego':
                    total_force[0] -= collision['normal_force'] * collision['contact_normal'][0]
                    total_force[1] -= collision['normal_force'] * collision['contact_normal'][1]

                self.paddles['paddle_ego_force'] = total_force
        
        # Calculate acceleration and jerk AFTER the entire action step
        # Use the total time step and initial velocity for proper acceleration calculation
        final_vel = np.array([self.paddles['paddle_ego'].linearVelocity[0], self.paddles['paddle_ego'].linearVelocity[1]])
        current_acceleration = (final_vel - initial_vel) / self.time_per_step
        self.paddles['paddle_ego_acceleration'] = current_acceleration
        
        # Calculate jerk (derivative of acceleration)
        self.jerk = (current_acceleration - self.previous_acceleration) / self.time_per_step
        self.paddles['paddle_ego_jerk'] = self.jerk
        self.previous_acceleration = current_acceleration.copy()
        
        # Debug: Print acceleration values occasionally (remove this after testing)
        # if self.timestep % 100 == 0 and np.linalg.norm(current_acceleration) > 0:
        #     print(f"Debug - Timestep {self.timestep}: initial_vel={initial_vel}, final_vel={final_vel}, time_per_step={self.time_per_step}")
        #     print(f"Debug - Acceleration: {current_acceleration}, magnitude: {np.linalg.norm(current_acceleration):.6f}")
        
        self.timestep += 1
        self.last_action = action

        return state_info
    
    def get_multiagent_transition(self, joint_action):
        raise NotImplementedError

    def get_contacts(self):
        contacts = list()
        shape_pointers = ([self.paddles[bn] for bn in self.paddles.keys()]  + \
                         [self.pucks[bn] for bn in self.pucks.keys()] + [self.blocks[pn] for pn in self.blocks.keys()])
                        #  [self.obstacles[pn][0] for pn in self.obstacles.keys()] + [self.targets[pn][0] for pn in self.targets.keys()])
        names = self.paddle_names + self.puck_names + self.block_names# + self.obstacle_names + self.target_names
        contact_names = {n: list() for n in names}
        for bn in names:
            all_contacts = np.zeros(len(shape_pointers)).astype(bool)
            for contact in self.object_dict[bn].contacts:
                if contact.contact.touching:
                    contact_bool = np.array([(contact.other == bp and contact.contact.touching) for bp in shape_pointers])
                    contact_names[bn] += [sn for sn, bp in zip(names, shape_pointers) if (contact.other == bp)]
                else:
                    contact_bool = np.zeros(len(shape_pointers)).astype(bool)
                all_contacts += contact_bool
            contacts.append(all_contacts)
        return np.stack(contacts, axis=0), contact_names

    def respond_contacts(self, contact_names):
        hit_a_puck = list()
        for tn in self.target_names:
            for cn in contact_names[tn]: 
                if cn.find("puck") != -1:
                    hit_a_puck.append(cn)
        if self.absorb_target:
            for cn in hit_a_puck:
                self.world.DestroyBody(self.object_dict[cn])
                del self.object_dict[cn]
        return hit_a_puck # TODO: record a destroyed flag

    def get_collision_forces(self):
        # Extract forces from the collision listener
        return self.collision_listener.collision_forces
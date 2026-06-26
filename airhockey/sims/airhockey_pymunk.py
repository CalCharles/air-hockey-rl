"""Pymunk physics backend for AirHockey.

Internal coordinate convention: identical to AirHockeyBox2D (rotated 90° from base frame):
  box2d_x = base_y,  box2d_y = -base_x

All body positions / velocities are stored in this box2d frame. ``get_current_state()``
converts to base frame via ``convert_from_box2d_coords()``, matching AirHockeyBox2D output
byte-for-byte.  This means AirHockeyBaseEnv, task classes, renderers, and wrappers all
work without modification when ``simulator="pymunk"`` is selected.
"""
from __future__ import annotations

import copy
from collections import deque
from types import SimpleNamespace

import numpy as np

import pymunk

from ..utils import dict_to_namespace
from ..observation_homography import make_sine_y_warp_fn
from .airhockey_box2d import PIDController, _COLLISION_TIERS, _make_empty_tier_stats

# ---------------------------------------------------------------------------
# Collision-type tags (pymunk shape.collision_type)
# ---------------------------------------------------------------------------
_CT_WALL = 0
_CT_DYNAMIC = 1   # paddles, pucks, blocks, obstacles


class AirHockeyPymunk:
    """Pymunk drop-in for AirHockeyBox2D.

    The public interface (attributes, method signatures, return-value formats) is
    identical to AirHockeyBox2D so AirHockeyBaseEnv and all task/wrapper classes
    work without changes.
    """

    @staticmethod
    def from_dict(state_dict: dict) -> "AirHockeyPymunk":
        return AirHockeyPymunk(**state_dict)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, **kwargs):
        explicit_delay_seconds = "delay_seconds" in kwargs
        explicit_action_lag = "action_lag" in kwargs
        defaults = {
            "action_x_scaling": 1.0,
            "action_y_scaling": 1.0,
            "rmax_x": 0.26,
            "rmax_y": 0.12,
            "x_min_lim": -0.85,
            "x_max_lim": -0.45,
            "y_min": -0.37,
            "y_max": 0.37,
            "top_abs": 0.8,
            "bot_abs": 0.1,
            "max_bias_p": -0.15,
            "max_bias_m": -0.15,
            "hist_len": 2,
            "render_masks": False,
            "gravity": -5,
            "paddle_density": 1000,
            "paddle_mass_reference_radius": None,
            "puck_density": 250,
            "puck_mass_reference_radius": None,
            "block_density": 1000,
            "max_paddle_vel": 2,
            "time_frequency": 20,
            "step_frequency": 20,
            "action_step_lag": 0,
            "paddle_bounds": [],
            "paddle_edge_bounds": [],
            "center_offset_constant": 1.2,
            "absorb_target": False,
            "puck_restitution": 1.0,
            "paddle_restitution": 1.0,
            "side_wall_restitution": 0.99,
            "end_wall_restitution": 0.7,
            "puck_wall_restitution_threshold_speed": 0.25,
            "puck_wall_min_rebound_speed_below_threshold": 0.1,
            "enable_action_delay": False,
            "enable_observation_delay": False,
            "delay_seconds": 0.025,
            "action_lag": 0.0,
            "puck_noise": False,
            "puck_noise_std": 0.005,
            "enable_random_occlusions": False,
            "random_occlusion_rate": 0.05,
            "random_occlusion_length_weights": [75, 39, 18, 9, 4, 2, 1],
            "derivative_min_dt": 1e-6,
            "acceleration_ema_alpha": 0.35,
            "jerk_ema_alpha": 0.35,
            "max_acceleration_norm": 0.0,
            "max_jerk_norm": 0.0,
            "puck_obs_sine_warp_amplitude": 0.0,
            "puck_obs_sine_warp_y_left": None,
            "puck_obs_sine_warp_y_right": None,
            "enable_fixed_state_velocity_jerk": False,
            "fixed_state_paddle_velocity": (0.0, 0.0),
            "fixed_state_paddle_jerk": (0.0, 0.0),
            "fixed_state_puck_velocity": (0.0, 0.0),
            "mask_puck_velocity": True,
            "enable_puck_delay_interpolation": False,
            "puck_delay_interpolation_min": 0.75,
            "puck_delay_interpolation_max": 1.25,
            "triangle_obstacle_size": 0.08,
            "triangle_obstacle_restitution": 1.15,
            "obstacle_shape": "triangle",
        }
        kwargs = {**defaults, **kwargs}
        config = dict_to_namespace(kwargs)

        # --- warp / fixed-state ---
        warp_amp = float(config.puck_obs_sine_warp_amplitude)
        warp_y_left = config.puck_obs_sine_warp_y_left
        warp_y_right = config.puck_obs_sine_warp_y_right
        if warp_y_left is None:
            warp_y_left = -float(config.width) / 2.0
        if warp_y_right is None:
            warp_y_right = float(config.width) / 2.0
        self.puck_obs_warp_fn = make_sine_y_warp_fn(warp_amp, float(warp_y_left), float(warp_y_right))
        self.enable_fixed_state_velocity_jerk = bool(config.enable_fixed_state_velocity_jerk)
        self.mask_puck_velocity = bool(config.mask_puck_velocity)
        self.fixed_state_paddle_velocity = self._coerce_fixed_xy_pair(
            config.fixed_state_paddle_velocity, "fixed_state_paddle_velocity"
        )
        self.fixed_state_paddle_jerk = self._coerce_fixed_xy_pair(
            config.fixed_state_paddle_jerk, "fixed_state_paddle_jerk"
        )
        self.fixed_state_puck_velocity = self._coerce_fixed_xy_pair(
            config.fixed_state_puck_velocity, "fixed_state_puck_velocity"
        )

        # --- physics params (identical to Box2D) ---
        self.length, self.width = config.length, config.width
        self.paddle_radius = config.paddle_radius
        self.puck_radius = config.puck_radius
        self.block_width = config.block_width
        self.max_force_timestep = config.max_force_timestep
        self.step_frequency = config.step_frequency
        self.time_frequency = config.time_frequency
        self.time_per_step = 1 / self.time_frequency
        self.force_scaling = config.force_scaling
        self.absorb_target = config.absorb_target
        self.paddle_damping = config.paddle_damping
        self.puck_damping = config.puck_damping
        self.gravity = config.gravity
        self.puck_restitution = config.puck_restitution
        self.paddle_restitution = config.paddle_restitution
        self.side_wall_restitution = float(config.side_wall_restitution)
        self.end_wall_restitution = float(config.end_wall_restitution)
        self.puck_wall_restitution_threshold_speed = max(
            float(config.puck_wall_restitution_threshold_speed), 0.0
        )
        self.puck_wall_min_rebound_speed_below_threshold = max(
            float(config.puck_wall_min_rebound_speed_below_threshold), 0.0
        )
        self.puck_min_height = (-config.length / 2) + (config.length / 3)
        self.paddle_max_height = 0
        self.block_min_height = 0
        self.max_speed_start = config.width
        self.min_speed_start = -config.width
        self.paddle_density = float(config.paddle_density)
        self.paddle_mass_reference_radius = config.paddle_mass_reference_radius
        if (
            self.paddle_mass_reference_radius is not None
            and float(self.paddle_mass_reference_radius) > 0.0
            and self.paddle_radius > 0.0
        ):
            ref = float(self.paddle_mass_reference_radius)
            self.paddle_density *= (ref / self.paddle_radius) ** 2
        self._paddle_density_base = self.paddle_density
        self.puck_density = float(config.puck_density)
        self.puck_mass_reference_radius = config.puck_mass_reference_radius
        if (
            self.puck_mass_reference_radius is not None
            and float(self.puck_mass_reference_radius) > 0.0
            and self.puck_radius > 0.0
        ):
            ref = float(self.puck_mass_reference_radius)
            self.puck_density *= (ref / self.puck_radius) ** 2
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
        self.edge_lims = (self.top_abs, self.bot_abs, self.max_bias_p, self.max_bias_m)
        self.hist_len = config.hist_len
        self.center_offset_constant = config.center_offset_constant
        self.enable_action_delay = bool(config.enable_action_delay)
        self.enable_observation_delay = bool(config.enable_observation_delay)
        self.action_lag = float(config.action_lag)
        assert 0.0 <= self.action_lag <= 1.0, "action_lag must be in [0, 1]"
        base_delay_seconds = float(config.delay_seconds)
        if self.enable_action_delay and (not explicit_delay_seconds) and explicit_action_lag:
            base_delay_seconds = self.action_lag * self.time_per_step
        self.delay_seconds = max(base_delay_seconds, 0.0)
        self.derivative_min_dt = max(float(config.derivative_min_dt), 1e-8)
        self.acceleration_ema_alpha = float(np.clip(config.acceleration_ema_alpha, 0.0, 1.0))
        self.jerk_ema_alpha = float(np.clip(config.jerk_ema_alpha, 0.0, 1.0))
        self.max_acceleration_norm = float(config.max_acceleration_norm)
        self.max_jerk_norm = float(config.max_jerk_norm)
        self.puck_noise = config.puck_noise
        self.puck_noise_std = float(config.puck_noise_std)
        self.enable_puck_delay_interpolation = bool(config.enable_puck_delay_interpolation)
        self.puck_delay_interpolation_min = float(config.puck_delay_interpolation_min)
        self.puck_delay_interpolation_max = float(config.puck_delay_interpolation_max)
        self._prev_puck_positions_box2d = {}
        self.triangle_obstacle_size = float(config.triangle_obstacle_size)
        self.triangle_obstacle_restitution = float(config.triangle_obstacle_restitution)
        self.obstacle_shape = str(config.obstacle_shape).lower()
        self.enable_random_occlusions = bool(config.enable_random_occlusions)
        self.random_occlusion_rate = float(np.clip(config.random_occlusion_rate, 0.0, 1.0))
        self.random_occlusion_length_weights = np.array(
            config.random_occlusion_length_weights, dtype=float
        ).reshape(-1)
        if self.random_occlusion_length_weights.size == 0:
            self.random_occlusion_length_weights = np.array([1.0], dtype=float)
        self._occlusion_max_run = int(self.random_occlusion_length_weights.size)
        self._occlusion_run_remaining = {}
        self._occlusion_last_visible_base = {}
        self._occlusion_prev_occluded = {}

        self.last_action = np.zeros(2)
        self.last_target_position = None
        self.previous_acceleration = np.zeros(2)
        self.filtered_acceleration = np.zeros(2)
        self.filtered_jerk = np.zeros(2)
        self._has_prev_acceleration = False
        self.jerk = np.zeros(2)
        self.pose_hist = deque(maxlen=self.hist_len)
        self.dpose_hist = deque(maxlen=self.hist_len)
        self.observation_state_info = None
        self.observation_puck_history = None
        self.observation_paddle_history = None
        self.last_step_delay_seconds = 0.0

        pid_kp = kwargs.get("pid_kp", 1000.0)
        pid_ki = kwargs.get("pid_ki", 50.0)
        pid_kd = kwargs.get("pid_kd", 100.0)
        self.pid_controller = PIDController(Kp=pid_kp, Ki=pid_ki, Kd=pid_kd, dt=self.time_per_step)

        self.paddle_mass = self.paddle_density * np.pi * self.paddle_radius ** 2
        self.puck_mass = self.puck_density * np.pi * self.puck_radius ** 2
        self.chump_dict = {}
        self.max_paddle_vel = config.max_paddle_vel
        max_a = self.max_paddle_vel / self.time_per_step
        max_f = self.paddle_mass * max_a
        puck_max_a = max_f / self.puck_mass
        self.max_puck_vel = puck_max_a * self.time_per_step

        # --- render params ---
        self.ppm = config.render_size / self.width
        self.render_width = int(config.render_size)
        self.render_length = int(self.ppm * self.length)
        self.render_masks = config.render_masks

        # --- table bounds (box2d frame): x=width dir, y=-length dir ---
        self.table_x_min = -self.width / 2
        self.table_x_max = self.width / 2
        self.table_y_min = -self.length / 2
        self.table_y_max = self.length / 2

        self.min_goal_radius = self.width / 16
        self.max_goal_radius = self.width / 4
        self.metadata = {}

        # --- collision stats / forces ---
        self._wall_scales = [1.0, 1.0, 1.0]
        self._paddle_scales = [1.0, 1.0, 1.0]
        self._speed_breakpoints = (0.25, 0.75)
        self._episode_stats = {"wall": _make_empty_tier_stats(), "paddle": _make_empty_tier_stats()}
        self._collision_forces: list = []
        # pending restitution corrections keyed by puck body id
        self._pending_wall_restitution: dict = {}
        self._pending_paddle_puck: dict = {}

        # --- contact tracking ---
        self._contact_names: dict = {}   # {name: set(names)}

        # --- pymunk space (created once; bodies added/removed per reset) ---
        self._space = pymunk.Space()
        # Disable global gravity and damping; applied manually per-body each sub-step.
        self._space.gravity = (0.0, 0.0)
        self._space.damping = 1.0
        self._current_gravity_y = self.gravity if not isinstance(self.gravity, list) else -0.5
        self._setup_collision_handlers()

        # --- static wall body (segments, recreated in reset) ---
        self._wall_body: pymunk.Body | None = None
        self._dynamic_bodies: list = []   # all non-static bodies spawned this episode

        # Trigger initial reset (creates walls, clears dicts)
        self.reset(config.seed)

        self.total_timesteps = 0

    # ------------------------------------------------------------------
    # Pymunk space setup
    # ------------------------------------------------------------------

    def _setup_collision_handlers(self) -> None:
        """Register collision handlers on the space (called once at init).

        Pymunk 7.x API: space.on_collision(type_a, type_b, begin=..., ...)
        Callbacks receive (arbiter, space, data) and return None.
        """
        # Dynamic–dynamic: contact tracking + paddle-puck restitution
        self._space.on_collision(
            _CT_DYNAMIC, _CT_DYNAMIC,
            begin=_dd_begin,
            separate=_dd_separate,
            pre_solve=_dd_pre_solve,
            post_solve=_dd_post_solve,
            data=self,
        )
        # Dynamic–wall: restitution correction + force accumulation
        self._space.on_collision(
            _CT_DYNAMIC, _CT_WALL,
            pre_solve=_dw_pre_solve,
            post_solve=_dw_post_solve,
            data=self,
        )

    def _create_walls(self) -> None:
        """Create (or replace) four wall segments in box2d frame coordinates."""
        if self._wall_body is not None:
            self._space.remove(self._wall_body, *self._wall_body.shapes)

        wall_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        wall_body.body_tag = "table_wall"
        segments = [
            # Left / right side walls (constant x, vary y)
            ((self.table_x_min, self.table_y_min), (self.table_x_min, self.table_y_max), self.side_wall_restitution),
            ((self.table_x_max, self.table_y_min), (self.table_x_max, self.table_y_max), self.side_wall_restitution),
            # Bottom / top end walls (vary x, constant y)
            ((self.table_x_min, self.table_y_min), (self.table_x_max, self.table_y_min), self.end_wall_restitution),
            ((self.table_x_min, self.table_y_max), (self.table_x_max, self.table_y_max), self.end_wall_restitution),
        ]
        shapes = []
        for p1, p2, restitution in segments:
            seg = pymunk.Segment(wall_body, p1, p2, radius=0.0)
            seg.elasticity = float(restitution)
            seg.friction = 0.0
            seg.collision_type = _CT_WALL
            shapes.append(seg)
        self._space.add(wall_body, *shapes)
        self._wall_body = wall_body

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_callbacks(self, **kwargs) -> None:
        pass

    def instantiate_objects(self) -> None:
        pass

    def set_object_links(self) -> None:
        self.paddle_names = [
            n for n in self.paddles
            if n not in ("paddle_ego_acceleration", "paddle_ego_force", "paddle_ego_jerk")
        ]
        self.puck_names = list(self.pucks.keys())
        self.block_names = list(self.blocks.keys())
        self.obstacle_names = list(self.obstacles.keys())
        self.paddle_names.sort()
        self.puck_names.sort()
        self.block_names.sort()
        self.obstacle_names.sort()

    def reset(self, seed, **kwargs) -> dict:
        self.rng = np.random.RandomState(seed)
        self.timestep = 0

        # Remove all dynamic bodies from previous episode
        for body in self._dynamic_bodies:
            shapes = list(body.shapes)
            if shapes:
                self._space.remove(body, *shapes)
            else:
                self._space.remove(body)
        self._dynamic_bodies = []

        # Re-create walls (handles domain-randomized gravity)
        if isinstance(self.gravity, list):
            self._current_gravity_y = self.rng.uniform(
                low=self.gravity[0], high=self.gravity[1]
            )
        else:
            self._current_gravity_y = self.gravity
        self._create_walls()

        # Reset contact / collision state
        self._contact_names = {}
        self._collision_forces = []
        self._pending_wall_restitution = {}
        self._pending_paddle_puck = {}
        self._episode_stats = {"wall": _make_empty_tier_stats(), "paddle": _make_empty_tier_stats()}

        # Object dictionaries
        self.paddles: dict = {}
        self.pucks: dict = {}
        self.blocks: dict = {}
        self.block_initial_positions: dict = {}
        self.obstacles: dict = {}
        self.targets: dict = {}
        self.object_dict: dict = {}
        self.multiagent = False

        self.puck_history: list = []
        self.paddle_history: list = []
        self.paddle_attrs = None
        self.target_attrs = None

        self.last_action = np.zeros(2)
        self.last_target_position = None
        self.previous_acceleration = np.zeros(2)
        self.filtered_acceleration = np.zeros(2)
        self.filtered_jerk = np.zeros(2)
        self._has_prev_acceleration = False
        self.jerk = np.zeros(2)
        self.pose_hist = deque(maxlen=self.hist_len)
        self.dpose_hist = deque(maxlen=self.hist_len)
        self.observation_state_info = None
        self.observation_puck_history = None
        self.observation_paddle_history = None
        self.last_step_delay_seconds = 0.0
        self._prev_puck_positions_box2d = {}
        self._occlusion_run_remaining = {}
        self._occlusion_last_visible_base = {}
        self._occlusion_prev_occluded = {}

        if hasattr(self, "pid_controller"):
            self.pid_controller.reset()

        return self.get_current_state()

    # ------------------------------------------------------------------
    # Spawn helpers
    # ------------------------------------------------------------------

    def spawn_paddle(self, pos, vel, name, affected_by_gravity=False, movable=True):
        assert name in ("paddle_ego", "paddle_alt")
        pos = self.base_coord_to_box2d(pos)
        vel = self.base_coord_to_box2d(vel)
        mass = self.paddle_mass
        moment = pymunk.moment_for_circle(mass, 0, self.paddle_radius)
        body = pymunk.Body(mass, moment)
        body.position = (float(pos[0]), float(pos[1]))
        body.velocity = (float(vel[0]), float(vel[1]))
        body._name = name
        body._damping = self.paddle_damping
        body._affected_by_gravity = bool(affected_by_gravity)
        shape = pymunk.Circle(body, self.paddle_radius)
        shape.elasticity = self.paddle_restitution
        shape.friction = 0.0
        shape.collision_type = _CT_DYNAMIC
        shape._body_name = name
        self._space.add(body, shape)
        self._dynamic_bodies.append(body)
        self.paddles[name] = body
        if name == "paddle_ego":
            self.paddles["paddle_ego_acceleration"] = (0, 0)
            self.paddles["paddle_ego_force"] = (0, 0)
            self.paddles["paddle_ego_jerk"] = (0, 0)
        self.object_dict[name] = body
        self.paddle_history += [(-2 + self.center_offset_constant, 0, 1) for _ in range(5)]
        if "paddle_ego" in self.paddles and "paddle_alt" in self.paddles:
            self.multiagent = True

    def spawn_puck(self, pos, vel, name, affected_by_gravity=True, movable=True):
        pos = self.base_coord_to_box2d(pos)
        vel = self.base_coord_to_box2d(vel)
        mass = self.puck_mass
        moment = pymunk.moment_for_circle(mass, 0, self.puck_radius)
        body = pymunk.Body(mass, moment)
        body.position = (float(pos[0]), float(pos[1]))
        body.velocity = (float(vel[0]), float(vel[1]))
        body._name = name
        body._damping = self.puck_damping
        body._affected_by_gravity = bool(affected_by_gravity)
        shape = pymunk.Circle(body, self.puck_radius)
        shape.elasticity = self.puck_restitution
        shape.friction = 0.0
        shape.collision_type = _CT_DYNAMIC
        shape._body_name = name
        self._space.add(body, shape)
        self._dynamic_bodies.append(body)
        self.pucks[name] = body
        self.object_dict[name] = body
        self.puck_history += [(-2 + self.center_offset_constant, 0, 1) for _ in range(5)]

    def spawn_block(self, pos, vel, name, affected_by_gravity=False, movable=True):
        pos = self.base_coord_to_box2d(pos)
        vel = self.base_coord_to_box2d(vel)
        hw = self.block_width / 2
        verts = [(-hw, -hw), (hw, -hw), (hw, hw), (-hw, hw)]
        mass = self.block_density * (self.block_width ** 2) if movable else 0.0
        moment = pymunk.moment_for_poly(mass, verts) if movable else float("inf")
        btype = pymunk.Body.DYNAMIC if movable else pymunk.Body.STATIC
        body = pymunk.Body(mass, moment, body_type=btype)
        body.position = (float(pos[0]), float(pos[1]))
        body.velocity = (float(vel[0]), float(vel[1])) if movable else (0.0, 0.0)
        body._name = name
        body._damping = self.puck_damping if movable else 0.0
        body._affected_by_gravity = bool(affected_by_gravity)
        shape = pymunk.Poly(body, verts)
        shape.elasticity = 1.0
        shape.friction = 0.0
        shape.collision_type = _CT_DYNAMIC
        shape._body_name = name
        self._space.add(body, shape)
        self._dynamic_bodies.append(body)
        self.blocks[name] = body
        self.block_initial_positions[name] = (float(pos[0]), float(pos[1]))
        self.object_dict[name] = body

    def spawn_obstacle(self, pos, name, size=None, affected_by_gravity=False, movable=False):
        """Spawn a static isosceles triangle obstacle (same geometry as Box2D version)."""
        if size is None:
            size = self.triangle_obstacle_size
        size = float(size)
        h = (np.sqrt(3.0) / 2.0) * size
        half_base = 0.75 * size

        local_base = [
            np.array([-2.0 * h / 3.0, 0.0], dtype=float),
            np.array([h / 3.0, -half_base], dtype=float),
            np.array([h / 3.0, half_base], dtype=float),
        ]
        verts_b2d = [tuple(self.base_coord_to_box2d(v)) for v in local_base]
        pos_b2d = self.base_coord_to_box2d(pos)

        btype = pymunk.Body.DYNAMIC if movable else pymunk.Body.STATIC
        mass = self.block_density * size ** 2 if movable else 0.0
        moment = pymunk.moment_for_poly(mass, verts_b2d) if movable else float("inf")
        body = pymunk.Body(mass, moment, body_type=btype)
        body.position = (float(pos_b2d[0]), float(pos_b2d[1]))
        body._name = name
        body._damping = 0.0
        body._affected_by_gravity = False
        shape = pymunk.Poly(body, verts_b2d)
        shape.elasticity = self.triangle_obstacle_restitution
        shape.friction = 0.0
        shape.collision_type = _CT_DYNAMIC
        shape._body_name = name
        self._space.add(body, shape)
        self._dynamic_bodies.append(body)

        center_base = np.asarray(pos, dtype=float)
        verts_base = [tuple((center_base + v).tolist()) for v in local_base]
        self.obstacles[name] = {
            "body": body,
            "center_base": tuple(center_base.tolist()),
            "vertices_base": verts_base,
            "size": size,
        }
        self.object_dict[name] = body

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_current_state(self) -> dict:
        state_info = {}

        if "paddle_ego" in self.paddles:
            body = self.paddles["paddle_ego"]
            pos = body.position
            vel = body.velocity
            state_info["paddles"] = {
                "paddle_ego": {
                    "position": (float(pos.x), float(pos.y)),
                    "velocity": (float(vel.x), float(vel.y)),
                    "acceleration": tuple(float(v) for v in self.paddles.get("paddle_ego_acceleration", (0, 0))),
                    "force": tuple(float(v) for v in self.paddles.get("paddle_ego_force", (0, 0))),
                    "jerk": tuple(float(v) for v in self.paddles.get("paddle_ego_jerk", (0, 0))),
                }
            }

        if "paddle_alt" in self.paddles:
            body = self.paddles["paddle_alt"]
            pos = body.position
            vel = body.velocity
            state_info.setdefault("paddles", {})["paddle_alt"] = {
                "position": (float(pos.x), float(pos.y)),
                "velocity": (float(vel.x), float(vel.y)),
            }

        if self.blocks:
            state_info["blocks"] = []
            for block_name in self.blocks:
                body = self.blocks[block_name]
                bpos = body.position
                ipos = self.block_initial_positions[block_name]
                state_info["blocks"].append({
                    "current_position": (float(bpos.x), float(bpos.y)),
                    "initial_position": (float(ipos[0]), float(ipos[1])),
                })

        if self.obstacles:
            state_info["obstacles"] = []
            for obs_name, obs_data in self.obstacles.items():
                body = obs_data["body"]
                center = body.position
                world_verts = []
                for shape in body.shapes:
                    if isinstance(shape, pymunk.Poly):
                        for lv in shape.get_vertices():
                            wv = body.local_to_world(lv)
                            world_verts.append((float(wv.x), float(wv.y)))
                state_info["obstacles"].append({
                    "name": obs_name,
                    "position": (float(center.x), float(center.y)),
                    "vertices": world_verts,
                    "size": float(obs_data.get("size", self.triangle_obstacle_size)),
                })

        if self.pucks:
            state_info["pucks"] = []
            for puck_name in self.pucks:
                body = self.pucks[puck_name]
                px, py = float(body.position.x), float(body.position.y)
                px, py = self._apply_puck_delay_interpolation(puck_name, (px, py))
                px, py = self._get_noisy_puck_position((px, py))
                puck_base_xy_true = self._box2d_to_base_coords((px, py))
                occluded, puck_base_xy_obs = self._update_random_occlusion(puck_name, puck_base_xy_true)
                puck_b2d_obs = self._base_to_box2d_coords(puck_base_xy_obs)
                vx, vy = float(body.velocity.x), float(body.velocity.y)
                state_info["pucks"].append({
                    "position": (float(puck_b2d_obs[0]), float(puck_b2d_obs[1])),
                    "velocity": (vx, vy),
                    "occluded": int(occluded),
                })

        state_info = self.convert_from_box2d_coords(state_info)
        return self._apply_fixed_state_velocity_jerk(state_info)

    # ------------------------------------------------------------------
    # Physics step
    # ------------------------------------------------------------------

    def get_transition(self, action, other_action=None):
        if self.multiagent:
            raise NotImplementedError("Multiagent not implemented for pymunk backend")
        action = self.convert_to_box2d_coords(action)
        return self._get_singleagent_transition(action)

    def _get_singleagent_transition(self, action: np.ndarray) -> dict:  # action in box2d frame
        collision_start_idx = len(self._collision_forces)
        self.observation_state_info = None
        self.observation_puck_history = None
        self.observation_paddle_history = None

        use_delay = self.enable_action_delay or self.enable_observation_delay
        t_delay = float(np.clip(self.delay_seconds, 0.0, self.time_per_step)) if use_delay else 0.0
        self.last_step_delay_seconds = t_delay
        t_action = t_delay if self.enable_action_delay else 0.0
        t_obs = t_delay if self.enable_observation_delay else None

        breakpoints = [0.0, self.time_per_step, t_action]
        if t_obs is not None:
            breakpoints.append(t_obs)
        breakpoints = sorted(set(float(np.clip(t, 0.0, self.time_per_step)) for t in breakpoints))

        paddle_body = self.paddles["paddle_ego"]
        initial_vel = np.array([paddle_body.velocity.x, paddle_body.velocity.y])
        obs_snapshot_recorded = False
        if t_obs is not None and t_obs <= 0.0:
            self.observation_state_info = copy.deepcopy(self.get_current_state())
            self.observation_puck_history = list(self.puck_history)
            self.observation_paddle_history = list(self.paddle_history)
            obs_snapshot_recorded = True

        for start_t, end_t in zip(breakpoints[:-1], breakpoints[1:]):
            sim_time = end_t - start_t
            if sim_time <= 0.0:
                continue
            act = np.copy(self.last_action) if end_t <= (t_action + 1e-12) else np.copy(action)

            pos = np.array([paddle_body.position.x, paddle_body.position.y])

            # Boundary: prevent crossing into opponent's half (same logic as Box2D)
            if pos[1] > 0 - 3 * self.paddle_radius:
                act[1] = min(act[1], 0)

            target_pos = self._compute_pid_target_pos(pos, act)
            self.pose_hist.append(np.array(pos, dtype=float))
            self.dpose_hist.append(np.array(target_pos, dtype=float))
            target_pos = self._filter_update()
            self.last_target_position = self._box2d_to_base_coords(target_pos)

            current_vel = np.array([paddle_body.velocity.x, paddle_body.velocity.y])
            force = self.pid_controller.compute(target_pos, pos, current_vel)

            force_mag = np.linalg.norm(force)
            force_unit = force / (force_mag + 1e-8)
            if force_mag > self.max_force_timestep:
                force = force_unit * self.max_force_timestep
            if self.force_scaling > 0:
                force = force * self.force_scaling
            force = force.astype(float)

            # Opponent-half override (same as Box2D)
            if paddle_body.position.y > 0:
                new_force = self.force_scaling * paddle_body.mass * act[1]
                if new_force < -self.max_force_timestep:
                    new_force = -self.max_force_timestep
                force[1] = min(new_force, 0)
            else:
                force = force * np.array([self.action_x_scaling, self.action_y_scaling])

            force_mag = np.linalg.norm(force)
            if force_mag > self.max_force_timestep:
                force = force / force_mag * self.max_force_timestep

            # Apply force to paddle
            paddle_body.apply_force_at_world_point(
                (float(force[0]), float(force[1])), paddle_body.position
            )

            # --- Sub-step: apply gravity to pucks before stepping ---
            for puck_body in self.pucks.values():
                if puck_body._affected_by_gravity:
                    grav_force = puck_body.mass * self._current_gravity_y
                    puck_body.apply_force_at_world_point((0.0, grav_force), puck_body.position)

            self._space.step(sim_time)

            # --- Post-step: apply per-body damping ---
            for body in self._dynamic_bodies:
                damping = getattr(body, "_damping", 0.0)
                if damping > 0.0:
                    scale = max(0.0, 1.0 - damping * sim_time)
                    body.velocity = (body.velocity.x * scale, body.velocity.y * scale)

            # Reset blocks at t=0 (same as Box2D)
            if self.timestep == 0 and self.blocks:
                for block_name, block_body in self.blocks.items():
                    ix, iy = self.block_initial_positions[block_name]
                    block_body.position = pymunk.Vec2d(ix, iy)
                    block_body.velocity = (0.0, 0.0)

            # Paddle velocity cap
            vel = np.array([paddle_body.velocity.x, paddle_body.velocity.y])
            vel_mag = np.linalg.norm(vel)
            if vel_mag > self.max_paddle_vel:
                paddle_body.velocity = pymunk.Vec2d(
                    vel[0] / vel_mag * self.max_paddle_vel,
                    vel[1] / vel_mag * self.max_paddle_vel,
                )

            # Paddle position clip to workspace
            pos = np.array([paddle_body.position.x, paddle_body.position.y])
            pos = self._clip_pid_target_to_workspace(pos)
            paddle_body.position = pymunk.Vec2d(float(pos[0]), float(pos[1]))

            # History update
            state_info = self.get_current_state()
            if "pucks" in state_info:
                for puck in state_info["pucks"]:
                    self.puck_history.append(list(puck["position"]) + [int(puck.get("occluded", 0))])
            else:
                for _ in self.pucks:
                    self.puck_history.append([-2 + self.center_offset_constant, 0, 1])

            if "paddles" in state_info:
                for pname, pdata in state_info["paddles"].items():
                    self.paddle_history.append(list(pdata["position"]) + [0])
            else:
                for _ in (n for n in self.paddles if "acceleration" not in n and "force" not in n and "jerk" not in n):
                    self.paddle_history.append([-2 + self.center_offset_constant, 0, 1])

            # Aggregate paddle force from collision forces
            total_force = np.array(force)
            step_forces = self._collision_forces[collision_start_idx:]
            for cf in step_forces:
                if cf["bodyA"] == "paddle_ego":
                    total_force[0] += cf["normal_force"] * cf["contact_normal"][0]
                    total_force[1] += cf["normal_force"] * cf["contact_normal"][1]
                elif cf["bodyB"] == "paddle_ego":
                    total_force[0] -= cf["normal_force"] * cf["contact_normal"][0]
                    total_force[1] -= cf["normal_force"] * cf["contact_normal"][1]
            self.paddles["paddle_ego_force"] = (float(total_force[0]), float(total_force[1]))

            if t_obs is not None and (not obs_snapshot_recorded) and (end_t >= t_obs - 1e-12):
                self.observation_state_info = copy.deepcopy(state_info)
                self.observation_puck_history = list(self.puck_history)
                self.observation_paddle_history = list(self.paddle_history)
                obs_snapshot_recorded = True

        # Final acceleration / jerk
        final_vel = np.array([paddle_body.velocity.x, paddle_body.velocity.y])
        current_acceleration, current_jerk = self._update_motion_derivatives(initial_vel, final_vel)
        self.paddles["paddle_ego_acceleration"] = (float(current_acceleration[0]), float(current_acceleration[1]))
        self.paddles["paddle_ego_jerk"] = (float(current_jerk[0]), float(current_jerk[1]))

        state_info = self.get_current_state()
        if t_obs is not None and not obs_snapshot_recorded:
            self.observation_state_info = copy.deepcopy(state_info)
            self.observation_puck_history = list(self.puck_history)
            self.observation_paddle_history = list(self.paddle_history)

        # Paddle–puck contact count for this step
        step_forces = self._collision_forces[collision_start_idx:]
        step_contacted_pucks: set = set()
        for cf in step_forces:
            if cf["bodyA"] == "paddle_ego" and str(cf["bodyB"]).startswith("puck"):
                step_contacted_pucks.add(cf["bodyB"])
            elif cf["bodyB"] == "paddle_ego" and str(cf["bodyA"]).startswith("puck"):
                step_contacted_pucks.add(cf["bodyA"])
        state_info["paddle_puck_collision_count"] = int(len(step_contacted_pucks))

        # Triangle side-hit analysis (same helper as Box2D)
        triangle_side_hits, triangle_hit_details = self._compute_triangle_side_hits(step_forces)
        state_info["triangle_side_hits"] = triangle_side_hits
        state_info["triangle_hit_details"] = triangle_hit_details

        self.timestep += 1
        self.last_action = action
        return state_info

    # ------------------------------------------------------------------
    # Contacts & forces
    # ------------------------------------------------------------------

    def get_contacts(self):
        names = self.paddle_names + self.puck_names + self.block_names + self.obstacle_names
        n = len(names)
        contact_matrix = np.zeros((n, n), dtype=bool)
        contact_names = {nm: [] for nm in names}
        for i, na in enumerate(names):
            for j, nb in enumerate(names):
                if nb in self._contact_names.get(na, set()):
                    contact_matrix[i, j] = True
                    contact_names[na].append(nb)
        return contact_matrix, contact_names

    def get_collision_forces(self) -> list:
        return self._collision_forces

    def set_collision_scales(self, wall_scales, paddle_scales, speed_breakpoints=None) -> None:
        self._wall_scales = [float(s) for s in wall_scales]
        self._paddle_scales = [float(s) for s in paddle_scales]
        if speed_breakpoints is not None:
            self._speed_breakpoints = (float(speed_breakpoints[0]), float(speed_breakpoints[1]))

    def get_episode_collision_stats(self) -> dict:
        stats = {}
        for surface in ("wall", "paddle"):
            stats[surface] = {}
            for tier in _COLLISION_TIERS:
                bucket = self._episode_stats[surface][tier]
                count = bucket["count"]
                stats[surface][tier] = {
                    "count": count,
                    "mean_speed_in": bucket["speed_in_sum"] / count if count > 0 else 0.0,
                    "mean_speed_out": bucket["speed_out_sum"] / count if count > 0 else 0.0,
                }
        self._episode_stats = {"wall": _make_empty_tier_stats(), "paddle": _make_empty_tier_stats()}
        return stats

    def respond_contacts(self, contact_names):
        hit_a_puck = []
        for tn in getattr(self, "target_names", []):
            for cn in contact_names.get(tn, []):
                if cn.find("puck") != -1:
                    hit_a_puck.append(cn)
        if self.absorb_target:
            for cn in hit_a_puck:
                body = self.object_dict.get(cn)
                if body is not None:
                    shapes = list(body.shapes)
                    self._space.remove(body, *shapes)
                    self._dynamic_bodies = [b for b in self._dynamic_bodies if b is not body]
                    del self.object_dict[cn]
        return hit_a_puck

    # ------------------------------------------------------------------
    # Coordinate helpers (same as AirHockeyBox2D)
    # ------------------------------------------------------------------

    def base_coord_to_box2d(self, coord):
        return (coord[1], -coord[0])

    def convert_to_box2d_coords(self, action):
        return np.array((action[1], -action[0]))

    def _box2d_to_base_coords(self, coord):
        return np.array((-coord[1], coord[0]), dtype=float)

    def _base_to_box2d_coords(self, coord):
        return np.array((coord[1], -coord[0]), dtype=float)

    def convert_from_box2d_coords(self, state_info):
        for key, value in state_info.items():
            if isinstance(value, list):
                for i in range(len(value)):
                    if isinstance(value[i], dict):
                        for key2, value2 in value[i].items():
                            if isinstance(value2, tuple) and len(value2) == 2:
                                state_info[key][i][key2] = (-value2[1], value2[0])
                            elif (
                                isinstance(value2, list)
                                and len(value2) > 0
                                and all(isinstance(v, tuple) and len(v) == 2 for v in value2)
                            ):
                                state_info[key][i][key2] = [(-v[1], v[0]) for v in value2]
            elif isinstance(value, dict):
                for key2, value2 in value.items():
                    if isinstance(value2, dict):
                        for key3, value3 in value2.items():
                            if isinstance(value3, tuple) and len(value3) == 2:
                                state_info[key][key2][key3] = (-value3[1], value3[0])
        return state_info

    @staticmethod
    def _coerce_fixed_xy_pair(value, field_name):
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size != 2:
            raise ValueError(f"{field_name} must contain exactly 2 numeric values.")
        if not np.isfinite(arr).all():
            raise ValueError(f"{field_name} must contain only finite numeric values.")
        return float(arr[0]), float(arr[1])

    def _apply_fixed_state_velocity_jerk(self, state_info):
        if not self.enable_fixed_state_velocity_jerk:
            return state_info
        paddles = state_info.get("paddles", {})
        paddle_ego = paddles.get("paddle_ego", {})
        if isinstance(paddle_ego, dict):
            paddle_ego["velocity"] = self.fixed_state_paddle_velocity
            if "jerk" in paddle_ego:
                paddle_ego["jerk"] = self.fixed_state_paddle_jerk
        if self.mask_puck_velocity:
            for puck in state_info.get("pucks", []):
                if isinstance(puck, dict):
                    puck["velocity"] = self.fixed_state_puck_velocity
        return state_info

    # --- workspace clipping (identical to Box2D) ---

    def _get_edge(self, x, y, w, h):
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
        x_min_lim, x_max_lim, y_min, y_max = self.lims
        top_abs, bot_abs, max_bias_m, max_bias_p = self.edge_lims
        x_raw = x - self.center_offset_constant
        y = np.clip(y, y_min, y_max)
        x_min = x_min_lim
        x_max = min(x_max_lim, max_bias_m - top_abs * y, max_bias_p + top_abs * y)
        x_raw = np.clip(x_raw, x_min, x_max)
        return np.array([x_raw + self.center_offset_constant, y], dtype=float)

    def _compute_pid_target_pos(self, pos, act):
        pos_base = self._box2d_to_base_coords(pos)
        act_base = self._box2d_to_base_coords(act)
        move_vector_base = np.array(act_base, dtype=float) * self.move_lims
        target_raw_base = pos_base + move_vector_base
        rel_base = target_raw_base - pos_base
        edge_rel = self._get_edge(rel_base[0], rel_base[1], self.move_lims[0], self.move_lims[1])
        target_rect_base = pos_base + edge_rel
        target_clipped_base = self._clip_limits(target_rect_base[0], target_rect_base[1])
        return self._base_to_box2d_coords(target_clipped_base)

    def _clip_pid_target_to_workspace(self, target_pos):
        target_base = self._box2d_to_base_coords(target_pos)
        clipped_base = self._clip_limits(target_base[0], target_base[1])
        return self._base_to_box2d_coords(clipped_base)

    def _filter_update(self):
        pose_vel = np.array(self.dpose_hist[-1], dtype=float) - np.array(self.pose_hist[-1], dtype=float)
        transform_vel = pose_vel
        if len(self.dpose_hist) > 1:
            pose_vels = [
                np.array(self.dpose_hist[i], dtype=float) - np.array(self.pose_hist[i], dtype=float)
                for i in range(len(self.dpose_hist))
            ]
            transform_vel = np.mean(pose_vels, axis=0)
        return np.array(self.pose_hist[-1], dtype=float) + transform_vel

    @staticmethod
    def _clip_vector_norm(vec, max_norm):
        if max_norm <= 0:
            return vec
        norm = np.linalg.norm(vec)
        if norm <= max_norm or norm <= 1e-8:
            return vec
        return vec * (max_norm / norm)

    def _update_motion_derivatives(self, initial_vel, final_vel):
        dt = max(float(self.time_per_step), self.derivative_min_dt)
        raw_acceleration = (final_vel - initial_vel) / dt
        raw_acceleration = self._clip_vector_norm(raw_acceleration, self.max_acceleration_norm)
        if not self._has_prev_acceleration:
            filtered_acceleration = raw_acceleration
            filtered_jerk = np.zeros_like(raw_acceleration)
            self._has_prev_acceleration = True
        else:
            alpha_a = self.acceleration_ema_alpha
            filtered_acceleration = alpha_a * raw_acceleration + (1.0 - alpha_a) * self.filtered_acceleration
            raw_jerk = (filtered_acceleration - self.filtered_acceleration) / dt
            raw_jerk = self._clip_vector_norm(raw_jerk, self.max_jerk_norm)
            alpha_j = self.jerk_ema_alpha
            filtered_jerk = alpha_j * raw_jerk + (1.0 - alpha_j) * self.filtered_jerk
        self.filtered_acceleration = filtered_acceleration
        self.filtered_jerk = filtered_jerk
        self.previous_acceleration = filtered_acceleration.copy()
        self.jerk = filtered_jerk.copy()
        return filtered_acceleration, filtered_jerk

    # --- puck observation helpers (identical to Box2D) ---

    def _apply_puck_delay_interpolation(self, puck_name, current_position):
        current = np.array(current_position, dtype=float)
        prev = self._prev_puck_positions_box2d.get(puck_name)
        self._prev_puck_positions_box2d[puck_name] = current.copy()
        if not self.enable_puck_delay_interpolation or prev is None:
            return float(current[0]), float(current[1])
        factor = self.rng.uniform(self.puck_delay_interpolation_min, self.puck_delay_interpolation_max)
        interpolated = prev + factor * (current - prev)
        return float(interpolated[0]), float(interpolated[1])

    def _get_noisy_puck_position(self, position):
        if not self.puck_noise:
            return position
        noise = self.rng.normal(loc=0.0, scale=self.puck_noise_std, size=2)
        noisy = np.array(position, dtype=float) + noise
        return float(noisy[0]), float(noisy[1])

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
            self._occlusion_last_visible_base[puck_name] = (float(true_puck_base_xy[0]), float(true_puck_base_xy[1]))
            self._occlusion_prev_occluded[puck_name] = False
            return False, np.array(true_puck_base_xy, dtype=float)
        remaining = int(self._occlusion_run_remaining.get(puck_name, 0))
        prev_occluded = bool(self._occlusion_prev_occluded.get(puck_name, False))
        if remaining > 0:
            self._occlusion_run_remaining[puck_name] = remaining - 1
            observed = self._occlusion_last_visible_base.get(puck_name, (-2.0 + self.center_offset_constant, 0.0))
            self._occlusion_prev_occluded[puck_name] = True
            return True, np.array(observed, dtype=float)
        if prev_occluded:
            self._occlusion_last_visible_base[puck_name] = (float(true_puck_base_xy[0]), float(true_puck_base_xy[1]))
            self._occlusion_prev_occluded[puck_name] = False
            return False, np.array(true_puck_base_xy, dtype=float)
        if self.rng.uniform(0.0, 1.0) < self.random_occlusion_rate:
            run_len = self._sample_occlusion_run_length()
            self._occlusion_run_remaining[puck_name] = max(run_len - 1, 0)
            observed = self._occlusion_last_visible_base.get(puck_name, (-2.0 + self.center_offset_constant, 0.0))
            self._occlusion_prev_occluded[puck_name] = True
            return True, np.array(observed, dtype=float)
        self._occlusion_last_visible_base[puck_name] = (float(true_puck_base_xy[0]), float(true_puck_base_xy[1]))
        self._occlusion_prev_occluded[puck_name] = False
        return False, np.array(true_puck_base_xy, dtype=float)

    # --- triangle side helpers (identical to Box2D) ---

    @staticmethod
    def _point_segment_distance(point, seg_a, seg_b):
        point = np.asarray(point, dtype=float)
        seg_a = np.asarray(seg_a, dtype=float)
        seg_b = np.asarray(seg_b, dtype=float)
        ab = seg_b - seg_a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-12:
            return float(np.linalg.norm(point - seg_a))
        t = float(np.clip(np.dot(point - seg_a, ab) / denom, 0.0, 1.0))
        return float(np.linalg.norm(point - (seg_a + t * ab)))

    def _triangle_side_from_contact(self, obstacle_name, contact_point_b2d):
        obstacle = self.obstacles.get(obstacle_name)
        if obstacle is None:
            return None
        verts = obstacle.get("vertices_base", [])
        if len(verts) != 3:
            return None
        contact_base = self._box2d_to_base_coords(contact_point_b2d)
        top, left, right = [np.asarray(v, dtype=float) for v in verts]
        dists = {
            "left": self._point_segment_distance(contact_base, top, left),
            "right": self._point_segment_distance(contact_base, top, right),
            "base": self._point_segment_distance(contact_base, left, right),
        }
        return min(dists, key=dists.get)

    def _compute_triangle_side_hits(self, step_collisions):
        side_counts = {"left": 0, "right": 0, "base": 0}
        details = []
        for cf in step_collisions:
            body_a = str(cf.get("bodyA", ""))
            body_b = str(cf.get("bodyB", ""))
            contact_point = cf.get("contact_point", None)
            if contact_point is None:
                continue
            obstacle_name = None
            hit_body = None
            if body_a.startswith("triangle_obstacle_"):
                obstacle_name, hit_body = body_a, body_b
            elif body_b.startswith("triangle_obstacle_"):
                obstacle_name, hit_body = body_b, body_a
            if obstacle_name is None:
                continue
            side = self._triangle_side_from_contact(obstacle_name, contact_point)
            if side is None:
                continue
            side_counts[side] += 1
            details.append({"obstacle": obstacle_name, "side": side, "hit_body": hit_body})
        return side_counts, details

    # --- speed-tier helper ---
    def _speed_tier(self, speed):
        low, high = self._speed_breakpoints
        if speed < low:
            return "low"
        if speed < high:
            return "mid"
        return "high"


# ---------------------------------------------------------------------------
# Pymunk collision handler callbacks (module-level functions, not methods)
# ---------------------------------------------------------------------------

def _body_name(shape) -> str | None:
    return getattr(shape, "_body_name", getattr(shape.body, "_name", None))


# Dynamic–dynamic: contact tracking
def _dd_begin(arbiter, space, sim):
    # sim is passed directly as data= in on_collision()
    na = _body_name(arbiter.shapes[0])
    nb = _body_name(arbiter.shapes[1])
    if na and nb:
        sim._contact_names.setdefault(na, set()).add(nb)
        sim._contact_names.setdefault(nb, set()).add(na)


def _dd_separate(arbiter, space, sim):
    na = _body_name(arbiter.shapes[0])
    nb = _body_name(arbiter.shapes[1])
    if na and nb:
        sim._contact_names.get(na, set()).discard(nb)
        sim._contact_names.get(nb, set()).discard(na)


def _dd_pre_solve(arbiter, space, sim):
    """Track approach speed for paddle-puck before solver applies impulse."""
    na = _body_name(arbiter.shapes[0])
    nb = _body_name(arbiter.shapes[1])
    if na is None or nb is None:
        return
    is_paddle_a = str(na).startswith("paddle")
    is_puck_a = str(na).startswith("puck")
    is_paddle_b = str(nb).startswith("paddle")
    is_puck_b = str(nb).startswith("puck")
    if not ((is_paddle_a and is_puck_b) or (is_puck_a and is_paddle_b)):
        return

    paddle_body = arbiter.shapes[0].body if is_paddle_a else arbiter.shapes[1].body
    puck_body = arbiter.shapes[0].body if is_puck_a else arbiter.shapes[1].body
    paddle_name = na if is_paddle_a else nb
    puck_name = nb if is_puck_a else na

    # Normal points from shape_a → shape_b in pymunk
    normal = np.array(arbiter.normal)
    if is_puck_a:
        normal = -normal   # make it point from paddle toward puck

    v_paddle = np.array(paddle_body.velocity)
    v_puck = np.array(puck_body.velocity)
    v_rel_n = float(np.dot(v_puck - v_paddle, normal))
    approach_speed = max(0.0, -v_rel_n)

    if approach_speed > 1e-8:
        e_a = arbiter.shapes[0].elasticity
        e_b = arbiter.shapes[1].elasticity
        combined_e = max(e_a, e_b)
        contact_id = (paddle_name, puck_name)
        prev = sim._pending_paddle_puck.get(contact_id)
        if prev is None or approach_speed > prev["approach_speed"]:
            sim._pending_paddle_puck[contact_id] = {
                "approach_speed": approach_speed,
                "normal": normal,
                "restitution": combined_e,
            }


def _dd_post_solve(arbiter, space, sim):
    na = _body_name(arbiter.shapes[0])
    nb = _body_name(arbiter.shapes[1])
    if na is None or nb is None:
        return

    normal = arbiter.normal  # pymunk.Vec2d, from shape A toward shape B
    total_impulse = arbiter.total_impulse
    impulse_mag = float(total_impulse.length)
    dt = sim.time_per_step
    normal_force = impulse_mag / max(dt, 1e-8)

    # --- deterministic paddle-puck restitution (mirrors Box2D PostSolve) ---
    is_paddle_a = str(na).startswith("paddle")
    is_puck_a = str(na).startswith("puck")
    is_paddle_b = str(nb).startswith("paddle")
    is_puck_b = str(nb).startswith("puck")
    is_paddle_puck = (is_paddle_a and is_puck_b) or (is_puck_a and is_paddle_b)

    if is_paddle_puck:
        paddle_body = arbiter.shapes[0].body if is_paddle_a else arbiter.shapes[1].body
        puck_body = arbiter.shapes[0].body if is_puck_a else arbiter.shapes[1].body
        paddle_name = na if is_paddle_a else nb
        puck_name = nb if is_puck_a else na
        contact_id = (paddle_name, puck_name)

        pending = sim._pending_paddle_puck.pop(contact_id, None)
        if pending is not None:
            approach_speed = float(pending["approach_speed"])
            if approach_speed > 1e-8:
                normal_pp = np.array(pending["normal"])
                e = float(pending["restitution"])
                tier = sim._speed_tier(approach_speed)
                tier_idx = _COLLISION_TIERS.index(tier)
                scale = sim._paddle_scales[tier_idx]

                bucket = sim._episode_stats["paddle"][tier]
                bucket["count"] += 1
                bucket["speed_in_sum"] += approach_speed

                v_paddle_post = np.array(paddle_body.velocity)
                v_puck_post = np.array(puck_body.velocity)
                v_rel_n_post = float(np.dot(v_puck_post - v_paddle_post, normal_pp))
                v_rel_n_desired = e * approach_speed * scale

                bucket["speed_out_sum"] += v_rel_n_desired

                delta = v_rel_n_desired - v_rel_n_post
                if abs(delta) > 1e-8:
                    m_p = float(paddle_body.mass)
                    m_k = float(puck_body.mass)
                    j = delta * m_p * m_k / (m_p + m_k)
                    j_vec = normal_pp * j
                    puck_body.apply_impulse_at_world_point(
                        (float(j_vec[0]), float(j_vec[1])), puck_body.position
                    )
                    paddle_body.apply_impulse_at_world_point(
                        (-float(j_vec[0]), -float(j_vec[1])), paddle_body.position
                    )

    # Record collision force
    cp_set = arbiter.contact_point_set
    if cp_set.points:
        cp = cp_set.points[0]
        contact_pt = (float(cp.point_a.x + cp.point_b.x) / 2, float(cp.point_a.y + cp.point_b.y) / 2)
    else:
        contact_pt = None

    sim._collision_forces.append({
        "bodyA": na,
        "bodyB": nb,
        "normal_force": normal_force,
        "contact_normal": (float(normal.x), float(normal.y)),
        "contact_point": contact_pt,
    })


# Dynamic–wall handlers
def _dw_pre_solve(arbiter, space, sim):
    """Capture pre-collision puck speed for deterministic wall restitution."""
    na = _body_name(arbiter.shapes[0])
    if na is None:
        return
    if not str(na).startswith("puck"):
        return  # only special-case puck-wall

    puck_body = arbiter.shapes[0].body
    wall_shape = arbiter.shapes[1]
    # arbiter.normal: from shape_a (dynamic) toward shape_b (wall) = outward from puck
    normal = np.array(arbiter.normal)
    puck_vel = np.array(puck_body.velocity)
    # positive dot = puck moving toward wall (in direction of normal = outward)
    vn = float(np.dot(puck_vel, normal))
    incoming_speed = max(0.0, vn)

    prev = sim._pending_wall_restitution.get(id(puck_body))
    if prev is None or incoming_speed > prev["incoming_speed"]:
        sim._pending_wall_restitution[id(puck_body)] = {
            "incoming_speed": incoming_speed,
            "normal": normal,
            "restitution": float(wall_shape.elasticity),
            "puck_name": na,
        }


def _dw_post_solve(arbiter, space, sim):
    """Apply corrective impulse to achieve deterministic puck-wall restitution."""
    na = _body_name(arbiter.shapes[0])
    if na is None:
        return

    # Collision force accumulation for all dynamics hitting walls
    normal = arbiter.normal
    total_impulse = arbiter.total_impulse
    impulse_mag = float(total_impulse.length)
    dt = sim.time_per_step
    sim._collision_forces.append({
        "bodyA": na,
        "bodyB": "table_wall",
        "normal_force": impulse_mag / max(dt, 1e-8),
        "contact_normal": (float(normal.x), float(normal.y)),
        "contact_point": None,
    })

    if not str(na).startswith("puck"):
        return

    puck_body = arbiter.shapes[0].body
    pending = sim._pending_wall_restitution.pop(id(puck_body), None)
    if pending is None:
        return

    incoming_speed = float(pending["incoming_speed"])
    if incoming_speed <= 1e-8:
        return

    normal_out = np.array(pending["normal"])  # points from puck toward wall (outward)
    restitution = pending["restitution"]

    tier = sim._speed_tier(incoming_speed)
    tier_idx = _COLLISION_TIERS.index(tier)
    scale = sim._wall_scales[tier_idx]

    if incoming_speed < sim.puck_wall_restitution_threshold_speed:
        target_outgoing = sim.puck_wall_min_rebound_speed_below_threshold
    else:
        target_outgoing = incoming_speed * restitution * scale

    # Record stat
    bucket = sim._episode_stats["wall"][tier]
    bucket["count"] += 1
    bucket["speed_in_sum"] += incoming_speed
    bucket["speed_out_sum"] += target_outgoing

    puck_vel_post = np.array(puck_body.velocity)
    # After collision puck moves away from wall: component in -normal_out direction
    current_outgoing = max(0.0, float(np.dot(puck_vel_post, -normal_out)))

    if target_outgoing > current_outgoing + 1e-8:
        delta = target_outgoing - current_outgoing
        impulse_mag = puck_body.mass * delta
        # Push puck away from wall (in -normal_out direction)
        imp = -normal_out * impulse_mag
        puck_body.apply_impulse_at_world_point((float(imp[0]), float(imp[1])), puck_body.position)

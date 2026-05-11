from Box2D.b2 import world, contactListener
from Box2D import (b2CircleShape, b2EdgeShape, b2FixtureDef, b2PolygonShape,
                   b2_dynamicBody, b2_staticBody, b2Filter, b2Vec2)
import numpy as np
from collections import deque
import copy
import yaml
import inspect
from types import SimpleNamespace
from ..utils import dict_to_namespace
from ..observation_homography import make_sine_y_warp_fn

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

_COLLISION_TIERS = ("low", "mid", "high")

def _make_empty_tier_stats():
    return {t: {"count": 0, "speed_in_sum": 0.0, "speed_out_sum": 0.0} for t in _COLLISION_TIERS}


class CollisionForceListener(contactListener):
    def __init__(
        self,
        wall_tag="table_wall",
        puck_wall_restitution_threshold_speed=0.25,
        puck_wall_min_rebound_speed_below_threshold=0.1,
        speed_breakpoints=(0.25, 0.75),
        wall_scales=(1.0, 1.0, 1.0),
        paddle_scales=(1.0, 1.0, 1.0),
        rng=None,
    ):
        contactListener.__init__(self)
        self.collision_forces = list()
        self.wall_tag = wall_tag
        self.puck_wall_restitution_threshold_speed = max(
            float(puck_wall_restitution_threshold_speed), 0.0
        )
        self.puck_wall_min_rebound_speed_below_threshold = max(
            float(puck_wall_min_rebound_speed_below_threshold), 0.0
        )
        self._pending_wall_restitution = {}
        self._pending_paddle_puck = {}
        self.speed_breakpoints = (float(speed_breakpoints[0]), float(speed_breakpoints[1]))
        self.wall_scales = [float(s) for s in wall_scales]
        self.paddle_scales = [float(s) for s in paddle_scales]
        self.rng = rng if rng is not None else np.random.default_rng()
        self._episode_stats = {"wall": _make_empty_tier_stats(), "paddle": _make_empty_tier_stats()}

    def set_rng(self, rng):
        """Kept for compatibility with the sim's reset hook (no-op without per-collision randomization)."""
        if rng is not None:
            self.rng = rng

    def set_scales(self, wall_scales, paddle_scales, speed_breakpoints=None):
        """Update per-tier restitution multipliers. Safe to call between episodes."""
        self.wall_scales = [float(s) for s in wall_scales]
        self.paddle_scales = [float(s) for s in paddle_scales]
        if speed_breakpoints is not None:
            self.speed_breakpoints = (float(speed_breakpoints[0]), float(speed_breakpoints[1]))

    def _speed_tier(self, speed):
        low_thresh, high_thresh = self.speed_breakpoints
        if speed < low_thresh:
            return "low"
        if speed < high_thresh:
            return "mid"
        return "high"

    def _tier_index(self, tier):
        return _COLLISION_TIERS.index(tier)

    def get_and_reset_episode_stats(self):
        """Return accumulated collision stats for this episode and reset counters."""
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

    def reset(self):
        del self.collision_forces
        self.collision_forces = list()
        self._pending_wall_restitution = {}
        self._pending_paddle_puck = {}
        self._episode_stats = {"wall": _make_empty_tier_stats(), "paddle": _make_empty_tier_stats()}

    @staticmethod
    def _is_puck(body):
        return body.userData is not None and "puck" in str(body.userData)

    @staticmethod
    def _is_paddle(body):
        return body.userData is not None and "paddle" in str(body.userData)

    def _is_wall(self, body):
        return body.userData == self.wall_tag

    def PreSolve(self, contact, oldManifold):
        fixtureA = contact.fixtureA
        fixtureB = contact.fixtureB
        bodyA = fixtureA.body
        bodyB = fixtureB.body
        wall_fixture = fixtureA if self._is_wall(bodyA) else (fixtureB if self._is_wall(bodyB) else None)

        # For any wall contact, use wall restitution (not mixed restitution).
        # Puck-wall contacts are still handled by custom impulse logic below.
        if wall_fixture is not None:
            contact.restitution = float(wall_fixture.restitution)

        # --- Paddle-puck: bypass Box2D's b2_velocityThreshold by disabling
        # built-in restitution and applying it manually in PostSolve. ---
        is_paddle_puck = (
            (self._is_paddle(bodyA) and self._is_puck(bodyB))
            or (self._is_puck(bodyA) and self._is_paddle(bodyB))
        )
        if is_paddle_puck:
            self._presolve_paddle_puck(contact, fixtureA, fixtureB, bodyA, bodyB)
            return

        # Enforce a deterministic restitution threshold for puck-wall contacts.
        # This avoids relying on global Box2D velocityThreshold behavior.
        is_puck_wall = (self._is_puck(bodyA) and self._is_wall(bodyB)) or (self._is_puck(bodyB) and self._is_wall(bodyA))
        if not is_puck_wall:
            return

        point_count = int(contact.manifold.pointCount)
        if point_count <= 0:
            return

        world_manifold = contact.worldManifold
        normal = np.array([world_manifold.normal.x, world_manifold.normal.y], dtype=float)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1e-8:
            return
        normal_unit = normal / normal_norm
        puck_body = bodyA if self._is_puck(bodyA) else bodyB
        puck_name = str(puck_body.userData)
        puck_pos = np.array([puck_body.position[0], puck_body.position[1]], dtype=float)
        center_vec = -puck_pos
        normal_inward = normal_unit if float(np.dot(center_vec, normal_unit)) >= 0.0 else -normal_unit
        puck_vel = np.array([puck_body.linearVelocity[0], puck_body.linearVelocity[1]], dtype=float)
        # Positive vn means moving away from wall (toward table center).
        vn_inward = float(np.dot(puck_vel, normal_inward))
        incoming_speed = max(0.0, -vn_inward)
        if incoming_speed <= 1e-8:
            # Not an incoming wall impact: do not arm any custom impulse.
            self._pending_wall_restitution.pop(puck_name, None)
            contact.restitution = 0.0
            return

        prev = self._pending_wall_restitution.get(puck_name)
        if prev is None or incoming_speed > prev["incoming_speed"]:
            self._pending_wall_restitution[puck_name] = {
                "incoming_speed": incoming_speed,
                "normal_inward": normal_inward,
                "restitution": float(wall_fixture.restitution) if wall_fixture is not None else max(float(fixtureA.restitution), float(fixtureB.restitution)),
            }

        # Disable built-in restitution for puck-wall so we can enforce
        # a deterministic threshold ourselves in PostSolve.
        contact.restitution = 0.0

    def _presolve_paddle_puck(self, contact, fixtureA, fixtureB, bodyA, bodyB):
        point_count = int(contact.manifold.pointCount)
        if point_count <= 0:
            return

        paddle_body = bodyA if self._is_paddle(bodyA) else bodyB
        puck_body = bodyA if self._is_puck(bodyA) else bodyB

        world_manifold = contact.worldManifold
        # Box2D normal points from fixtureA to fixtureB.
        # Standardize to point from paddle toward puck.
        raw_normal = np.array(
            [world_manifold.normal.x, world_manifold.normal.y], dtype=float
        )
        if self._is_paddle(bodyA):
            normal = raw_normal
        else:
            normal = -raw_normal
        n_norm = float(np.linalg.norm(normal))
        if n_norm <= 1e-8:
            return
        normal = normal / n_norm

        v_paddle = np.array(
            [paddle_body.linearVelocity[0], paddle_body.linearVelocity[1]], dtype=float
        )
        v_puck = np.array(
            [puck_body.linearVelocity[0], puck_body.linearVelocity[1]], dtype=float
        )
        # Relative velocity of puck w.r.t. paddle along the normal.
        # Negative means approaching (puck moving toward paddle).
        v_rel_n = float(np.dot(v_puck - v_paddle, normal))
        approach_speed = max(0.0, -v_rel_n)
        if approach_speed <= 1e-8:
            contact.restitution = 0.0
            return

        combined_e = max(
            float(fixtureA.restitution), float(fixtureB.restitution)
        )

        contact_id = (str(paddle_body.userData), str(puck_body.userData))
        prev = self._pending_paddle_puck.get(contact_id)
        if prev is None or approach_speed > prev["approach_speed"]:
            self._pending_paddle_puck[contact_id] = {
                "approach_speed": approach_speed,
                "normal": normal,
                "restitution": combined_e,
                "paddle_body": paddle_body,
                "puck_body": puck_body,
            }

        contact.restitution = 0.0

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

                # Apply deterministic puck-wall rebound model.
                is_puck_wall = (self._is_puck(bodyA) and self._is_wall(bodyB)) or (self._is_puck(bodyB) and self._is_wall(bodyA))
                if is_puck_wall:
                    puck_body = bodyA if self._is_puck(bodyA) else bodyB
                    puck_name = str(puck_body.userData)
                    pending = self._pending_wall_restitution.pop(puck_name, None)
                    if pending is not None:
                        incoming_speed = float(pending.get("incoming_speed", 0.0))
                        if incoming_speed <= 1e-8:
                            continue
                        normal_unit = np.array(pending.get("normal_inward", (0.0, 0.0)), dtype=float)
                        n_norm = float(np.linalg.norm(normal_unit))
                        if n_norm > 1e-8:
                            normal_unit = normal_unit / n_norm
                            restitution = max(float(pending.get("restitution", 0.0)), 0.0)
                            tier = self._speed_tier(incoming_speed)
                            scale = self.wall_scales[self._tier_index(tier)]
                            if incoming_speed >= self.puck_wall_restitution_threshold_speed:
                                target_outgoing = incoming_speed * restitution * scale
                            else:
                                # For low-speed incoming wall impacts, enforce a deterministic
                                # minimum rebound speed (applies to all wall orientations).
                                target_outgoing = self.puck_wall_min_rebound_speed_below_threshold

                            # Record stat
                            bucket = self._episode_stats["wall"][tier]
                            bucket["count"] += 1
                            bucket["speed_in_sum"] += incoming_speed
                            bucket["speed_out_sum"] += target_outgoing

                            post_vel = np.array(
                                [puck_body.linearVelocity[0], puck_body.linearVelocity[1]],
                                dtype=float,
                            )
                            current_outgoing = max(0.0, float(np.dot(post_vel, normal_unit)))
                            if target_outgoing > current_outgoing + 1e-8:
                                delta_outgoing = target_outgoing - current_outgoing
                                impulse_mag = float(puck_body.mass) * delta_outgoing
                                impulse_vec = normal_unit * impulse_mag
                                puck_body.ApplyLinearImpulse(
                                    b2Vec2(float(impulse_vec[0]), float(impulse_vec[1])),
                                    puck_body.worldCenter,
                                    True,
                                )

                # Apply deterministic paddle-puck restitution model.
                is_paddle_puck = (
                    (self._is_paddle(bodyA) and self._is_puck(bodyB))
                    or (self._is_puck(bodyA) and self._is_paddle(bodyB))
                )
                if is_paddle_puck:
                    paddle_body = bodyA if self._is_paddle(bodyA) else bodyB
                    puck_body = bodyA if self._is_puck(bodyA) else bodyB
                    contact_id = (str(paddle_body.userData), str(puck_body.userData))
                    pending = self._pending_paddle_puck.pop(contact_id, None)
                    if pending is not None:
                        approach_speed = float(pending["approach_speed"])
                        if approach_speed <= 1e-8:
                            continue
                        normal_pp = pending["normal"]
                        e = float(pending["restitution"])

                        tier = self._speed_tier(approach_speed)
                        scale = self.paddle_scales[self._tier_index(tier)]

                        v_paddle_post = np.array(
                            [paddle_body.linearVelocity[0], paddle_body.linearVelocity[1]],
                            dtype=float,
                        )
                        v_puck_post = np.array(
                            [puck_body.linearVelocity[0], puck_body.linearVelocity[1]],
                            dtype=float,
                        )
                        v_rel_n_post = float(np.dot(v_puck_post - v_paddle_post, normal_pp))
                        v_rel_n_desired = e * approach_speed * scale

                        # Record stat
                        bucket = self._episode_stats["paddle"][tier]
                        bucket["count"] += 1
                        bucket["speed_in_sum"] += approach_speed
                        bucket["speed_out_sum"] += v_rel_n_desired
                        delta = v_rel_n_desired - v_rel_n_post
                        if abs(delta) > 1e-8:
                            m_paddle = float(paddle_body.mass)
                            m_puck = float(puck_body.mass)
                            j = delta * m_paddle * m_puck / (m_paddle + m_puck)
                            j_vec = normal_pp * j
                            puck_body.ApplyLinearImpulse(
                                b2Vec2(float(j_vec[0]), float(j_vec[1])),
                                puck_body.worldCenter,
                                True,
                            )
                            paddle_body.ApplyLinearImpulse(
                                b2Vec2(float(-j_vec[0]), float(-j_vec[1])),
                                paddle_body.worldCenter,
                                True,
                            )

class AirHockeyBox2D:
    """Box2D backend.

    Paddle workspace: ``x_min_lim`` … ``y_max`` and ``top_abs`` … ``max_bias_m`` become
    ``self.lims`` / ``self.edge_lims`` and are applied inside this sim (PID / target clip).

    ``AirHockeyBaseEnv`` also clips the policy action *before* ``get_transition`` using
    top-level env config ``paddle_bounds`` / ``paddle_edge_bounds``; those should match
    the same geometry. Keys ``paddle_bounds`` / ``paddle_edge_bounds`` may appear in
    ``kwargs`` (env copies them onto ``simulator_params``) but are not read here.
    """

    def __init__(self, **kwargs):
        explicit_delay_seconds = 'delay_seconds' in kwargs
        explicit_action_lag = 'action_lag' in kwargs
        defaults = {
            'action_x_scaling': 1.0,
            'action_y_scaling': 1.0,
            'rmax_x': 0.26,
            'rmax_y': 0.12,
            # Real-equivalent workspace and edge shaping limits (base frame).
            'x_min_lim': -0.85,
            'x_max_lim': -0.45, # a lot more narrow
            'y_min': -0.37,
            'y_max': 0.37,
            'top_abs': 0.8,
            'bot_abs': 0.1,
            'max_bias_p': -0.15,
            'max_bias_m': -0.15,
            'hist_len': 2,
            'render_masks': False,
            'gravity': -5,
            'paddle_density': 1000,
            # When non-None, the effective paddle density is scaled at init so
            # mass = paddle_density * pi * paddle_mass_reference_radius**2 is
            # preserved regardless of the actual paddle_radius. Use this to
            # perturb paddle_radius without changing paddle mass. Default None
            # leaves behavior unchanged from the prior implementation.
            'paddle_mass_reference_radius': None,
            'puck_density': 250,
            # Same mass-preservation knob for the puck. When non-None,
            # effective puck density is scaled to keep
            # mass = puck_density * pi * puck_mass_reference_radius**2 fixed
            # as puck_radius is varied.
            'puck_mass_reference_radius': None,
            'block_density': 1000,
            'max_paddle_vel': 2,
            'time_frequency': 20,
            'step_frequency': 20,
            'action_step_lag': 0,
            # Ignored in this class; env uses air_hockey.paddle_* for pre-step action clip.
            'paddle_bounds': [],
            'paddle_edge_bounds': [],
            'center_offset_constant': 1.2,
            # When True, paddle targets that overshoot are damped toward the table (see step logic).
            'absorb_target': False,
            'puck_restitution': 1.0,
            'paddle_restitution': 1.0,
            # Wall restitution defaults: side rails (x-min/x-max) are livelier
            # than top/bottom rails (y-min/y-max).
            'side_wall_restitution': 0.99,
            'end_wall_restitution': 0.7,
            # Deterministic puck-wall restitution gate based on relative normal speed.
            'puck_wall_restitution_threshold_speed': 0.25,
            # For low-speed incoming wall impacts, enforce this minimum rebound speed (m/s).
            'puck_wall_min_rebound_speed_below_threshold': 0.1,
            'enable_action_delay': False,
            'enable_observation_delay': False,
            'delay_seconds': 0.025,
            'action_lag': 0.0,
            'puck_noise': False,
            'puck_noise_std': 0.005,
            # Optional stochastic puck occlusions in observations. Per-step
            # start probability is uniform everywhere on the table; once a run
            # starts, its length is drawn from random_occlusion_length_weights.
            'enable_random_occlusions': False,
            'random_occlusion_rate': 0.05,
            # Run-length weights for lengths 1..N (N also sets max consecutive occlusions).
            'random_occlusion_length_weights': [75, 39, 18, 9, 4, 2, 1],
            # Robust derivative-estimation controls used for accel/jerk readouts.
            'derivative_min_dt': 1e-6,
            'acceleration_ema_alpha': 0.35,
            'jerk_ema_alpha': 0.35,
            # <= 0 disables norm clipping.
            'max_acceleration_norm': 0.0,
            'max_jerk_norm': 0.0,
            # Optional crude e-stop simulation using jerk magnitude (m/s^3).
            'simulate_jerk_estop': False,
            'jerk_estop_consecutive_steps': 10,
            'jerk_estop_consecutive_threshold': 18.0,
            'jerk_estop_avg_window_steps': 50,
            'jerk_estop_avg_threshold': 15.0,
            # Edge-preserving sine warp on the puck-y observation only
            # (paddle obs untouched). Models a systematic perception error
            # in the lateral axis. Disabled when amplitude == 0.
            # Defaults for y_left / y_right are the table side walls.
            'puck_obs_sine_warp_amplitude': 0.0,
            'puck_obs_sine_warp_y_left': None,
            'puck_obs_sine_warp_y_right': None,
            # Optionally replace returned state-info motion values to mimic
            # unavailable real-world velocity/jerk sensing.
            'enable_fixed_state_velocity_jerk': False,
            'fixed_state_paddle_velocity': (0.0, 0.0),
            'fixed_state_paddle_jerk': (0.0, 0.0),
            'fixed_state_puck_velocity': (0.0, 0.0),
            'mask_puck_velocity': True,
            # Puck position delay interpolation: simulate timing jitter by
            # interpolating between the previous and current puck position
            # with a random factor uniformly drawn from [min, max].
            # Factor 1.0 = exact current position; <1 = lagging; >1 = extrapolated.
            'enable_puck_delay_interpolation': False,
            'puck_delay_interpolation_min': 0.75,
            'puck_delay_interpolation_max': 1.25,
            # Triangle obstacles: isosceles, point-up; base length = 1.5 * this scale.
            'triangle_obstacle_size': 0.08,
            # High vs puck so b2MixRestitution favors bounce on triangle edges (no listener changes).
            'triangle_obstacle_restitution': 1.15,
            'obstacle_shape': 'triangle',
        }

        kwargs = {**defaults, **kwargs}
        config = dict_to_namespace(kwargs)
        warp_amp = float(config.puck_obs_sine_warp_amplitude)
        warp_y_left = config.puck_obs_sine_warp_y_left
        warp_y_right = config.puck_obs_sine_warp_y_right
        if warp_y_left is None:
            warp_y_left = -float(config.width) / 2.0
        if warp_y_right is None:
            warp_y_right = float(config.width) / 2.0
        self.puck_obs_warp_fn = make_sine_y_warp_fn(
            warp_amp, float(warp_y_left), float(warp_y_right)
        )
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
        self.paddle_restitution = config.paddle_restitution
        self.side_wall_restitution = float(config.side_wall_restitution)
        self.end_wall_restitution = float(config.end_wall_restitution)
        self.puck_wall_restitution_threshold_speed = max(float(config.puck_wall_restitution_threshold_speed), 0.0)
        self.puck_wall_min_rebound_speed_below_threshold = max(
            float(config.puck_wall_min_rebound_speed_below_threshold), 0.0
        )
        self.puck_min_height = (-config.length / 2) + (config.length / 3)
        self.paddle_max_height = 0
        self.block_min_height = 0
        self.max_speed_start = config.width
        self.min_speed_start = -config.width
        self.paddle_density = float(config.paddle_density)
        # Optional mass preservation: if a reference radius is supplied, scale
        # the effective paddle density so mass stays at
        # paddle_density_input * pi * reference**2 regardless of paddle_radius.
        # Density fluctuations (if enabled) act multiplicatively on this
        # mass-preserving baseline.
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
        # Keep tuple ordering consistent with real environment construction.
        self.edge_lims = (self.top_abs, self.bot_abs, self.max_bias_p, self.max_bias_m)
        self.hist_len = config.hist_len
        self.center_offset_constant = config.center_offset_constant
        self.enable_action_delay = bool(config.enable_action_delay)
        self.enable_observation_delay = bool(config.enable_observation_delay)
        self.action_lag = float(config.action_lag)
        assert self.action_lag >= 0 and self.action_lag <= 1, "Action lag must be between 0 and 1"
        base_delay_seconds = float(config.delay_seconds)
        if self.enable_action_delay and (not explicit_delay_seconds) and explicit_action_lag:
            base_delay_seconds = self.action_lag * self.time_per_step
        self.delay_seconds = max(base_delay_seconds, 0.0)
        self.derivative_min_dt = max(float(config.derivative_min_dt), 1e-8)
        self.acceleration_ema_alpha = float(np.clip(config.acceleration_ema_alpha, 0.0, 1.0))
        self.jerk_ema_alpha = float(np.clip(config.jerk_ema_alpha, 0.0, 1.0))
        self.max_acceleration_norm = float(config.max_acceleration_norm)
        self.max_jerk_norm = float(config.max_jerk_norm)
        self.simulate_jerk_estop = bool(config.simulate_jerk_estop)
        self.jerk_estop_consecutive_steps = max(1, int(config.jerk_estop_consecutive_steps))
        self.jerk_estop_consecutive_threshold = float(config.jerk_estop_consecutive_threshold)
        self.jerk_estop_avg_window_steps = max(1, int(config.jerk_estop_avg_window_steps))
        self.jerk_estop_avg_threshold = float(config.jerk_estop_avg_threshold)
        self._jerk_estop_window_len = max(
            self.jerk_estop_consecutive_steps,
            self.jerk_estop_avg_window_steps,
        )
        self._jerk_mag_history = deque(maxlen=self._jerk_estop_window_len)
        self._jerk_estop_latched = False
        self._jerk_estop_reason = None
        self.puck_noise = config.puck_noise
        self.puck_noise_std = float(config.puck_noise_std)
        self.enable_puck_delay_interpolation = bool(config.enable_puck_delay_interpolation)
        self.puck_delay_interpolation_min = float(config.puck_delay_interpolation_min)
        self.puck_delay_interpolation_max = float(config.puck_delay_interpolation_max)
        self._prev_puck_positions_box2d = {}
        self.triangle_obstacle_size = float(config.triangle_obstacle_size)
        self.triangle_obstacle_restitution = float(config.triangle_obstacle_restitution)
        self.obstacle_shape = str(config.obstacle_shape).lower()
        # random occlusion simulation — uniform per-step start probability,
        # run length sampled from random_occlusion_length_weights.
        self.enable_random_occlusions = bool(config.enable_random_occlusions)
        self.random_occlusion_rate = float(np.clip(config.random_occlusion_rate, 0.0, 1.0))
        self.random_occlusion_length_weights = np.array(config.random_occlusion_length_weights, dtype=float).reshape(-1)
        if self.random_occlusion_length_weights.size == 0:
            self.random_occlusion_length_weights = np.array([1.0], dtype=float)
        self._occlusion_max_run = int(self.random_occlusion_length_weights.size)
        self._occlusion_run_remaining = {}
        self._occlusion_last_visible_base = {}
        self._occlusion_prev_occluded = {}
        self._jerk_mag_history.clear()
        self._jerk_estop_latched = False
        self._jerk_estop_reason = None
        self._prev_puck_positions_box2d = {}

        self.last_action = np.zeros(2) # keep the last action taken, used for action lag
        self.last_target_position = None  # base-frame target used for visualization/debugging
        self.previous_acceleration = np.zeros(2)  # for jerk calculation
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

        # Create walls as four fixtures so side vs end restitution can differ.
        self.ground_body = self.world.CreateBody(
            type=b2_staticBody,
            userData="table_wall",
        )
        wall_segments = [
            # Left / right side walls.
            ((self.table_x_min, self.table_y_min), (self.table_x_min, self.table_y_max), self.side_wall_restitution),
            ((self.table_x_max, self.table_y_min), (self.table_x_max, self.table_y_max), self.side_wall_restitution),
            # Bottom / top end walls.
            ((self.table_x_min, self.table_y_min), (self.table_x_max, self.table_y_min), self.end_wall_restitution),
            ((self.table_x_min, self.table_y_max), (self.table_x_max, self.table_y_max), self.end_wall_restitution),
        ]
        for p1, p2, restitution in wall_segments:
            self.ground_body.CreateFixture(
                b2FixtureDef(
                    shape=b2EdgeShape(vertices=[p1, p2]),
                    restitution=float(restitution),
                    friction=0.0,
                )
            )
        self.reset(config.seed)

        # Initialize the contact listener
        self.collision_listener = CollisionForceListener(
            wall_tag="table_wall",
            puck_wall_restitution_threshold_speed=self.puck_wall_restitution_threshold_speed,
            puck_wall_min_rebound_speed_below_threshold=self.puck_wall_min_rebound_speed_below_threshold,
            speed_breakpoints=(0.25, 0.75),
            wall_scales=(1.0, 1.0, 1.0),
            paddle_scales=(1.0, 1.0, 1.0),
            rng=np.random.default_rng(config.seed),
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
        
        if hasattr(self, "collision_listener"):
            self.collision_listener.reset()
            self.collision_listener.set_rng(np.random.default_rng(seed))

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
        self._occlusion_run_remaining = {}
        self._occlusion_last_visible_base = {}
        self._occlusion_prev_occluded = {}
        self._jerk_mag_history.clear()
        self._jerk_estop_latched = False
        self._jerk_estop_reason = None
        
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
        self.obstacle_names = list(self.obstacles.keys())

        self.paddle_names.sort()
        self.puck_names.sort()
        self.block_names.sort()
        self.obstacle_names.sort()

        
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
                        elif (
                            isinstance(value2, list)
                            and len(value2) > 0
                            and all(isinstance(v, tuple) and len(v) == 2 for v in value2)
                        ):
                            state_info[key][i][key2] = [(-v[1], v[0]) for v in value2]
            else:
                for key2, value2 in value.items():
                    for key3, value3 in value2.items():
                        state_info[key][key2][key3] = (-value3[1], value3[0])
        return state_info
    
    def base_coord_to_box2d(self, coord):
        return (coord[1], -coord[0])

    def _coerce_fixed_xy_pair(self, value, field_name):
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

        if len(self.obstacles) > 0:
            state_info["obstacles"] = []
            for obstacle_name, obstacle_data in self.obstacles.items():
                body = obstacle_data["body"]
                center_box2d = (float(body.position[0]), float(body.position[1]))
                # Rotate local Box2D vertices into world, then convert to base frame.
                world_vertices = []
                for fixture in body.fixtures:
                    shape = fixture.shape
                    for local_v in shape.vertices:
                        world_v = body.GetWorldPoint(local_v)
                        world_vertices.append((float(world_v[0]), float(world_v[1])))
                state_info["obstacles"].append(
                    {
                        "name": obstacle_name,
                        "position": center_box2d,
                        "vertices": world_vertices,
                        "size": float(obstacle_data.get("size", self.triangle_obstacle_size)),
                    }
                )

        if len(self.pucks) > 0:
            state_info['pucks'] = []
            for puck_name in self.pucks:
                puck_x_pos_true = self.pucks[puck_name].position[0]
                puck_y_pos_true = self.pucks[puck_name].position[1]
                puck_x_pos_true, puck_y_pos_true = self._apply_puck_delay_interpolation(
                    puck_name, (puck_x_pos_true, puck_y_pos_true)
                )
                puck_x_pos_true, puck_y_pos_true = self._get_noisy_puck_position((puck_x_pos_true, puck_y_pos_true))
                puck_base_xy_true = self._box2d_to_base_coords((puck_x_pos_true, puck_y_pos_true))
                occluded, puck_base_xy_observed = self._update_random_occlusion(
                    puck_name, puck_base_xy_true
                )
                puck_box2d_observed = self._base_to_box2d_coords(puck_base_xy_observed)
                puck_x_vel = self.pucks[puck_name].linearVelocity[0]
                puck_y_vel = self.pucks[puck_name].linearVelocity[1]
                state_info['pucks'].append({'position': (float(puck_box2d_observed[0]), float(puck_box2d_observed[1])),
                                'velocity': (puck_x_vel, puck_y_vel),
                                'occluded': int(occluded)})

        state_info = self.convert_from_box2d_coords(state_info)
        return self._apply_fixed_state_velocity_jerk(state_info)

    def _apply_puck_delay_interpolation(self, puck_name, current_position):
        current = np.array(current_position, dtype=float)
        prev = self._prev_puck_positions_box2d.get(puck_name)
        self._prev_puck_positions_box2d[puck_name] = current.copy()
        if not self.enable_puck_delay_interpolation or prev is None:
            return float(current[0]), float(current[1])
        factor = self.rng.uniform(
            self.puck_delay_interpolation_min,
            self.puck_delay_interpolation_max,
        )
        interpolated = prev + factor * (current - prev)
        return float(interpolated[0]), float(interpolated[1])

    def _get_noisy_puck_position(self, position):
        if not self.puck_noise:
            return position
        noise = self.rng.normal(loc=0.0, scale=self.puck_noise_std, size=2)
        noisy_position = np.array(position, dtype=float) + noise
        return float(noisy_position[0]), float(noisy_position[1])

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

        if self.rng.uniform(0.0, 1.0) < self.random_occlusion_rate:
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
                restitution = self.paddle_restitution,
                filter=b2Filter (maskBits=1,
                                 categoryBits=1)),
            bullet=True,
            position=pos,
            linearVelocity=vel,
            linearDamping=self.paddle_damping,
            userData=name,
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
        body_type = b2_dynamicBody if movable else b2_staticBody
        block = self.world.CreateBody(
            type=body_type,
            position=pos,
            linearVelocity=vel if movable else (0, 0),
            linearDamping=self.puck_damping if movable else 0.0,
            bullet=bool(movable),
            userData=name,
        )
        block.CreateFixture(
            b2FixtureDef(
                shape=b2PolygonShape(vertices=vertices),
                density=self.block_density if movable else 0.0,
                restitution=1.0,
                filter=b2Filter(maskBits=1, categoryBits=1),
                friction=0.0,
            )
        )
        if not affected_by_gravity:
            block.gravityScale = 0
        self.blocks[name] = block
        self.block_initial_positions[name] = pos
        self.object_dict[name] = block

    def spawn_obstacle(self, pos, name, size=None, affected_by_gravity=False, movable=False):
        """Spawn an isosceles triangle obstacle (wide base, single apex).

        ``size`` scales the shape: bottom edge length is ``1.5 * size`` (``0.75 * 2 *
        size``). Apex uses the same height ``h`` as an equilateral triangle of side
        ``size``.

        Vertices are laid out in **base frame** so the base spans **base-y** (horizontal
        on screen after ``render_x = base_y``) and the apex lies toward **negative base-x**
        (goal / ``table_x_top``). The opposite sign (+apex) reads as pointing down after
        the table bitmap’s rotation in ``get_frame()``; keep apex at ``-2h/3``.
        """
        if size is None:
            size = self.triangle_obstacle_size
        size = float(size)
        h = (np.sqrt(3.0) / 2.0) * size
        half_base = 0.75 * size  # full base length = 1.5 * size (along base-y on screen)

        # Centroid at origin; apex toward goal (-x); flat base toward center (+x).
        local_base = [
            np.array([-2.0 * h / 3.0, 0.0], dtype=float),
            np.array([h / 3.0, -half_base], dtype=float),
            np.array([h / 3.0, half_base], dtype=float),
        ]
        vertices_local_box2d = [tuple(self.base_coord_to_box2d(v)) for v in local_base]
        pos_box2d = self.base_coord_to_box2d(pos)

        body_type = b2_dynamicBody if movable else b2_staticBody
        triangle_body = self.world.CreateBody(
            type=body_type,
            position=pos_box2d,
            userData=name,
        )
        triangle_body.CreateFixture(
            b2FixtureDef(
                shape=b2PolygonShape(vertices=vertices_local_box2d),
                density=self.block_density if movable else 0.0,
                restitution=self.triangle_obstacle_restitution,
                filter=b2Filter(maskBits=1, categoryBits=1),
                friction=0.0,
            )
        )
        if not affected_by_gravity:
            triangle_body.gravityScale = 0

        center_base = np.asarray(pos, dtype=float)
        vertices_base = [tuple((center_base + v).tolist()) for v in local_base]
        self.obstacles[name] = {
            "body": triangle_body,
            "center_base": tuple(center_base.tolist()),
            "vertices_base": vertices_base,
            "size": size,
        }
        self.object_dict[name] = triangle_body

    def convert_to_box2d_coords(self, action):
        action = np.array((action[1], -action[0]))
        return action

    def _box2d_to_base_coords(self, coord):
        return np.array((-coord[1], coord[0]), dtype=float)

    def _base_to_box2d_coords(self, coord):
        return np.array((coord[1], -coord[0]), dtype=float)

    @staticmethod
    def _point_segment_distance(point, seg_a, seg_b):
        point = np.asarray(point, dtype=float)
        seg_a = np.asarray(seg_a, dtype=float)
        seg_b = np.asarray(seg_b, dtype=float)
        ab = seg_b - seg_a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-12:
            return float(np.linalg.norm(point - seg_a))
        t = float(np.dot(point - seg_a, ab) / denom)
        t = float(np.clip(t, 0.0, 1.0))
        proj = seg_a + t * ab
        return float(np.linalg.norm(point - proj))

    def _triangle_side_from_contact(self, obstacle_name, contact_point_box2d):
        obstacle = self.obstacles.get(obstacle_name)
        if obstacle is None:
            return None
        verts = obstacle.get("vertices_base", [])
        if len(verts) != 3:
            return None
        contact_base = self._box2d_to_base_coords(contact_point_box2d)
        top, left, right = [np.asarray(v, dtype=float) for v in verts]
        d_left = self._point_segment_distance(contact_base, top, left)
        d_right = self._point_segment_distance(contact_base, top, right)
        d_base = self._point_segment_distance(contact_base, left, right)
        dists = {"left": d_left, "right": d_right, "base": d_base}
        return min(dists, key=dists.get)

    def _compute_triangle_side_hits(self, step_collisions):
        side_counts = {"left": 0, "right": 0, "base": 0}
        details = []
        for collision in step_collisions:
            body_a = str(collision.get("bodyA", ""))
            body_b = str(collision.get("bodyB", ""))
            contact_point = collision.get("contact_point", None)
            if contact_point is None:
                continue
            obstacle_name = None
            hit_body = None
            if body_a.startswith("triangle_obstacle_"):
                obstacle_name = body_a
                hit_body = body_b
            elif body_b.startswith("triangle_obstacle_"):
                obstacle_name = body_b
                hit_body = body_a
            if obstacle_name is None:
                continue
            side = self._triangle_side_from_contact(obstacle_name, contact_point)
            if side is None:
                continue
            side_counts[side] += 1
            details.append({"obstacle": obstacle_name, "side": side, "hit_body": hit_body})
        return side_counts, details

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

    @staticmethod
    def _clip_vector_norm(vec, max_norm):
        if max_norm <= 0:
            return vec
        norm = np.linalg.norm(vec)
        if norm <= max_norm or norm <= 1e-8:
            return vec
        return vec * (max_norm / norm)

    def _update_motion_derivatives(self, initial_vel, final_vel):
        """
        Robust acceleration/jerk estimates with optional EMA smoothing.
        Keeps estimator lightweight: O(1) state per environment step.
        """
        dt = max(float(self.time_per_step), self.derivative_min_dt)
        raw_acceleration = (final_vel - initial_vel) / dt
        raw_acceleration = self._clip_vector_norm(raw_acceleration, self.max_acceleration_norm)

        if not self._has_prev_acceleration:
            filtered_acceleration = raw_acceleration
            filtered_jerk = np.zeros_like(raw_acceleration)
            self._has_prev_acceleration = True
        else:
            alpha_a = self.acceleration_ema_alpha
            filtered_acceleration = (
                alpha_a * raw_acceleration + (1.0 - alpha_a) * self.filtered_acceleration
            )
            raw_jerk = (filtered_acceleration - self.filtered_acceleration) / dt
            raw_jerk = self._clip_vector_norm(raw_jerk, self.max_jerk_norm)
            alpha_j = self.jerk_ema_alpha
            filtered_jerk = alpha_j * raw_jerk + (1.0 - alpha_j) * self.filtered_jerk

        self.filtered_acceleration = filtered_acceleration
        self.filtered_jerk = filtered_jerk
        self.previous_acceleration = filtered_acceleration.copy()
        self.jerk = filtered_jerk.copy()
        return filtered_acceleration, filtered_jerk

    def _update_simulated_jerk_estop(self, jerk_vector):
        jerk_mag = float(np.linalg.norm(np.asarray(jerk_vector, dtype=float)))
        self._jerk_mag_history.append(jerk_mag)
        if self._jerk_estop_latched:
            return
        if not self.simulate_jerk_estop:
            return

        recent_vals = list(self._jerk_mag_history)
        if len(recent_vals) < 1:
            return

        if len(recent_vals) >= self.jerk_estop_consecutive_steps:
            tail = recent_vals[-self.jerk_estop_consecutive_steps:]
            if all(v > self.jerk_estop_consecutive_threshold for v in tail):
                self._jerk_estop_latched = True
                self._jerk_estop_reason = "jerk_consecutive"
                return

        if len(recent_vals) >= self.jerk_estop_avg_window_steps:
            tail = recent_vals[-self.jerk_estop_avg_window_steps:]
            if float(np.mean(tail)) > self.jerk_estop_avg_threshold:
                self._jerk_estop_latched = True
                self._jerk_estop_reason = "jerk_avg"

    def get_singleagent_transition(self, action):
        collision_start_idx = len(self.collision_listener.collision_forces)
        self.observation_state_info = None
        self.observation_puck_history = None
        self.observation_paddle_history = None

        use_delay_logic = self.enable_action_delay or self.enable_observation_delay
        t_delay = float(np.clip(self.delay_seconds, 0.0, self.time_per_step)) if use_delay_logic else 0.0
        self.last_step_delay_seconds = t_delay
        t_action = t_delay if self.enable_action_delay else 0.0
        t_obs = t_delay if self.enable_observation_delay else None
        breakpoints = [0.0, self.time_per_step, t_action]
        if t_obs is not None:
            breakpoints.append(t_obs)
        breakpoints = sorted(set(float(np.clip(t, 0.0, self.time_per_step)) for t in breakpoints))
        
        # Store initial velocity for acceleration calculation over the entire step
        initial_vel = np.array([self.paddles['paddle_ego'].linearVelocity[0], self.paddles['paddle_ego'].linearVelocity[1]])
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
            if end_t <= (t_action + 1e-12):
                act = np.copy(self.last_action)
            else:
                act = np.copy(action)
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
                # Keep overlay target consistent with commanded move geometry even
                # when using the legacy force controller (non-PID).
                target_pos = self._compute_pid_target_pos(pos, act)
                self.last_target_position = self._box2d_to_base_coords(target_pos)
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
            paddle_body = self.paddles["paddle_ego"]
            # pybox2d: assigning ``position`` routes through the internal SetTransform.
            paddle_body.position = (float(pos[0]), float(pos[1]))
            
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

            if t_obs is not None and (not obs_snapshot_recorded) and (end_t >= (t_obs - 1e-12)):
                # Mid-step snapshots reuse currently tracked derivative fields.
                # Acceleration/jerk are only fully refreshed at end-of-step.
                self.observation_state_info = copy.deepcopy(state_info)
                self.observation_puck_history = list(self.puck_history)
                self.observation_paddle_history = list(self.paddle_history)
                obs_snapshot_recorded = True
        
        # Calculate robust acceleration and jerk AFTER the entire action step.
        final_vel = np.array([self.paddles['paddle_ego'].linearVelocity[0], self.paddles['paddle_ego'].linearVelocity[1]])
        current_acceleration, current_jerk = self._update_motion_derivatives(initial_vel, final_vel)
        self.paddles['paddle_ego_acceleration'] = current_acceleration
        self.paddles['paddle_ego_jerk'] = current_jerk
        self._update_simulated_jerk_estop(current_jerk)

        # Refresh state so acceleration/jerk in returned transition are current-step values.
        state_info = self.get_current_state()
        if self._jerk_estop_latched:
            state_info["protective_stop"] = True
            if self._jerk_estop_reason is not None:
                state_info["protective_stop_reason"] = str(self._jerk_estop_reason)
        if t_obs is not None and not obs_snapshot_recorded:
            self.observation_state_info = copy.deepcopy(state_info)
            self.observation_puck_history = list(self.puck_history)
            self.observation_paddle_history = list(self.paddle_history)
        # Count unique paddle<->puck contacts per env step.
        # PostSolve can emit multiple entries for the same physical collision
        # (multiple manifold points / sub-steps), which overestimates counts.
        step_contacted_pucks = set()
        step_collisions = self.collision_listener.collision_forces[collision_start_idx:]
        for collision in step_collisions:
            body_a = str(collision.get("bodyA", ""))
            body_b = str(collision.get("bodyB", ""))
            if body_a == "paddle_ego" and body_b.startswith("puck"):
                step_contacted_pucks.add(body_b)
            elif body_b == "paddle_ego" and body_a.startswith("puck"):
                step_contacted_pucks.add(body_a)
        state_info["paddle_puck_collision_count"] = int(len(step_contacted_pucks))
        triangle_side_hits, triangle_hit_details = self._compute_triangle_side_hits(step_collisions)
        state_info["triangle_side_hits"] = triangle_side_hits
        state_info["triangle_hit_details"] = triangle_hit_details
        
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
                         [self.pucks[bn] for bn in self.pucks.keys()] + \
                         [self.blocks[pn] for pn in self.blocks.keys()] + \
                         [self.obstacles[on]["body"] for on in self.obstacles.keys()])
                        #  [self.obstacles[pn][0] for pn in self.obstacles.keys()] + [self.targets[pn][0] for pn in self.targets.keys()])
        names = self.paddle_names + self.puck_names + self.block_names + self.obstacle_names
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

    def set_collision_scales(self, wall_scales, paddle_scales, speed_breakpoints=None):
        """Update per-tier restitution multipliers. Call at episode boundaries.

        Args:
            wall_scales: sequence of 3 floats [low, mid, high] multiplied onto
                wall restitution for each speed tier.
            paddle_scales: sequence of 3 floats [low, mid, high] multiplied onto
                paddle restitution for each speed tier.
            speed_breakpoints: optional (low_thresh, high_thresh) in m/s.
                Defaults to (0.25, 0.75) if not provided.
        """
        self.collision_listener.set_scales(wall_scales, paddle_scales, speed_breakpoints)

    def get_episode_collision_stats(self):
        """Return per-tier collision stats accumulated since last call and reset counters.

        Returns a dict:
            {
              "wall":   {"low": {"count", "mean_speed_in", "mean_speed_out"}, "mid": ..., "high": ...},
              "paddle": {"low": ..., "mid": ..., "high": ...},
            }
        """
        return self.collision_listener.get_and_reset_episode_stats()
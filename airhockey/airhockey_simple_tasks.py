import math

import numpy as np
from gymnasium.spaces import Box
from .airhockey_base import AirHockeyBaseEnv
from .airhockey_rewards import AirHockeyPuckCatchReward, AirHockeyPuckVelReward, AirHockeyPuckTouchReward, AirHockeyPuckHeightReward, AirHockeyPuckJuggleReward, AirHockeyPuckJuggleLinearTopReward, AirHockeyPuckJuggleNoBaseReward, AirHockeyPuckJuggleUpperHalfReward, AirHockeyPuckJuggleUpperHalfMidBandReward, AirHockeyPuckStrikeReward, AirHockeyStrikeCrowdReward, AirHockeyPaddleFreeMovementReward, AirHockeyPinballTriangleSideReward, AirHockeyTopEdgeSlotGoalReward, AirHockeyTopEdgeVelocityScaledGoalReward

class AirHockeyPuckVelEnv(AirHockeyBaseEnv):
    def get_puck_configuration(self, bad_regions=None):
        # Same spawn as juggle: uniform over the top fraction of the table with a
        # random low-speed heading, rather than a fixed drop from the top edge.
        return self.sample_puck_spawn_top_fraction(bad_regions=bad_regions)

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
    def __init__(self, **kwargs):
        # Multi-puck (num_pucks > 1) staggered spawn. Pucks are placed along a
        # single ballistic "juggle cycle" so the times at which they fall into the
        # paddle-reachable region are evenly spaced, instead of every puck being
        # dropped from the top edge at once.
        self.multipuck_stagger = bool(kwargs.get("multipuck_stagger", True))
        # Apex of the shared cycle, as a fraction of the distance between the
        # reach line and the top edge (1.0 = puck just grazes the top wall).
        self.multipuck_stagger_apex_frac_min = float(
            kwargs.get("multipuck_stagger_apex_frac_min", 0.65)
        )
        self.multipuck_stagger_apex_frac_max = float(
            kwargs.get("multipuck_stagger_apex_frac_max", 0.95)
        )
        if self.multipuck_stagger_apex_frac_max < self.multipuck_stagger_apex_frac_min:
            raise ValueError(
                "multipuck_stagger_apex_frac_max must be >= multipuck_stagger_apex_frac_min"
            )
        # x of the "reachable area" boundary the arrival times are measured to.
        # Defaults to self.paddle_x_min (the far edge of the paddle's workspace).
        reach_x = kwargs.get("multipuck_stagger_reach_x", None)
        self.multipuck_stagger_reach_x = None if reach_x is None else float(reach_x)
        # Lateral (y) speed given to each puck, sampled from [-max, max].
        self.multipuck_stagger_lateral_speed_max = float(
            max(0.0, kwargs.get("multipuck_stagger_lateral_speed_max", 0.1))
        )
        # Minimum |dy| between pucks. They share one x corridor at different
        # times, so separate lanes keep them from colliding mid-flight and
        # destroying the stagger. ``None`` => 4 puck radii.
        min_y_separation = kwargs.get("multipuck_stagger_min_y_separation_m", None)
        self._multipuck_stagger_min_y_separation_m = (
            None if min_y_separation is None else float(max(0.0, min_y_separation))
        )
        # Fraction of one arrival-time slot by which each phase may be jittered.
        self.multipuck_stagger_phase_jitter = float(
            np.clip(kwargs.get("multipuck_stagger_phase_jitter", 0.0), 0.0, 0.5)
        )
        super().__init__(**kwargs)

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
        if self.num_pucks > 1 and self.multipuck_stagger and self._multipuck_stagger_supported():
            self._create_world_objects_staggered_multipuck()
            return

        for i in range(self.num_pucks):
            name = 'puck_{}'.format(i)
            pos, vel = self.get_puck_configuration()
            self.simulator.spawn_puck(pos, vel, name)

        name = 'paddle_ego'
        pos, vel = self.get_paddle_configuration(name)
        self.simulator.spawn_paddle(pos, vel, name)
        self._spawn_triangle_obstacles()

    # ------------------------------------------------------------------
    # Staggered multi-puck spawning
    # ------------------------------------------------------------------
    def _multipuck_fall_dynamics(self):
        """(a, d): down-table acceleration (base +x, m/s^2) and puck linear damping (1/s).

        Base-frame +x points from the goal edge toward the paddle, and Box2D
        gravity (0, g) maps to base +x acceleration of |g| (see
        ``base_coord_to_box2d``), so the puck "falls" toward the paddle.
        """
        world = getattr(self.simulator, "world", None)
        gravity = getattr(world, "gravity", None) if world is not None else None
        if gravity is not None:
            accel = abs(float(gravity[1]))
        else:
            accel = abs(float(np.mean(getattr(self.simulator, "gravity", 0.0))))
        damping = float(getattr(self.simulator, "puck_damping", 0.0) or 0.0)
        return accel, max(0.0, damping)

    def _multipuck_stagger_supported(self):
        accel, _ = self._multipuck_fall_dynamics()
        return accel > 1e-6 and self._multipuck_reach_x() > (self.table_x_top + 3 * self.puck_radius)

    def _multipuck_reach_x(self):
        if self.multipuck_stagger_reach_x is not None:
            return float(self.multipuck_stagger_reach_x)
        return float(self.paddle_x_min)

    @property
    def multipuck_stagger_min_y_separation_m(self):
        if self._multipuck_stagger_min_y_separation_m is not None:
            return self._multipuck_stagger_min_y_separation_m
        return 4.0 * self.puck_radius

    @staticmethod
    def _multipuck_ballistic_state(accel, damping, v0, t):
        """Displacement / velocity along +x at time ``t`` after launch with ``v0``."""
        if damping <= 1e-9:
            return 0.5 * accel * t * t + v0 * t, v0 + accel * t
        v_terminal = accel / damping
        decay = math.exp(-damping * t)
        dx = v_terminal * t + (v0 - v_terminal) * (1.0 - decay) / damping
        return dx, v_terminal + (v0 - v_terminal) * decay

    def _multipuck_apex(self, accel, damping, v0):
        """(t_apex, rise) for an upward (v0 < 0) launch; ``rise`` > 0 is against +x."""
        if v0 >= 0.0:
            return 0.0, 0.0
        if damping <= 1e-9:
            t_apex = -v0 / accel
        else:
            t_apex = math.log(1.0 - v0 * damping / accel) / damping
        dx, _ = self._multipuck_ballistic_state(accel, damping, v0, t_apex)
        return t_apex, -dx

    def _multipuck_launch_speed_for_rise(self, accel, damping, rise):
        """Upward launch speed whose apex is ``rise`` metres above the launch point."""
        lo = 0.0
        hi = max(math.sqrt(2.0 * accel * rise), 1e-3)  # exact when undamped, low otherwise
        for _ in range(60):
            if self._multipuck_apex(accel, damping, -hi)[1] >= rise:
                break
            hi *= 1.5
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if self._multipuck_apex(accel, damping, -mid)[1] < rise:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _multipuck_cycle_time(self, accel, damping, v0, t_apex):
        """Time for the launched puck to fall back to its launch x."""
        lo = t_apex
        hi = max(2.0 * t_apex, t_apex + 1e-3)
        for _ in range(60):
            if self._multipuck_ballistic_state(accel, damping, v0, hi)[0] >= 0.0:
                break
            hi *= 1.5
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if self._multipuck_ballistic_state(accel, damping, v0, mid)[0] < 0.0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _multipuck_y_lanes(self, n):
        """``n`` shuffled y-intervals, one per puck, plus the separation they guarantee.

        The pucks share one x corridor at different times, so they are given
        separate y lanes rather than merely non-overlapping spawn points — two
        pucks in the same lane collide mid-flight and destroy the stagger.
        """
        y_low = self.table_y_left + self.puck_radius
        y_high = self.table_y_right - self.puck_radius
        lane_width = (y_high - y_low) / n
        # Shrink the request if the table is too narrow to honour it.
        separation = min(self.multipuck_stagger_min_y_separation_m, 0.8 * lane_width)
        lanes = [
            (
                y_low + i * lane_width + 0.5 * separation,
                y_low + (i + 1) * lane_width - 0.5 * separation,
            )
            for i in range(n)
        ]
        order = self.rng.permutation(n)
        return [lanes[i] for i in order], separation

    def _multipuck_sample_y_in_lane(self, x_pos, lane, paddle_pos):
        """Sample y inside ``lane``, preferring placements clear of the paddle."""
        paddle_clearance = self.puck_radius + self.paddle_radius + 0.01
        best_y, best_margin = None, -np.inf
        for _ in range(32):
            y_pos = float(self.rng.uniform(low=lane[0], high=lane[1]))
            margin = math.hypot(x_pos - paddle_pos[0], y_pos - paddle_pos[1]) - paddle_clearance
            if margin >= 0.0:
                return y_pos
            if margin > best_margin:
                best_y, best_margin = y_pos, margin
        return best_y

    def _multipuck_lateral_speed(self, time_to_reach, separation):
        """Lateral speed bounded so a puck stays in its lane until it arrives."""
        drift_budget = 0.25 * separation / max(float(time_to_reach), 0.25)
        speed_max = min(self.multipuck_stagger_lateral_speed_max, drift_budget)
        return float(self.rng.uniform(low=-speed_max, high=speed_max))

    def _sample_staggered_multipuck_configurations(self, paddle_pos):
        """Puck states along one juggle cycle, evenly spaced in time-to-reach.

        ``puck_i`` reaches the paddle-reachable region at ``(i + 1) / n`` of the
        cycle, so ``puck_0`` is the one falling in soonest and ``puck_{n-1}`` was
        just launched upward. With two pucks that is one rising and one falling;
        with three, two rising (one near the apex, one just launched) and one
        falling; and so on.
        """
        accel, damping = self._multipuck_fall_dynamics()
        x_reach = self._multipuck_reach_x()

        rise_max = x_reach - (self.table_x_top + 2.0 * self.puck_radius)
        apex_frac = float(
            self.rng.uniform(
                low=self.multipuck_stagger_apex_frac_min,
                high=self.multipuck_stagger_apex_frac_max,
            )
        )
        rise = max(1e-3, apex_frac * rise_max)
        v0 = -self._multipuck_launch_speed_for_rise(accel, damping, rise)
        t_apex, _ = self._multipuck_apex(accel, damping, v0)
        cycle_time = self._multipuck_cycle_time(accel, damping, v0, t_apex)

        n = int(self.num_pucks)
        slot = cycle_time / n
        lanes, separation = self._multipuck_y_lanes(n)
        configurations = []
        for i in range(n):
            time_to_reach = (i + 1) * slot
            if self.multipuck_stagger_phase_jitter > 0.0:
                jitter = self.multipuck_stagger_phase_jitter * slot
                time_to_reach += float(self.rng.uniform(low=-jitter, high=jitter))
            phase = float(np.clip(cycle_time - time_to_reach, 0.0, cycle_time))
            dx, vx = self._multipuck_ballistic_state(accel, damping, v0, phase)
            x_pos = float(
                np.clip(
                    x_reach + dx,
                    self.table_x_top + self.puck_radius,
                    self.table_x_bot - self.puck_radius,
                )
            )
            y_pos = self._multipuck_sample_y_in_lane(x_pos, lanes[i], paddle_pos)
            vy = self._multipuck_lateral_speed(time_to_reach, separation)
            configurations.append(((x_pos, y_pos), (float(vx), vy)))
        return configurations

    def _create_world_objects_staggered_multipuck(self):
        paddle_name = 'paddle_ego'
        paddle_pos, paddle_vel = self.get_paddle_configuration(paddle_name)
        configurations = self._sample_staggered_multipuck_configurations(paddle_pos)
        for i, (pos, vel) in enumerate(configurations):
            self.simulator.spawn_puck(pos, vel, 'puck_{}'.format(i))
        self.simulator.spawn_paddle(paddle_pos, paddle_vel, paddle_name)
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
    """Juggle with a uniformly random start.

    Puck: uniform over the top ``puck_spawn_top_fraction`` of the table, speed
    uniform in ``[puck_spawn_speed_min, puck_spawn_speed_max]`` at a random
    heading (``AirHockeyBaseEnv.sample_puck_spawn_top_fraction``).  Paddle:
    uniform over the reachable workspace.  The old two-regime spawn (85 %
    "linear top" with x cutoffs / 15 % "near paddle") was collapsed into this
    single scheme, so ``puck_spawn_near_paddle_prob`` and
    ``puck_linear_top_spawn_*`` no longer do anything.
    """

    random_paddle_spawn_default = True

    def __init__(self, **kwargs):
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
        del paddle_pos, spawn_near_paddle  # single spawn scheme, no near-paddle regime
        if self.puck_spawn_fixed_x_from_goal_frac is not None:
            return self._sample_puck_fixed_x_from_goal_frac_zero_vel()
        return self.sample_puck_spawn_top_fraction(bad_regions=bad_regions)


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


class AirHockeyPuckTopEdgeGoalTrianglesEnv(AirHockeyPuckJuggleEnv):
    """
    Score in the center 1/5 of the goal-side (top) edge; episode terminates on success.
    Two static triangle obstacles (same Box2D path as pinball / juggle).
    """

    def __init__(self, **kwargs):
        self.top_edge_goal_line_tolerance_m = float(kwargs.get("top_edge_goal_line_tolerance_m", 0.025))
        self.top_edge_goal_reward_bonus = float(kwargs.get("top_edge_goal_reward_bonus", 100.0))
        self.top_edge_goal_visual_offset_m = float(kwargs.get("top_edge_goal_visual_offset_m", 0.012))
        self.top_edge_goal_visual_half_depth_m = float(
            kwargs.get("top_edge_goal_visual_half_depth_m", 0.018)
        )
        super().__init__(**kwargs)

    def initialize_spaces(self, obs_type):
        low, high = self.init_observation(obs_type)
        self.action_space = self.single_action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.reward_range = Box(low=-1, high=1)
        self.count_hit = False
        self.hits = 0
        self.reward = AirHockeyTopEdgeSlotGoalReward(self)
        gx = float(self.table_x_top) + self.top_edge_goal_visual_offset_m
        self.goal_pos = (gx, 0.0)
        self.goal_radius = (self.top_edge_goal_visual_half_depth_m, float(self.width) / 10.0)
        self.goal_draw_shape = "rect"

    @staticmethod
    def from_dict(state_dict):
        return AirHockeyPuckTopEdgeGoalTrianglesEnv(**state_dict)

    def validate_configuration(self):
        assert self.num_pucks == 1
        assert self.num_blocks == 0
        assert self.num_obstacles == 2
        assert self.num_targets == 0
        assert self.num_paddles == 1
        assert self.simulator_name == "box2d"
        if len(self.obstacle_positions) > self.num_obstacles:
            raise ValueError("obstacle_positions has more entries than num_obstacles.")

    def get_puck_configuration(self, bad_regions=None):
        """Spawn on the paddle side with low speed toward the top / obstacles."""
        x_lo = self.table_x_top + 0.28 * float(self.length)
        x_hi = self.table_x_bot * 0.82
        x_pos = float(self.rng.uniform(x_lo, x_hi))
        y_pos = float(self.rng.uniform(-self.width / 3.0, self.width / 3.0))
        speed = float(self.rng.uniform(0.0, 0.35))
        angle = float(self.rng.uniform(-math.pi, math.pi))
        vx = speed * math.cos(angle)
        vy = speed * math.sin(angle)
        return (x_pos, y_pos), (vx, vy)

    def puck_scored_top_edge_goal(self, state_info):
        if "pucks" not in state_info or len(state_info["pucks"]) == 0:
            return False
        px, py = state_info["pucks"][0]["position"]
        vx = state_info["pucks"][0]["velocity"][0]
        if vx >= 0.0:
            return False
        if px > self.table_x_top + self.puck_radius + self.top_edge_goal_line_tolerance_m:
            return False
        half_slot_y = float(self.width) / 10.0
        if abs(float(py)) > half_slot_y + 1e-9:
            return False
        return True

    def has_finished(self, state_info, multiagent=False):
        terminated, truncated, a, b, c, d = super().has_finished(state_info, multiagent)
        if self.puck_scored_top_edge_goal(state_info):
            terminated = True
            self._last_done_reasons["terminated"] = list(
                dict.fromkeys(self._last_done_reasons.get("terminated", []) + ["top_edge_goal"])
            )
        return terminated, truncated, a, b, c, d


class AirHockeyPuckScoreEnv(AirHockeyPuckJuggleLinearTopEnv):
    """
    Score in a band along the goal-side (top) edge; episode terminates on success.
    By default the band is the middle third in ``y`` (half-width ``width/6``).
    Override with ``top_edge_goal_slot_half_width_y_m``. No obstacles. Success
    reward scales with puck x-velocity into the top edge.

    With ``puck_spawn_mode: linear_top`` the puck uses the juggle spawn (uniform
    over the top ``puck_spawn_top_fraction`` of the table, random heading);
    ``puck_spawn_fixed_x_from_goal_frac`` still overrides it. Default
    ``puck_spawn_mode`` is ``score_default`` (legacy paddle-side low-speed spawn).
    """

    def __init__(self, **kwargs):
        self.top_edge_goal_line_tolerance_m = float(kwargs.get("top_edge_goal_line_tolerance_m", 0.025))
        self.top_edge_goal_velocity_reward_scale = float(
            kwargs.get("top_edge_goal_velocity_reward_scale", 10.0)
        )
        self.top_edge_goal_visual_offset_m = float(kwargs.get("top_edge_goal_visual_offset_m", 0.012))
        self.top_edge_goal_visual_half_depth_m = float(
            kwargs.get("top_edge_goal_visual_half_depth_m", 0.018)
        )
        # None => after ``width`` is known, use middle third: half-width ``width / 6``.
        self._user_top_edge_goal_slot_half_width_y_m = kwargs.pop(
            "top_edge_goal_slot_half_width_y_m", None
        )
        self.puck_spawn_mode = str(kwargs.get("puck_spawn_mode", "score_default")).strip().lower()
        self.puck_spawn_affected_by_gravity = bool(
            kwargs.get("puck_spawn_affected_by_gravity", True)
        )
        if self.puck_spawn_mode not in ("score_default", "linear_top"):
            raise ValueError(
                f"Unknown puck_spawn_mode={self.puck_spawn_mode!r} for puck_score. "
                "Expected one of: score_default, linear_top."
            )
        super().__init__(**kwargs)

    def initialize_spaces(self, obs_type):
        low, high = self.init_observation(obs_type)
        self.action_space = self.single_action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.reward_range = Box(low=-1, high=1)
        self.count_hit = False
        self.hits = 0
        self.reward = AirHockeyTopEdgeVelocityScaledGoalReward(self)
        if self._user_top_edge_goal_slot_half_width_y_m is None:
            self.top_edge_goal_slot_half_width_y_m = float(self.width) / 6.0
        else:
            self.top_edge_goal_slot_half_width_y_m = float(
                self._user_top_edge_goal_slot_half_width_y_m
            )
            if self.top_edge_goal_slot_half_width_y_m <= 0.0:
                raise ValueError("top_edge_goal_slot_half_width_y_m must be positive.")
        gx = float(self.table_x_top) + self.top_edge_goal_visual_offset_m
        self.goal_pos = (gx, 0.0)
        self.goal_radius = (
            self.top_edge_goal_visual_half_depth_m,
            float(self.top_edge_goal_slot_half_width_y_m),
        )
        self.goal_draw_shape = "rect"

    @staticmethod
    def from_dict(state_dict):
        return AirHockeyPuckScoreEnv(**state_dict)

    def validate_configuration(self):
        assert self.num_pucks == 1
        assert self.num_blocks == 0
        assert self.num_obstacles == 0
        assert self.num_targets == 0
        assert self.num_paddles == 1
        if len(self.obstacle_positions) > self.num_obstacles:
            raise ValueError("obstacle_positions has more entries than num_obstacles.")

    def create_world_objects(self):
        """Same ordering as juggle/linear-top, but honor ``puck_spawn_affected_by_gravity``."""
        g = self.puck_spawn_affected_by_gravity
        for i in range(self.num_pucks):
            name = "puck_{}".format(i)
            pos, vel = self.get_puck_configuration()
            self.simulator.spawn_puck(pos, vel, name, affected_by_gravity=g)
        name = "paddle_ego"
        pos, vel = self.get_paddle_configuration(name)
        self.simulator.spawn_paddle(pos, vel, name)
        self._spawn_triangle_obstacles()

    def _sample_puck_score_default(self):
        """Legacy score spawn: paddle side, low random speed."""
        x_lo = self.table_x_top + 0.28 * float(self.length)
        x_hi = self.table_x_bot * 0.82
        x_pos = float(self.rng.uniform(x_lo, x_hi))
        y_pos = float(self.rng.uniform(-self.width / 3.0, self.width / 3.0))
        speed = float(self.rng.uniform(0.0, 0.35))
        angle = float(self.rng.uniform(-math.pi, math.pi))
        vx = speed * math.cos(angle)
        vy = speed * math.sin(angle)
        return (x_pos, y_pos), (vx, vy)

    def get_puck_configuration(
        self,
        bad_regions=None,
        paddle_pos=None,
        spawn_near_paddle=False,
    ):
        """Juggle-compatible spawn: fixed-x / linear-top / score default."""
        if self.puck_spawn_fixed_x_from_goal_frac is not None:
            return super().get_puck_configuration(
                bad_regions=bad_regions,
                paddle_pos=paddle_pos,
                spawn_near_paddle=spawn_near_paddle,
            )
        if self.puck_spawn_mode == "linear_top":
            return super().get_puck_configuration(
                bad_regions=bad_regions,
                paddle_pos=paddle_pos,
                spawn_near_paddle=False,
            )
        del bad_regions, paddle_pos, spawn_near_paddle
        return self._sample_puck_score_default()

    def get_top_edge_goal_entry_vx(self, state_info):
        if "pucks" not in state_info or len(state_info["pucks"]) == 0:
            return None
        px, py = state_info["pucks"][0]["position"]
        vx = float(state_info["pucks"][0]["velocity"][0])
        if vx >= 0.0:
            return None
        if px > self.table_x_top + self.puck_radius + self.top_edge_goal_line_tolerance_m:
            return None
        half_slot_y = float(self.top_edge_goal_slot_half_width_y_m)
        if abs(float(py)) > half_slot_y + 1e-9:
            return None
        return vx

    def puck_scored_top_edge_goal(self, state_info):
        return self.get_top_edge_goal_entry_vx(state_info) is not None

    def has_finished(self, state_info, multiagent=False):
        terminated, truncated, a, b, c, d = super().has_finished(state_info, multiagent)
        if self.puck_scored_top_edge_goal(state_info):
            terminated = True
            self._last_done_reasons["terminated"] = list(
                dict.fromkeys(self._last_done_reasons.get("terminated", []) + ["top_edge_goal"])
            )
        return terminated, truncated, a, b, c, d


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


class AirHockeyPuckStrikeEnv(AirHockeyBaseEnv):
    def __init__(self, **kwargs):
        self.puck_spawn_mode = str(
            kwargs.get("puck_spawn_mode", "strike_default")
        ).strip().lower()
        self.puck_spawn_affected_by_gravity = bool(
            kwargs.get("puck_spawn_affected_by_gravity", False)
        )
        self.puck_spawn_near_paddle_prob = float(
            np.clip(kwargs.get("puck_spawn_near_paddle_prob", 0.0), 0.0, 1.0)
        )
        self.puck_near_paddle_offset_min_m = float(
            max(0.0, kwargs.get("puck_near_paddle_offset_min_m", 0.025))
        )
        self.puck_near_paddle_offset_max_m = float(
            max(0.0, kwargs.get("puck_near_paddle_offset_max_m", 0.05))
        )
        if self.puck_near_paddle_offset_max_m < self.puck_near_paddle_offset_min_m:
            self.puck_near_paddle_offset_min_m, self.puck_near_paddle_offset_max_m = (
                self.puck_near_paddle_offset_max_m,
                self.puck_near_paddle_offset_min_m,
            )
        self.puck_near_paddle_horizontal_std_m = float(
            max(0.0, kwargs.get("puck_near_paddle_horizontal_std_m", 0.015))
        )
        self.puck_near_paddle_speed_min_m_s = float(
            max(0.0, kwargs.get("puck_near_paddle_speed_min_m_s", 0.0))
        )
        self.puck_near_paddle_speed_max_m_s = float(
            max(0.0, kwargs.get("puck_near_paddle_speed_max_m_s", 0.2))
        )
        if self.puck_near_paddle_speed_max_m_s < self.puck_near_paddle_speed_min_m_s:
            raise ValueError(
                "puck_near_paddle_speed_max_m_s must be >= puck_near_paddle_speed_min_m_s"
            )
        self.puck_linear_top_spawn_center_cutoff_x = float(
            kwargs.get("puck_linear_top_spawn_center_cutoff_x", 0.2)
        )
        linear_goal_cutoff_x = kwargs.get("puck_linear_top_spawn_goal_cutoff_x", None)
        self.puck_linear_top_spawn_goal_cutoff_x = (
            None if linear_goal_cutoff_x is None else float(linear_goal_cutoff_x)
        )
        self.puck_linear_top_spawn_speed_min = float(
            max(0.0, kwargs.get("puck_linear_top_spawn_speed_min", 0.0))
        )
        self.puck_linear_top_spawn_speed_max = float(
            max(0.0, kwargs.get("puck_linear_top_spawn_speed_max", 0.5))
        )
        if self.puck_linear_top_spawn_speed_max < self.puck_linear_top_spawn_speed_min:
            raise ValueError(
                "puck_linear_top_spawn_speed_max must be >= puck_linear_top_spawn_speed_min"
            )
        frac = kwargs.get("puck_spawn_fixed_x_from_goal_frac", None)
        self.puck_spawn_fixed_x_from_goal_frac = (
            None if frac is None else float(np.clip(float(frac), 0.0, 1.0))
        )
        if self.puck_spawn_mode not in ("strike_default", "linear_top"):
            raise ValueError(
                f"Unknown puck_spawn_mode={self.puck_spawn_mode!r}. "
                "Expected one of: strike_default, linear_top."
            )
        super().__init__(**kwargs)

    def initialize_spaces(self, obs_type):
        # setup observation / action / reward spaces
        low, high = self.init_observation(obs_type)
        self.action_space = self.single_action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32) # 2D action space
        self.reward_range = Box(low=-1, high=1) # need to make sure rewards are between 0 and 1
        self.reward = AirHockeyPuckStrikeReward(self)
        
    @staticmethod
    def from_dict(state_dict):
        return AirHockeyPuckStrikeEnv(**state_dict)

    def _sample_puck_speed_velocity(self, min_speed, max_speed):
        speed = self.rng.uniform(low=min_speed, high=max_speed)
        heading = self.rng.uniform(low=0.0, high=2 * math.pi)
        return (speed * math.cos(heading), speed * math.sin(heading))

    def _sample_puck_strike_default(self):
        puck_x_low = self.length / 5
        puck_x_high = self.length / 3
        puck_y_low = -self.width / 2 + self.puck_radius
        puck_y_high = self.width / 2 - self.puck_radius
        puck_x = self.rng.uniform(low=puck_x_low, high=puck_x_high)
        puck_y = self.rng.uniform(low=puck_y_low, high=puck_y_high)
        return (puck_x, puck_y), (0.0, 0.0)

    def _sample_puck_upper_half_linear_top(self):
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

    def _sample_puck_with_mode(self, paddle_pos=None, spawn_near_paddle=False):
        if self.puck_spawn_fixed_x_from_goal_frac is not None:
            return self._sample_puck_fixed_x_from_goal_frac_zero_vel()
        if spawn_near_paddle and paddle_pos is not None:
            return self._sample_puck_near_paddle(paddle_pos)
        if self.puck_spawn_mode == "linear_top":
            return self._sample_puck_upper_half_linear_top()
        return self._sample_puck_strike_default()

    def create_world_objects(self):
        paddle_pos, paddle_vel = self.get_paddle_configuration("paddle_ego")
        spawn_near_paddle = (
            self.rng.uniform(low=0.0, high=1.0) < self.puck_spawn_near_paddle_prob
        )
        puck_pos, puck_vel = self._sample_puck_with_mode(
            paddle_pos=paddle_pos,
            spawn_near_paddle=spawn_near_paddle,
        )

        self.simulator.spawn_puck(
            puck_pos,
            puck_vel,
            "puck_0",
            affected_by_gravity=self.puck_spawn_affected_by_gravity,
        )
        self.simulator.spawn_paddle(paddle_pos, paddle_vel, "paddle_ego")
    
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
    random_paddle_spawn_default = True

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

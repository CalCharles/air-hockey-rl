import numpy as np
from gymnasium.spaces import Box
from gymnasium import spaces
from airhockey.airhockey_base import AirHockeyBaseEnv
from abc import ABC, abstractmethod
from collections.abc import Iterable

class AirHockeyGoalEnv(AirHockeyBaseEnv, ABC):        
    @abstractmethod
    def initialize_spaces(self, obs_type):
        pass

    @abstractmethod
    def create_world_objects(self):
        pass
    
    @abstractmethod
    def get_achieved_goal(self, state_info):
        pass
    
    @abstractmethod
    def get_desired_goal(self):
        pass

    @abstractmethod
    def get_observation(self, state_info):
        pass
    
    @abstractmethod
    def set_goals(self, goal_radius_type, goal_pos=None, alt_goal_pos=None, goal_set=None):
        pass
    
    @abstractmethod
    def validate_configuration(self):
        pass
    
    @abstractmethod
    def from_dict(state_dict):
        pass
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.reward.compute_reward(achieved_goal, desired_goal, info)
    
    def get_base_reward(self, state_info):
        return self.reward.get_base_reward(state_info)
        
    def get_goal_obs_space(self, low: list, high: list, goal_low: list, goal_high: list):
        return spaces.Dict(dict(
            observation=Box(low=np.array(low), high=np.array(high), dtype=float),
            desired_goal=Box(low=np.array(goal_low), high=np.array(goal_high), dtype=float),
            achieved_goal=Box(low=np.array(goal_low), high=np.array(goal_high), dtype=float)
        ))
        
    # Tasks that can end the episode the moment their goal is met opt in by
    # overriding ``goal_reached``; ``terminate_on_goal_reached`` (config key of
    # the same name) then turns that check into a termination.
    terminate_on_goal_reached = False

    def goal_reached(self, state_info):
        """Whether the goal is met in ``state_info``. Off by default."""
        return False

    def has_finished(self, state_info, multiagent=False):
        result = super().has_finished(state_info, multiagent)
        terminated, truncated = result[0], result[1]
        if (
            self.terminate_on_goal_reached
            and not terminated
            and not truncated
            and self.goal_reached(state_info)
        ):
            reasons = self._last_done_reasons.get("terminated", [])
            self._last_done_reasons["terminated"] = list(
                dict.fromkeys(list(reasons) + ["goal_reached"])
            )
            return (True, truncated) + tuple(result[2:])
        return result

    def get_paddle_workspace_bounds(self, margin=0.0, y=None):
        """Bounds of the paddle centre positions that are actually reachable.

        The env-level ``paddle_bounds`` used for action masking are far looser
        than the real limits: every PID target is re-clipped by the simulator to
        the robot workspace (``x_min_lim`` / ``x_max_lim`` in raw robot x, plus
        ``y_min`` / ``y_max`` and a ``top_abs`` corner wedge), so that clip is
        what binds.  Goal-conditioned tasks sample from here so a goal is never
        outside the paddle's reach.

        Args:
            margin: inset applied on every side, e.g. the goal radius, or the
                distance the paddle needs to accelerate to / decelerate from a
                target velocity.
            y: if given, tighten ``x_max`` by the corner wedge at that y.

        Returns:
            ``(x_min, x_max, y_min, y_max)`` in the env (centered) frame.
        """
        sim = self.simulator
        offset = float(getattr(sim, "center_offset_constant", 0.0))
        x_min_lim = getattr(sim, "x_min_lim", None)
        x_max_lim = getattr(sim, "x_max_lim", None)
        y_min_lim = getattr(sim, "y_min", None)
        y_max_lim = getattr(sim, "y_max", None)

        if None in (x_min_lim, x_max_lim, y_min_lim, y_max_lim):
            # Backend without an explicit workspace clip: fall back to the
            # action-masking bounds, inset by the paddle radius.
            x_lo = self.paddle_x_min + self.paddle_radius
            x_hi = self.paddle_x_max - self.paddle_radius
            y_lo = self.paddle_y_min + self.paddle_radius
            y_hi = self.paddle_y_max - self.paddle_radius
        else:
            x_lo = float(x_min_lim) + offset
            x_hi = float(x_max_lim) + offset
            y_lo, y_hi = float(y_min_lim), float(y_max_lim)
            if y is not None:
                x_hi = min(x_hi, self._workspace_x_max_at_y(y))

        margin = float(margin)
        x_lo, x_hi = x_lo + margin, x_hi - margin
        y_lo, y_hi = y_lo + margin, y_hi - margin
        # A margin wider than the workspace collapses to its centre rather than
        # inverting the sampling range.
        if x_lo > x_hi:
            x_lo = x_hi = 0.5 * (x_lo + x_hi)
        if y_lo > y_hi:
            y_lo = y_hi = 0.5 * (y_lo + y_hi)
        return x_lo, x_hi, y_lo, y_hi

    def _workspace_x_max_at_y(self, y):
        """``x_max`` (centered frame) allowed by the simulator corner wedge at ``y``.

        Mirrors ``x_max = min(x_max_lim, max_bias_m - top_abs * y,
        max_bias_p + top_abs * y)``, taking the smaller of the two bias terms so
        the result holds regardless of how the two are ordered.
        """
        sim = self.simulator
        offset = float(getattr(sim, "center_offset_constant", 0.0))
        top_abs = getattr(sim, "top_abs", None)
        biases = [b for b in (getattr(sim, "max_bias_p", None), getattr(sim, "max_bias_m", None)) if b is not None]
        if not top_abs or not biases:
            return float("inf")
        return float(min(biases)) - float(top_abs) * abs(float(y)) + offset

    def _sync_goal_marker_to_simulator(self):
        """Push ``self.goal_pos`` to the simulator for on-screen visualization.

        No-op when the simulator backend doesn't implement ``set_goal_marker``
        (e.g. Box2D already renders the goal itself via ``AirHockeyRenderer``).
        Called after every ``set_goals`` so non-goal tasks — which never enter
        this class — leave the marker untouched.
        """
        set_marker = getattr(self.simulator, "set_goal_marker", None)
        if not callable(set_marker):
            return
        goal_pos = getattr(self, "goal_pos", None)
        if goal_pos is None:
            return
        set_marker(goal_pos, getattr(self, "goal_radius", None))

    def reset(self, seed=None, **kwargs):
        self.set_goals(self.goal_radius_type)
        obs, success = super().reset(seed, **kwargs)
        self._sync_goal_marker_to_simulator()
        achieved_goal = self.get_achieved_goal(self.current_state)
        desired_goal = self.get_desired_goal()
        if self.return_goal_obs:
            return {"observation": obs, "desired_goal": desired_goal, "achieved_goal": achieved_goal}, success
        else:
            obs = np.concatenate([obs, desired_goal])
            return obs, success

    def reset_from_state_and_goal(self, state_vector, goal_vector, seed=None):
        self.set_goals(None, goal_pos=goal_vector)

        obs, success = super().reset_from_state(state_vector, seed)
        self._sync_goal_marker_to_simulator()

        achieved_goal = self.get_achieved_goal(self.current_state)
        desired_goal = self.get_desired_goal()
        if self.return_goal_obs:
            return {"observation": obs, "desired_goal": desired_goal, "achieved_goal": achieved_goal}, success
        else:
            obs = np.concatenate([obs, desired_goal])
            return obs, success

        
    def set_goal_set(self, goal_set):
        self.goal_set = goal_set

    def step(self, action):
        obs, reward, is_finished, truncated, info = super().step(action)
        info['ego_goal'] = self.goal_pos
        achieved_goal = self.get_achieved_goal(self.current_state)
        desired_goal = self.get_desired_goal()
        if self.return_goal_obs:
            return {"observation": obs, "desired_goal": desired_goal, "achieved_goal": achieved_goal}, reward, is_finished, truncated, info
        else:
            obs = np.concatenate([obs, desired_goal])
            return obs, reward, is_finished, truncated, info
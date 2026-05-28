import numpy as np
from gymnasium.spaces import Box
from gymnasium import spaces
from airhockey.airhockey_base import AirHockeyBaseEnv
from airhockey.sims.real.overlay_utils import enlarged_goal_marker_radius_m
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
        
    def goal_marker_radius_m(self):
        """Radius drawn on the real-camera goal ring (enlarged for visibility)."""
        return enlarged_goal_marker_radius_m(self.simulator, getattr(self, "goal_radius", None))

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
        set_marker(goal_pos, self.goal_marker_radius_m())

    def set_goal_sequence(self, goals):
        """Inject a deterministic, repeatable sequence of goal positions.

        While a sequence is set, every ``reset`` / ``soft_reset`` consumes the
        next ``(x, y)`` from ``goals`` (wrapping with modulo once exhausted)
        and passes it to ``set_goals(goal_pos=...)``, bypassing the random
        uniform sampler. The sequence is independent of ``self.rng``, so the
        same checkpoint evaluated twice will visit the same goals in the
        same order regardless of any other RNG advances.

        Pass ``None`` (or an empty list) to clear the sequence and restore
        the default uniform-sampling behavior. Setting a new sequence also
        resets the internal index to 0.
        """
        if goals is None or len(goals) == 0:
            self._goal_sequence = None
            self._goal_sequence_idx = 0
            return
        self._goal_sequence = [
            np.asarray(g, dtype=float).reshape(2) for g in goals
        ]
        self._goal_sequence_idx = 0

    def rewind_goal_sequence(self) -> None:
        """Undo the last goal-sequence advance.

        Eval uses this when a trajectory is discarded (e.g. too short): the
        inter-episode reset would otherwise consume the next scripted goal even
        though no kept episode completed on the current one.
        """
        seq = getattr(self, "_goal_sequence", None)
        if not seq:
            return
        self._goal_sequence_idx = max(0, int(getattr(self, "_goal_sequence_idx", 0)) - 1)

    def prepare_goal_sequence_for_kept_index(self, kept_index: int) -> None:
        """Position a scripted goal sequence for resuming eval mid-run.

        ``kept_index`` is 0-based: after ``K`` kept episodes the next policy
        episode should use ``goal[K]``. Sets ``_goal_sequence_idx`` to
        ``kept_index + 1`` so a subsequent discard rewind + reset stays aligned.
        """
        seq = getattr(self, "_goal_sequence", None)
        if not seq:
            return
        k = int(kept_index)
        if k < 0:
            k = 0
        goal = seq[k % len(seq)]
        self._goal_sequence_idx = k + 1
        self.set_goals(self.goal_radius_type, goal_pos=goal)
        self._sync_goal_marker_to_simulator()

    def _next_goal_pos_from_sequence(self):
        """Return the next scripted goal (advancing the counter), or None."""
        seq = getattr(self, "_goal_sequence", None)
        if not seq:
            return None
        idx = getattr(self, "_goal_sequence_idx", 0)
        goal = seq[idx % len(seq)]
        self._goal_sequence_idx = idx + 1
        return np.array(goal, dtype=float)

    def _apply_set_goals_with_optional_sequence(self):
        """Either consume the next scripted goal or fall back to random."""
        next_goal = self._next_goal_pos_from_sequence()
        if next_goal is not None:
            self.set_goals(self.goal_radius_type, goal_pos=next_goal)
        else:
            self.set_goals(self.goal_radius_type)

    def reset(self, seed=None, **kwargs):
        self._apply_set_goals_with_optional_sequence()
        obs, success = super().reset(seed, **kwargs)
        self._sync_goal_marker_to_simulator()
        achieved_goal = self.get_achieved_goal(self.current_state)
        desired_goal = self.get_desired_goal()
        if self.return_goal_obs:
            return {"observation": obs, "desired_goal": desired_goal, "achieved_goal": achieved_goal}, success
        else:
            obs = np.concatenate([obs, desired_goal])
            return obs, success

    def soft_reset(self):
        # The base ``soft_reset`` only bumps episode counters and refreshes
        # ``self.current_state``; it does NOT re-sample the goal. Without
        # this override the SOFT reset path (used between most real-world
        # eval / training episodes) leaves ``self.goal_pos`` frozen, so the
        # policy is evaluated on the same goal for many episodes in a row.
        obs, info = super().soft_reset()
        self._apply_set_goals_with_optional_sequence()
        self._sync_goal_marker_to_simulator()
        achieved_goal = self.get_achieved_goal(self.current_state)
        desired_goal = self.get_desired_goal()
        if self.return_goal_obs:
            return {"observation": obs, "desired_goal": desired_goal, "achieved_goal": achieved_goal}, info
        else:
            obs = np.concatenate([obs, desired_goal])
            return obs, info

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
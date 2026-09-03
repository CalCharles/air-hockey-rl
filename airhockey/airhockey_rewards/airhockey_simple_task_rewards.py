from .airhockey_reward_base import AirHockeyRewardBase
import numpy as np

class AirHockeyPuckVelReward(AirHockeyRewardBase):
    """Per-step upward displacement of the puck, from positions only.

    The real robot's vision stack reports puck positions, not velocities, so the
    reward is measured the way the robot can measure it: the change in puck
    position between two consecutive frames.  Base-frame x grows toward the
    agent, so upward motion is a decreasing x and ``prev_x - x`` is the upward
    displacement in metres.  Downward motion pays nothing.

    A step scores zero whenever the displacement cannot actually be measured --
    the first step of an episode, and any step where the puck was occluded at
    either end, since an occluded reading is a placeholder rather than a
    position.  Scale the whole thing with the ``base_reward_scaling`` config key
    if the raw metres are too small (the old velocity-based reward was ~100x
    this at 20 Hz).
    """

    def __init__(self, task_env):
        super().__init__(task_env)
        self._prev_puck_x = None
        self._prev_timestep = None

    @staticmethod
    def _is_occluded(puck):
        occluded = puck.get('occluded', 0)
        return float(np.asarray(occluded).reshape(-1)[0]) > 0.5

    def get_base_reward(self, state_info):
        puck = state_info['pucks'][0]
        puck_x = float(puck['position'][0])
        occluded = self._is_occluded(puck)
        timestep = self.task_env.current_timestep

        prev_x, prev_timestep = self._prev_puck_x, self._prev_timestep
        # current_timestep restarts at 0 every episode, so a non-consecutive
        # timestep also covers the episode boundary.
        self._prev_puck_x = None if occluded else puck_x
        self._prev_timestep = timestep

        measurable = (
            not occluded
            and prev_x is not None
            and prev_timestep == timestep - 1
        )
        upward_displacement = (prev_x - puck_x) if measurable else 0.0

        reward = max(upward_displacement, 0.0)
        puck_height = -puck_x
        success = puck_height > 0.5 and timestep > 25
        return reward, success


class AirHockeyPuckHeightReward(AirHockeyRewardBase):
    def __init__(self, task_env):
        super().__init__(task_env)
        
    def get_base_reward(self, state_info):
        puck_height = -state_info['pucks'][0]['position'][0]
        puck_vel = -state_info['pucks'][0]['velocity'][0]
        puck_pos = state_info['pucks'][0]['position']

        paddle_pos = state_info['paddles']['paddle_ego']['position']
        min_dist = self.task_env.paddle_radius + self.task_env.puck_radius
        dist = np.linalg.norm(np.array(puck_pos) - np.array(paddle_pos))

        reward = max(puck_vel, 0) * 5 if puck_height < 0 else max(puck_vel, 0) * -10
        success = puck_height > 0 and self.task_env.current_timestep > 25

        if dist - min_dist < 0.05:
            if not self.task_env.touching:
                reward += 20
                self.task_env.num_touches += 1
            self.task_env.touching = True
        else:
            self.task_env.touching = False

        if success:
            reward = 60
        return reward, success


class AirHockeyPuckCatchReward(AirHockeyRewardBase):
    def __init__(self, task_env):
        super().__init__(task_env)

    def get_base_reward(self, state_info):
        # reward for getting close to the puck, but make sure not to displace it
        puck_pos = state_info['pucks'][0]['position']
        paddle_pos = state_info['paddles']['paddle_ego']['position']
        dist = np.linalg.norm(np.array(puck_pos) - np.array(paddle_pos))
        max_dist = 0.16 * self.task_env.width
        min_dist = self.task_env.paddle_radius + self.task_env.puck_radius
        reward = 1 - ((dist - min_dist) / (max_dist - min_dist))
        reward = max(reward, 0)
        success = reward >= 0.9 and self.task_env.current_timestep > 75
        return reward, success


class AirHockeyPuckJuggleReward(AirHockeyRewardBase):
    def __init__(self, task_env):
        super().__init__(task_env)
        self.hit_counter = 0
        self.hit_cooldown = False

    def get_base_reward(self, state_info):
        reward = self.original_region_reward(state_info) + self.top_bumping_reward(state_info)
        success = reward > 0 and self.task_env.current_timestep > 50
        return reward, success
    
    def top_bumping_reward(self, state_info):
        bump_top = state_info['paddles']['paddle_ego']['position'][0] < 0 + 4 * self.task_env.paddle_radius
        
        if bump_top:
            return -5
        
        return 0

    def original_region_reward(self, state_info):
        reward = 0
        
        for puck in state_info["pucks"]:
            x_pos = puck['position'][0]
            x_higher = self.task_env.table_x_top
            x_lower = self.task_env.table_x_bot
            if x_higher / 4 < x_pos < 0:
                reward += 15 / len(state_info["pucks"])
            elif x_pos < x_higher / 4:
                reward -= 1 / len(state_info["pucks"])
        
        return reward

    def vel_reward(self, state_info):
        reward = 0
        max_vel = self.task_env.table_x_bot * 2
        for puck in state_info["pucks"]:
            vel = puck['velocity'][0]
            if 0.1 < vel < 0.5:
                reward += (vel / max_vel) * 5 / len(state_info["pucks"])
            elif vel > 0.5:
                reward -= (vel / max_vel) * 5 / len(state_info["pucks"])
        return reward

    def low_vel_x_correc_region_reward(self, state_info, min_vel=0, max_vel=0.3):
        reward = 0
        max_expected = self.task_env.table_x_bot * 2
        for puck in state_info["pucks"]:
            vel = puck['velocity'][0]
            x_pos = puck['position'][0]
            if min_vel < vel < max_vel and self.task_env.table_x_top / 4 < x_pos < 0:
                reward += 1 / len(state_info["pucks"])
            else:
                reward -= 0.05
        return reward
    
    def x_potential_reward(self, state_info):
        reward = 0
        max_distance = self.task_env.table_x_bot * 2
        for puck in state_info["pucks"]:
            x_pos = puck['position'][0]
            target_pos = self.task_env.table_x_top * 1 / 2
            distance_to_target = abs(x_pos - target_pos)
            reward += (distance_to_target / max_distance) * 10 / len(state_info["pucks"])
        return reward

    def hit_reward(self, state_info):
        reward = 0
        paddle_pos = state_info['paddles']['paddle_ego']['position']
        min_dist = self.task_env.paddle_radius + self.task_env.puck_radius

        for puck in state_info["pucks"]:
            puck_pos = puck['position']
            dist = np.linalg.norm(np.array(puck_pos) - np.array(paddle_pos))
            if not self.hit_cooldown and abs(dist - min_dist) < 0.02:
                self.hit_counter += 1
                self.hit_cooldown = True
                reward += 2  # 2 points for each hit
            elif self.hit_cooldown and dist > (min_dist + 0.1):
                self.hit_cooldown = False
        
        return reward

    def hit_low_vel_potential(self, state_info, min_vel=0, max_vel=0.3):
        reward = self.hit_reward(state_info)
        max_distance = self.task_env.table_x_bot * 2
        for puck in state_info["pucks"]:
            x_pos = puck['position'][0]
            target_pos = self.task_env.table_x_top * 3 / 4
            distance_to_target = abs(x_pos - target_pos)
            vel = puck['velocity'][0]
            if min_vel < vel < max_vel:
                reward += (distance_to_target / max_distance) * 10 / len(state_info["pucks"])
        return reward

    def y_position_reward(self, state_info):
        reward = 0
        for puck in state_info["pucks"]:
            y_pos = puck['position'][1]
            if -self.task_env.width / 4 < y_pos < self.task_env.width / 4:
                reward += 5 / len(state_info["pucks"])
        return reward

    def combo_hits_reward(self, state_info):
        reward = self.hit_reward(state_info)
        if self.hit_counter > 1:
            reward = 5 * (self.hit_counter - 1)
        return reward

class AirHockeyPuckJuggleLinearTopReward(AirHockeyPuckJuggleReward):
    def original_region_reward(self, state_info):
        reward = 0

        for puck in state_info["pucks"]:
            x_pos = puck["position"][0]
            x_higher = self.task_env.table_x_top
            x_optimal_start = x_higher / 4

            if x_optimal_start < x_pos < 0:
                reward += 3 / len(state_info["pucks"])
            elif x_pos <= x_optimal_start:
                # Linear shaping over the top band: x_higher -> -1 and x_higher/4 -> +3.
                t = (x_pos - x_higher) / (x_optimal_start - x_higher)
                t = np.clip(t, 0.0, 1.0)
                shaped_reward = -1 + 4 * t
                reward += shaped_reward / len(state_info["pucks"])

        return reward


class AirHockeyPuckJuggleNoBaseReward(AirHockeyPuckJuggleLinearTopReward):
    def get_base_reward(self, state_info):
        # Preserve success logic from linear-top shaping while removing base reward signal.
        _, success = super().get_base_reward(state_info)
        return 0.0, success


class AirHockeyPuckJuggleUpperHalfReward(AirHockeyPuckJuggleLinearTopReward):
    def get_base_reward(self, state_info):
        _, success = super().get_base_reward(state_info)
        reward = self.upper_half_reward(state_info)
        return reward, success

    def upper_half_reward(self, state_info):
        bonus_reward = 0.0
        x_midpoint = (self.task_env.table_x_top + self.task_env.table_x_bot) / 2.0
        num_pucks = max(len(state_info["pucks"]), 1)

        for puck in state_info["pucks"]:
            if puck["position"][0] <= x_midpoint:
                bonus_reward += 1.0 / num_pucks

        return bonus_reward


class AirHockeyPuckJuggleUpperHalfMidBandReward(AirHockeyPuckJuggleLinearTopReward):
    """
    Reward pattern along x (bottom -> top): 0, +1, 0.
    +1 is only within the lower 3/4 of the upper half, excluding the top 1/4.
    """

    def get_base_reward(self, state_info):
        _, success = super().get_base_reward(state_info)
        reward = self.upper_half_mid_band_reward(state_info)
        return reward, success

    def upper_half_mid_band_reward(self, state_info):
        bonus_reward = 0.0
        x_top = self.task_env.table_x_top
        x_bot = self.task_env.table_x_bot
        x_midpoint = (x_top + x_bot) / 2.0
        upper_half_height = x_midpoint - x_top
        top_quarter_cutoff = x_top + 0.25 * upper_half_height
        num_pucks = max(len(state_info["pucks"]), 1)

        for puck in state_info["pucks"]:
            x_pos = puck["position"][0]
            # Reward only in middle band of the upper half.
            if top_quarter_cutoff <= x_pos <= x_midpoint:
                bonus_reward += 1.0 / num_pucks

        return bonus_reward


class AirHockeyPuckStrikeReward(AirHockeyRewardBase):
    def __init__(self, task_env):
        super().__init__(task_env)
        
    def get_base_reward(self, state_info):
        x_vel = state_info['pucks'][0]['velocity'][0]
        y_vel = state_info['pucks'][0]['velocity'][1]
        vel_mag = np.linalg.norm(np.array([x_vel, y_vel]))
        reward = vel_mag
        max_rew = 2  # estimated max vel
        min_rew = 0  # min acceptable good velocity

        initial_pos = self.task_env.puck_initial_position
        current_pos = state_info['pucks'][0]['position']
        dist = np.linalg.norm(np.array(initial_pos) - np.array(current_pos))
        has_moved = dist > 0.01

        if reward <= min_rew and not has_moved:
            return -5, False  # negative rew for standing still and hasn't moved
        reward = min(reward, max_rew)
        reward = (reward - min_rew) / (max_rew - min_rew)
        success = reward > (0.1)  # means the puck is moving
        if reward > 0:
            reward *= 10
        return reward, success


class AirHockeyPuckTouchReward(AirHockeyRewardBase):
    """+1 on the step the paddle touches the puck, 0 otherwise.

    The touch threshold is the one ``terminate_on_puck_hit_paddle`` uses, so with
    that flag on (the default for the puck_touch configs) the episode ends on the
    touch and the +1 is the whole episode return.  The bonus is awarded once per
    episode either way.
    """

    def __init__(self, task_env):
        super().__init__(task_env)

    def get_base_reward(self, state_info):
        puck_pos = state_info['pucks'][0]['position']
        paddle_pos = state_info['paddles']['paddle_ego']['position']
        dist = np.linalg.norm(np.array(puck_pos) - np.array(paddle_pos))
        touch_dist = self.task_env.paddle_radius + self.task_env.puck_radius + 0.02

        success = bool(dist < touch_dist)
        reward = 1.0 if (success and not self.task_env.success_in_ep) else 0.0
        return reward, success


class AirHockeyPaddleFreeMovementReward(AirHockeyRewardBase):
    def __init__(self, task_env):
        super().__init__(task_env)

    def get_base_reward(self, state_info):
        # Return zero reward - pure AMP learning without task rewards
        return 0.0, False

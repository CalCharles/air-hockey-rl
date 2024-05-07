import numpy as np
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
class RewardRegion():
    def __init__(self, reward_value_range, scale_range, limits, rad_limits, shapes, reset=True):
        self.reward_value_range = reward_value_range
        self.scale_range = scale_range
        self.shapes = shapes
        self.shape_onehot_helper = np.eye(len(shapes))
        self.limits = limits
        self.limit_range = self.limits[1] - self.limits[0]
        self.rad_limits = rad_limits
        if reset: self.reset()

    def reset(self):
        self.state = np.random.rand(*self.limits[0].shape) * self.limit_range + self.limits[0]
        self.reward_value = np.random.rand() * (self.reward_value_range[1] - self.reward_value_range[0]) + self.reward_value_range[0]
        self.scale = np.random.rand() * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
        self.shape_idx = np.random.randint(len(self.shapes))
        self.shape = self.shapes[self.shape_idx]
        self.radius = np.random.rand(*self.rad_limits[1].shape) * (self.rad_limits[1] - self.rad_limits[0]) + self.rad_limits[0]
        if (self.shape == "circle" or self.shape == "square") and len(self.rad_limits[0]) > 1: self.radius = self.radius[0]

    def get_state(self):
        if isinstance(self.radius, Iterable):
            radius = np.pad(self.radius, (0,3-len(self.radius)), constant_values=self.radius[0])
        else:
            radius = np.pad(np.array([self.radius]), (0,2), constant_values=self.radius)
        return np.concatenate([self.state, [self.scale], [self.reward_value], radius, self.shape_onehot_helper[self.shape_idx]])

    def check_reward(self, obj_state):
        if self.shape == "circle" or self.shape == "ellipse":
            norm_dist = np.sum(np.square(obj_state - self.state) / np.square(self.radius))
        elif self.shape == "diamond":
            norm_dist = np.sum(np.abs(obj_state - self.state) / self.radius)
        elif self.shape == "rect" or self.shape == "rectangle" or self.shape == "square":
            norm_dist = np.max(np.abs(obj_state - self.state) / self.radius)

        return float(norm_dist <= 1) * np.exp(-self.scale * norm_dist) * self.reward_value


class DynamicRewardRegion(RewardRegion):
    def __init__(self, reward_value_range, scale_range, limits, rad_limits, shapes, movement_patterns, velocity_limits, use_reset=True):
        super().__init__(reward_value_range, scale_range, limits, rad_limits, shapes, reset=False)
        self.movement_patterns = movement_patterns
        self.movement_onehot_helper = np.eye(len(movement_patterns))
        self.velocity_limits = velocity_limits
        self.velocity_limit_range = self.velocity_limits[1] - self.velocity_limits[0]
        if use_reset: self.reset()

    def reset(self):
        super().reset()
        self.velocity = np.random.rand(self.velocity_limits[0].shape[0]) * self.velocity_limit_range + self.velocity_limits[0]
        self.movement_idx = np.random.randint(len(self.movement_patterns))
        self.movement = self.movement_patterns[self.movement_idx]

    def get_state(self):
        radius_obs = [self.radius] if not isinstance(self.radius, Iterable) else self.radius
        return np.concatenate([self.state, self.velocity, [self.scale], [self.reward_value], radius_obs, self.shape_onehot_helper[self.shape_idx], self.movement_onehot_helper[self.movement_idx]])

    def step(self, env_state=None, action=None):
        next_state = self.state + self.velocity
        hit_top_lim = next_state[1] > self.limits[1][1]
        hit_bot_lim = next_state[1] < self.limits[0][1]
        hit_right_lim = next_state[0] > self.limits[1][0]
        hit_left_lim = next_state[0] < self.limits[0][0]
        hit = hit_top_lim or hit_bot_lim or hit_right_lim or hit_left_lim
        
        if hit:
            if self.movement == "bounce":
                if hit_top_lim or hit_bot_lim:
                    self.velocity[1] = -self.velocity[1]
                    next_state = self.state + self.velocity
                if hit_right_lim or hit_left_lim:
                    self.velocity[0] = -self.velocity[0]
                    next_state = self.state + self.velocity
            elif self.movement == "through":
                if hit_top_lim:
                    next_state[1] = self.limits[0][1]
                if hit_bot_lim:
                    next_state[1] = self.limits[1][1]
                if hit_right_lim:
                    next_state[0] = self.limits[0][0]
                if hit_left_lim:
                    next_state[0] = self.limits[1][0]
            elif self.movement == "top_reset":
                next_state[1] = self.limits[0][1]
                next_state[0] = np.random.rand() * (self.limits[0][1] - self.limits[0][0]) + self.limits[0][0]
        self.state = next_state
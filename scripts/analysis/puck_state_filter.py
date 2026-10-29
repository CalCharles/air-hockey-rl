"""Experimental puck observer; retained for a negative offline diagnostic.

Uses only the latest noisy puck xy and its occlusion flag, with fixed canonical
mean physics. No environment, simulator body, sampled DR factor, reward or
action is accessed. It never modifies an observation or emits a control action.
It is not connected to a trainer or deployed policy.
"""
from __future__ import annotations

import numpy as np


class PuckStateFilter:
    def __init__(self, *, dt=.05, latency=.025, accel_std=4., measurement_std=.01,
                 gravity=.73, damping=.11, walls=True):
        self.dt, self.latency = float(dt), float(latency)
        self.accel_std, self.measurement_std = float(accel_std), float(measurement_std)
        self.gravity, self.damping, self.walls = float(gravity), float(damping), bool(walls)
        if not all(np.isfinite(v) for v in (dt, latency, accel_std, measurement_std, gravity, damping)):
            raise ValueError("Filter parameters must be finite")
        if dt <= 0 or latency < 0 or accel_std < 0 or measurement_std <= 0 or damping < 0:
            raise ValueError("Invalid filter times/noise/damping")
        self.state = np.zeros(4, dtype=np.float64)
        self.covariance = np.eye(4, dtype=np.float64)
        self.initialized = False

    def _motion(self, dt):
        drag = 1. / (1. + self.damping * dt)
        matrix = np.eye(4)
        matrix[0, 2] = matrix[1, 3] = .5 * dt * (1. + drag)
        matrix[2, 2] = matrix[3, 3] = drag
        force = np.array([.5 * drag * self.gravity * dt ** 2, 0., drag * self.gravity * dt, 0.])
        return matrix, force

    def _reflect(self, state):
        # Known table geometry, not the episode's hidden randomized parameters.
        derivative = np.ones(4)
        if self.walls:
            for axis, bound, restitution in ((0, 1.9304 / 2 - .03175, .55),
                                              (1, .8636 / 2 - .03175, .90)):
                if abs(state[axis]) > bound:
                    side = np.sign(state[axis])
                    state[axis] = 2 * side * bound - state[axis]
                    state[axis + 2] = -side * abs(state[axis + 2]) * restitution
                    derivative[axis], derivative[axis + 2] = -1., -restitution
        return derivative

    def estimate(self):
        matrix, force = self._motion(self.latency)
        result = matrix @ self.state + force
        self._reflect(result)
        return result.astype(np.float32)

    def reset(self, observation):
        self.state[:] = 0.
        self.covariance = np.diag([.01, .01, 1., 1.])
        self.initialized = False
        return self.update(observation, advance=False)

    def update(self, observation, *, advance=True):
        obs = np.asarray(observation, dtype=np.float64).reshape(-1)
        if len(obs) < 30:
            raise ValueError("Expected canonical 30-dimensional history observation")
        if advance:
            matrix, force = self._motion(self.dt)
            self.state = matrix @ self.state + force
            noise_map = np.array([[.5 * self.dt ** 2, 0.], [0., .5 * self.dt ** 2],
                                  [self.dt, 0.], [0., self.dt]])
            self.covariance = matrix @ self.covariance @ matrix.T + self.accel_std ** 2 * noise_map @ noise_map.T
            derivative = self._reflect(self.state)
            self.covariance *= derivative[:, None] * derivative[None, :]
        if obs[29] <= .5 and np.isfinite(obs[27:29]).all():
            if not self.initialized:
                self.state[:2] = obs[27:29]
                self.initialized = True
            innovation = self.covariance[:2, :2] + np.eye(2) * self.measurement_std ** 2
            gain = np.linalg.solve(innovation, self.covariance[:2, :]).T
            self.state += gain @ (obs[27:29] - self.state[:2])
            # Joseph form preserves positive semidefinite covariance.
            residual_map = np.eye(4)
            residual_map[:, :2] -= gain
            self.covariance = (residual_map @ self.covariance @ residual_map.T
                               + self.measurement_std ** 2 * gain @ gain.T)
        return self.estimate()

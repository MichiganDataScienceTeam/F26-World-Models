"""Torque-controlled inverted pendulum, matching Gymnasium's nominal dynamics."""

from dataclasses import dataclass, replace

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .common import broadcast_inputs, vector


@dataclass(frozen=True)
class PendulumPhysics:
    gravity: float = 10.0
    mass: float = 1.0
    length: float = 1.0
    damping: float = 0.0  # angular damping

    def __post_init__(self):
        for name in ("gravity", "mass", "length", "damping"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0 or (name in ("mass", "length") and value == 0):
                raise ValueError(f"Invalid {name}")


class PendulumEnv(gym.Env):
    """obs=[cos(theta),sin(theta),angular_velocity], action=[torque]; upright theta=0."""

    metadata = {"render_modes": []}

    def __init__(self, physics=None, dt=.05, max_steps=200, max_torque=2., max_speed=8.):
        if not np.isfinite([dt, max_torque, max_speed]).all() or min(dt, max_torque, max_speed) <= 0:
            raise ValueError("dt, max_steps, max_torque and max_speed must be positive")
        if max_steps is not None and (
                isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1):
            raise ValueError("dt, max_steps, max_torque and max_speed must be positive")

        self.physics = physics or PendulumPhysics()
        self.dt, self.max_steps = dt, max_steps
        self.max_torque, self.max_speed = max_torque, max_speed

        self.action_space = spaces.Box(-max_torque, max_torque, (1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            np.array([-1., -1., -max_speed], np.float32),
            np.array([1., 1., max_speed], np.float32), dtype=np.float32)

        self.state = None
        self.steps, self._done = 0, False

    @property
    def observation(self):
        if self.state is None:
            raise RuntimeError("Call reset() first")

        theta, velocity = self.state
        return np.array([np.cos(theta), np.sin(theta), velocity], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if set(options) - {"state"}:
            raise ValueError("reset option is state=[theta,angular_velocity]")

        state = (vector(options["state"], 2, "state") if "state" in options else
                 self.np_random.uniform([-np.pi, -1.], [np.pi, 1.]))
        if abs(state[1]) > self.max_speed:
            raise ValueError("Initial speed exceeds max_speed")

        self.state, self.steps, self._done = state, 0, False
        return self.observation, {"elapsed_time": 0.}

    def predict(self, observation, action):
        """Predict the next observation without advancing the live environment."""

        obs, action = broadcast_inputs(observation, action, (3,), 1)
        p = self.physics
        theta = np.arctan2(obs[..., 1], obs[..., 0])
        torque = np.clip(action[..., 0], -self.max_torque, self.max_torque)

        acceleration = (3*p.gravity/(2*p.length)*np.sin(theta)
                        + 3*torque/(p.mass*p.length**2) - p.damping*obs[..., 2])
        velocity = np.clip(obs[..., 2] + self.dt*acceleration, -self.max_speed, self.max_speed)
        theta = theta + self.dt*velocity
        return np.stack((np.cos(theta), np.sin(theta), velocity), axis=-1)

    def cost(self, observation, action):
        obs, action = broadcast_inputs(observation, action, (3,), 1)
        theta = np.arctan2(obs[..., 1], obs[..., 0])
        torque = np.clip(action[..., 0], -self.max_torque, self.max_torque)
        return theta**2 + .1*obs[..., 2]**2 + .001*torque**2

    def project(self, observation):
        """Normalize the angle encoding and limit angular velocity."""

        obs = np.asarray(observation).copy()
        norm = np.linalg.norm(obs[..., :2], axis=-1, keepdims=True)
        obs[..., :2] = np.divide(obs[..., :2], norm, out=np.zeros_like(obs[..., :2]), where=norm > 1e-8)
        obs[..., 0] = np.where(norm[..., 0] > 1e-8, obs[..., 0], 1.)
        obs[..., 2] = np.clip(obs[..., 2], -self.max_speed, self.max_speed)
        return obs

    def step(self, action):
        if self.state is None or self._done:
            raise RuntimeError("Call reset() before stepping a new or finished episode")

        applied = np.clip(vector(action, 1, "action"), -self.max_torque, self.max_torque)
        reward = -float(self.cost(self.observation, applied))
        obs = self.predict(self.observation, applied)
        self.state = np.array([np.arctan2(obs[1], obs[0]), obs[2]])

        self.steps += 1
        self._done = self.max_steps is not None and self.steps >= self.max_steps
        return self.observation, reward, False, self._done, {
            "elapsed_time": self.steps*self.dt, "applied_action": applied.copy()}

    def set_physics(self, **changes):
        self.physics = replace(self.physics, **changes)

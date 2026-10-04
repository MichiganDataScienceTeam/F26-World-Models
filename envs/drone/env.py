from dataclasses import replace

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ..common import broadcast_inputs, vector
from .config import DronePhysics, DroneSimulation, DroneTask
from .dynamics import integrate, running_cost


class DroneEnv(gym.Env):
    """obs=[x,z,theta,vx,vz,omega,target_x,target_z], action=[left,right].

    Throttle is in [0,1]; only the first six values are physical state.
    """

    metadata = {"render_modes": []}

    def __init__(self, physics=None, simulation=None, task=None):
        self.physics = physics or DronePhysics()
        self.simulation = simulation or DroneSimulation()
        self.task = task or DroneTask()

        self.action_space = spaces.Box(0., 1., shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)

        self.state = None
        self.target = np.array(self.task.target, dtype=float)
        self.steps, self._done = 0, False

    @property
    def dt(self):
        return self.simulation.dt

    @property
    def observation(self):
        if self.state is None:
            raise RuntimeError("Call reset() first")

        return np.r_[self.state, self.target].astype(np.float32)

    def _info(self):
        return {"elapsed_time": self.steps * self.dt,
                "position_error": float(np.linalg.norm(self.state[:2] - self.target))}

    def is_failure(self, observation):
        s = np.asarray(observation)
        outside = ((s[..., 0] <= self.task.x_bounds[0]) | (s[..., 0] >= self.task.x_bounds[1])
                   | (s[..., 1] <= self.task.z_bounds[0]) | (s[..., 1] >= self.task.z_bounds[1]))
        return outside & self.task.terminate_on_bounds

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if set(options) - {"state", "target"}:
            raise ValueError("reset options are state and target")

        target = vector(options.get("target", self.task.target), 2, "target")
        if "state" in options:
            state = vector(options["state"], 6, "state")
        else:
            width = np.asarray(self.task.reset_noise)
            state = np.r_[target, 0., 0., 0., 0.] + self.np_random.uniform(-width, width)

        if self.is_failure(state):
            raise ValueError("Initial state is outside task bounds")
        self.state, self.target = state, target
        self.steps, self._done = 0, False
        return self.observation, self._info()

    def predict(self, observation, action):
        """Predict the next observation without advancing the live environment."""

        obs, action = broadcast_inputs(observation, action, (6, 8), 2)
        state = integrate(obs[..., :6], action, self.physics, self.simulation)
        return np.concatenate((state, obs[..., 6:]), axis=-1)

    def cost(self, observation, action):
        obs = np.asarray(observation)
        target = obs[..., 6:8] if obs.shape[-1] == 8 else self.target
        return self.dt * running_cost(obs[..., :6], action, target, self.task)

    def project(self, observation):
        return np.asarray(observation).copy()

    def step(self, action):
        if self.state is None or self._done:
            raise RuntimeError("Call reset() before stepping a new or finished episode")

        applied = np.clip(vector(action, 2, "action"), 0, 1)
        self.state = integrate(self.state, applied, self.physics, self.simulation)
        self.steps += 1

        terminated = bool(self.is_failure(self.state))
        truncated = self.simulation.max_steps is not None and self.steps >= self.simulation.max_steps
        reward = -float(self.cost(self.observation, applied))
        if terminated:
            reward -= self.task.failure_penalty

        self._done = terminated or truncated
        return self.observation, reward, terminated, truncated, {
            **self._info(), "applied_action": applied.copy()}

    def set_physics(self, **changes):
        self.physics = replace(self.physics, **changes)

    def set_target(self, target):
        if self.state is None:
            raise RuntimeError("Call reset() first")
        self.target = vector(target, 2, "target")
        return self.observation

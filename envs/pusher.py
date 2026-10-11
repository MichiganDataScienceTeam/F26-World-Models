"""A rectangular slider with sustained contact on its left face.

A quadratic limit surface maps contact wrench to body velocity. Coulomb friction
allows sticking or sliding. Inertia, free contact, and impacts are omitted.
"""

from dataclasses import dataclass, replace

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .common import broadcast_inputs, vector


@dataclass(frozen=True)
class PusherPhysics:
    drag: float = 4.
    angular_drag: float = .3
    contact_friction: float = .5
    patch_strength: float = 0.
    patch_width: float = .22
    patch_center: float = 0.
    pressure_offset: float = 0.

    def __post_init__(self):
        for name in ("drag", "angular_drag", "contact_friction", "patch_width"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite([self.patch_strength, self.patch_center, self.pressure_offset]).all():
            raise ValueError("Patch settings must be finite")
        if abs(self.patch_strength) > 2:
            raise ValueError("Patch strength must be between -2 and 2")
        if abs(self.pressure_offset) > .2:
            raise ValueError("Pressure offset must be between -.2 and .2 meters")

    def patch(self, x):
        return np.exp(-.5 * ((np.asarray(x) - self.patch_center) / self.patch_width)**2)

    def resistance(self, x):
        """A smooth strip changes drag by exp(patch_strength) at its center."""

        return np.exp(self.patch_strength * self.patch(x))


class PusherEnv(gym.Env):
    """State [x,y,theta,contact_y], action [normal_force,tangential_speed].

    The observation appends target [x,y,theta]. A guide keeps the pusher on the
    face; contact_y is projected at the edges. Units are meters, seconds, newtons.
    """

    metadata = {"render_modes": []}
    width, height = .4, .6
    initial_state = (-.6, -.1, 0., 0.)
    action_low = np.array([0., -.4])
    action_high = np.array([2., .4])

    def __init__(self, physics=None, dt=.1, max_steps=None, target=(.45, .3, np.pi / 2)):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
            raise ValueError("max_steps must be a positive integer or None")
        self.physics = physics or PusherPhysics()
        self.dt, self.max_steps = dt, max_steps
        self.target = vector(target, 3, "target")
        self.action_space = spaces.Box(self.action_low.astype(np.float32), self.action_high.astype(np.float32))
        self.observation_space = spaces.Box(-np.inf, np.inf, (7,), dtype=np.float32)
        self.state = None
        self.steps, self._done = 0, False

    @property
    def observation(self):
        if self.state is None:
            raise RuntimeError("Call reset() first")
        return np.concatenate((self.state, self.target)).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if set(options) - {"state", "target"}:
            raise ValueError("Reset options are state and target")
        state = vector(options.get("state", self.initial_state), 4, "state")
        if abs(state[3]) > self.height / 2:
            raise ValueError("Contact must lie on the slider face")
        target = vector(options.get("target", self.target), 3, "target")
        self.state, self.target = state, target
        self.steps, self._done = 0, False
        return self.observation, {"elapsed_time": 0.}

    def predict(self, observation, action):
        """Batched true dynamics; the target is carried through unchanged."""

        obs, action = broadcast_inputs(observation, action, (4, 7), 2)
        force, speed = np.moveaxis(np.clip(action, self.action_low, self.action_high), -1, 0)
        p = self.physics
        contact = np.clip(obs[..., 3], -self.height / 2, self.height / 2)
        arm = -self.width / 2
        resistance = p.resistance(obs[..., 0])
        mobility = 1 / (p.drag * resistance)
        angular_mobility = 1 / (p.angular_drag * resistance)
        pressure_offset = p.pressure_offset * p.patch(obs[..., 0])
        contact_arm = contact - pressure_offset

        # Contact wrench
        tangent = (speed + arm * angular_mobility * contact_arm * force) / (mobility + arm**2 * angular_mobility)
        tangent = np.clip(tangent, -p.contact_friction * force, p.contact_friction * force)
        vx, vy = mobility * force, mobility * tangent
        omega = angular_mobility * (arm * tangent - contact_arm * force)
        vx += pressure_offset * omega
        slip = speed - (vy + arm * omega)

        # World velocity
        cosine, sine = np.cos(obs[..., 2]), np.sin(obs[..., 2])
        delta = self.dt * np.stack((cosine * vx - sine * vy, sine * vx + cosine * vy, omega, slip), -1)
        state = obs[..., :4] + delta
        state[..., 3] = np.clip(state[..., 3], -self.height / 2, self.height / 2)
        return np.concatenate((state, obs[..., 4:]), -1)

    def project(self, observation):
        observation = np.asarray(observation).copy()
        observation[..., 3] = np.clip(observation[..., 3], -self.height / 2, self.height / 2)
        return observation

    def cost(self, observation, action):
        obs, action = broadcast_inputs(observation, action, (7,), 2)
        position = obs[..., :2] - obs[..., 4:6]
        angle = obs[..., 2] - obs[..., 6]
        return 10 * (position**2).sum(-1) + 2 * (1 - np.cos(angle)) + .005 * (action**2).sum(-1)

    def step(self, action):
        if self.state is None or self._done:
            raise RuntimeError("Call reset() before stepping a new or finished episode")
        applied = np.clip(vector(action, 2, "action"), self.action_low, self.action_high)
        reward = -float(self.cost(self.observation, applied))
        self.state = self.predict(self.observation, applied)[:4]
        self.steps += 1
        self._done = self.max_steps is not None and self.steps >= self.max_steps
        return self.observation, reward, False, self._done, {
            "elapsed_time": self.steps * self.dt, "applied_action": applied.copy()}

    def set_physics(self, **changes):
        self.physics = replace(self.physics, **changes)

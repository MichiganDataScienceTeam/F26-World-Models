"""Immutable drone settings in SI units."""

from dataclasses import dataclass

import numpy as np

from ..common import vector


@dataclass(frozen=True)
class DronePhysics:
    mass: float = 1.0
    inertia: float = 0.02
    arm_length: float = 0.2  # centre to rotor
    gravity: float = 9.81
    max_thrust: float = 10.0  # per rotor
    motor_efficiency: tuple = (1.0, 1.0)  # left, right
    linear_drag: tuple = (0.1, 0.1)  # x, z
    angular_drag: float = 0.01
    external_force: tuple = (0.0, 0.0)  # world coordinates

    def __post_init__(self):
        for name in ("mass", "inertia", "arm_length", "max_thrust"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("gravity", "angular_drag"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")

        for name in ("motor_efficiency", "linear_drag", "external_force"):
            values = vector(getattr(self, name), 2, name, nonnegative=name != "external_force")
            object.__setattr__(self, name, tuple(values.tolist()))


@dataclass(frozen=True)
class DroneSimulation:
    dt: float = 0.05
    substeps: int = 5  # RK4 steps
    max_steps: int | None = 200  # None disables truncation

    def __post_init__(self):
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt must be finite and positive")
        for name in ("substeps", "max_steps"):
            value = getattr(self, name)
            if name == "max_steps" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class DroneTask:
    target: tuple = (0.0, 1.5)
    reset_noise: tuple = (0.25, 0.25, 0.1, 0.1, 0.1, 0.1)  # uniform reset offsets
    state_weights: tuple = (1.0, 1.0, 0.2, 0.1, 0.1, 0.02)
    action_weight: float = 0.01
    failure_penalty: float = 10.0
    x_bounds: tuple = (-5.0, 5.0)
    z_bounds: tuple = (0.0, 5.0)
    terminate_on_bounds: bool = True

    def __post_init__(self):
        for name, size in (("target", 2), ("reset_noise", 6), ("state_weights", 6),
                           ("x_bounds", 2), ("z_bounds", 2)):
            values = vector(getattr(self, name), size, name,
                            nonnegative=name in ("reset_noise", "state_weights"))
            object.__setattr__(self, name, tuple(values.tolist()))

        for name in ("x_bounds", "z_bounds"):
            if getattr(self, name)[0] >= getattr(self, name)[1]:
                raise ValueError(f"{name} must have increasing endpoints")
        for name in ("action_weight", "failure_penalty"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")

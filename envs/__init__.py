"""Project environments with Gymnasium's Env and spaces interfaces."""

from .drone import DroneEnv, DronePhysics, DroneSimulation, DroneTask
from .pendulum import PendulumEnv, PendulumPhysics


def make(name, **kwargs):
    """Create a project environment by name."""

    environments = {"drone": DroneEnv, "pendulum": PendulumEnv}
    if name not in environments:
        raise ValueError(f"Unknown environment {name!r}; choose drone or pendulum")
    return environments[name](**kwargs)

"""Project environments with Gymnasium's Env and spaces interfaces."""

from .drone import DroneEnv, DronePhysics, DroneSimulation, DroneTask
from .pendulum import PendulumEnv, PendulumPhysics
from .pusher import PusherEnv, PusherPhysics


def make(name, **kwargs):
    """Create a project environment by name."""

    environments = {"drone": DroneEnv, "pendulum": PendulumEnv, "pusher": PusherEnv}
    if name not in environments:
        raise ValueError(f"Unknown environment {name!r}; choose {', '.join(environments)}")
    return environments[name](**kwargs)

import numpy as np


def vector(value, size, name, nonnegative=False):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError(f"{name} must have shape ({size},) and contain finite numbers")
    if nonnegative and (value < 0).any():
        raise ValueError(f"{name} must contain nonnegative numbers")

    return value.copy()


def broadcast_inputs(observation, action, observation_sizes, action_size):
    observation = np.asarray(observation, dtype=float)
    action = np.asarray(action, dtype=float)
    if observation.ndim < 1 or observation.shape[-1] not in observation_sizes:
        raise ValueError(f"Observation must end in one of {observation_sizes}")
    if action.ndim < 1 or action.shape[-1] != action_size:
        raise ValueError(f"Action must end in {action_size}")
    if not np.isfinite(observation).all() or not np.isfinite(action).all():
        raise ValueError("Observations and actions must be finite")

    shape = np.broadcast_shapes(observation.shape[:-1], action.shape[:-1])
    return (np.broadcast_to(observation, shape + (observation.shape[-1],)),
            np.broadcast_to(action, shape + (action_size,)))

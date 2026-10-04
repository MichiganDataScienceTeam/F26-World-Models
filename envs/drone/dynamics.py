"""Planar drone dynamics: state=[x,z,theta,vx,vz,omega], action=[left,right].

Positive theta tilts thrust toward -x. Right thrust produces positive torque.
"""

import numpy as np

from ..common import broadcast_inputs
from .config import DronePhysics, DroneSimulation, DroneTask


def _inputs(state, action):
    state, action = broadcast_inputs(state, action, (6,), 2)
    return state, np.clip(action, 0, 1)


def _thrust(action, physics):
    thrust = action * physics.max_thrust * np.asarray(physics.motor_efficiency)
    return thrust.sum(axis=-1), physics.arm_length * (thrust[..., 1] - thrust[..., 0])


def _derivative(state, thrust, torque, physics):
    result = np.empty_like(state)
    result[..., :3] = state[..., 3:]

    force_x = physics.external_force[0] - physics.linear_drag[0] * state[..., 3]
    force_z = physics.external_force[1] - physics.linear_drag[1] * state[..., 4]
    result[..., 3] = (-thrust * np.sin(state[..., 2]) + force_x) / physics.mass
    result[..., 4] = (thrust * np.cos(state[..., 2]) + force_z) / physics.mass - physics.gravity
    result[..., 5] = (torque - physics.angular_drag * state[..., 5]) / physics.inertia
    return result


def derivative(state, action, physics=DronePhysics()):
    """Continuous derivative for one state or a batch."""

    state, action = _inputs(state, action)
    thrust, torque = _thrust(action, physics)
    return _derivative(state, thrust, torque, physics)


def integrate(state, action, physics=DronePhysics(), simulation=DroneSimulation()):
    """Advance one control interval with fixed-step RK4."""

    state, action = _inputs(state, action)
    thrust, torque = _thrust(action, physics)
    h = simulation.dt / simulation.substeps

    for _ in range(simulation.substeps):
        k1 = _derivative(state, thrust, torque, physics)
        k2 = _derivative(state + h/2*k1, thrust, torque, physics)
        k3 = _derivative(state + h/2*k2, thrust, torque, physics)
        k4 = _derivative(state + h*k3, thrust, torque, physics)
        state = state + h/6 * (k1 + 2*k2 + 2*k3 + k4)

    if not np.isfinite(state).all():
        raise FloatingPointError("Drone integration diverged; reduce dt or increase substeps")
    return state


def running_cost(state, action, target=(0.0, 1.5), task=DroneTask()):
    """Quadratic cost per second, with periodic orientation error."""

    state, action = _inputs(state, action)
    error = state.copy()
    error[..., :2] -= np.asarray(target)
    error[..., 2] = np.arctan2(np.sin(error[..., 2]), np.cos(error[..., 2]))
    return ((np.square(error) * task.state_weights).sum(axis=-1)
            + task.action_weight * (action*action).sum(axis=-1))

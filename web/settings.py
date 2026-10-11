from dataclasses import replace

import numpy as np

from envs import PusherPhysics

# key, label, min, max, step, unit
CONTROLS = {
    "drone": [
        ("gravitational_force", "Gravitational force", 8.1, 11.55, .01, "N"),
        ("horizontal_bias", "Horizontal bias", -100., 100., 5., "%"),
    ],
    "pendulum": [
        ("gravitational_force", "Gravitational force", 7., 15., .1, "N"),
        ("length", "Length", .7, 1.3, .01, "m"),
        ("torque_bias", "Torque bias", -.5, .5, .05, "N·m"),
    ],
    "pusher": [
        ("rotation_resistance", "Rotation resistance", .5, 2., .05, "×"),
        ("patch_center", "Patch position", -.6, .8, .05, "m"),
    ],
}


def control_values(environment, physics):
    if environment == "pusher":
        return {"rotation_resistance": physics.angular_drag / PusherPhysics.angular_drag,
                "patch_center": physics.patch_center}
    values = {"gravitational_force": physics.mass * physics.gravity}
    if environment == "drone":
        values["horizontal_bias"] = 100 * physics.external_force[0] / .3
    else:
        values.update(length=physics.length, torque_bias=physics.torque_bias)
    return values


def update_physics(environment, physics, values):
    """Return physics with validated slider changes."""

    limits = {key: (low, high) for key, _, low, high, *_ in CONTROLS[environment]}
    if not isinstance(values, dict) or set(values) - limits.keys():
        raise ValueError("Unknown physics parameter")

    settings = {}
    for key, value in values.items():
        value = float(value)
        low, high = limits[key]
        if not np.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{key} is outside the slider range")
        settings[key] = value

    changes = {}
    if "gravitational_force" in settings:
        changes["gravity"] = settings["gravitational_force"] / physics.mass

    if environment == "drone":
        if "horizontal_bias" in settings:
            bias = settings["horizontal_bias"] / 100
            changes["external_force"] = (.3 * bias, physics.external_force[1])
            changes["motor_efficiency"] = (1 + .05 * bias, 1 - .05 * bias)
    elif environment == "pendulum":
        changes.update({key: settings[key] for key in ("length", "torque_bias")
                        if key in settings})
    else:
        if "patch_center" in settings:
            changes["patch_center"] = settings["patch_center"]
        if "rotation_resistance" in settings:
            changes["angular_drag"] = PusherPhysics.angular_drag * settings["rotation_resistance"]
    return replace(physics, **changes)

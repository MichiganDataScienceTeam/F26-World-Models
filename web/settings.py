from dataclasses import replace

import numpy as np

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
}


def control_values(environment, physics):
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
    else:
        changes.update({key: settings[key] for key in ("length", "torque_bias")
                        if key in settings})
    return replace(physics, **changes)

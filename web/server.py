from collections import deque
from dataclasses import replace
from threading import Lock

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request

from adaptation import AdaptiveWorldModel, BayesianLinearRegression
from control import make_mpc, PlanningError
from envs import make, DronePhysics, DroneSimulation, DroneTask, PendulumPhysics
from models import load_pretrained

WINDOW = 400
MODES = ("frozen", "bayes", "oracle")

# key, label, min, max, step, unit
PHYSICS = {
    "drone": [
        ("mass", "Mass", .9, 1.1, .01, "kg"),
        ("gravity", "Gravity", 9., 10.5, .01, "m/s²"),
        ("force_x", "Horizontal force", -.3, .3, .02, "N"),
        ("motor_left", "Left motor", .95, 1.05, .01, "×"),
        ("motor_right", "Right motor", .95, 1.05, .01, "×"),
    ],
    "pendulum": [
        ("gravity", "Gravity", 9., 11., .1, "m/s²"),
        ("mass", "Mass", .9, 1.1, .01, "kg"),
        ("length", "Length", .95, 1.05, .01, "m"),
        ("damping", "Damping", 0., .4, .02, "1/s"),
    ],
}


def physics_values(environment, physics):
    values = {key: getattr(physics, key) for key in ("mass", "gravity")}
    if environment == "drone":
        values.update(force_x=physics.external_force[0],
                      motor_left=physics.motor_efficiency[0],
                      motor_right=physics.motor_efficiency[1])
    else:
        values.update(length=physics.length, damping=physics.damping)
    return values


class Run:
    def __init__(self, environment, mode, physics, frozen):
        if environment == "drone":
            self.env = make(environment, physics=physics,
                            simulation=DroneSimulation(max_steps=None),
                            task=DroneTask(terminate_on_bounds=False))
            initial = [-.7, 1.2, .05, 0., 0., 0.]
            self.action = np.array([.4905, .4905])
        else:
            self.env = make(environment, physics=physics, max_steps=None)
            initial = [.25, 0.]
            self.action = np.zeros(1)
        self.env.reset(seed=0, options={"state": initial})

        self.mode = mode
        self.state_size = frozen.state_size
        self.model = frozen if mode == "frozen" else self.env
        if mode == "bayes":
            scale = frozen.ys.detach().cpu().numpy().astype(float)
            noise_std = np.maximum(frozen.residual_rms.detach().cpu().numpy() / scale, 1e-4)
            correction = BayesianLinearRegression(frozen.width + 1, frozen.state_size,
                                                  prior_std=.1, noise_std=noise_std)
            self.model = AdaptiveWorldModel(frozen, correction, output_scale=scale)
        self.controller = make_mpc(self.env, self.model, seed=0)

        self.trace = deque(maxlen=WINDOW)
        self.remaining_plan = deque()
        self.predicted_path = []
        self.status = ""
        self.active = True
        if mode == "bayes":
            try:
                self.model.predict(self.env.observation, self.action)
            except NotImplementedError:
                self.disable()

    def invalidate_plan(self):
        self.remaining_plan.clear()
        self.predicted_path = []

    def reset_posterior(self):
        self.model.reset()
        self.controller = make_mpc(self.env, self.model, seed=0)
        self.invalidate_plan()

    def disable(self):
        self.active = False
        self.status = "Not implemented"
        self.invalidate_plan()

    def plan(self, observation):
        try:
            action = self.controller.predict(observation)
        except PlanningError:
            self.predicted_path = []
            if self.remaining_plan:
                self.status = "MPC: using previous plan"
                return self.remaining_plan.popleft()
            self.status = "MPC: holding previous action"
            return self.action.copy()

        self.remaining_plan = deque(self.controller.plan[1:].copy())
        self.predicted_path = self.controller.predicted_states.tolist()
        self.status = ""
        return action

    def prepare(self):
        if not self.active:
            return None

        observation = self.env.observation
        try:
            action = self.plan(observation)
            prediction = self.model.predict(observation, action)
        except NotImplementedError:
            self.disable()
            return None
        return observation, action, prediction

    def advance(self, prepared):
        if prepared is None:
            return None

        observation, action, prediction = prepared
        next_observation, _, _, _, _ = self.env.step(action)
        self.action = action
        residual = prediction[..., :self.state_size] - next_observation[..., :self.state_size]
        mse = float(np.mean(residual**2))

        if self.mode == "bayes":
            try:
                self.model.partial_fit(observation, action, next_observation)
            except NotImplementedError:
                self.disable()
                return None
        self.trace.append(self.env.state[:2].tolist())
        return mse

    def state(self):
        return {
            "physical_state": self.env.state.tolist(),
            "predicted_path": self.predicted_path,
            "target": self.env.target.tolist() if hasattr(self.env, "target") else None,
            "trace": list(self.trace),
            "status": self.status,
            "active": self.active,
        }


class Playground:
    def __init__(self):
        self.models = {}
        self.worlds = {"drone": DronePhysics(), "pendulum": PendulumPhysics()}
        self.environment = "drone"
        self.reset_on_change = False
        self.runs = {}
        self.reset()

    def reset(self, environment=None):
        name = self.environment if environment is None else environment
        if name not in PHYSICS:
            raise ValueError("Choose drone or pendulum")

        if name not in self.models:
            self.models[name] = load_pretrained(name)
        runs = {mode: Run(name, mode, self.worlds[name], self.models[name]) for mode in MODES}
        for run in self.runs.values():
            run.env.close()

        self.environment = name
        self.runs = runs
        self.history = deque(maxlen=WINDOW)

    def set_physics(self, values, reset_on_change=None):
        if reset_on_change is not None and not isinstance(reset_on_change, bool):
            raise ValueError("reset_on_change must be a boolean")
        limits = {key: (low, high) for key, _, low, high, *_ in PHYSICS[self.environment]}
        if not isinstance(values, dict) or set(values) - limits.keys():
            raise ValueError("Unknown physics parameter")

        settings = physics_values(self.environment, self.worlds[self.environment])
        for key, value in values.items():
            value = float(value)
            low, high = limits[key]
            if not np.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{key} is outside the slider range")
            settings[key] = value

        current = self.worlds[self.environment]
        if self.environment == "drone":
            physics = replace(current, mass=settings["mass"], gravity=settings["gravity"],
                              external_force=(settings["force_x"], current.external_force[1]),
                              motor_efficiency=(settings["motor_left"], settings["motor_right"]))
        else:
            physics = replace(current, **settings)

        if reset_on_change is not None:
            self.reset_on_change = reset_on_change
        self.apply_physics(physics)

    def apply_physics(self, physics):
        if physics == self.worlds[self.environment]:
            return

        self.worlds[self.environment] = physics
        for run in self.runs.values():
            run.env.physics = physics
            run.invalidate_plan()
        if self.reset_on_change:
            self.reset_posterior()

    def reset_posterior(self):
        self.runs["bayes"].reset_posterior()

    def restore(self):
        self.apply_physics(DronePhysics() if self.environment == "drone" else PendulumPhysics())

    def step(self):
        prepared = {mode: run.prepare() for mode, run in self.runs.items()}
        mse = {mode: self.runs[mode].advance(data) for mode, data in prepared.items()}
        self.history.append({"time": self.time, **mse})

    @property
    def time(self):
        env = self.runs["frozen"].env
        return env.steps * env.dt

    def state(self):
        return {
            "environment": self.environment,
            "dt": self.runs["frozen"].env.dt,
            "time": self.time,
            "physics": physics_values(self.environment, self.worlds[self.environment]),
            "physics_controls": PHYSICS[self.environment],
            "reset_on_change": self.reset_on_change,
            "runs": {mode: run.state() for mode, run in self.runs.items()},
            "history": list(self.history),
        }


def create_app():
    torch.set_num_threads(1)
    app = Flask(__name__)
    playground = Playground()
    lock = Lock()

    def body(allowed):
        payload = request.get_json()
        if not isinstance(payload, dict) or set(payload) - set(allowed):
            raise ValueError("Unexpected request fields")
        return payload

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/state")
    def state():
        with lock:
            return jsonify(playground.state())

    @app.post("/api/reset")
    def reset():
        payload = body(("environment",))
        with lock:
            playground.reset(payload.get("environment"))
            return jsonify(playground.state())

    @app.post("/api/configure")
    def configure():
        payload = body(("physics", "reset_on_change"))
        if "reset_on_change" in payload and not isinstance(payload["reset_on_change"], bool):
            raise ValueError("reset_on_change must be a boolean")
        with lock:
            playground.set_physics(payload.get("physics", {}), payload.get("reset_on_change"))
            return jsonify(playground.state())

    @app.post("/api/restore")
    def restore():
        body(())
        with lock:
            playground.restore()
            return jsonify(playground.state())

    @app.post("/api/reset-posterior")
    def reset_posterior():
        body(())
        with lock:
            playground.reset_posterior()
            return jsonify(playground.state())

    @app.post("/api/step")
    def step():
        payload = body(("steps",))
        steps = payload.get("steps", 1)
        if type(steps) is not int or not 1 <= steps <= 4:
            raise ValueError("steps must be an integer from 1 to 4")
        with lock:
            for _ in range(steps):
                playground.step()
            return jsonify(playground.state())

    @app.errorhandler(ValueError)
    @app.errorhandler(TypeError)
    def invalid_request(error):
        return jsonify(error=str(error)), 400

    @app.errorhandler(RuntimeError)
    def stalled(error):
        return jsonify(error=str(error)), 409

    return app

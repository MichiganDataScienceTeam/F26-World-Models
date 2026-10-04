from dataclasses import dataclass

import numpy as np

from envs.drone import DroneTask, running_cost
from .cem import minimize


class PlanningError(RuntimeError):
    """The planner could not find a usable action sequence."""


@dataclass(frozen=True)
class MPCConfig:
    horizon: int = 30
    population: int = 256
    elites: int = 24
    iterations: int = 5
    knots: int = 8
    collective_std: float = .04
    differential_std: float = .008
    terminal_seconds: float = 1.0
    max_tilt: float = np.pi/3


class DroneMPC:
    """Plan throttle sequences using only a supplied next-state model.

    Collective and differential thrust use separate search scales. Knot noise is
    interpolated across the horizon to explore smooth action sequences.
    """

    def __init__(self, predict, dt=.05, task=None, config=None, seed=0):
        self.transition = predict
        self.dt = dt
        self.task = task or DroneTask()
        self.config = config or MPCConfig()
        config = self.config
        if not (config.horizon >= 2 and 2 <= config.knots <= config.horizon
                and 2 <= config.elites < config.population and config.iterations >= 1):
            raise ValueError("Invalid MPC dimensions")

        self.rng = np.random.default_rng(seed)
        self.mean = np.tile([.4905, 0.], (config.horizon, 1))
        self.trim = np.array([0., .4905, .4905])  # angle, left, right
        self.plan = None
        self.predicted_states = None

        knots = np.linspace(0, config.horizon-1, config.knots)
        self.interpolation = np.stack([
            np.interp(np.arange(config.horizon), knots, row)
            for row in np.eye(config.knots)
        ], axis=-1)

    def trim_at(self, target):
        """Find a steady angle and motor balance with bounded Newton steps."""

        trim = self.trim.copy()

        def residual(candidate):
            candidate = np.atleast_2d(candidate)
            states = np.zeros((len(candidate), 6))
            states[:, :2] = target
            states[:, 2] = candidate[:, 0]
            return self.transition(states, candidate[:, 1:])[:, 3:] / self.dt

        for _ in range(10):
            error = residual(trim)[0]
            if np.linalg.norm(error) < 1e-5:
                break

            delta = np.eye(3)*1e-4
            jacobian = ((residual(trim+delta)-residual(trim-delta))/2e-4).T
            step = np.linalg.solve(jacobian.T@jacobian + 1e-6*np.eye(3), jacobian.T@error)
            proposals = np.clip(
                trim - np.array([1., .5, .25, .1])[:, None]*step,
                [-self.config.max_tilt, 0, 0], [self.config.max_tilt, 1, 1])

            costs = np.square(residual(proposals)).sum(-1)
            best = int(np.argmin(costs))
            if costs[best] >= error@error:
                break
            trim = proposals[best]

        return trim

    @staticmethod
    def actions(parameters):
        collective, differential = parameters[..., 0], parameters[..., 1]
        return np.clip(np.stack([collective-differential, collective+differential], -1), 0, 1)

    @staticmethod
    def parameters(actions):
        return np.stack([actions.mean(-1), (actions[..., 1]-actions[..., 0])/2], -1)

    def score(self, state, actions, target, return_path=False):
        """Accumulate tracking and stopping costs; penalize infeasible rollouts."""

        states = np.broadcast_to(state, (len(actions), 6)).copy()
        costs = np.zeros(len(actions))
        failed = np.zeros(len(actions), dtype=bool)
        path = [states.copy()] if return_path else None

        for t in range(actions.shape[1]):
            states = self.transition(states, actions[:, t])
            angle = np.arctan2(np.sin(states[:, 2]), np.cos(states[:, 2]))
            failed |= np.abs(angle) > self.config.max_tilt
            if self.task.terminate_on_bounds:
                failed |= ((states[:, 0] <= self.task.x_bounds[0])
                           | (states[:, 0] >= self.task.x_bounds[1])
                           | (states[:, 1] <= self.task.z_bounds[0])
                           | (states[:, 1] >= self.task.z_bounds[1]))

            costs += self.dt * running_cost(states, actions[:, t], target, self.task)
            if return_path:
                path.append(states.copy())

        # Approximate stopping.
        error = states.copy()
        error[:, :2] -= target
        error[:, 2] = np.arctan2(np.sin(error[:, 2]), np.cos(error[:, 2]))
        costs += self.config.terminal_seconds * (error*error * [1, 1, .3, .5, .5, .05]).sum(-1)

        # Rank failed proposals so CEM can improve them; predict rejects the final plan.
        costs[failed] += 1e6
        costs[~np.isfinite(costs)] = np.inf
        return (costs, np.stack(path, axis=1)) if return_path else costs

    def predict(self, observation):
        observation = np.asarray(observation, dtype=float)
        state, target = observation[:6], observation[6:8]
        config = self.config

        trim = self.trim_at(target)
        self.mean += self.parameters(trim[1:])-self.parameters(self.trim[1:])
        self.trim = trim
        std = np.tile([config.collective_std, config.differential_std], (config.horizon, 1))

        def project(parameters):
            return self.parameters(self.actions(parameters))

        best, mean = minimize(
            lambda parameters: self.score(state, self.actions(parameters), target),
            self.mean.copy(), std,
            population=config.population, elites=config.elites, iterations=config.iterations,
            min_std=[.012, .003], rng=self.rng, interpolation=self.interpolation,
            broad_std=[.2, .08], project=project)

        candidates = self.actions(np.stack([best, mean]))
        costs, paths = self.score(state, candidates, target, return_path=True)
        best = int(np.argmin(costs))
        if not np.isfinite(costs[best]) or costs[best] >= 1e6:
            raise PlanningError("MPC found no feasible plan")

        self.plan, self.predicted_states = candidates[best], paths[best]
        parameters = self.parameters(self.plan)
        self.mean = np.concatenate([parameters[1:], parameters[-1:]], axis=0)
        return self.plan[0].copy()


class PendulumMPC:
    """Plan torques using a supplied next-observation model."""

    def __init__(self, env, predict, horizon=20, population=400, elites=40, iterations=5, seed=0):
        if horizon < 2:
            raise ValueError("horizon must be at least 2")

        self.env = env
        self.transition = predict
        self.horizon = horizon
        self.population = population
        self.elites = elites
        self.iterations = iterations

        self.rng = np.random.default_rng(seed)
        self.mean = np.zeros((horizon, 1))
        self.plan = self.predicted_states = None

    def score(self, observation, actions, return_path=False):
        state = np.broadcast_to(observation, (len(actions), 3)).copy()
        costs = np.zeros(len(actions))
        path = [state.copy()]

        for t in range(self.horizon):
            costs += self.env.cost(state, actions[:, t])
            state = self.env.project(self.transition(state, actions[:, t]))
            if return_path:
                path.append(state.copy())

        costs[~np.isfinite(costs)] = np.inf
        return (costs, np.stack(path, 1)) if return_path else costs

    def predict(self, observation):
        best, mean = minimize(
            lambda actions: self.score(observation, actions), self.mean, np.ones_like(self.mean),
            population=self.population, elites=self.elites, iterations=self.iterations,
            lower=-self.env.max_torque, upper=self.env.max_torque, rng=self.rng)

        candidates = np.stack((best, mean))
        costs, paths = self.score(observation, candidates, return_path=True)
        best = int(np.argmin(costs))
        if not np.isfinite(costs[best]):
            raise PlanningError("MPC found no finite plan")

        self.plan, self.predicted_states = candidates[best], paths[best]
        self.mean = np.concatenate((self.plan[1:], np.zeros((1, 1))))
        return self.plan[0].copy()


def make_mpc(env, model=None, seed=0):
    """Use the supplied model, or the environment's oracle when model is None."""

    from envs import DroneEnv, PendulumEnv

    transition = env.predict if model is None else model.predict
    if isinstance(env, DroneEnv):
        return DroneMPC(transition, dt=env.dt, task=env.task, seed=seed)
    if isinstance(env, PendulumEnv):
        return PendulumMPC(env, transition, seed=seed)
    raise TypeError("No planner configuration for this environment")

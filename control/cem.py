import numpy as np


def minimize(score, mean, std, *, population=256, elites=24, iterations=5,
             lower=-np.inf, upper=np.inf, min_std=.001, rng=None,
             interpolation=None, broad_std=None, project=None):
    """Fit a Gaussian to low-cost plans; return the best plan and final mean.

    Plans have shape [population, horizon, action]. Optional interpolation smooths
    the sampled noise, and project maps proposals to feasible search parameters.
    """

    mean = np.asarray(mean, dtype=float).copy()
    std = np.asarray(std, dtype=float)
    if mean.ndim != 2 or not 2 <= elites < population or iterations < 1:
        raise ValueError("Expected a [horizon,action] mean and valid CEM sizes")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("CEM mean/std must be finite and std positive")

    std = np.broadcast_to(std, mean.shape).copy()
    rng = rng or np.random.default_rng()
    best_plan, best_cost = mean.copy(), np.inf
    noise_steps = mean.shape[0] if interpolation is None else interpolation.shape[1]

    for _ in range(iterations):
        noise = rng.normal(size=(population, noise_steps, mean.shape[1]))
        if interpolation is not None:
            noise = np.einsum("hk,nkd->nhd", interpolation, noise)

        proposals = mean + std*noise
        if broad_std is not None:
            count = max(2, population//8)
            proposals[-count:] = mean + np.asarray(broad_std)*noise[-count:]
        proposals[0], proposals[1] = mean, best_plan
        proposals = np.clip(proposals, lower, upper)
        if project is not None:
            proposals = project(proposals)

        costs = np.asarray(score(proposals))
        costs = np.where(np.isfinite(costs), costs, np.inf)
        indices = np.argsort(costs)[:elites]
        if costs[indices[0]] < best_cost:
            best_plan = proposals[indices[0]].copy()
            best_cost = float(costs[indices[0]])

        elite_plans = proposals[indices]
        mean = elite_plans.mean(0)
        std = np.maximum(elite_plans.std(0), min_std)

    return best_plan, mean

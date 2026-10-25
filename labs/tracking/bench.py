"""Synthetic worlds: a world model's error over time, changing at step `at`.

y: observed error (true error + sensor noise), e: true error, z: operating point (action, state).
Scores compare against the true error e. 'push' is not validated.
"""
from dataclasses import dataclass

import numpy as np

TASKS = {   # outputs, steps, change step, size of the error before the change
    'hover': dict(n=6, steps=300, at=100, wander=3e-4, white=1e-5),
    'swing': dict(n=3, steps=300, at=100, wander=1.5e-3, white=9e-4),
    'push': dict(n=4, steps=150, at=0, wander=8e-3, white=2.8e-2),
}


def signals(task, rng):
    """Stylized action u and state x of each task's controller."""
    steps, at = TASKS[task]['steps'], TASKS[task]['at']
    t, u, x = np.arange(steps), np.zeros(steps), np.zeros(steps)
    if task == 'hover':                  # like the drone: hovers, flies to new targets at 150 and 225
        for start in (150, 225):
            k = t - start
            u[(k >= 0) & (k < 8)], u[(k >= 8) & (k < 16)] = 1., -1.
        x = np.cumsum(u)/8
    elif task == 'swing':                # like the pendulum: knocked down at the start and at the change,
                                         # swings up (bang-bang), then balances
        for start in (0, at):
            k = t - start
            swing = (k >= 0) & (k < 50)
            u[swing] = np.sign(np.sin(2*np.pi*(k[swing] + .5)/30))
            x[swing] = k[swing]/50*np.sin(2*np.pi*k[swing]/30)
    else:                                # like the pusher: pushes hard with brief releases, then small corrections
        u = np.where(t < 40, rng.random(steps) > .15, .3*rng.random(steps))
        x = t/steps
    return u, x


@dataclass
class Stream:
    y: np.ndarray       # (steps, n) observed error
    e: np.ndarray       # (steps, n) true error
    z: np.ndarray       # (steps, 2) operating point
    at: int             # step of the change


def stream(task, seed, sensor=0., change=True):
    p = TASKS[task]
    rng = np.random.default_rng(seed)
    u, x = signals(task, rng)
    J, G, K = rng.integers(0, 2, (3, p['n']))*rng.choice([-1., 1.], (3, p['n']))*10**rng.uniform(-2, .3, (3, p['n']))
    e = (np.cumsum(p['wander']*rng.standard_normal((p['steps'], p['n'])), 0)
         + p['white']*rng.standard_normal((p['steps'], p['n'])))
    if change:
        e[p['at']:] += J + np.outer(u[p['at']:], G) + np.outer(x[p['at']:], K)
    eps = rng.standard_normal((p['steps'] + 1, p['n']))
    return Stream(e + sensor*(eps[1:] - eps[:-1]), e, np.stack([u, x], -1), p['at'])


def run(model, s):
    """Means and SDs, each predicted before seeing that step."""
    model.reset(s.y.shape[1])
    means, sds = [], []
    for t in range(len(s.y)):
        x = np.r_[t, s.z[t]]
        mean, sd = model.predict(x)
        means.append(mean); sds.append(sd)
        model.update(x, s.y[t])
    return np.array(means), np.array(sds)


def scores(model, s):
    mean, sd = run(model, s)
    after = slice(s.at, None)
    return dict(error=np.sqrt(np.mean((mean - s.e)[after]**2)),
                inside=np.mean(np.abs(s.y - mean)[after] <= 2*sd[after]))


def evaluate(make, task, sensor=0., worlds=10):
    """Average scores over worlds; make() returns a fresh model."""
    changed = [scores(make(), stream(task, seed, sensor)) for seed in range(worlds)]
    calm = [scores(make(), stream(task, seed, sensor, change=False)) for seed in range(worlds)]
    return {'error after change': np.exp(np.mean([np.log(r['error']) for r in changed])),
            'inside 2 SD': np.mean([r['inside'] for r in changed]),
            'error, no change': np.exp(np.mean([np.log(r['error']) for r in calm]))}


def plot(models, task='hover', seed=0, sensor=0., outputs=(0, 1, 2), path=None):
    """One world: observed readings, true error, and each model's mean (line) and +-2 SD band (dashed)."""
    import matplotlib.pyplot as plt
    s = stream(task, seed, sensor)
    fig, axes = plt.subplots(len(outputs), 1, figsize=(9, 2.2*len(outputs)), sharex=True, squeeze=False)
    fits = {name: run(model, s) for name, model in models.items()}
    for ax, j in zip(axes[:, 0], outputs):
        ax.plot(s.y[:, j], '.', color='0.75', ms=3, label='observed')
        ax.plot(s.e[:, j], 'k', lw=1, label='true error')
        for k, (name, (mean, sd)) in enumerate(fits.items()):
            ax.plot(mean[:, j], color=f'C{k}', lw=1.5, label=name)
            for side in (-2, 2):                                  # the +-2 SD band, dashed
                ax.plot(mean[:, j] + side*sd[:, j], color=f'C{k}', lw=.7, ls='--')
        ax.axvline(s.at, color='0.5', ls=':', lw=1)
        ax.set_ylabel(f'output {j}')
        lo, hi = np.percentile(np.r_[s.y[:, j], s.e[:, j]], [1, 99])
        ax.set_ylim(lo - .3*(hi - lo + .1), hi + .3*(hi - lo + .1))
    axes[0, 0].set_title(f'{task}, world {seed}, sensor noise {sensor}')
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=3)
    axes[-1, 0].set_xlabel('step')
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig

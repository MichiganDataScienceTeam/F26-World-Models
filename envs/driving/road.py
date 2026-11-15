"""Random road: straights and arcs joined by clothoids (linear curvature change), as in road design.

Sampled on a 1 m grid along the centre line. A twisty track: straights 20-200 m, arc radius 15-80 m (log-uniform;
15 m is a hairpin at about 45 km/h, 80 m a bend at about 100 km/h), turns of 30-150 degrees, clothoids 15-40 m.
"""
import math

import numpy as np
import torch

HALF_WIDTH = 3.5                       # two standard 3.5 m lanes
SHAPE = dict(straight=(20, 200), radius=(15, 80), angle=(30, 150), clothoid=(15, 40))


class Road:
    def __init__(self, seed, length=6000):
        rng, sh = np.random.default_rng(seed), SHAPE
        k = [np.zeros(100)]                                           # start on a straight
        while sum(map(len, k)) < length:
            R = math.exp(rng.uniform(math.log(sh['radius'][0]), math.log(sh['radius'][1])))*rng.choice([-1, 1])
            arc = int(abs(R)*math.radians(rng.uniform(*sh['angle'])))
            cl = int(rng.uniform(*sh['clothoid']))
            k += [np.linspace(0, 1/R, cl), np.full(arc, 1/R), np.linspace(1/R, 0, cl), np.zeros(int(rng.uniform(*sh['straight'])))]
        self.kappa = np.concatenate(k)[:length]
        self.yaw = np.cumsum(self.kappa)                              # 1 m steps
        self.x = np.cumsum(np.cos(self.yaw))
        self.y = np.cumsum(np.sin(self.yaw))
        self.t = {n: torch.tensor(getattr(self, n), dtype=torch.float32) for n in ('x', 'y', 'yaw', 'kappa')}

    def locate(self, x, i0):
        """Nearest centre-line index (searched near i0) and signed lateral offset for cars x (B, 7)."""
        idx = (i0[:, None] + torch.arange(-5, 16)).clamp(0, len(self.x) - 1)
        dx, dy = x[:, :1] - self.t['x'][idx], x[:, 1:2] - self.t['y'][idx]
        i = idx.gather(1, (dx**2 + dy**2).argmin(1, keepdim=True)).squeeze(1)
        yaw = self.t['yaw'][i]
        ex, ey = x[:, 0] - self.t['x'][i], x[:, 1] - self.t['y'][i]
        s = i + ex*torch.cos(yaw) + ey*torch.sin(yaw)                 # distance along the road [m]
        e = -ex*torch.sin(yaw) + ey*torch.cos(yaw)                    # left of the centre line is positive
        return i, s, e

    def ideal(self, T, v_max, a_brake, accel):
        """Distance a point mass on the centre line covers in T s (lateral, braking and engine limits)."""
        v = np.minimum(v_max, np.sqrt(a_brake/np.maximum(np.abs(self.kappa), 1e-9)))
        v[0] = 0.                                                     # from rest
        for i in range(1, len(v)):                                    # forward: engine
            v[i] = min(v[i], math.sqrt(v[i - 1]**2 + 2*accel(v[i - 1])))
        for i in range(len(v) - 2, -1, -1):                           # backward: brakes
            v[i] = min(v[i], math.sqrt(v[i + 1]**2 + 2*a_brake))
        t = np.cumsum(2/(v[:-1] + v[1:]))                             # time for each 1 m step
        self.v_ideal = v
        return int(np.searchsorted(t, T))

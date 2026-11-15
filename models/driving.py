"""The driving agent: a learned model of the car and a planner that searches over key presses.

Model: a small network. Input: the car's motion (forward speed, sideways speed, turning rate), the three commands
it carried out, and whether it is on grass. Output, one decision (0.05 s) later: the change in motion, the heading
change, and how far the car moved. It knows nothing about the road; the road's geometry is exact.

Planner: no policy and no value function. A plan is one key combination per block of 4 decisions (0.2 s), for 26
blocks (5.2 s: the time to brake from top speed). Each search round samples 512 plans, scores each one with the
model (metres gained on the road), and moves the sampling probabilities towards the best 64.

The planner can plan through a command map: the car is assumed to carry out a = clip(M c + b) for command c.
"""

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from envs.driving import car
from envs.driving.road import HALF_WIDTH

WEIGHTS = Path(__file__).parent / "pretrained" / "driving.pt"
KEYS = torch.tensor([[s, t, b] for s in (-1., 0., 1.) for t in (0., 1.) for b in (0., 1.)])   # (12, 3): steer, throttle, brake
COAST = 4                                            # KEYS[4]: no keys held
BLOCK = round(car.RAMP / car.DT)                     # decisions per block: keys need RAMP seconds to take full effect
HORIZON = math.ceil((car.V_MAX / car.GRIP + car.RAMP) / (BLOCK * car.DT))    # blocks: braking time from top speed
SAMPLES, ELITES, FLOOR = 512, 64, .05


class Dynamics(nn.Module):
    def __init__(self, stats):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(7, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 6))
        for k, v in stats.items():                                   # input and output scales, from the training data
            self.register_buffer(k, v)

    def forward(self, motion, a, grass):
        """One decision. motion (B, 3), executed commands a (B, 3), grass (B,). Returns (B, 6): change in forward
        speed, sideways speed, turning rate; heading change; forward and sideways move in the car's frame."""
        z = torch.cat([(motion - self.m_mean) / self.m_std, a, grass.float()[:, None]], 1)
        return self.net(z) * self.d_std + self.d_mean

    def decide(self, x, cmd, keys, grass, M=None, b=None):
        """Same as car.decide, with the network in place of the physics."""
        cmd = car.smooth(cmd, keys)
        a = cmd if M is None else car.fault(cmd, M, b)
        d = self(x[:, 3:6], a, grass)
        c, s = torch.cos(x[:, 2]), torch.sin(x[:, 2])
        X, Y = x[:, 0] + c * d[:, 4] - s * d[:, 5], x[:, 1] + s * d[:, 4] + c * d[:, 5]
        motion = x[:, 3:6] + d[:, :3]
        motion = torch.cat([motion[:, :1].clamp(min=0.), motion[:, 1:]], 1)      # no reversing, as the car
        return torch.cat([X[:, None], Y[:, None], (x[:, 2] + d[:, 3])[:, None], motion, x[:, 6:]], 1), cmd


def load():
    ck = torch.load(WEIGHTS)
    model = Dynamics(ck["stats"])
    model.load_state_dict(ck["state"])
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval()


def outputs(x0, x1):
    """What the model predicts, measured from two car states: (B, 6) as Dynamics.forward."""
    dX, dY = x1[:, 0] - x0[:, 0], x1[:, 1] - x0[:, 1]
    c, s = torch.cos(x0[:, 2]), torch.sin(x0[:, 2])
    return torch.stack([*(x1[:, 3:6] - x0[:, 3:6]).unbind(1), x1[:, 2] - x0[:, 2], c * dX + s * dY, -s * dX + c * dY], 1)


def grass_speed():
    """Top speed on grass [m/s]: where half engine power equals drag plus the grass's rolling resistance."""
    lo, hi = 1., car.V_MAX
    for _ in range(50):
        v = (lo + hi) / 2
        push = (car.ENGINE_GRASS * car.A_MAX * min(1., car.V_SWITCH / v) - .5 * car.RHO * car.CDA * v * v / car.M
                - car.ROLL_GRASS * car.G)
        lo, hi = (v, hi) if push > 0 else (lo, v)
    return v


def ideal_distance(road, seconds):
    """Metres the healthy car could cover on this road from rest (grip, engine and brake limits)."""
    return road.ideal(seconds, car.V_MAX, car.GRIP, lambda v: car.A_MAX * min(1, car.V_SWITCH / max(v, 1e-3)))


def way_back(road, x, i, e):
    """Length [m] of the shortest drivable path back to the road for cars that are off it: an arc at the car's
    tightest turn towards the road, then straight. Zero on the road. (Counting only the distance leaves a car that
    stopped facing away from the road stuck: every start of a U-turn first takes it further away.)"""
    d = (e.abs() - HALF_WIDTH).clamp(min=0.)
    to_road = road.t["yaw"][i] - torch.sign(e) * math.pi / 2         # the direction straight at the road
    turn = torch.atan2(torch.sin(x[:, 2] - to_road), torch.cos(x[:, 2] - to_road)).abs()      # to face the road
    R, sin = car.R_MIN, torch.sin(turn)
    arc = R * (turn - torch.asin((sin - d / R).clamp(-1., 1.)))      # the arc alone reaches the road
    return torch.where(d <= 0, torch.zeros_like(d), torch.where(d <= R * sin, arc, R * turn + d - R * sin))


def lost_per_metre(road):
    """Road metres lost for each metre of the way back: it takes (metres) / (top speed on grass) seconds, in which
    a car on the road covers that time x the road's ideal average speed."""
    if not hasattr(road, "lost"):
        ideal_distance(road, 1e9)
        v = road.v_ideal
        road.lost = (len(v) - 1) / np.sum(2 / (v[:-1] + v[1:])) / grass_speed()
    return road.lost


def rollout(road, model, x, cmd, i, plans, M=None, b=None):
    """Score of each plan: metres gained along the road while on it, minus the metres lost getting back if the plan
    ends off the road (way_back). plans (B, H) key indices; x, cmd, i already (B, ...)."""
    score = torch.zeros(len(x))
    _, s, e = road.locate(x, i)
    for h in range(plans.shape[1]):
        keys = KEYS[plans[:, h]]
        for _ in range(BLOCK):
            x, cmd = model.decide(x, cmd, keys, e.abs() > HALF_WIDTH, M, b)
            i, s2, e = road.locate(x, i)
            score += torch.where(e.abs() <= HALF_WIDTH, (s2 - s).clamp(min=0.), torch.zeros_like(s))
            s = s2
    return score - lost_per_metre(road) * way_back(road, x, i, e)


def ahead(road, model, x, cmd, i, k, M, b, n):
    """The car n decisions from now if keys k stay held, by the model."""
    m = (None, None) if M is None else (M[None], b[None])
    for _ in range(n):
        i, _, e = road.locate(x, i)
        x, cmd = model.decide(x, cmd, KEYS[k][None], e.abs() > HALF_WIDTH, *m)
    return x, cmd, road.locate(x, i)[0]


class Planner:
    def __init__(self, road, model, seed=0):
        self.road, self.model = road, model
        self.rng = torch.Generator().manual_seed(seed)
        self.p = torch.full((HORIZON, len(KEYS)), 1 / len(KEYS))      # P(key combination) for each block
        self.best = torch.full((HORIZON,), COAST)

    def round(self, x, cmd, i, M=None, b=None):
        """One search round from car state x (7,), commands cmd (3,), road index i."""
        plans = torch.multinomial(self.p, SAMPLES, replacement=True, generator=self.rng).T           # (512, 26)
        plans[0] = self.best
        rep = lambda v: None if v is None else v[None].expand(SAMPLES, *v.shape).clone()
        scores = rollout(self.road, self.model, rep(x), rep(cmd), i.expand(SAMPLES).clone(), plans, rep(M), rep(b))
        self.best = plans[scores.argmax()]
        elites = plans[scores.topk(ELITES).indices]                   # the best plans: how often each key combination
        share = torch.nn.functional.one_hot(elites, len(KEYS)).float().mean(0)       # appears in each block (26, 12)
        self.p = (1 - FLOOR) * share + FLOOR / len(KEYS)

    def commit(self):
        """The keys to hold for the next block; everything then shifts one block (warm start)."""
        k = int(self.best[0])
        self.p = torch.cat([self.p[1:], torch.full((1, len(KEYS)), 1 / len(KEYS))])
        self.best = torch.cat([self.best[1:], self.best[-1:]])
        return k

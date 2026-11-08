"""
Fill in the six TODOs.

Check them by running this: python -m labs.walker.check.
"""

import math
from pathlib import Path

import numpy as np
import torch

from adaptation.fast_blr import FastBLR

# Constants that you should assume I have chosen well
CALIBRATION = dict(np.load(Path(__file__).with_name("calibration.npz")))
NOISE = dict(noise_std=float(CALIBRATION["scale"]) ** .5, nu=float(CALIBRATION["nu"]), eta=float(CALIBRATION["eta"]))
A = 6 # dimension of action space
STRUCTURES = ("none", "common", "gains", "gain+offset", "general")
RUNS = 20 # candidate change times kept (Fearnhead & Liu 2007 resampling)


def structures():
    """
    dict of structures {name: 6 x 7 integer grid of labels}.
    Each grid is [M - I | b]
    """

    raise NotImplementedError("TODO: structures")


def fit(X, y, e):
    """
    FastBLR(42, **NOISE) fitted on X (64, 42), y (64,), with error e (64,).
    """

    raise NotImplementedError("TODO: fit")


def structure_evidences(blr, grids):
    """
    log evidence of each structure (5,)
    """

    raise NotImplementedError("TODO: structure_evidences")


def posterior(E):
    """
    probabilities from log scores E, equal prior (used for structures and for change times)
    """

    raise NotImplementedError("TODO: posterior")


def structure_estimates(blr, grids, theta_nom):
    """
    each structure's posterior mean of theta (5, 42)
    """

    raise NotImplementedError("TODO: structure_estimates")


def average(weights, thetas):
    """
    sum_k weights[k] * thetas[k] (used over structures, then over change times)
    """

    raise NotImplementedError("TODO: average")


# Provided for you

def logsumexp(x):
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m))) if np.isfinite(m) else m


def use_solutions():
    from solutions import walker_adapter
    for name in ("structures", "fit", "structure_evidences", "posterior", "structure_estimates", "average"):
        globals()[name] = getattr(walker_adapter, name)


def run_length_update(logR, pred, fresh, hazard):
    """Bayes' rule over the change time (new candidate first)"""
    new = np.r_[math.log(hazard) + logsumexp(logR) + fresh, math.log(1 - hazard) + logR + pred]
    return new - logsumexp(new)


def expected_clip(mu, sd):
    """For u ~ N(mu, sd^2): E[clip(u, -1, 1)] and P(-1 < u < 1)"""
    erf = np.vectorize(math.erf)
    cdf = lambda x: .5 * (1 + erf(x / math.sqrt(2)))
    pdf = lambda x: np.exp(-.5 * x**2) / math.sqrt(2 * math.pi)
    lo, hi = (-1 - mu) / sd, (1 - mu) / sd
    inside = cdf(hi) - cdf(lo)
    return -cdf(lo) + (1 - cdf(hi)) + mu * inside + sd * (pdf(lo) - pdf(hi)), inside


class Run:
    """One candidate change time and the data since then"""

    def __init__(self, start, blr):
        self.start, self.blr = start, blr

    def mixture_evidence(self):
        """log p(data since start), averaging over the structures"""
        return logsumexp(self.E) - math.log(len(STRUCTURES))


class Adapter:
    def __init__(self, model, hazard, seed=0):
        self.model, self.hazard = model, hazard
        self.grids = None                                     # set on the first update
        self.W = torch.tensor(CALIBRATION["W"], dtype=torch.float32)
        self.theta_nom = CALIBRATION["theta_nom"].astype(float)
        self.theta, self.cov = self.theta_nom.copy(), CALIBRATION["S_nom"] + np.eye(A * (A + 1))
        self.runs, self.logR = [], np.zeros(0)
        self.rng, self.t = np.random.default_rng(seed), 0
        self.R, self.S = np.ones(1), np.eye(len(STRUCTURES))[:1]
        self.thetas = np.tile(self.theta_nom, (1, len(STRUCTURES), 1))

    def map(self, theta=None):
        """(M, b) from theta"""
        grid = (self.theta if theta is None else theta).reshape(A, A + 1)
        return np.eye(A) + grid[:, :A], grid[:, A]

    def rows(self, z, c, nz):
        """regression rows X, y and whitened error e for one decision"""
        M, b = self.map()
        cx = np.r_[c, 1.]
        cov_rows = self.cov.reshape(A, A + 1, A, A + 1)[range(A), :, range(A), :]
        a, slope = expected_clip(M @ c + b, np.sqrt(np.maximum(cx @ cov_rows @ cx, 1e-12)))
        zt, at = torch.as_tensor(z, dtype=torch.float32), torch.as_tensor(a, dtype=torch.float32)
        f = lambda v: self.model.next(zt, v)
        with torch.no_grad():
            pred = f(at)
        J = (self.W @ torch.func.jacfwd(f)(at).detach()).double().numpy() * slope       # (64, 6)
        e = (self.W @ (torch.as_tensor(nz, dtype=torch.float32) - pred)).double().numpy()
        X = (J[:, :, None] * cx).reshape(len(e), A * (A + 1))
        y = e + X @ (self.theta - self.theta_nom)
        return X, y, e

    def resample(self, logR):
        """Keep at most RUNS runs"""
        w = np.exp(logR - logsumexp(logR))
        if len(w) <= RUNS:
            keep = w > 1e-6
            keep[np.argmax(w)] = True
            return keep, logR
        order = np.argsort(w)[::-1]
        ws = w[order]
        tail = np.cumsum(ws[::-1])[::-1]
        for k in range(RUNS):
            c = tail[k] / (RUNS - k)
            if ws[k] < c:
                break
        big = np.zeros(len(w), bool)
        big[order[:k]] = True
        small = order[k:]
        u = (self.rng.random() + np.arange(RUNS - k)) * c
        pick = small[np.minimum(np.searchsorted(np.cumsum(w[small]), u), len(small) - 1)]
        keep = big.copy()
        keep[pick] = True
        new = np.full(len(w), -np.inf)
        new[big], new[pick] = np.log(w[big]), np.log(c)
        return keep, new

    def update(self, z, c, nz):
        """update with latent z, command c, next latent nz"""
        if self.grids is None:
            self.grids = structures()
        d = fit(*self.rows(z, c, nz))
        fresh = Run(self.t, d)
        self.t += 1
        fresh.E = structure_evidences(d, self.grids)
        pred = []
        for r in self.runs:
            before = r.mixture_evidence()
            r.blr = r.blr + d
            r.E = structure_evidences(r.blr, self.grids)
            pred.append(r.mixture_evidence() - before)
        logR = run_length_update(self.logR, np.array(pred), fresh.mixture_evidence(), self.hazard) if self.runs \
            else np.zeros(1)
        keep, logR = self.resample(logR)
        self.runs = [r for r, k in zip([fresh] + self.runs, keep) if k]
        self.R = posterior(logR[keep])                                    # P(change time)
        self.logR = np.log(self.R)
        self.S = np.array([posterior(r.E) for r in self.runs])            # P(structure | change time)
        self.thetas = np.array([structure_estimates(r.blr, self.grids, self.theta_nom) for r in self.runs])
        per_run = np.array([average(S, thetas) for S, thetas in zip(self.S, self.thetas)])
        self.theta = average(self.R, per_run)
        i, j = np.unravel_index(np.argmax(self.R[:, None] * self.S), self.S.shape)   # most probable run, structure
        self.cov = CALIBRATION["S_nom"] + self.runs[i].blr.cov(self.grids[STRUCTURES[j]])

    def change_time(self):
        """The most probable decision of the last change"""
        start = self.runs[int(np.argmax(self.R))].start
        return start if start > 0 else None

    def structure_probabilities(self):
        """P(structure) averaged over the change times"""
        return average(self.R, self.S)

    def structure_maps(self):
        """structure estimates averaged over the change times"""
        return [self.map(theta) for theta in average(self.R, self.thetas)]

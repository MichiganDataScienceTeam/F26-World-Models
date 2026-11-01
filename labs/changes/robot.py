"""A 2-motor robot that may break at step `change`: commands a go in, the motors deliver y = M a + noise.

M = I before the change and M_after from then on. r = y - a is the error of the nominal model.
"""
import numpy as np

SIGMA = .3      # noise SD
WORLDS = {'none': np.eye(2), 'weak': np.diag([.5, .5]), 'swap': np.array([[0., 1.], [1., 0.]])}


def stream(M_after, T=300, change=100, seed=0, outliers=0.):
    """Commands a (T, 2), outputs y and errors r = y - a. outliers: fraction of steps with an extra
    10-sigma error."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((T, 2))
    M = np.where(np.arange(T)[:, None, None] < change, np.eye(2), M_after)
    y = np.einsum('tij,tj->ti', M, a) + SIGMA*rng.standard_normal((T, 2))
    hit = rng.random(T) < outliers
    y[hit] += 10*SIGMA*rng.choice([-1, 1], (hit.sum(), 2))
    return a, y, y - a


def closed_form_evidence(X, y, prior_std):
    """log N(y; 0, SIGMA^2 I + prior_std^2 X X^T) for one output: to check your sequential evidence."""
    C = SIGMA**2*np.eye(len(y)) + prior_std**2*X@X.T
    return -.5*(y@np.linalg.solve(C, y) + np.linalg.slogdet(C)[1] + len(y)*np.log(2*np.pi))

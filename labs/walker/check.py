"""
1. Tests each function on made-up inputs.
2. A walk with the left knee jammed at decision 60. Expect "gain+offset" by ~decision 120.

python -m labs.walker.check [--solutions]
"""

import argparse

import numpy as np
import torch

from envs.walker import WalkerEnv
from adaptation.fast_blr import FastBLR
from labs.walker import adapter as student
from labs.walker.demo import HAZARD, HORIZON
from models.tdmpc2 import Planner, load

CHANGE, T = 60, 160


def report(name, ok):
    print(f"   {name}: {'OK' if ok else 'MISMATCH'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solutions", action="store_true")
    if parser.parse_args().solutions:
        student.use_solutions()

    print("1. the six functions")
    diag, offset = np.arange(6) * 8, np.arange(6) * 7 + 6       # positions in theta
    same = {S: np.zeros((42, 42)) for S in student.STRUCTURES}
    same["common"][np.ix_(diag, diag)] = 1
    same["gains"][diag, diag] = same["gain+offset"][diag, diag] = same["gain+offset"][offset, offset] = 1
    same["general"] = np.eye(42)
    grids = student.structures()
    for S in student.STRUCTURES:
        g = np.ravel(grids.get(S, [])).astype(int)
        report(f"structures[{S!r}]", g.size == 42 and np.array_equal((g[:, None] == g) & (g[:, None] >= 0), same[S]))
    rng = np.random.default_rng(0)
    X, y, e = rng.standard_normal((30, 42)), rng.standard_normal(30), rng.standard_normal(30)
    blr = student.fit(X, y, e)
    nu, sd2, eta = student.NOISE["nu"], student.NOISE["noise_std"] ** 2, student.NOISE["eta"]
    w = eta * (nu + 30) / (nu + e @ e / sd2) / sd2
    report("fit", isinstance(blr, FastBLR) and blr.n_features == 42 and blr.prior_std == 1
           and np.allclose(blr.G, w * X.T @ X) and np.allclose(blr.g, w * X.T @ y))
    blr, C = FastBLR(42).partial_fit(X, y), {S: np.eye(30) + X @ same[S] @ X.T for S in student.STRUCTURES}
    direct = [-.5 * (y @ np.linalg.solve(C[S], y) + np.linalg.slogdet(C[S])[1] + 30 * np.log(2 * np.pi))
              for S in student.STRUCTURES]
    report("structure_evidences", np.allclose(student.structure_evidences(blr, grids), direct))
    E = np.array([-3., 0., 1., 1000., 2.])
    P = np.exp(E - E.max())
    report("posterior", np.allclose(student.posterior(E), P / P.sum()))
    m0 = rng.standard_normal(42)
    want = [m0 + same[S] @ X.T @ np.linalg.solve(C[S], y) for S in student.STRUCTURES]
    report("structure_estimates", np.allclose(student.structure_estimates(blr, grids, m0), want))
    w, th = rng.dirichlet(np.ones(5)), rng.standard_normal((5, 42))
    R, thetas = np.array([.7, .3]), rng.standard_normal((2, 5, 42))           # thetas can have any shape
    report("average", np.allclose(student.average(w, th), w @ th)
           and np.allclose(student.average(R, thetas), .7 * thetas[0] + .3 * thetas[1]))

    print(f"2. a walk with the left knee jammed at decision {CHANGE}")
    model = load()
    env, planner, ad = WalkerEnv(seed=0), Planner(model, 1, HORIZON), student.Adapter(model, HAZARD)
    obs = env.reset()
    print(f"   decision   P({', '.join(student.STRUCTURES)})   last change at")
    for t in range(T):
        if t == CHANGE:
            env.set_fault(gain=[1, 1, 1, 1, 0, 1], offset=[0, 0, 0, 0, .8, 0])
        c = planner.act(obs[None], [t == 0])[0]
        nobs, _ = env.step(c)
        with torch.no_grad():
            ad.update(model.encode(torch.tensor(obs)).numpy(), c.astype(np.float64),
                      model.encode(torch.tensor(nobs)).numpy())
        if t + 1 - CHANGE in (-1, 1, 2, 5, 10, 20, 40, 60, 80, 100):
            print(f"   {t + 1:>8}   {np.round(ad.structure_probabilities(), 2)}   {ad.change_time()}")
        obs = nobs
    M, b = ad.map()
    print(f"   gains   {np.round(np.diag(M), 2)}\n   offsets {np.round(b, 2)}")


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as err:
        print(f"   {err}: fill it in, then run this again")

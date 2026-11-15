"""
1. Tests each function on made-up inputs.
2. A drive with the steering reversed at decision 200. Expect "gains" and a steering gain near -1 by ~decision 210.

python -m labs.driving.check [--solutions]
"""

import argparse

import numpy as np
import torch

from envs.driving import car
from envs.driving.road import HALF_WIDTH, Road
from labs.driving import adapter as student
from adaptation.fast_blr import FastBLR
from labs.driving.demo import FAULTS, HAZARD
from models.driving import BLOCK, COAST, KEYS, Planner, ahead, load, outputs

CHANGE, T = 200, 300


def report(name, ok):
    print(f"   {name}: {'OK' if ok else 'MISMATCH'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solutions", action="store_true")
    if parser.parse_args().solutions:
        student.use_solutions()

    print("1. the six functions")
    diag, offset = np.arange(3) * 5, np.arange(3) * 4 + 3       # positions in theta
    same = {S: np.zeros((12, 12)) for S in student.STRUCTURES}
    same["common"][np.ix_(diag, diag)] = 1
    same["gains"][diag, diag] = same["gain+offset"][diag, diag] = same["gain+offset"][offset, offset] = 1
    same["general"] = np.eye(12)
    grids = student.structures()
    for S in student.STRUCTURES:
        g = np.ravel(grids.get(S, [])).astype(int)
        report(f"structures[{S!r}]", g.size == 12 and np.array_equal((g[:, None] == g) & (g[:, None] >= 0), same[S]))
    rng = np.random.default_rng(0)
    X, y, e = rng.standard_normal((30, 12)), rng.standard_normal(30), rng.standard_normal(30)
    blr = student.fit(X, y, e)
    nu, sd2, eta = student.NOISE["nu"], student.NOISE["noise_std"] ** 2, student.NOISE["eta"]
    w = eta * (nu + 30) / (nu + e @ e / sd2) / sd2
    report("fit", isinstance(blr, FastBLR) and blr.n_features == 12 and blr.prior_std == 1
           and np.allclose(blr.G, w * X.T @ X) and np.allclose(blr.g, w * X.T @ y))
    blr, C = FastBLR(12).partial_fit(X, y), {S: np.eye(30) + X @ same[S] @ X.T for S in student.STRUCTURES}
    direct = [-.5 * (y @ np.linalg.solve(C[S], y) + np.linalg.slogdet(C[S])[1] + 30 * np.log(2 * np.pi))
              for S in student.STRUCTURES]
    report("structure_evidences", np.allclose(student.structure_evidences(blr, grids), direct))
    E = np.array([-3., 0., 1., 1000., 2.])
    P = np.exp(E - E.max())
    report("posterior", np.allclose(student.posterior(E), P / P.sum()))
    m0 = rng.standard_normal(12)
    want = [m0 + same[S] @ X.T @ np.linalg.solve(C[S], y) for S in student.STRUCTURES]
    report("structure_estimates", np.allclose(student.structure_estimates(blr, grids, m0), want))
    w, th = rng.dirichlet(np.ones(5)), rng.standard_normal((5, 12))
    R, thetas = np.array([.7, .3]), rng.standard_normal((2, 5, 12))           # thetas can have any shape
    report("average", np.allclose(student.average(w, th), w @ th)
           and np.allclose(student.average(R, thetas), .7 * thetas[0] + .3 * thetas[1]))

    print(f"2. a drive with the steering reversed at decision {CHANGE}")
    torch.set_num_threads(1)
    model, road = load(), Road(0)
    planner, ad = Planner(road, model), student.Adapter(model, HAZARD)
    x, cmd, i = car.start(1, road), torch.zeros(1, 3), torch.zeros(1, dtype=torch.long)
    grass, keys, fault = torch.zeros(1, dtype=torch.bool), COAST, None
    print(f"   decision   P({', '.join(student.STRUCTURES)})   last change at   steering gain")
    for t in range(T):
        if t == CHANGE:
            fault = FAULTS["steering reversed"]
        if t % BLOCK == 0 and t:
            keys = planner.commit()
        Mp, bp = (torch.tensor(v, dtype=torch.float32) for v in ad.planner_map())
        with torch.no_grad():                               # one search round per decision, for the next block
            planner.round(*(v[0] for v in ahead(road, model, x, cmd, i, keys, Mp, bp, BLOCK - t % BLOCK)), Mp, bp)
        x0, grass0 = x, grass
        cmd = car.smooth(cmd, KEYS[keys][None])
        c = cmd[0].double().numpy()
        a = cmd if fault is None else car.fault(cmd, *(torch.tensor(v, dtype=torch.float32)[None] for v in fault))
        x = car.drive(x, a, grass)
        i, _, e = road.locate(x, i)
        grass = e.abs() > HALF_WIDTH
        ad.update(x0[0, 3:6], grass0[0], c, outputs(x0, x)[0])
        if t + 1 - CHANGE in (-1, 1, 2, 5, 10, 20, 40, 60, 100):
            print(f"   {t + 1:>8}   {np.round(ad.structure_probabilities(), 2)}   {ad.change_time()}   "
                  f"{ad.planner_map()[0][0, 0]:.2f}")
    print(f"   speed {3.6 * x[0, 3].item():.0f} km/h, {'on the road' if not grass.item() else 'off the road'}")


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as err:
        print(f"   {err}: fill it in, then run this again")

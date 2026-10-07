"""The plan closest to our play that keeps a share of a variant's gain: the calendar LP with a second objective.

Among the plans of a variant (z fixed, a set of columns as played) that cost at most J_played - keep * (J_played - J*),
the one whose flows differ least from ours, each unit of difference weighed by the good's customs value (L1). The
calendar LP has many optima that differ from our play in ways worth nothing; this one differs only where it pays.

uv run python closest.py run --task=small --only=17,12 --variant=cal --keep=0.9
"""

import pickle
import time

import fire
import numpy as np
import scipy.sparse as sp

import common as C
import solve as S


def closest(cal, zpi, z, fixed, J_star, J_played, keep=0.9, limit=900.0):
    """The column vector (model columns only) of the closest plan, its cost and the weighted distance."""
    from scipy.optimize._highspy import _core as hs
    from shockbench_flow.oracle.lp import lp_usd

    inst = cal.inst
    K = inst.commodities
    nn, nz, nc, T = cal.nn, cal.nz, cal.nc, cal.T
    xcols = [j for j, key in enumerate(cal.keys) if key[0] == "x"]
    w_tmpl = np.array([K[cal.keys[j][2]].v for j in xcols])
    D = (np.arange(T)[:, None] * nc + np.array(xcols)[None, :]).ravel()
    w = np.tile(w_tmpl, T)
    if fixed is not None:
        keepD = ~fixed[D]
        D, w = D[keepD], w[keepD]
    nd = len(D)
    n0 = nn + nz
    # columns: the model's, then dp, dm
    A0 = cal.A.tocsr()
    rows_d = sp.csr_matrix((np.concatenate([np.ones(nd), -np.ones(nd), np.ones(nd)]), (np.tile(np.arange(nd), 3), np.concatenate([D, n0 + np.arange(nd), n0 + nd + np.arange(nd)]))), shape=(nd, n0 + 2 * nd))
    cost_row = sp.csr_matrix((cal.cost, (np.zeros(n0, dtype=int), np.arange(n0))), shape=(1, n0 + 2 * nd))
    A = sp.vstack([sp.hstack([A0, sp.csr_matrix((A0.shape[0], 2 * nd))], format="csr"), rows_d, cost_row], format="csc")
    bound = J_played - keep * (J_played - J_star) - cal.offset
    lo = np.concatenate([cal.lo, zpi[D], [-np.inf]])
    up = np.concatenate([cal.up, zpi[D], [bound]])
    lb, ub = cal.lb.copy(), cal.ub.copy()
    if fixed is not None:
        idx = np.flatnonzero(fixed)
        v = zpi[idx]
        d = C.BAND * np.abs(v)
        new_ub = np.minimum(ub[idx], v + d)
        new_lb = np.minimum(np.maximum(lb[idx], v - d), new_ub)
        lb[idx], ub[idx] = new_lb, np.maximum(new_ub, new_lb)
    lb[nn:], ub[nn:] = z.ravel(), z.ravel()
    lb = np.concatenate([lb, np.zeros(2 * nd)])
    ub = np.concatenate([ub, np.full(2 * nd, np.inf)])
    scale = 1.0 / max(float(w.max()), 1.0)
    cost = np.concatenate([np.zeros(n0), w * scale, w * scale])
    inf = hs.kHighsInf
    lp = hs.HighsLp()
    lp.num_col_, lp.num_row_ = A.shape[1], A.shape[0]
    lp.col_cost_ = cost
    lp.col_lower_ = np.where(np.isinf(lb), -inf, lb)
    lp.col_upper_ = np.where(np.isinf(ub), inf, ub)
    lp.row_lower_ = np.where(np.isinf(lo), -inf, lo)
    lp.row_upper_ = np.where(np.isinf(up), inf, up)
    lp.a_matrix_.format_ = hs.MatrixFormat.kColwise
    lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = A.shape[1], A.shape[0]
    lp.a_matrix_.start_ = A.indptr.astype(np.int32)
    lp.a_matrix_.index_ = A.indices.astype(np.int32)
    lp.a_matrix_.value_ = A.data.astype(np.float64)
    h = hs._Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(limit))
    h.passModel(lp)
    h.run()
    status = h.modelStatusToString(h.getModelStatus())
    x = np.array(h.getSolution().col_value, dtype=np.float64)
    if status != "Optimal":
        return None, None, None, status
    return x[:nn], float(lp_usd(cal.model, x[:nn])[0]), float(h.getInfo().objective_function_value / scale), status


def one(task, n, variant, keep):
    start = time.perf_counter()
    d = C.load_traj(task, n)
    cal = C.Cal(task, n)
    zpi = np.concatenate([d["zpi"], np.zeros(cal.nz)])
    z, _both = cal.calendar(d["zpi"])
    zpi[cal.nn :] = z.ravel()
    V = S.variants(cal, task)
    fixed = V[variant]
    J_star, status, _s, _x = cal.solve(zpi, z, fixed)
    J_played = d["J_cents"] / 100
    x, J, dist, status2 = closest(cal, zpi, z, fixed, J_star, J_played, keep)
    out = {"episode": n, "variant": variant, "keep": keep, "J_star": J_star, "J_played": J_played, "J": J, "dist": dist, "status": (status, status2), "x": None if x is None else x.reshape(cal.T, cal.nc), "keys": cal.keys, "z": z}
    (C.HERE / "res").mkdir(exist_ok=True)
    (C.HERE / "res" / f"{task}_{n}_close_{variant}_{int(round(keep * 100))}.pkl").write_bytes(pickle.dumps(out))
    return n, J_star, J_played, J, dist, status, status2, round(time.perf_counter() - start, 1)


def run(task="small", only=None, variant="cal", keep=0.9, n_jobs=2):
    from joblib import Parallel, delayed

    refs = C.references(task)
    ns = sorted(refs) if only is None else ([only] if isinstance(only, int) else [int(v) for v in only])
    results = Parallel(n_jobs=n_jobs)(delayed(one)(task, n, variant, float(keep)) for n in ns)
    for n, J_star, J_played, J, dist, status, status2, seconds in results:
        r = refs[n]
        D = (r["J_naive_cents"] - r["J_oracle_cents"]) / 100
        sc = lambda v: float("nan") if v is None else (r["J_naive_cents"] / 100 - v) / D  # noqa: E731
        print(f"episode {n}: played {sc(J_played):.4f}, {variant} {sc(J_star):.4f}, closest plan {sc(J):.4f} at distance {0 if dist is None else dist / 1e9:.2f} bn of goods moved differently ({status}, {status2}, {seconds} s)", flush=True)


if __name__ == "__main__":
    fire.Fire({"run": run})

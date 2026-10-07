"""The realizable clairvoyant: the whole-episode clairvoyant LP with "base load before fabs" made exact.

The model is the package's own clairvoyant LP (``shockbench_flow.oracle.lp.build_lp(inst, marks)``, the plan that scores
1) plus, for every grid with fabs under ``base_first`` and every week, one binary z:

    ysh[g, t]            <= ybar[g, t] * (1 - z[g, t])        z = 1: the grid sheds no base load this week
    sum_f E[f, t]        <= Emax[g, t] * z[g, t]              z = 0: its fabs get no energy this week

with Emax the fabs' draw at capacity, e_f * alpha_f(t) * cap0_f over the grid's fabs with R_f(t) > 0, and no more than
the grid's power above its base load. This is ``dynamics.production.allocate_energy`` for ``base_first``: y = min(ybar,
G_av), and the fabs share G_av - y. Every other freedom of the clairvoyant LP stays (lot starts, releases, the mix of
segments), so the optimum is still a bound: no agent can cost less.

uv run python milp_real.py validate --task=small --episode=0
uv run python milp_real.py run --task=small --only=0,2 --limit=120
"""

import json
import pickle
import sys
import time
from pathlib import Path

import fire
import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp


ROOT = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT / "experiments" / "approaches"))
OUT = HERE / "milp"


def world(task, entropy, n):
    from shockbench_flow.evaluation.cache import default_cache_dir
    from shockbench_flow.policies.naive_fq import REPLICATIONS
    from shockbench_flow_agent.scoring import _world

    return _world(task, entropy, n, REPLICATIONS, str(default_cache_dir()))


def extend(inst, marks, model):
    """The clairvoyant LP's arrays with the binaries appended: (cost, offset, A, lo, hi, lb, ub, integrality, info)."""
    T, nc, tmpl = model.T, model.meta["nc"], model.meta["template"]
    n = len(model.lb)
    grids = []
    for go, g in enumerate(inst.grids):
        ga = inst.nodes[g].grid
        fabs = [fo for fo in inst.grid_fabs[go] if inst.nodes[inst.fabs[fo]].fab.e > 0]
        if ga.priority == "base_first" and fabs:
            grids.append((go, fabs, sum(ga.shares.values())))
    nz = len(grids) * T
    rows, cols, vals, hi = [], [], [], []
    r = 0
    emax = np.zeros((len(grids), T))
    for gi, (go, fabs, S) in enumerate(grids):
        ybar, Gbar = np.asarray(marks.y_bar)[:, go], np.asarray(marks.G_bar)[:, go]
        cap = np.zeros(T)
        for fo in fabs:
            fa = inst.nodes[inst.fabs[fo]].fab
            cap += np.where(np.asarray(marks.R)[:, fo] > 0, fa.e * np.asarray(marks.alpha_bar)[:, fo] * fa.cap0, 0.0)
        cap = np.minimum(cap, np.maximum(Gbar * S - ybar, 0.0))
        emax[gi] = cap
        for t in range(T):
            z = n + gi * T + t
            rows += [r, r]  # ysh + ybar z <= ybar
            cols += [t * nc + tmpl[("ysh", go)], z]
            vals += [1.0, float(ybar[t])]
            hi.append(float(ybar[t]))
            r += 1
            for fo in fabs:  # sum E - Emax z <= 0
                rows.append(r)
                cols.append(t * nc + tmpl[("E", fo)])
                vals.append(1.0)
            rows.append(r)
            cols.append(z)
            vals.append(-float(cap[t]))
            hi.append(0.0)
            r += 1
    extra = sp.csr_matrix((vals, (rows, cols)), shape=(r, n + nz))
    pad = lambda A: sp.hstack([A, sp.csr_matrix((A.shape[0], nz))], format="csr")  # noqa: E731
    A = sp.vstack([pad(model.A_ub), pad(model.A_eq), extra], format="csr")
    lo = np.concatenate([np.full(model.A_ub.shape[0], -np.inf), model.b_eq, np.full(r, -np.inf)])
    up = np.concatenate([model.b_ub, model.b_eq, np.array(hi)])
    cost = np.concatenate([model.objective(), np.zeros(nz)])
    lb = np.concatenate([model.lb, np.zeros(nz)])
    ub = np.concatenate([model.ub, np.ones(nz)])
    integrality = np.concatenate([np.zeros(n), np.ones(nz)])
    offset = -float(np.sum(model.meta["salvage_const"]))
    return cost, offset, A, lo, up, lb, ub, integrality, {"grids": grids, "n": n, "nz": nz, "emax": emax}


def solve(arrays, limit, gap=1e-4, fix=None):
    cost, offset, A, lo, up, lb, ub, integrality, info = arrays
    lb, ub = lb.copy(), ub.copy()
    if fix:  # column -> value
        idx = np.fromiter(fix.keys(), dtype=int)
        val = np.fromiter(fix.values(), dtype=float)
        lb[idx], ub[idx] = val, val
    start = time.perf_counter()
    res = milp(cost, constraints=LinearConstraint(A, lo, up), integrality=integrality, bounds=Bounds(lb, ub), options={"time_limit": float(limit), "mip_rel_gap": gap, "disp": False})
    out = {
        "status": int(res.status),
        "message": str(res.message)[:60],
        "seconds": round(time.perf_counter() - start, 1),
        "J": None if res.x is None else float(res.fun) + offset,
        "bound": None if getattr(res, "mip_dual_bound", None) is None else float(res.mip_dual_bound) + offset,
        "x": res.x,
    }
    return out


def solve_hs(arrays, limit, gap=1e-4, fix=None, start=None, relax=False):
    """The same model through the HiGHS that SciPy bundles, directly: a starting solution, the proven bound.

    ``fix``: column -> value; ``relax``: every column continuous (with the binaries fixed: the best plan under a given
    on/off calendar); ``start``: a feasible column vector to start the search from.
    """
    from scipy.optimize._highspy import _core as hs

    cost, offset, A, lo, up, lb, ub, integrality, info = arrays
    lb, ub = lb.copy(), ub.copy()
    if fix:
        idx = np.fromiter(fix.keys(), dtype=int)
        val = np.fromiter(fix.values(), dtype=float)
        lb[idx], ub[idx] = val, val
    A = sp.csc_matrix(A)
    inf = hs.kHighsInf
    lp = hs.HighsLp()
    lp.num_col_, lp.num_row_ = A.shape[1], A.shape[0]
    lp.col_cost_ = cost
    lp.col_lower_ = np.where(np.isinf(lb), -inf, lb)
    lp.col_upper_ = np.where(np.isinf(ub), inf, ub)
    lp.row_lower_ = np.where(np.isinf(lo), -inf, lo)
    lp.row_upper_ = np.where(np.isinf(up), inf, up)
    lp.offset_ = float(offset)
    lp.a_matrix_.format_ = hs.MatrixFormat.kColwise
    lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = A.shape[1], A.shape[0]
    lp.a_matrix_.start_ = A.indptr.astype(np.int32)
    lp.a_matrix_.index_ = A.indices.astype(np.int32)
    lp.a_matrix_.value_ = A.data.astype(np.float64)
    if not relax:
        lp.integrality_ = [hs.HighsVarType.kInteger if v else hs.HighsVarType.kContinuous for v in integrality]
    h = hs._Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(limit))
    h.setOptionValue("mip_rel_gap", float(gap))
    h.passModel(lp)
    if start is not None:
        sol = hs.HighsSolution()
        sol.col_value = np.asarray(start, dtype=np.float64)
        h.setSolution(sol)
    t0 = time.perf_counter()
    h.run()
    hinfo = h.getInfo()
    status = h.modelStatusToString(h.getModelStatus())
    x = np.array(h.getSolution().col_value, dtype=np.float64)
    ok = x.shape == (A.shape[1],) and bool(np.all(np.isfinite(x))) and hinfo.primal_solution_status == 2
    return {
        "message": status,
        "status": 0 if status == "Optimal" else 1,
        "seconds": round(time.perf_counter() - t0, 1),
        "J": float(hinfo.objective_function_value) if ok else None,
        "bound": None if relax else float(hinfo.mip_dual_bound),
        "x": x if ok else None,
    }


def calendar_of(task, n, info, T):
    """Our agent's own on/off calendar from its traced episode: z = 1 where the grid shed no base load."""
    path = HERE / "trace" / f"{task}_v2" / f"{n}.pkl"
    if path.exists():
        d = pickle.loads(path.read_bytes())
        shed, ybar, J = d["played"]["shed"], d["marks"]["y_bar"], d["played"]["J"]
    else:  # not traced: play the episode again
        traj = played(task, 12345, n)
        shed = np.array([r.shed for r in traj.records])
        ybar = np.asarray(world(task, 12345, n)[2].y_bar)
        J = traj.J_cents / 100
    z = np.zeros((len(info["grids"]), T))
    for gi, (go, _fabs, _S) in enumerate(info["grids"]):
        z[gi] = (shed[:, go] <= 1e-9 * np.maximum(ybar[:, go], 1.0)).astype(float)
    return z, J


def one_warm(task, entropy, n, limit, gap, mip):
    """The best plan under our own on/off calendar (an LP), then the MILP started from it."""
    from shockbench_flow.oracle.lp import build_lp, lp_usd

    inst, omega, marks, _fallback = world(task, entropy, n)
    model = build_lp(inst, marks)
    arrays = extend(inst, marks, model)
    info = arrays[8]
    nn, T = info["n"], model.T
    z, J_played = calendar_of(task, n, info, T)
    fix = {nn + i: float(v) for i, v in enumerate(z.ravel())}
    cal = solve_hs(arrays, 900, fix=fix, relax=True)
    out = {"episode": n, "J_played": J_played, "cal_message": cal["message"], "cal_seconds": cal["seconds"]}
    out["J_calendar"] = None if cal["x"] is None else float(lp_usd(model, cal["x"][:nn])[0])
    if mip:
        res = solve_hs(arrays, limit, gap, start=cal["x"])
        x = res.pop("x")
        out.update(res)
        if x is not None:
            out["J_usd"] = float(lp_usd(model, x[:nn])[0])
            OUT.mkdir(exist_ok=True)
            (OUT / f"{task}_{n}.pkl").write_bytes(
                pickle.dumps({"x": x[:nn].reshape(T, model.meta["nc"]), "z": x[nn:].reshape(len(info["grids"]), T), "columns": list(model.columns), "grids": info["grids"], "J": out["J_usd"], "bound": out["bound"]})
            )
    return out


def warm(task="full", entropy=12345, only=None, limit=900, gap=1e-3, n_jobs=2, mip=True, tag="_warm"):
    """Per episode: our play, the best plan under our own on/off calendar, and (``mip``) the MILP started from it."""
    from joblib import Parallel, delayed

    total = {"small": 64, "full": 16}[task]
    refs = {r["episode"]: r for r in json.loads((ROOT / "outputs" / "recovery_tune" / f"{task}-e{entropy}-n{total}" / "references.json").read_text())}
    traced = sorted(int(p.stem) for p in (HERE / "trace" / f"{task}_v2").glob("*.pkl"))
    ns = traced if only is None else ([only] if isinstance(only, int) else [int(v) for v in only])
    start = time.perf_counter()
    results = Parallel(n_jobs=n_jobs)(delayed(one_warm)(task, entropy, n, limit, gap, mip) for n in ns)
    path = HERE / f"milp_{task}{tag}.json"
    done = json.loads(path.read_text()) if path.exists() else {}
    for res in results:
        n = res["episode"]
        r = refs[n]
        naive, oracle = r["J_naive_cents"] / 100, r["J_oracle_cents"] / 100
        share = lambda J: None if J is None else round((naive - J) / (naive - oracle), 4)  # noqa: E731
        res["stratum"] = r["stratum"]
        res["score_played"], res["score_calendar"] = share(res["J_played"]), share(res["J_calendar"])
        line = f"episode {n} level {r['stratum']}: played {res['score_played']}, our calendar with the best plan {res['score_calendar']} ({res['cal_message']}, {res['cal_seconds']} s)"
        if mip:
            res["score"], res["score_bound"] = share(res.get("J_usd")), share(res.get("bound"))
            line += f"; realizable incumbent {res['score']}, bound {res['score_bound']} ({res['message']}, {res['seconds']} s)"
        done[str(n)] = res
        print(line, flush=True)
    path.write_text(json.dumps(done, indent=1))
    print(f"{time.perf_counter() - start:.0f} s")

def played(task, entropy, n):
    """Our agent's episode again (agents/ft_buffer_v2 as it is): the trajectory, for the replay check."""
    from shockbench_flow.dynamics.env import rollout
    from shockbench_flow.hosting.docker import without_secret_like
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _metered_shim, _policy_seed
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    inst, omega, marks, fallback = world(task, entropy, n)
    with without_secret_like():
        factory = load_agent_class(str(ROOT / "agents" / "ft_buffer_v2"), "submission_milp")
        sys.modules[factory.__module__].PARAMS.update({"strict": True})
        shim = _metered_shim(factory, None)
        traj = rollout(inst, shim, omega, "standard", _policy_seed(entropy, n, NO_ZIP_SHA256), marks=marks, fallback=fallback)
    unload_agent()
    return traj


def validate(task="small", episode=0, entropy=12345, limit=120):
    """Our own trajectory in the model: every row holds, the cost is ours; then the model with our flows fixed."""
    from shockbench_flow.oracle.lp import build_lp, lp_usd
    from shockbench_flow.oracle.replay import replay, trajectory_vector

    inst, omega, marks, _fallback = world(task, entropy, episode)
    model = build_lp(inst, marks)
    traj = played(task, entropy, episode)
    rep = replay(model, traj)
    print(f"replay of our trajectory in the clairvoyant LP: feasible {rep.feasible} (worst residual {rep.worst_residual:.2e} at {rep.worst_row}), cost equal {rep.cost_equal}")
    arrays = extend(inst, marks, model)
    cost, offset, A, lo, up, lb, ub, integrality, info = arrays
    zpi = trajectory_vector(model, traj)
    T, nc, tmpl = model.T, model.meta["nc"], model.meta["template"]
    zb = np.zeros(info["nz"])
    both = 0
    for gi, (go, fabs, _S) in enumerate(info["grids"]):
        for t in range(T):
            shed = zpi[t * nc + tmpl[("ysh", go)]]
            E = sum(zpi[t * nc + tmpl[("E", fo)]] for fo in fabs)
            ybar = float(np.asarray(marks.y_bar)[t, go])
            zb[gi * T + t] = 1.0 if shed <= 1e-9 * max(ybar, 1.0) else 0.0
            both += shed > 1e-6 * ybar and E > 1e-6
    full = np.concatenate([zpi, zb])
    Ax = A @ full
    scale = np.maximum(1.0, np.maximum(np.abs(np.where(np.isfinite(up), up, 0.0)), np.abs(Ax)))
    worst = float(np.max(np.maximum(Ax - up, lo - Ax) / scale))
    J_pi = traj.J_cents / 100
    print(f"with the binaries set from the records: grid-weeks that shed and powered fabs {both}; worst row residual of the whole model {worst:.2e}")
    print(f"our cost {J_pi / 1e9:.3f} bn; the model's cost of our vector {(float(cost @ full) + offset) / 1e9:.3f} bn; lp_usd {lp_usd(model, zpi)[0] / 1e9:.3f} bn")
    # the same with the agent's own decisions fixed: flows from non-chokepoint nodes and the releases at straits
    fix = {}
    for key, j in tmpl.items():
        if key[0] == "x":
            for t in range(T):
                fix[t * nc + j] = float(zpi[t * nc + j])
    res = solve(arrays, limit, fix=fix)
    refs = {r["episode"]: r for r in json.loads((ROOT / "outputs" / "recovery_tune" / f"{task}-e{entropy}-n{'64' if task == 'small' else '16'}" / "references.json").read_text())}
    r = refs[episode]
    naive, oracle = r["J_naive_cents"] / 100, r["J_oracle_cents"] / 100
    share = lambda J: (naive - J) / (naive - oracle)  # noqa: E731
    print(f"with our flows fixed the model's best is {res['J'] / 1e9 if res['J'] else float('nan'):.3f} bn ({res['message']}, {res['seconds']} s): score {share(res['J']) if res['J'] else float('nan'):.4f} against ours {share(J_pi):.4f}")


def one(task, entropy, n, limit, gap):
    from shockbench_flow.oracle.lp import build_lp, lp_usd

    inst, omega, marks, _fallback = world(task, entropy, n)
    model = build_lp(inst, marks)
    arrays = extend(inst, marks, model)
    res = solve(arrays, limit, gap)
    x = res.pop("x")
    out = dict(res, episode=n)
    if x is not None:
        nn = arrays[8]["n"]
        out["J_usd"] = float(lp_usd(model, x[:nn])[0])
        OUT.mkdir(exist_ok=True)
        (OUT / f"{task}_{n}.pkl").write_bytes(
            pickle.dumps({"x": x[:nn].reshape(model.T, model.meta["nc"]).astype(np.float64), "z": x[nn:].reshape(len(arrays[8]["grids"]), model.T), "columns": list(model.columns), "grids": arrays[8]["grids"], "J": out["J_usd"], "bound": out["bound"]})
        )
    return out


def run(task="small", entropy=12345, only=None, limit=120, gap=1e-4, n_jobs=2, tag=""):
    from joblib import Parallel, delayed

    total = {"small": 64, "full": 16}[task]
    refs = {r["episode"]: r for r in json.loads((ROOT / "outputs" / "recovery_tune" / f"{task}-e{entropy}-n{total}" / "references.json").read_text())}
    ns = sorted(refs) if only is None else ([only] if isinstance(only, int) else [int(v) for v in only])
    start = time.perf_counter()
    results = Parallel(n_jobs=n_jobs)(delayed(one)(task, entropy, n, limit, gap) for n in ns)
    path = HERE / f"milp_{task}{tag}.json"
    done = json.loads(path.read_text()) if path.exists() else {}
    for res in results:
        n = res["episode"]
        r = refs[n]
        naive, oracle = r["J_naive_cents"] / 100, r["J_oracle_cents"] / 100
        share = lambda J: None if J is None else (naive - J) / (naive - oracle)  # noqa: E731
        res["score"], res["score_bound"], res["stratum"] = share(res.get("J_usd", res["J"])), share(res["bound"]), r["stratum"]
        done[str(n)] = res
        print(f"episode {n} level {r['stratum']}: incumbent {res['score'] if res['score'] is None else round(res['score'], 4)}, bound {res['score_bound'] if res['score_bound'] is None else round(res['score_bound'], 4)} ({res['message']}, {res['seconds']} s)", flush=True)
    path.write_text(json.dumps(done, indent=1))
    print(f"{time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    fire.Fire({"validate": validate, "run": run, "warm": warm})

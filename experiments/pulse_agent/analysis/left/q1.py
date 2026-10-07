"""Question 1: is the optimum's opening a reaction to what week 1 shows, or to events still to come?

Per episode, three worlds with the episode's own demand: N (every event removed), V (only the events that had started
by the instant week 1 observes, with their true ends) and the true one. In each, the best plan under our own fab
calendar (an LP), and among its near-optimal plans the one whose flows of weeks 1..8 are closest to ours (so it differs
from our play only where that pays). The opening flows of each plan (weeks 1..4 and 1..8) are then put into the true
world (moved to the closest flows the true network allows) and the rest is optimised with full knowledge.

uv run python q1.py run --task=small --n_jobs=2
uv run python q1.py report --task=small
"""

import json
import pickle
import time

import fire
import numpy as np
import scipy.sparse as sp

import common as C
import worlds as W


OUT = C.HERE / "q1"
OPEN = (4, 8)


def l1(cal, lb, ub, cols, target, w, cap=None, limit=900.0):
    """min sum w |x[cols] - target| over the model's rows and the bounds, with cost @ x <= cap when given.

    Returns (x over the model's columns or None, the weighted distance, the status).
    """
    from scipy.optimize._highspy import _core as hs

    C.one_thread()
    n0, nd = cal.nn + cal.nz, len(cols)
    A0 = cal.A.tocsr()
    rows_d = sp.csr_matrix(
        (np.concatenate([np.ones(nd), -np.ones(nd), np.ones(nd)]), (np.tile(np.arange(nd), 3), np.concatenate([cols, n0 + np.arange(nd), n0 + nd + np.arange(nd)]))),
        shape=(nd, n0 + 2 * nd),
    )
    blocks = [sp.hstack([A0, sp.csr_matrix((A0.shape[0], 2 * nd))], format="csr"), rows_d]
    lo, up = [cal.lo, target], [cal.up, target]
    if cap is not None:
        blocks.append(sp.csr_matrix((cal.cost, (np.zeros(n0, dtype=int), np.arange(n0))), shape=(1, n0 + 2 * nd)))
        lo.append([-np.inf])
        up.append([cap - cal.offset])
    A = sp.vstack(blocks, format="csc")
    lo, up = np.concatenate(lo), np.concatenate(up)
    lbx = np.concatenate([lb, np.zeros(2 * nd)])
    ubx = np.concatenate([ub, np.full(2 * nd, np.inf)])
    scale = 1.0 / max(float(np.max(w)), 1.0)
    cost = np.concatenate([np.zeros(n0), w * scale, w * scale])
    inf = hs.kHighsInf
    lp = hs.HighsLp()
    lp.num_col_, lp.num_row_ = A.shape[1], A.shape[0]
    lp.col_cost_ = cost
    lp.col_lower_ = np.where(np.isinf(lbx), -inf, lbx)
    lp.col_upper_ = np.where(np.isinf(ubx), inf, ubx)
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
    if status != "Optimal":
        return None, None, status
    x = np.array(h.getSolution().col_value, dtype=np.float64)
    return x[:n0], float(h.getInfo().objective_function_value / scale), status


def xcols(cal, weeks):
    """(columns of the flows of the given weeks, their customs values)."""
    K = cal.inst.commodities
    tm = [j for j, key in enumerate(cal.keys) if key[0] == "x"]
    v = np.array([K[cal.keys[j][2]].v for j in tm])
    ts = np.asarray(list(weeks), dtype=int)
    return (ts[:, None] * cal.nc + np.array(tm)[None, :]).ravel(), np.tile(v, len(ts))


def near(cal, z, zpi, weeks=8, tol=2e-4, D=1.0):
    """The best plan under calendar z, and among the plans within tol * D of it the one closest to our flows of the first weeks."""
    lb, ub = cal.bounds(z=z)
    J, status, _s, x = cal.lp(lb, ub)
    if J is None:
        # this world does not allow our calendar (a week we ran whole is short of fuel in it): keep our calendar for
        # the first weeks only and let the later weeks' z take any value between 0 and 1
        zr = np.asarray(z, dtype=float).copy()
        zr[:, weeks:] = np.nan
        lb, ub = cal.bounds(z=zr)
        J, status2, _s, x = cal.lp(lb, ub, fresh=True)
        status = f"{status}; later weeks relaxed: {status2}"
        if J is None:
            return None, None, None, status
    cols, w = xcols(cal, range(weeks))
    obj = float(cal.cost @ x + cal.offset)
    xc, dist, status2 = l1(cal, lb, ub, cols, zpi[cols], w, cap=obj + tol * D)
    if xc is None:
        return J, x, None, f"{status}; closest: {status2}"
    return J, xc, dist, status


def transplant(cal, z, plan, weeks, also=None):
    """In ``cal``'s world: the flows of the first weeks moved as close to ``plan``'s as the world allows, then fixed and
    the rest optimised. ``also``: a mask of more columns to hold at ``plan`` (not projected). Returns (J, distance moved)."""
    lb, ub = cal.bounds(z=z)
    cols, w = xcols(cal, range(weeks))
    full = np.concatenate([plan, z.ravel()]) if len(plan) == cal.nn else plan
    xp, dist, status = l1(cal, lb, ub, cols, full[cols], w)
    if xp is None:
        return None, None, status
    fixed = np.zeros(cal.nn + cal.nz, dtype=bool)
    fixed[cols] = True
    values = xp.copy()
    lb2, ub2 = cal.bounds(z=z, fixed=fixed, values=values, band=1e-9)
    J, status, _s, _x = cal.lp(lb2, ub2)
    if J is None:
        lb2, ub2 = cal.bounds(z=z, fixed=fixed, values=values, band=1e-7)
        J, status, _s, _x = cal.lp(lb2, ub2, fresh=True)
    return J, dist, status


def one(task, n, kinds=("N", "V", "P"), opens=OPEN):
    start = time.perf_counter()
    refs = C.references(task)[n]
    D = (refs["J_naive_cents"] - refs["J_oracle_cents"]) / 100
    d = C.load_traj(task, n)
    true = C.Cal(task, n)
    z, _both = true.calendar(d["zpi"])
    zpi = np.concatenate([d["zpi"], z.ravel()])
    inst, omega = C.world(task, n)[:2]
    evs = W.events(inst, omega)
    out = {"episode": n, "D": D, "J_played": d["J_cents"] / 100, "events": len(evs), "carried": sum(q.onset <= 0 for q in evs), "J": {}, "dist": {}, "status": {}}
    plans = {}
    # the true world: the optimum, and the near-optimal plan closest to our opening
    J, x, dist, status = near(true, z, zpi, D=D)
    out["J"]["true"], out["dist"]["true"], out["status"]["true"] = J, dist, status
    plans["true"] = x
    for kind in kinds:
        # N: no event; V: the events begun by the instant week 1 observes, with their true ends; P: those of them still
        # acting at that instant, each lasting the whole episode (the network of week 1's observation, kept)
        if kind == "N":
            marks = W.subset_marks(inst, omega, [])
        elif kind == "V":
            marks = W.subset_marks(inst, omega, [q.row for q in evs if q.onset <= 0.0])
        else:
            marks = W.subset_marks(inst, omega, [q.row for q in evs if q.onset <= 0.0 < q.end], until=true.T)
        cw = C.Cal(task, n, marks=marks)
        if cw.keys != true.keys or cw.nn != true.nn:
            raise RuntimeError("the counterfactual world's LP has other columns than the true one")
        J, x, dist, status = near(cw, z, zpi, D=D)
        out["J"][f"world_{kind}"], out["dist"][f"world_{kind}"], out["status"][f"world_{kind}"] = J, dist, status
        plans[kind] = x
        del cw
    plans["ours"] = zpi
    for name, plan in plans.items():
        if plan is None:
            continue
        for t in opens:
            J, dist, status = transplant(true, z, plan[: true.nn], t)
            out["J"][f"{name}_{t}"], out["dist"][f"{name}_{t}"], out["status"][f"{name}_{t}"] = J, dist, status
    out["seconds"] = round(time.perf_counter() - start, 1)
    OUT.mkdir(exist_ok=True)
    (OUT / f"{task}_{n}.json").write_text(json.dumps(out, indent=1))
    (OUT / f"{task}_{n}.pkl").write_bytes(pickle.dumps({k: (None if p is None else p[: true.nn].reshape(true.T, true.nc)[:16].astype(np.float32)) for k, p in plans.items()}))
    return out


def run(task="small", only=None, n_jobs=2, redo=False, kinds="N,V,P", opens="4,8"):
    from joblib import Parallel, delayed

    refs = C.references(task)
    ns = sorted(refs) if only is None else ([only] if isinstance(only, int) else [int(v) for v in only])
    todo = [n for n in ns if redo or not (OUT / f"{task}_{n}.json").exists()]
    kinds = tuple(kinds) if isinstance(kinds, (list, tuple)) else tuple(v for v in str(kinds).split(",") if v)
    opens = tuple(int(v) for v in (opens if isinstance(opens, (list, tuple)) else str(opens).split(",")))
    start = time.perf_counter()
    for i in range(0, len(todo), 2 * n_jobs):
        for out in Parallel(n_jobs=n_jobs)(delayed(one)(task, n, kinds, opens) for n in todo[i : i + 2 * n_jobs]):
            D, naive = out["D"], refs[out["episode"]]["J_naive_cents"] / 100
            sc = lambda J: float("nan") if J is None else (naive - J) / D  # noqa: E731
            print(f"episode {out['episode']}: " + " ".join(f"{k} {sc(v):.4f}" for k, v in out["J"].items() if not k.startswith("world")) + f"  ({out['seconds']} s)", flush=True)
    print(f"{time.perf_counter() - start:.0f} s")


def report(task="small"):
    refs = C.references(task)
    res = {}
    for n in sorted(refs):
        p = OUT / f"{task}_{n}.json"
        if p.exists():
            res[n] = json.loads(p.read_text())
    ns = sorted(res)
    base = {n: json.loads((C.HERE / "res" / f"{task}_{n}.json").read_text()) for n in ns}
    print(f"{task}: {len(ns)} episodes; events per episode {np.mean([res[n]['events'] for n in ns]):.1f}, carried in {np.mean([res[n]['carried'] for n in ns]):.1f}")
    names = sorted({k for n in ns for k in res[n]["J"] if not k.startswith("world")})
    bad = {k: [n for n in ns if res[n]["J"].get(k) is None] for k in names}
    relaxed = {kind: [n for n in ns if "relaxed" in str(res[n]["status"].get(f"world_{kind}", ""))] for kind in ("N", "V", "P")}
    print(f"  worlds that did not allow our calendar (their later weeks' calendar was relaxed): { {k: len(v) for k, v in relaxed.items()} }")
    print(f"  unsolved: { {k: v for k, v in bad.items() if v} }")
    sc = {}
    for k in names:
        if bad[k]:
            continue
        sc[k], _ = C.pooled(refs, {n: res[n]["J"][k] * 100 for n in ns})
    for k in ("played", "cal", "pre_4", "prex_4", "pre_8", "prex_8", "fix_x"):
        J = {n: (base[n]["J_played"] if k == "played" else base[n]["J"][k]) * 100 for n in ns}
        sc[k], _ = C.pooled(refs, J)
    print(f"  played {sc['played']:.4f}; calendar optimum {sc['cal']:.4f}; with our flows of weeks 1-4 {sc['prex_4']:.4f} (loss {sc['cal'] - sc['prex_4']:.4f}); of weeks 1-8 {sc['prex_8']:.4f} (loss {sc['cal'] - sc['prex_8']:.4f})")
    print("  the opening flows of each plan put into the true world, the rest optimised with full knowledge (score; share of the opening's loss recovered)")
    for t in OPEN:
        lossx = sc["cal"] - sc[f"prex_{t}"]
        print(f"  weeks 1-{t}:")
        for name, label in (("ours", "ours (check: equals the line above)"), ("true", "the true optimum's, closest to ours"), ("V", "the optimum that knows only the events begun by week 1 (true ends)"), ("P", "the optimum of week 1's network kept for the whole episode"), ("N", "the optimum of the network without any event")):
            k = f"{name}_{t}"
            if k in sc:
                moved = np.mean([res[n]["dist"][k] for n in ns]) / 1e9
                print(f"    {label:68s} {sc[k]:.4f}  recovered {(sc[k] - sc[f'prex_{t}']) / lossx:5.0%}   (moved to fit the true network: {moved:.2f} bn of goods per episode)")
    # by whether a new physical event starts early
    w = {}
    from report import weights

    w = weights(task, ns)
    ev = {r["n"]: r for r in json.loads((C.HERE / f"opening_events_{task}.json").read_text())}
    for label, pick in (("a new event starts in weeks 1-12", lambda n: bool(ev[n]["new"])), ("none starts", lambda n: not ev[n]["new"])):
        sub = [n for n in ns if pick(n)]
        if not sub:
            continue
        for t in OPEN:
            tot = sum(w[n] * (base[n]["J"][f"prex_{t}"] - base[n]["J"]["cal"]) for n in sub)
            line = f"  {label} ({len(sub)} episodes), weeks 1-{t}: the opening's loss {tot:.4f} pooled;"
            for name in ("true", "V", "P", "N"):
                have = [n for n in sub if res[n]["J"].get(f"{name}_{t}") is not None]
                rec = sum(w[n] * (base[n]["J"][f"prex_{t}"] - res[n]["J"][f"{name}_{t}"]) for n in have)
                tot_h = sum(w[n] * (base[n]["J"][f"prex_{t}"] - base[n]["J"]["cal"]) for n in have)
                line += f" {name} recovers {rec:.4f} ({rec / tot_h if tot_h else float('nan'):.0%}{'' if len(have) == len(sub) else f', {len(have)} episodes'})"
            print(line)
    print("  per episode, weeks 1-4 (bn USD): the opening's loss with our flows; what each plan's opening recovers")
    rows = []
    for n in ns:
        loss = (base[n]["J"]["prex_4"] - base[n]["J"]["cal"]) / 1e9
        rows.append((loss, n, *[((base[n]["J"]["prex_4"] - res[n]["J"][f"{name}_4"]) / 1e9 if res[n]["J"].get(f"{name}_4") is not None else float("nan")) for name in ("true", "V", "P", "N")]))
    rows.sort(reverse=True)
    for loss, n, a, b, p, c in rows[:24]:
        print(f"    episode {n:2d} level {refs[n]['stratum']}: loss {loss:7.2f}; true {a:7.2f}  V {b:7.2f}  P {p:7.2f}  N {c:7.2f}   new events early {len(ev[n]['new'])}")


if __name__ == "__main__":
    fire.Fire({"run": run, "report": report})

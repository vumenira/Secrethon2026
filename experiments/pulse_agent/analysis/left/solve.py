"""The calendar LP of every episode of agents/mpc_pulse_rec_v6 under each variant; results in left/res.

A variant fixes a set of the LP's columns to what we played (z, the fab calendar, is always ours) and frees the rest.

uv run python solve.py run --task=small --n_jobs=2
"""

import json
import pickle
import time

import fire
import numpy as np

import common as C


PREFIX = {"small": (4, 8, 13, 20, 26, 32, 38, 44, 48), "full": (4, 8, 16, 26, 52, 78, 92)}
OPEN = (4, 8)  # the opening: weeks 1..4 and 1..8
KEEP = ("cal", "fix_x", "prex_4", "pre_4")


def variants(cal, task, sets="all"):
    cls = C.classes(cal.inst)
    m = lambda name, weeks=None: cal.mask(cls[name], weeks)  # noqa: E731
    nonstate, x = m("nonstate"), m("x")
    if sets == "core":  # the short list for Full, where one LP takes a minute
        V = {"all_ns": nonstate, "cal": None, "fix_x": x}
        for t in (4, 8, 16):
            V[f"pre_{t}"] = m("nonstate", range(t))
            V[f"prex_{t}"] = m("x", range(t))
        return V
    V = {"all_ns": nonstate, "cal": None, "fix_x": x, "fix_x_lots": x | m("lots"), "fix_x_grid": x | m("grid"), "fix_lots": m("lots")}
    for g in ("fuel+", "mat+", "chip+"):
        V[f"fix_{g}"] = m(g)
    for t in PREFIX[task]:
        V[f"pre_{t}"] = m("nonstate", range(t))  # weeks 1..t as played (decisions and rules), the rest free
        V[f"prex_{t}"] = m("x", range(t))  # the flows of weeks 1..t as played, the rest free
    for t in OPEN:
        w = range(t)
        V[f"o{t}_lots"] = m("lots", w)  # our lot starts of the opening, everything else free
        V[f"o{t}_grid"] = m("grid", w)  # our burn, fab energy and shed of the opening
        V[f"o{t}_rules"] = m("rules", w)
        V[f"o{t}_fuel"] = m("fuel+", w)  # one good's flows of the opening as played
        V[f"o{t}_mat"] = m("mat+", w)
        V[f"o{t}_chip"] = m("chip+", w)
        V[f"o{t}_x_lots"] = m("x", w) | m("lots", w)
        V[f"o{t}_x_grid"] = m("x", w) | m("grid", w)
        V[f"win_0_{t}"] = nonstate & ~m("nonstate", w)  # only the opening free (gain over played)
        V[f"winx_0_{t}"] = x & ~m("x", w)  # only the opening's flows and every week's rules free (gain over fix_x)
    return V


def one(task, n, limit=900.0, sets="all"):
    start = time.perf_counter()
    d = C.load_traj(task, n)
    cal = C.Cal(task, n)
    zpi = np.concatenate([d["zpi"], np.zeros(cal.nz)])
    z, both = cal.calendar(d["zpi"])
    zpi[cal.nn :] = z.ravel()
    out = {"episode": n, "J_played": d["J_cents"] / 100, "both": both, "residual": cal.residual(zpi), "J": {}, "status": {}, "seconds": {}, "fixed": {}}
    out["J_model"] = float(cal.cost @ zpi + cal.offset)
    sols = {}
    for name, fixed in variants(cal, task, sets).items():
        J, status, seconds, x = cal.solve(zpi, z, fixed, band=C.BAND, limit=limit)
        if J is None:  # a second try from a cold start (a warm start can leave HiGHS at "Unknown")
            J, status2, seconds2, x = cal.solve(zpi, z, fixed, band=C.BAND, limit=limit, fresh=True)
            status, seconds = f"{status}; cold: {status2}", seconds + seconds2
        if J is None:  # a third, cold with a wider band
            J, status3, seconds3, x = cal.solve(zpi, z, fixed, band=1e-7, limit=limit, fresh=True)
            status, seconds = f"{status}; wider band: {status3}", seconds + seconds3
        out["J"][name], out["status"][name], out["seconds"][name] = J, status, round(seconds, 2)
        out["fixed"][name] = 0 if fixed is None else int(fixed.sum())
        if x is not None and name in KEEP:
            sols[name] = x[: cal.nn].reshape(cal.T, cal.nc)
    out["total_seconds"] = round(time.perf_counter() - start, 1)
    res = C.HERE / "res"
    res.mkdir(exist_ok=True)
    (res / f"{task}_{n}.json").write_text(json.dumps(out, indent=1))
    (res / f"{task}_{n}_sol.pkl").write_bytes(pickle.dumps({"sols": sols, "keys": cal.keys, "z": z}))
    return out


def run(task="small", only=None, n_jobs=2, redo=False, limit=900.0, sets="all"):
    from joblib import Parallel, delayed

    refs = C.references(task)
    ns = sorted(refs) if only is None else ([only] if isinstance(only, int) else [int(v) for v in only])
    todo = [n for n in ns if redo or not (C.HERE / "res" / f"{task}_{n}.json").exists()]
    start = time.perf_counter()
    results = Parallel(n_jobs=n_jobs)(delayed(one)(task, n, limit, sets) for n in todo) if todo else []
    for out in results:
        n = out["episode"]
        r = refs[n]
        naive, oracle = r["J_naive_cents"] / 100, r["J_oracle_cents"] / 100
        sc = lambda J: float("nan") if J is None else (naive - J) / (naive - oracle)  # noqa: E731
        bad = [k for k, s in out["status"].items() if s != "Optimal"]
        print(
            f"episode {n} level {r['stratum']}: played {sc(out['J_played']):.4f}, all fixed {sc(out['J'].get('all_ns')):.4f}, calendar {sc(out['J'].get('cal')):.4f};"
            f" residual {out['residual']:.1e}; not optimal at first: {bad}; {out['total_seconds']} s",
            flush=True,
        )
    print(f"{time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    fire.Fire({"run": run})

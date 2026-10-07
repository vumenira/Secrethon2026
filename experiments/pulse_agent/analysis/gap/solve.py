"""The calendar LP of every episode under each variant (a set of columns fixed to what we played); results in gap/res.

uv run python solve.py validate --task=small --episode=0
uv run python solve.py run --task=small --n_jobs=2
"""

import json
import pickle
import time

import fire
import numpy as np

import common as C


PREFIX = {"small": (4, 8, 13, 20, 26, 32, 38, 44, 48), "full": (8, 16, 26, 39, 52, 65, 78, 84, 90, 96, 100)}
WINDOWS = {"small": ((0, 8), (8, 20), (20, 32), (32, 44), (44, 52)), "full": ((0, 16), (16, 39), (39, 65), (65, 84), (84, 104))}
PREFIX_CORE = {"small": (4, 8, 13, 26, 39, 46), "full": (4, 8, 16, 26, 52, 78, 92)}
GROUPS = ("fuel", "mat", "chip", "strait", "lots", "other")


def variants(cal, task, sets="all"):
    cls = C.classes(cal.inst)
    m = lambda name, weeks=None: cal.mask(cls[name], weeks)  # noqa: E731
    nonstate, x = m("nonstate"), m("x")
    if sets == "check":  # every decision and rule as played, the stocks free: the model must return our cost
        return {"all_ns": nonstate}
    if sets == "core":  # the shorter list for Full, where one LP takes a minute: the same definitions as below
        V = {"all_ns": nonstate, "cal": None, "fix_x": x, "fix_x_grid": x | m("grid"), "fix_x_lots": x | m("lots"), "fix_lots": m("lots")}
        for g in ("fuel+", "mat+", "chip+", "lng+", "crude+", "nucfuel+"):
            V[f"fix_{g}"] = m(g)
            V[f"freex_{g}"] = x & ~m(g)
        for g in ("fuel", "mat", "chip", "strait"):
            V[f"fix_{g}"] = m(g)
            V[f"freex_{g}"] = x & ~m(g)
        for t in PREFIX_CORE[task]:
            V[f"pre_{t}"] = m("nonstate", range(t))
            V[f"prex_{t}"] = m("x", range(t))
        return V
    V = {"all": cal.mask(cls["all"]), "cal": None}
    if sets in ("all", "groups"):
        for g in GROUPS:
            V[f"fix_{g}"] = m(g)  # this group as played, the rest free
            V[f"free_{g}"] = nonstate & ~m(g)  # only this group free
        V["fix_x"] = x  # every flow as played: what the simulator's own rules leave free
        V["fix_rules"] = m("rules")  # every flow free, production, burn, serving and disposal as played
        for g in ("fuel", "mat", "chip", "strait", "fuel+", "mat+", "chip+", "lng+", "crude+", "nucfuel+", "fuel_grid", "fuel_up", "strait_fuel", "strait_rest"):
            if g not in cls:
                continue
            if f"fix_{g}" not in V:
                V[f"fix_{g}"] = m(g)
            V[f"freex_{g}"] = x & ~m(g)  # the other flows as played; this group and the simulator's rules free
        for g in ("lots", "grid", "xi", "serve", "disp"):
            V[f"fix_x_{g}"] = x | m(g)  # every flow and this piece of the rules as played
    if sets in ("all", "time"):
        for t in PREFIX[task]:
            V[f"pre_{t}"] = m("nonstate", range(t))  # weeks 1..t as played, the rest free
            V[f"prex_{t}"] = m("x", range(t))  # the flows of weeks 1..t as played, the rest free
        for a, b in WINDOWS[task]:
            V[f"win_{a}_{b}"] = nonstate & ~m("nonstate", range(a, b))  # only weeks a+1..b free
            V[f"winx_{a}_{b}"] = x & ~m("x", range(a, b))  # only the flows of weeks a+1..b (and the rules) free
        for g in ("fuel+", "mat+", "chip+"):  # one good's flows of weeks 1..t as played, the rest free
            for _a, b in WINDOWS[task][:-1]:
                V[f"preg_{g}_{b}"] = m(g, range(b))
    return V


KEEP = ("cal", "fix_x", "freex_fuel+", "freex_mat+", "freex_chip+")


def one(task, n, sets="all", keep=("cal",), limit=900.0):
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
        # "all" fixes the stocks too, so every row has all its columns fixed: the band is the solver's own tolerance
        band = 1e-7 if name == "all" else C.BAND
        J, status, seconds, x = cal.solve(zpi, z, fixed, band=band, limit=limit)
        if J is None:  # a second try from a cold start with the same band (a warm start can leave HiGHS at "Unknown")
            J, status2, seconds2, x = cal.solve(zpi, z, fixed, band=band, limit=limit, fresh=True)
            status, seconds = f"{status}; cold: {status2}", seconds + seconds2
        if J is None and band < 1e-7:  # a third, cold with a wider band
            J, status3, seconds3, x = cal.solve(zpi, z, fixed, band=1e-7, limit=limit, fresh=True)
            status, seconds = f"{status}; wider band: {status3}", seconds + seconds3
        out["J"][name], out["status"][name], out["seconds"][name] = J, status, round(seconds, 2)
        out["fixed"][name] = 0 if fixed is None else int(fixed.sum())
        if x is not None and name in keep:
            sols[name] = x[: cal.nn].reshape(cal.T, cal.nc)
    out["total_seconds"] = round(time.perf_counter() - start, 1)
    res = C.HERE / "res"
    res.mkdir(exist_ok=True)
    (res / f"{task}_{n}_{sets}.json").write_text(json.dumps(out, indent=1))
    if sols:
        (res / f"{task}_{n}_sol.pkl").write_bytes(pickle.dumps({"sols": sols, "keys": cal.keys, "z": z}))
    return out


def validate(task="small", episode=0):
    out = one(task, int(episode), sets="groups")
    refs = C.references(task)[int(episode)]
    naive, oracle = refs["J_naive_cents"] / 100, refs["J_oracle_cents"] / 100
    sc = lambda J: float("nan") if J is None else (naive - J) / (naive - oracle)  # noqa: E731
    print(f"episode {episode}: played {out['J_played'] / 1e9:.6f} bn, the model's cost of our vector {out['J_model'] / 1e9:.6f} bn; worst row residual {out['residual']:.2e}; grid-weeks that shed and powered fabs {out['both']}")
    print(f"score played {sc(out['J_played']):.4f}")
    for name, J in out["J"].items():
        print(f"  {name:18s} {sc(J):.6f}  fixed {out['fixed'][name]:7d}  {out['status'][name]}  {out['seconds'][name]} s")
    print(f"{out['total_seconds']} s")


def run(task="small", sets="all", only=None, n_jobs=2, redo=False, limit=900.0):
    from joblib import Parallel, delayed

    refs = C.references(task)
    ns = sorted(refs) if only is None else ([only] if isinstance(only, int) else [int(v) for v in only])
    todo = [n for n in ns if redo or not (C.HERE / "res" / f"{task}_{n}_{sets}.json").exists()]
    start = time.perf_counter()
    results = Parallel(n_jobs=n_jobs)(delayed(one)(task, n, sets, KEEP, limit) for n in todo) if todo else []
    for out in results:
        n = out["episode"]
        r = refs[n]
        naive, oracle = r["J_naive_cents"] / 100, r["J_oracle_cents"] / 100
        sc = lambda J: float("nan") if J is None else (naive - J) / (naive - oracle)  # noqa: E731
        bad = [k for k, s in out["status"].items() if s != "Optimal"]
        print(
            f"episode {n} level {r['stratum']}: played {sc(out['J_played']):.4f}, all fixed {sc(out['J'].get('all')):.4f}, calendar {sc(out['J'].get('cal')):.4f};"
            f" residual {out['residual']:.1e}; not optimal at first: {bad}; {out['total_seconds']} s",
            flush=True,
        )
    print(f"{time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    fire.Fire({"validate": validate, "run": run})

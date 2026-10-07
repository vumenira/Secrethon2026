"""The realizable clairvoyant for agents/mpc_pulse_rec_v6: the calendar model with z free (a MILP, time-limited).

Started from the best of: our own calendar with the best plan under it (an LP), and the earlier analyst's incumbent
(``../milp/<task>_<n>.pkl``, found from version 2's calendar; its z is re-solved as an LP here).

uv run python real.py prep --task=small                 # every episode: our calendar's optimum and the old incumbent
uv run python real.py run --task=small --only=0,2 --limit=300
"""

import json
import pickle
import time

import fire
import numpy as np

import common as C


OLD = C.HERE.parent / "milp"
OUT = C.HERE / "milp"


def setup(task, n):
    d = C.load_traj(task, n)
    cal = C.Cal(task, n)
    z, _both = cal.calendar(d["zpi"])
    return d, cal, z


def old_z(task, n, cal):
    p = OLD / f"{task}_{n}.pkl"
    if not p.exists():
        return None
    m = pickle.loads(p.read_bytes())
    z = np.asarray(m["z"], dtype=float)
    if z.shape != (len(cal.grids), cal.T) or [g[0] for g in m["grids"]] != [g[0] for g in cal.grids]:
        return None
    return np.round(z)


def prep_one(task, n):
    d, cal, z = setup(task, n)
    out = {"episode": n, "J_played": d["J_cents"] / 100}
    J, status, seconds, _x = cal.lp(*cal.bounds(z=z))
    out["J_cal"], out["cal_status"] = J, status
    zo = old_z(task, n, cal)
    if zo is not None:
        J, status, seconds, _x = cal.lp(*cal.bounds(z=zo))
        out["J_old"], out["old_status"] = J, status
        out["old_differs"] = int((zo != z).sum())
    return out


def prep(task="small", n_jobs=2):
    from joblib import Parallel, delayed

    refs = C.references(task)
    results = Parallel(n_jobs=n_jobs)(delayed(prep_one)(task, n) for n in sorted(refs))
    (C.HERE / f"real_prep_{task}.json").write_text(json.dumps({r["episode"]: r for r in results}, indent=1))
    ns = [r["episode"] for r in results if r.get("J_old") is not None and r.get("J_cal") is not None]
    by = {r["episode"]: r for r in results}
    for key, label in (("J_played", "played"), ("J_cal", "our calendar, the best plan"), ("J_old", "the earlier incumbent's calendar, re-solved"), ("J_best", "the better of the two")):
        J = {n: (min(by[n]["J_old"], by[n]["J_cal"]) if key == "J_best" else by[n][key]) * 100 for n in ns}
        p, lev = C.pooled(refs, J)
        print(f"{label:46s} {p:.4f}  {' '.join(f'{s}: {v:.4f}' for s, v in lev.items())}  ({len(ns)} episodes)")
    for n in sorted(by):
        r, ref = by[n], refs[n]
        D = (ref["J_naive_cents"] - ref["J_oracle_cents"]) / 100
        sc = lambda J: float("nan") if J is None else (ref["J_naive_cents"] / 100 - J) / D  # noqa: E731
        print(f"  episode {n:2d} level {ref['stratum']}: played {sc(r['J_played']):.4f}  calendar {sc(r.get('J_cal')):.4f}  old incumbent {sc(r.get('J_old')):.4f}  ({r.get('old_differs')} grid-weeks differ)")


def one(task, n, limit, gap):
    start = time.perf_counter()
    d, cal, z = setup(task, n)
    out = {"episode": n, "J_played": d["J_cents"] / 100}
    Jc, status, _s, xc = cal.lp(*cal.bounds(z=z))
    out["J_cal"], out["cal_status"] = Jc, status
    best, Jb, src = xc, Jc, "ours"
    zo = old_z(task, n, cal)
    if zo is not None:
        Jo, status, _s, xo = cal.lp(*cal.bounds(z=zo))
        out["J_old"] = Jo
        if Jo is not None and (Jb is None or Jo < Jb):
            best, Jb, src = xo, Jo, "old"
    prev = OUT / f"{task}_{n}.pkl"
    if prev.exists():  # a run of this script before: continue from its incumbent
        m = pickle.loads(prev.read_bytes())
        Jp, status, _s, xp = cal.lp(*cal.bounds(z=np.round(m["z"])))
        if Jp is not None and (Jb is None or Jp < Jb):
            best, Jb, src = xp, Jp, "previous"
    out["start"], out["J_start"] = src, Jb
    res = cal.milp(*cal.bounds(), start=best, limit=limit, gap=gap)
    x = res.pop("x")
    out.update({"status": res["status"], "milp_seconds": res["seconds"], "bound": res["bound"], "J_milp": res["J"]})
    if x is not None:
        zm = np.round(x[cal.nn :].reshape(len(cal.grids), cal.T))
        Jm, status, _s, xm = cal.lp(*cal.bounds(z=zm))  # the best plan under the incumbent's calendar
        if Jm is None or (Jb is not None and Jb < Jm):
            Jm, xm, zm = Jb, best, np.round(best[cal.nn :].reshape(len(cal.grids), cal.T))
        out["J_real"] = Jm
        OUT.mkdir(exist_ok=True)
        prev.write_bytes(pickle.dumps({"x": xm[: cal.nn].reshape(cal.T, cal.nc), "z": zm, "z_ours": z, "x_cal": xc[: cal.nn].reshape(cal.T, cal.nc), "J": Jm, "J_cal": Jc, "bound": res["bound"], "keys": cal.keys}))
    out["seconds"] = round(time.perf_counter() - start, 1)
    return out


def run(task="small", only=None, limit=300.0, gap=1e-4, n_jobs=2, tag=""):
    from joblib import Parallel, delayed

    refs = C.references(task)
    ns = sorted(refs) if only is None else ([only] if isinstance(only, int) else [int(v) for v in only])
    start = time.perf_counter()
    path = C.HERE / f"real_{task}{tag}.json"
    for i in range(0, len(ns), n_jobs):  # pairs, so that results are written as they come
        results = Parallel(n_jobs=n_jobs)(delayed(one)(task, n, limit, gap) for n in ns[i : i + n_jobs])
        done = json.loads(path.read_text()) if path.exists() else {}
        for r in results:
            n = r["episode"]
            ref = refs[n]
            D = (ref["J_naive_cents"] - ref["J_oracle_cents"]) / 100
            sc = lambda J: None if J is None else round((ref["J_naive_cents"] / 100 - J) / D, 4)  # noqa: E731
            r["limit"] = limit
            r["score"] = {k: sc(r.get(f"J_{k}")) for k in ("played", "cal", "old", "start", "real")} | {"bound": sc(r.get("bound"))}
            done[str(n)] = r
            print(f"episode {n} level {ref['stratum']}: {r['score']} started from {r['start']}; {r['status']}, {r['milp_seconds']} s", flush=True)
        path.write_text(json.dumps(done, indent=1))
    print(f"{time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    fire.Fire({"prep": prep, "run": run})

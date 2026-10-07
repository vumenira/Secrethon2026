"""The foresight check: the fab calendar an optimum would choose at week t if the network seen then stayed as it is.

The world is the true one before week t and week t's observation kept from then on; our play is fixed before week t
and everything from week t on is free, the calendar included (a MILP, time-limited). Its calendar and its fuel use
from week t are set beside ours and the realizable plan's (left/diff), and its first weeks are valued in the true
world.

uv run python persist.py run --task=small --cases="41:1,12:1" --limit=180
"""

import json
import pickle
import time

import fire
import numpy as np

import common as C
import diff as Dm
import view as V
import worlds as W


OUT = C.HERE / "persist"


def one(task, n, t, limit, horizon=12):
    start = time.perf_counter()
    refs = C.references(task)[n]
    D = (refs["J_naive_cents"] - refs["J_oracle_cents"]) / 100
    d = C.load_traj(task, n)
    true = C.Cal(task, n)
    T, G = true.T, len(true.grids)
    z0, _both = true.calendar(d["zpi"])
    zpi = np.concatenate([d["zpi"], z0.ravel()])
    p = Dm.OUT / f"{task}_{n}.pkl"
    plan = pickle.loads(p.read_bytes()) if p.exists() else None
    zs = plan["z"] if plan is not None else None
    cp = C.Cal(task, n, marks=W.persist_marks(true.true_marks, t))
    cls = C.classes(cp.inst)
    fixed = cp.mask(cls["nonstate"], range(t - 1)) if t > 1 else None
    zfree = z0.astype(float).copy()
    zfree[:, t - 1 :] = np.nan
    lb, ub = cp.bounds(z=zfree, fixed=fixed, values=zpi)
    out = {"episode": n, "week": t, "D": D}
    # a start: our own calendar, or the realizable plan's, if the persisting world allows it
    best, Jb, src = None, None, None
    for label, zc in (("ours", z0), ("plan", zs)):
        if zc is None:
            continue
        zc2 = zc.astype(float).copy()
        zc2[:, : t - 1] = z0[:, : t - 1]
        J, _st, _s, x = cp.lp(*cp.bounds(z=zc2, fixed=fixed, values=zpi))
        out[f"J_{label}_in_world"] = J
        if J is not None and (Jb is None or J < Jb):
            best, Jb, src = x, J, label
    res = cp.milp(lb, ub, start=best, limit=limit, gap=1e-4)
    x = res.pop("x")
    out.update({"status": res["status"], "milp_seconds": res["seconds"], "J_world": res["J"], "bound_world": res["bound"], "start": src})
    if x is None:
        return out
    zp = np.round(x[cp.nn :].reshape(G, T))
    xp = x[: cp.nn].reshape(T, cp.nc)
    v = V.View(cp)
    xo = d["zpi"].reshape(T, true.nc)
    hz = slice(t - 1, min(T, t - 1 + horizon))
    rows = []
    for gi in range(G):
        go = true.grids[gi][0]
        so = Dm.grid_series(v, cp, xo, gi)
        sp_ = Dm.grid_series(v, cp, xp, gi)
        ss = Dm.grid_series(v, cp, plan["x"], gi) if plan is not None else None
        r = {"grid": v.node[cp.inst.grids[go]], "z_ours": "".join("#" if q else "." for q in z0[gi, hz]), "z_world": "".join("#" if q else "." for q in zp[gi, hz])}
        if zs is not None:
            r["z_plan"] = "".join("#" if q else "." for q in zs[gi, hz])
            dif = zs[gi, hz] != z0[gi, hz]
            r["differ"], r["world_as_plan"] = int(dif.sum()), int((zp[gi, hz][dif] == zs[gi, hz][dif]).sum())
        for label, s in (("ours", so), ("world", sp_), ("plan", ss)):
            if s is None:
                continue
            r[label] = {"E": float(s["E"][hz].sum()), "shed": float(s["shed"][hz].sum()), "E_all": float(s["E"][t - 1 :].sum()), "burn": {k: float(f["G"][hz].sum()) for k, f in s["fuel"].items() if k != "nucfuel"}, "burn4": {k: float(f["G"][t - 1 : t + 3].sum()) for k, f in s["fuel"].items() if k != "nucfuel"}}
        rows.append(r)
    out["grids"] = rows
    # the true world: the persisting world's calendar for the first weeks, then ours or the realizable plan's
    for H in (4, 8):
        for label, zc in (("ours", z0), ("plan", zs)):
            if zc is None:
                continue
            zt = zc.astype(float).copy()
            zt[:, : t - 1] = z0[:, : t - 1]
            zt[:, t - 1 : t - 1 + H] = zp[:, t - 1 : t - 1 + H]
            J, st, _s, _x = true.lp(*true.bounds(z=zt, fixed=(true.mask(cls["nonstate"], range(t - 1)) if t > 1 else None), values=zpi))
            out[f"J_true_{H}_{label}"] = J
    for label, zc in (("ours", z0), ("plan", zs)):
        if zc is None:
            continue
        zt = zc.astype(float).copy()
        zt[:, : t - 1] = z0[:, : t - 1]
        J, st, _s, _x = true.lp(*true.bounds(z=zt, fixed=(true.mask(cls["nonstate"], range(t - 1)) if t > 1 else None), values=zpi))
        out[f"J_true_{label}"] = J
    out["seconds"] = round(time.perf_counter() - start, 1)
    OUT.mkdir(exist_ok=True)
    (OUT / f"{task}_{n}_{t}.json").write_text(json.dumps(out, indent=1))
    (OUT / f"{task}_{n}_{t}.pkl").write_bytes(pickle.dumps({"x": xp[:20].astype(np.float32), "z": zp}))
    return out


def show(out, refs):
    n, D = out["episode"], out["D"]
    naive = refs[n]["J_naive_cents"] / 100
    sc = lambda J: float("nan") if J is None else (naive - J) / D  # noqa: E731
    print(f"episode {n} from week {out['week']}: {out['status']} in {out.get('milp_seconds')} s, started from {out.get('start')}")
    print(f"  in the true world, our play before week {out['week']}: our calendar {sc(out.get('J_true_ours')):.4f}; the realizable plan's {sc(out.get('J_true_plan')):.4f};"
          f" the persisting world's first 4 weeks then ours {sc(out.get('J_true_4_ours')):.4f}, then the plan's {sc(out.get('J_true_4_plan')):.4f}; first 8 weeks then ours {sc(out.get('J_true_8_ours')):.4f}, then the plan's {sc(out.get('J_true_8_plan')):.4f}")
    for r in out.get("grids", []):
        print(f"  {r['grid']:8s} ours {r['z_ours']}  world {r['z_world']}  plan {r.get('z_plan')}  (weeks the plan differs from ours {r.get('differ')}, the world agrees with the plan in {r.get('world_as_plan')})")
        for label in ("ours", "world", "plan"):
            if label in r:
                s = r[label]
                print(f"      {label:6s} fab energy in these weeks {s['E']:7.0f} (to the end {s['E_all']:8.0f}); shed {s['shed']:8.0f}; burnt {' '.join(f'{k} {x:8.0f}' for k, x in s['burn'].items())}; in the first 4 weeks {' '.join(f'{k} {x:8.0f}' for k, x in s['burn4'].items())}")


def run(task="small", cases="41:1", limit=180.0, n_jobs=2, horizon=12):
    from joblib import Parallel, delayed

    refs = C.references(task)
    items = cases if isinstance(cases, (list, tuple)) else str(cases).split(",")
    todo = [(int(a), int(b)) for a, b in (str(c).split("w") for c in items)]  # "41w1": episode 41 from week 1
    start = time.perf_counter()
    for i in range(0, len(todo), n_jobs):
        for out in Parallel(n_jobs=n_jobs)(delayed(one)(task, n, t, limit, horizon) for n, t in todo[i : i + n_jobs]):
            show(out, refs)
    print(f"{time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    fire.Fire({"run": run})

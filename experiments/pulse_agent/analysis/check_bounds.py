"""How firm the ideal-holding bound is, and what the clairvoyant plan's own fuel would allow.

Per grid and episode, capped at the plan's fab energy: the bound as reported (integer on/off weeks, 4 s), with a long
time limit, relaxed (a week may be partly on: an upper bound of the bound), and on the plan's fuel (each fuel as the
plan burns it, week by week) instead of ours.
"""

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import fire
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import analyze_v2 as A  # noqa: E402


def relaxed(gw, cap, limit=0.0):
    """The bound with every fuel held; ``limit`` 0: weeks may be partly on (LP), else integer with that time limit."""
    T, Ecap = gw["T"], gw["Ecap"]
    best = min(float(Ecap.sum()), float(cap))
    if best <= A.EPS:
        return best, "trivial"
    rows, hi = [], []
    for t in range(T):
        r = np.zeros(2 * T)
        r[t], r[T + t] = -Ecap[t], 1.0
        rows.append(r)
        hi.append(0.0)
    for fu in gw["fuels"].values():
        usable = fu["Ig"][0] + fu["It"][0] + np.cumsum(fu["a_direct"]) + np.concatenate([[0.0], np.cumsum(fu["a_term"])[:-1]])
        for t in range(T):
            r = np.zeros(2 * T)
            r[: t + 1] = fu["base_burn"][: t + 1]
            r[T : T + t + 1] = fu["per_E"]
            rows.append(r)
            hi.append(max(usable[t], 0.0) * (1 + 1e-9) + 1e-6)
    r = np.zeros(2 * T)
    r[T:] = 1.0
    rows.append(r)
    hi.append(float(cap))
    c = np.concatenate([np.zeros(T), -np.ones(T)])
    res = milp(
        c,
        constraints=LinearConstraint(np.array(rows), -np.inf, np.array(hi)),
        integrality=np.concatenate([np.ones(T) if limit else np.zeros(T), np.zeros(T)]),
        bounds=Bounds(np.zeros(2 * T), np.concatenate([(Ecap > A.EPS).astype(float), Ecap])),
        options={"time_limit": limit or 20.0, "mip_rel_gap": 1e-4},
    )
    return (float(res.x[T:].sum()) if res.x is not None else float("nan")), res.message[:40]


def one(task, tag, n):
    from shockbench_flow.hosting.tasks import task_generator

    inst, _ = task_generator(task)
    st = A.static(inst)
    d = pickle.loads((HERE / "trace" / f"{task}_{tag}" / f"{n}.pkl").read_bytes())
    out = {}
    for go in st:
        if not st[go]["fabs"]:
            continue
        gw = A.grid_weeks(inst, st, d, go)
        target = float(gw["Eplan"].sum())
        quick = A.bound(gw, set(gw["fuels"]), cap=target)
        lp, _ = relaxed(gw, target)
        long_, msg = relaxed(gw, target, limit=25.0) if lp - quick > 0.002 * max(target, 1.0) else (quick, "not needed")
        # the plan's fuel: each fuel arrives as the plan burns it (nothing before, nothing held over)
        gp = dict(gw)
        gp["fuels"] = {}
        for k, fu in gw["fuels"].items():
            f2 = dict(fu)
            f2["a_direct"], f2["a_term"] = fu["burn_plan"].copy(), np.zeros(gw["T"])
            f2["Ig"], f2["It"] = np.zeros(gw["T"] + 1), np.zeros(gw["T"] + 1)
            gp["fuels"][k] = f2
        plan_lp, _ = relaxed(gp, target)
        plan_int = A.bound(gp, set(gp["fuels"]), cap=target)
        out[st[go]["id"]] = {"plan": target, "quick": quick, "lp": lp, "long": long_, "msg": msg, "planfuel_lp": plan_lp, "planfuel": plan_int, "got": min(float(gw["E"].sum()), target)}
    return n, out


def main(task="full", tag="v2", n_jobs=2):
    from joblib import Parallel, delayed

    ns = sorted(int(p.stem) for p in (HERE / "trace" / f"{task}_{tag}").glob("*.pkl"))
    res = dict(Parallel(n_jobs=n_jobs)(delayed(one)(task, tag, n) for n in ns))
    tot = defaultdict(lambda: defaultdict(float))
    slow = 0
    for n, grids in res.items():
        for g, v in grids.items():
            for c in ("plan", "quick", "lp", "long", "planfuel_lp", "planfuel", "got"):
                tot[g][c] += v[c]
                tot["pooled"][c] += v[c]
            slow += v["msg"] not in ("not needed", "trivial") and "Optimal" not in v["msg"]
    print(f"{task}: fab energy, GWh, capped at the plan per grid and episode ({len(ns)} episodes)")
    print(f"{'grid':9s} {'plan':>9s} {'got':>9s} | ours: {'as reported':>11s} {'25 s':>9s} {'relaxed':>9s} | the plan's fuel: {'integer':>9s} {'relaxed':>9s}")
    for g, v in tot.items():
        print(f"{g:9s} {v['plan']:9.0f} {v['got']:9.0f} |       {v['quick']:11.0f} {v['long']:9.0f} {v['lp']:9.0f} |                  {v['planfuel']:9.0f} {v['planfuel_lp']:9.0f}")
    print(f"long solves that did not finish: {slow}")
    (HERE / f"check_bounds_{task}.json").write_text(json.dumps({g: dict(v) for g, v in tot.items()}))


if __name__ == "__main__":
    fire.Fire(main)

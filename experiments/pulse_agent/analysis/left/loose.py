"""How far the calendar model is from the simulator in weeks that shed base load.

In a week a grid sheds, the simulator runs every fuel segment at all it has (the load factor is 1): gas burns
min(share x power x ration, stock after arrivals) with the ration read off last week's closing stock, another fuel
min(share x power, stock after arrivals). The calendar model only bounds the burn from above, so a plan may keep fuel
at a grid through a shed week and burn it in a later whole week. This counts that under-burn in our play (must be 0),
in the best plan under our calendar and in the realizable plan, and the whole weeks of each plan that follow a shed
week in which the grid's gas stock stayed at or above the rationing threshold (which the simulator cannot do: a
gas-short week burns the grid's gas).

uv run python loose.py --task=small
"""

import json
import pickle

import fire
import numpy as np

import common as C
import diff as Dm
import view as V


def measure(cal, v, x, gi):
    inst, marks = cal.inst, cal.marks
    go, fabs, _S = cal.grids[gi]
    g = inst.grids[go]
    ga = inst.nodes[g].grid
    T = cal.T
    Gbar = np.asarray(marks.G_bar)[:, go]
    ybar = np.asarray(marks.y_bar)[:, go]
    shed = v.col(x, ("ysh", go))
    E = sum(v.col(x, ("E", fo)) for fo in fabs)
    off = shed > 1e-7 * ybar
    out = {"off_weeks": int(off.sum()), "on_weeks": int(((~off) & (E > 1e-6 * np.maximum(cal.emax[gi], 1.0))).sum()), "under": {}, "burn": {}}
    psi = inst.params.psi
    for k in ga.fuels:
        name = v.K[k]
        if name == "nucfuel":
            continue
        s = inst.slot_index[(g, k)]
        I = v.col(x, ("I", s))
        Gk = v.col(x, ("G", go, k))
        i0 = float(Dm_initial(inst)[s])
        Iprev = np.concatenate([[i0], I[:-1]])
        hand = I + Gk  # stock after arrivals, before the burn (disposal aside)
        cap = ga.shares[k] * Gbar
        if k == ga.rationed:
            thr = psi * ga.ibar[k]
            ration = np.minimum(1.0, Iprev / thr) if thr > 0 else np.ones(T)
            avail = np.minimum(cap * ration, hand)
            # whole weeks right after a shed week that left the grid's gas at or above the threshold
            kept = int(sum(1 for t in range(1, T) if (not off[t]) and E[t] > 1e-6 * max(cal.emax[gi][t], 1.0) and off[t - 1] and I[t - 1] >= thr * (1 - 1e-6)))
            out["kept_charge"] = kept
            # whole weeks right after a shed week at all (the simulator needs the stock at the threshold after that week)
            out["on_after_off"] = int(sum(1 for t in range(1, T) if (not off[t]) and E[t] > 1e-6 * max(cal.emax[gi][t], 1.0) and off[t - 1]))
        else:
            avail = np.minimum(cap, hand)
        under = np.where(off, np.maximum(avail - Gk, 0.0), 0.0)
        burn = ga.shares[k] * ga.deliverable
        out["under"][name] = float(under.sum() / burn)
        out["burn"][name] = float(Gk.sum() / burn)
    return out


_I0 = {}


def Dm_initial(inst):
    from shockbench_flow.dynamics.sim import initial_stock

    key = id(inst)
    if key not in _I0:
        _I0.clear()
        _I0[key] = initial_stock(inst)
    return _I0[key]


def main(task="small"):
    refs = C.references(task)
    acc = {}
    used = 0
    for n in sorted(refs):
        p = Dm.OUT / f"{task}_{n}.pkl"
        if not p.exists():
            continue
        plan = pickle.loads(p.read_bytes())
        cal = C.Cal(task, n)
        d = C.load_traj(task, n)
        v = V.View(cal)
        xs = {"ours": d["zpi"].reshape(cal.T, cal.nc), "cal": plan["x_cal"], "plan": plan["x"]}
        used += 1
        for gi, (go, _f, _S) in enumerate(cal.grids):
            gname = v.node[cal.inst.grids[go]]
            for label, x in xs.items():
                m = measure(cal, v, x, gi)
                a = acc.setdefault((gname, label), {"off": 0, "on": 0, "kept": 0, "after": 0, "lng": 0.0, "crude": 0.0, "lng_burn": 0.0, "crude_burn": 0.0})
                a["off"] += m["off_weeks"]
                a["on"] += m["on_weeks"]
                a["kept"] += m.get("kept_charge", 0)
                a["after"] += m.get("on_after_off", 0)
                for k in ("lng", "crude"):
                    a[k] += m["under"].get(k, 0.0)
                    a[f"{k}_burn"] += m["burn"].get(k, 0.0)
    (C.HERE / f"loose_{task}.json").write_text(json.dumps({f"{g}|{lab}": a for (g, lab), a in acc.items()}, indent=1))
    print(f"{task}, {used} episodes; per episode: weeks that shed, whole weeks with fab energy, whole weeks right after a shed week (and of those, with the grid's gas left at the threshold through that shed week), fuel not burnt in shed weeks though it was at the grid (weeks of a segment's full burn), fuel burnt")
    for (g, lab), a in acc.items():
        print(f"  {g} {lab:5s} shed weeks {a['off'] / used:5.1f}; whole weeks {a['on'] / used:5.1f}; whole right after a shed week {a['after'] / used:5.2f} (gas kept at the threshold through it: {a['kept'] / used:5.2f}); gas held back in shed weeks {a['lng'] / used:6.2f} of {a['lng_burn'] / used:5.1f} burnt; crude {a['crude'] / used:6.2f} of {a['crude_burn'] / used:5.1f}")


if __name__ == "__main__":
    fire.Fire(main)

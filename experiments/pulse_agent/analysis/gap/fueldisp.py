"""Fuel disposed above a grid's storage in our play: was it delivered that week over the zero-lead edge from the
terminal, and would the terminal have had room to keep it?

uv run python fueldisp.py --task=small
"""

from collections import defaultdict

import fire
import numpy as np

import common as C
import describe as Dsc


def main(task="small", show=12):
    refs = C.references(task)
    ns = sorted(refs)
    w = Dsc.weights(task, ns)
    acc = defaultdict(float)
    rows = []
    for n in ns:
        cal = C.Cal(task, n)
        inst, marks, T = cal.inst, cal.marks, cal.T
        d = C.load_traj(task, n)
        a = d["zpi"].reshape(T, cal.nc)
        v = Dsc.View(cal)
        chk, supply = set(inst.chokepoints), set(inst.supply_nodes)
        for go, g in enumerate(inst.grids):
            ga = inst.nodes[g].grid
            shed = v.col(a, ("ysh", go))
            for k in ga.fuels:
                s = inst.slot_index[(g, k)]
                O = v.col(a, ("O", s))
                if O.sum() <= 1e-6:
                    continue
                storage = inst.stock_slots[s].storage
                I = v.col(a, ("I", s))
                G = v.col(a, ("G", go, k))
                feed = [(e, inst.slot_index[(inst.edges[e].tail, k)]) for e in inst.in_edges[g] if k in inst.edges[e].K and inst.edges[e].tau == 0 and inst.edges[e].tail not in chk and inst.edges[e].tail not in supply and (inst.edges[e].tail, k) in inst.slot_index]
                other = [e for e in inst.in_edges[g] if k in inst.edges[e].K and e not in [f[0] for f in feed]]
                for t in np.flatnonzero(O > 1e-6):
                    deliv = sum(v.flow(a, e, k)[t] for e, _ts in feed)
                    room_t = sum(max(inst.stock_slots[ts].storage - v.col(a, ("I", ts))[t], 0.0) for _e, ts in feed)
                    direct = sum(v.flow(a, e, k)[t - inst.edges[e].tau] for e in other if t - inst.edges[e].tau >= 0)
                    kept = min(O[t], deliv, room_t)  # what holding the delivery back at the terminal would have kept
                    later_shed = shed[t:].sum()
                    key = v.K[k]
                    acc[f"{key} disposed at grids, GWh"] += O[t]
                    acc[f"{key} of it delivered that week from the terminal, GWh"] += min(O[t], deliv)
                    acc[f"{key} of it the terminal had room to keep, GWh"] += kept
                    acc[f"{key} disposed at grids at VOLL, pooled"] += w[n] * ga.voll * min(O[t], later_shed)
                    acc[f"{key} the terminal could have kept, at VOLL, pooled"] += w[n] * ga.voll * min(kept, later_shed)
                    rows.append((O[t], n, v.node[g], key, t + 1, I[t - 1] if t else float("nan"), storage, deliv, direct, G[t], ga.shares[k] * float(np.asarray(marks.G_bar)[t, go]), room_t, shed[t]))
    print(f"{task}, {len(ns)} episodes:")
    for key, x in acc.items():
        print(f"  {key:60s} {x:12,.4f}" if "pooled" in key else f"  {key:60s} {x:12,.0f}")
    rows.sort(reverse=True)
    print(f"the {show} largest (GWh): disposed, episode, grid, fuel, week, stock the week before / storage, delivered from the terminal, arrived otherwise, burnt / the segment's cap, room at the terminal, shed that week")
    for r in rows[:show]:
        print(f"  {r[0]:7.0f}  ep {r[1]:2d} {r[2]} {r[3]:6s} week {r[4]:2d}  stock {r[5]:7.0f}/{r[6]:.0f}  delivered {r[7]:7.0f}  other {r[8]:6.0f}  burnt {r[9]:6.0f}/{r[10]:.0f}  terminal room {r[11]:7.0f}  shed {r[12]:6.0f}")


if __name__ == "__main__":
    fire.Fire(main)

"""What a cap on fuel that can never be burnt is worth, from the records alone: no foresight, nothing else changes.

For a fuel a grid receives straight from a source (nuclear fuel): each week, the grid's stock plus what our play had
dispatched and not yet delivered is compared with the most the rest of the episode can burn (the fuel's share of the
grid's nominal power, times the weeks left). A dispatch that takes the total above that can never be burnt, whatever
happens (it arrives after everything dispatched before it, which already covers every week left at the full rate), so
cancelling it changes nothing but costs: its freight, war risk and tariff, the holding cost from its arrival to the
end, less the salvage credit at the end (at the grid or in transit, the same price). Shipments already under way at
week 1 are not counted as on their way, so a little less is cancelled than could be.

uv run python nuc.py --task=small
"""

from collections import defaultdict

import fire
import numpy as np

import common as C
import describe as Dsc


def main(task="small", fuel="nucfuel"):
    from shockbench_flow.dynamics.sim import initial_stock

    refs = C.references(task)
    ns = sorted(refs)
    w = Dsc.weights(task, ns)
    acc = defaultdict(float)
    by_grid = defaultdict(float)
    for n in ns:
        cal = C.Cal(task, n)
        inst, marks, T = cal.inst, cal.marks, cal.T
        d = C.load_traj(task, n)
        a = d["zpi"].reshape(T, cal.nc)
        v = Dsc.View(cal)
        k = next(i for i, c in enumerate(inst.commodities) if c.id == fuel)
        cost_c, cost_wr, tar = np.asarray(marks.c), np.asarray(marks.c_wr), np.asarray(marks.tariff)
        vk = inst.commodities[k].v
        i0 = np.asarray(initial_stock(inst), dtype=float)
        for go, g in enumerate(inst.grids):
            ga = inst.nodes[g].grid
            if k not in ga.fuels:
                continue
            s = inst.slot_index[(g, k)]
            slot = inst.stock_slots[s]
            burn = ga.shares[k] * ga.deliverable
            stock = v.col(a, ("I", s))
            edges = [e for e in inst.in_edges[g] if k in inst.edges[e].K]
            shipped = {e: v.flow(a, e, k) for e in edges}
            cancelled = 0.0  # cancelled so far: none of it would ever have been burnt, so stock plus transit is lower by it
            for t in range(T):
                prev = stock[t - 1] if t else float(i0[s])
                on_way = sum(shipped[e][max(t - inst.edges[e].tau, 0) : t].sum() for e in edges)  # dispatched before, not yet in
                room = max(burn * (T - t) - (prev + on_way - cancelled), 0.0)  # what weeks t..T could still burn of a new dispatch
                for e in edges:
                    q = shipped[e][t]
                    if q <= 0:
                        continue
                    cut = max(q - room, 0.0)
                    room = max(room - q, 0.0)
                    if cut <= 0:
                        continue
                    cancelled += cut
                    arrive = t + inst.edges[e].tau
                    acc["shipping (freight, war risk, tariff)"] += w[n] * cut * (cost_c[t, e] + cost_wr[t, e, k] + tar[t, e, k] * vk)
                    acc["holding to the end"] += w[n] * cut * slot.holding * max(T - arrive, 0)
                    acc["salvage credit lost"] += w[n] * cut * slot.salvage
                    by_grid[f"cancelled, {v.node[g]}"] += cut
            by_grid["shipped in the play"] += sum(x.sum() for x in shipped.values())
            by_grid["left at grids at the end in the play"] += stock[-1]
    net = acc["shipping (freight, war risk, tariff)"] + acc["holding to the end"] - acc["salvage credit lost"]
    print(f"{task}, {len(ns)} episodes: cancelling every {fuel} dispatch that the rest of the episode could not burn")
    for key, x in acc.items():
        print(f"  {key:44s} {x:+.4f}  (pooled score units)")
    print(f"  {'net saving':44s} {net:+.4f}  (pooled score units)")
    for key, x in by_grid.items():
        print(f"  {key:44s} {x:14,.0f} GWh")


if __name__ == "__main__":
    fire.Fire(main)

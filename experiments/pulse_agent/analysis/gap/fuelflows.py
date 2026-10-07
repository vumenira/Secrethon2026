"""Fuel by edge, terminal and grid, summed over episodes: played | fuel flows freed alone | calendar optimum.

uv run python fuelflows.py --task=small
"""

from collections import defaultdict

import fire
import numpy as np

import common as C
import describe as Dsc


def main(task="small", names="freex_fuel+,cal"):
    names = [v for v in str(names).split(",") if v]
    refs = C.references(task)
    ns = sorted(refs)
    T = C.Cal(task, ns[0]).T
    q = [0, 4, 8, 13, 26, 39, T] if task == "small" else [0, 8, 16, 26, 52, 78, T]
    P = list(zip(q, q[1:]))
    nv = 1 + len(names)
    edge = defaultdict(lambda: np.zeros((nv, len(P))))
    disp = defaultdict(lambda: np.zeros((nv, len(P))))
    stock = defaultdict(lambda: np.zeros((nv, len(P))))
    burn = defaultdict(lambda: np.zeros((nv, len(P))))
    shed = defaultdict(lambda: np.zeros((nv, len(P))))
    for n in ns:
        cal = C.Cal(task, n)
        inst = cal.inst
        d = C.load_traj(task, n)
        vecs = [d["zpi"].reshape(T, cal.nc)] + [Dsc.load_sol(task, n, nm)[0] for nm in names]
        v = Dsc.View(cal)
        fuels = {k for g in inst.grids for k in inst.nodes[g].grid.fuels}
        for vi, a in enumerate(vecs):
            for (e, k), _cols in v.flows.items():
                if k in fuels:
                    f = v.flow(a, e, k)
                    for pi, (s, e2) in enumerate(P):
                        edge[(v.K[k], v.edge_name(e), inst.edges[e].tau)][vi, pi] += f[s:e2].sum()
            for s_, slot in enumerate(inst.stock_slots):
                if slot.k not in fuels:
                    continue
                jO, jI = cal.tmpl.get(("O", s_)), cal.tmpl.get(("I", s_))
                key = (v.K[slot.k], v.node[slot.node])
                for pi, (s, e2) in enumerate(P):
                    if jO is not None:
                        disp[key][vi, pi] += a[s:e2, jO].sum()
                    if jI is not None and slot.node not in set(inst.supply_nodes):
                        stock[key][vi, pi] += a[s:e2, jI].sum()
            for go, g in enumerate(inst.grids):
                ga = inst.nodes[g].grid
                for k in ga.fuels:
                    G = v.col(a, ("G", go, k))
                    for pi, (s, e2) in enumerate(P):
                        burn[(v.K[k], v.node[g])][vi, pi] += G[s:e2].sum()
                sh = v.col(a, ("ysh", go))
                for pi, (s, e2) in enumerate(P):
                    shed[v.node[g]][vi, pi] += sh[s:e2].sum()
    head = "played | " + " | ".join(names)
    print(f"{task}, {len(ns)} episodes summed, thousand GWh: {head}; periods (weeks): " + ", ".join(f"{s + 1}-{e}" for s, e in P))

    def show(title, table, scale=1e3, weeks=False):
        print(title)
        for key in sorted(table):
            x = table[key]
            if np.abs(x).sum() < 1:
                continue
            div = np.array([e - s for s, e in P]) * len(ns) if weeks else 1.0
            label = " ".join(str(p) for p in key) if isinstance(key, tuple) else key
            print(f"  {label:44s} " + "  ".join("|".join(f"{x[vi, pi] / scale / (div[pi] if weeks else 1):7.1f}" for vi in range(nv)) for pi in range(len(P))) + "   total " + " | ".join(f"{x[vi].sum() / scale / (div.sum() if weeks else 1):9.1f}" for vi in range(nv)))

    show("base load shed by grid:", shed)
    show("fuel burnt by grid:", burn)
    show("fuel disposed above storage, by node:", disp)
    show("mean fuel stock per episode-week, by node (thousand GWh):", stock, weeks=True)
    show("fuel carried by edge (good, edge, lead):", edge)


if __name__ == "__main__":
    fire.Fire(main)

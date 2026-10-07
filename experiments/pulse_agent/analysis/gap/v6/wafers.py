"""Where the recorded plays dispose of wafers above a fab's storage: by fab, with the ways in and their lead times.

uv run python wafers.py --task=full
"""

import sys
from collections import defaultdict
from pathlib import Path

import fire
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common as C  # noqa: E402
import describe as Dsc  # noqa: E402


def main(task="full", detail=None):
    refs = C.references(task)
    ns = sorted(refs)
    total = defaultdict(float)
    weeks = defaultdict(int)
    printed = False
    for n in ns:
        cal = C.Cal(task, n)
        inst, T = cal.inst, cal.T
        d = C.load_traj(task, n)
        a = d["zpi"].reshape(T, cal.nc)
        v = Dsc.View(cal)
        chk, supply = set(inst.chokepoints), set(inst.supply_nodes)
        if not printed:
            printed = True
            for fo, f in enumerate(inst.fabs):
                fa = inst.nodes[f].fab
                s = inst.slot_index[(f, fa.input)]
                ways = []
                for e in inst.in_edges[f]:
                    ed = inst.edges[e]
                    if fa.input in ed.K:
                        kind = "strait" if ed.tail in chk else "supply" if ed.tail in supply else "node"
                        ways.append(f"{v.node[ed.tail]}({kind},tau {ed.tau},u0 {ed.u0:.0f})")
                print(f"{v.node[f]}: cap0 {fa.cap0:.0f} storage {inst.stock_slots[s].storage:.0f} disposal cost {inst.commodities[fa.input].disposal_cost:.0f} v {inst.commodities[fa.input].v:.0f} ways in: {', '.join(ways)}")
        for fo, f in enumerate(inst.fabs):
            fa = inst.nodes[f].fab
            s = inst.slot_index[(f, fa.input)]
            O = v.col(a, ("O", s))
            if O.sum() <= 0:
                continue
            total[v.node[f]] += O.sum()
            weeks[v.node[f]] += int((O > 0).sum())
            if detail is not None and int(detail) == n:
                I, p = v.col(a, ("I", s)), v.col(a, ("p", fo))
                go = inst.grid_ordinal[fa.grid]
                shed = v.col(a, ("ysh", go))
                arr = {}
                for e in inst.in_edges[f]:
                    if fa.input in inst.edges[e].K:
                        arr[e] = v.flow(a, e, fa.input)
                print(f"episode {n} {v.node[f]} storage {inst.stock_slots[s].storage:.0f}: week, stock end, lots, disposed, shed at grid, dispatched on each way in")
                for t in range(T):
                    if O[max(t - 6, 0) : t + 3].sum() > 0:
                        print(f"  {t + 1:3d} {I[t]:10.0f} {p[t]:9.0f} {O[t]:9.0f} {shed[t]:8.0f}  " + " ".join(f"{v.node[inst.edges[e].tail]}:{x[t]:.0f}" for e, x in arr.items()))
    print(f"{task}, {len(ns)} episodes: wafers disposed above storage, by fab (weeks with disposal)")
    for name, q in sorted(total.items(), key=lambda kv: -kv[1]):
        print(f"  {name:24s} {q:14,.0f} ({weeks[name]})")
    print(f"  total {sum(total.values()):,.0f}")


if __name__ == "__main__":
    fire.Fire(main)

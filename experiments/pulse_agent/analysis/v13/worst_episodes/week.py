"""Our play week by week for one grid and fuel: terminal and grid stock, delivery, burn, fab energy, queues (no play).

uv run python outputs/w13a/week.py --ep="12345:12" --grid=grid_jp --fuel=crude --weeks="1,40"
"""
import sys
from pathlib import Path

import fire
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cmp as C


def main(ep, grid, fuel, weeks="1,104", plan_name="b2"):
    from shockbench_flow.hosting.tasks import task_generator

    inst, _ = task_generator("full")
    N, E, K = inst.nodes, inst.edges, inst.commodities
    root, n = (int(v) for v in str(ep).split(":"))
    d, P = C.ours(root, n), C.plan(root, n, plan_name)
    g = next(i for i, nd in enumerate(N) if nd.id == grid)
    k = next(i for i, c in enumerate(K) if c.id == fuel)
    go = inst.grid_ordinal[g]
    ga = N[g].grid
    full = ga.shares[k] * ga.deliverable
    fabs = [fo for fo in inst.grid_fabs[go] if N[inst.fabs[fo]].fab.e > 0]
    draw = sum(N[inst.fabs[fo]].fab.e * N[inst.fabs[fo]].fab.cap0 for fo in fabs)
    term = [E[e].tail for e in inst.in_edges[g] if k in E[e].K and E[e].tau == 0 and E[e].tail not in inst.chokepoints]
    sg = inst.slot_index[(g, k)]
    st = inst.slot_index[(term[0], k)] if term else None
    slot = next((s for s, (e, kk, lane) in enumerate(inst.action_slots) if kk == k and E[e].head == g and term and E[e].tail == term[0]), None)
    chk = [(c, inst.slot_index[(c, k)]) for c in inst.chokepoints if (c, k) in inst.slot_index]
    a, b = (int(v) for v in (weeks if isinstance(weeks, (tuple, list)) else str(weeks).split(",")))
    x, col = (P["x"], P["col"]) if P else (None, None)
    print(f"{grid} {fuel}: full burn {full:.0f}/wk, fabs' draw {draw:.0f}; terminal storage {inst.stock_slots[st].storage if st is not None else 0:.0f}, grid storage {inst.stock_slots[sg].storage:.0f}")
    print("week  term   grid   sent   burn  share fabE | plan: term grid burn share fabE | queues ours " + " ".join(N[c].id[4:8] for c, _ in chk))
    for t in range(a - 1, b):
        seg = d["segment"][t].get((go, k), 0.0)
        Eo = float(d["energy"][t, fabs].sum()) / draw if fabs else 0.0
        line = f"{t + 1:4d} {d['stock'][t, st] if st is not None else 0:6.0f} {d['stock'][t, sg]:6.0f} {d['executed'][t].get(slot, 0.0):6.0f} {seg:6.0f}  {seg / full:4.2f} {Eo:4.2f}"
        if x is not None:
            Ep = float(sum(x[t, col[("E", fo)]] for fo in fabs)) / draw if fabs else 0.0
            line += f" | {x[t, col[('I', st)]] if st is not None else 0:6.0f} {x[t, col[('I', sg)]]:6.0f} {x[t, col[('G', go, k)]]:6.0f} {x[t, col[('G', go, k)]] / full:4.2f} {Ep:4.2f}"
        line += " | " + " ".join(f"{d['stock'][t, s]:5.0f}" for _, s in chk)
        print(line)


if __name__ == "__main__":
    fire.Fire(main)

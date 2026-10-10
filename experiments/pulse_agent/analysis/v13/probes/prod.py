"""Lost sales by product and lots started by fab (share of capacity) in recorded Full episodes."""
import sys, glob
import numpy as np
from shockbench_flow.hosting.tasks import task_generator
inst, _ = task_generator("full")
N, K, E = inst.nodes, inst.commodities, inst.edges
T = inst.T
chk = set(inst.chokepoints)
slots = [(s.node, s.k) for s in inst.stock_slots if s.node not in chk]
pos = {key: i for i, key in enumerate(slots)}
grids = list(inst.grids)
for ep in sorted(glob.glob(sys.argv[1] + "/record_ep*.npz")):
    z = np.load(ep, allow_pickle=True)
    dem = z["obs.last_week.sinks.demand"][1:T + 1]; lost = z["obs.last_week.sinks.lost"][1:T + 1]
    print(ep.split("record_")[1])
    byk = {}
    for di, d in enumerate(inst.demands):
        a = byk.setdefault(K[d.k].id, [0.0, 0.0, 0.0, 0.0])
        a[0] += dem[:, di].sum() * d.pi; a[1] += lost[:, di].sum() * d.pi
        a[2] += dem[52:, di].sum() * d.pi; a[3] += lost[52:, di].sum() * d.pi
    for k, a in byk.items():
        print(f"   {k:9s} demand {a[0]/1e9:6.0f} bn, lost {a[1]/1e9:6.0f} bn ({a[1]/a[0]:.1%}); second year lost {a[3]/a[2]:.1%}")
    # lots started per fab: new wip entries each week
    wn, wq, wo, wob = z["obs.wip.node"], z["obs.wip.qty"], z["obs.wip.out_week"], z["obs.wip.qty.observed"]
    cap = z["obs.graph_now.fab.cap_eff"][:T]
    shed = z["obs.last_week.shed.qty"][1:T + 1]
    line = []
    for fo, f in enumerate(inst.fabs):
        fa = N[f].fab
        started = np.zeros(T)
        for t in range(1, T + 1):  # obs row t: state after week t; lots started in week t leave in week t + tau
            m = (wob[t] > 0) & (wn[t] == f) & (wo[t] == t + fa.tau)
            started[t - 1] = wq[t][m].sum()
        gi = grids.index(fa.grid) if fa.grid in grids else inst.grid_ordinal[fa.grid]
        dark = shed[:, gi] > 1e-9
        waf = stock[:T, pos[(f, fa.input)]] if False else z["obs.stock.qty"][:T, pos[(f, fa.input)]]
        sellable = slice(0, T - 14)
        util = started[sellable].sum() / cap[sellable, fo].sum()
        uw = started[sellable][~dark[sellable]].sum() / max(cap[sellable, fo][~dark[sellable]].sum(), 1e-9)
        line.append(f"   {N[f].id:14s} {fa.cls:8s} {K[fa.product].id:12s} on {N[fa.grid].id:8s} e*cap {fa.e*fa.cap0:6.1f} GWh: lots/capacity {util:5.2f} (in lit weeks {uw:4.2f}, lit {1-dark[sellable].mean():4.2f} of weeks); wafers on hand, weeks of capacity: mean {np.mean(waf/np.maximum(cap[:, fo],1e-9)):4.1f}")
    print("\n".join(line))

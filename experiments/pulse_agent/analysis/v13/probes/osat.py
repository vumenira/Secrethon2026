"""OSATs and the routes out of fabs in a recorded Full episode."""
import sys
import numpy as np
from shockbench_flow.hosting.tasks import task_generator
inst, _ = task_generator("full")
N, K, E = inst.nodes, inst.commodities, inst.edges
T = inst.T
z = np.load(sys.argv[1], allow_pickle=True)
ex = z["obs.last_week.clip.executed"][1:T + 1]
u = z["obs.graph_now.u"][:T]; proh = z["obs.graph_now.prohibited"][:T]
thr = z["obs.graph_now.osat.thr_eff"][:T]
chk = set(inst.chokepoints)
slots = [(s.node, s.k) for s in inst.stock_slots if s.node not in chk]
store = {(s.node, s.k): s.storage for s in inst.stock_slots}
pos = {key: i for i, key in enumerate(slots)}
stock = z["obs.stock.qty"]
sell = slice(0, T - 6)
xs_of = {}
for x, (e, k, lane) in enumerate(inst.action_slots):
    xs_of.setdefault((e, k), []).append(x)
print("OSATs:")
for oo, o in enumerate(inst.osats):
    os_ = N[o].osat
    print(f"  {N[o].id}: throughput nominal {getattr(os_, 'thr', '?')} mean thr_eff {thr[:, oo].mean():,.0f}; attrs {[a for a in dir(os_) if not a.startswith('_')]}")
    for raw, pk in os_.packages.items():
        inflow = sum(ex[sell][:, xs_of.get((e, raw), [])].sum(1).mean() for e in inst.in_edges[o] if raw in E[e].K)
        outflow = sum(ex[sell][:, xs_of.get((e, pk), [])].sum(1).mean() for e in inst.out_edges[o] if pk in E[e].K)
        rs = stock[:T, pos[(o, raw)]].mean() if (o, raw) in pos else float('nan')
        ps = stock[:T, pos[(o, pk)]].mean() if (o, pk) in pos else float('nan')
        print(f"      {K[raw].id:12s} in {inflow:10,.0f}/wk  raw stock {rs:10,.0f} (store {store.get((o, raw), float('nan')):,.0f})   {K[pk].id:9s} out {outflow:10,.0f}/wk  packaged stock {ps:10,.0f} (store {store.get((o, pk), float('nan')):,.0f})")
print("routes out of fabs (raw chips):")
for f in inst.fabs:
    fa = N[f].fab
    parts = []
    for e in inst.out_edges[f]:
        if fa.product not in E[e].K: continue
        fl = ex[sell][:, xs_of.get((e, fa.product), [])].sum(1).mean()
        parts.append(f"{N[E[e].head].id}({E[e].id.split('.')[0]}, tau {E[e].tau}, u {u[:, e].mean()/fa.cap0:4.2f} of cap, banned {int(proh[:, e, fa.product].sum())}): {fl/fa.cap0:4.2f}")
    rs = stock[:T, pos[(f, fa.product)]].mean() / fa.cap0
    print(f"  {N[f].id:17s} raw chips on hand {rs:4.1f} wks of cap -> " + "; ".join(parts))

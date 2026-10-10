"""Wafers in a recorded Full episode: supply against what is lifted, and each fab's inbound routes."""
import sys
import numpy as np
from shockbench_flow.hosting.tasks import task_generator
inst, _ = task_generator("full")
N, K, E = inst.nodes, inst.commodities, inst.edges
T = inst.T
z = np.load(sys.argv[1], allow_pickle=True)
ex = z["obs.last_week.clip.executed"][1:T + 1]; rq = z["obs.last_week.clip.requested"][1:T + 1]
u = z["obs.graph_now.u"][:T]; proh = z["obs.graph_now.prohibited"][:T]; sup = z["obs.graph_now.supply.avail"][:T]
chk = set(inst.chokepoints)
slots = [(s.node, s.k) for s in inst.stock_slots if s.node not in chk]
pos = {key: i for i, key in enumerate(slots)}
stock = z["obs.stock.qty"]
inputs = sorted({N[f].fab.input for f in inst.fabs})
print("fab inputs:", [K[k].id for k in inputs])
out_slots = {}
for sl, (e, k, lane) in enumerate(inst.action_slots):
    out_slots.setdefault((E[e].tail, k), []).append(sl)
# supply slots: find the layout from the static block if the instance has no list
lay = None
for name in ("supply_slots",):
    lay = getattr(inst, name, None)
print("supply nodes:", [N[s].id for s in inst.supply_nodes])
sell = slice(0, T - 14)
for k in inputs:
    capsum = sum(N[f].fab.cap0 for f in inst.fabs if N[f].fab.input == k)
    print(f"== {K[k].id}: fab capacity for it {capsum:,.0f} lots/wk (wafers per lot: see w_scr)")
    for s in inst.supply_nodes:
        sl = out_slots.get((s, k))
        if not sl: continue
        lifted = ex[:, sl].sum(1)
        st = stock[:T, pos[(s, k)]] if (s, k) in pos else None
        print(f"   source {N[s].id:18s} lifted/wk mean {lifted[sell].mean():12,.0f} (y1 {lifted[:52].mean():12,.0f}, y2 {lifted[52:90].mean():12,.0f})  stock at source mean {('%12.0f' % st.mean()) if st is not None else 'n/a'}")
        for x in sl:
            e = inst.action_slots[x][0]
            print(f"        edge {E[e].id:44s} tau {E[e].tau} u0 {E[e].u0:10,.0f} mean u {u[:, e].mean():10,.0f} banned wks {int(proh[:, e, k].sum()):3d} executed {ex[sell, x].mean():10,.0f}")
    for f in inst.fabs:
        fa = N[f].fab
        if fa.input != k: continue
        tot = 0.0; parts = []
        for e in inst.in_edges[f]:
            if k not in E[e].K: continue
            xs = [x for x, (ee, kk, lane) in enumerate(inst.action_slots) if ee == e and kk == k]
            fl = ex[sell][:, xs].sum(1).mean()
            tot += fl
            parts.append(f"{N[E[e].tail].id}(tau {E[e].tau}, u {u[:, e].mean()/fa.cap0:4.2f} of cap, banned {int(proh[:, e, k].sum())}): {fl/fa.cap0:4.2f}")
        print(f"   fab {N[f].id:17s} cap0 {fa.cap0:9,.0f} w_scr {fa.w_scr}: wafers in / capacity {tot/fa.cap0:4.2f}  <- " + "; ".join(parts))
print("supply.avail columns (constant?):", [(round(float(sup[:, i].mean())), round(float(sup[:, i].min()))) for i in range(sup.shape[1])])

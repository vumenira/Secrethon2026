"""Edges into a terminal for one fuel in a recorded Full episode: capacity, bans, and what arrived."""
import sys
import numpy as np
from shockbench_flow.hosting.tasks import task_generator
inst, _ = task_generator("full")
N, K, E = inst.nodes, inst.commodities, inst.edges
T = inst.T
z = np.load(sys.argv[1], allow_pickle=True)
kid = {k.id: i for i, k in enumerate(K)}; nid = {n.id: i for i, n in enumerate(N)}
chk = set(inst.chokepoints)
slots = [(s.node, s.k) for s in inst.stock_slots if s.node not in chk]
pos = {key: i for i, key in enumerate(slots)}
stock = z["obs.stock.qty"]; ex = z["obs.last_week.clip.executed"][1:T + 1]
u = z["obs.graph_now.u"][:T]; proh = z["obs.graph_now.prohibited"][:T]
ovq = z["action.override_qty"]; rmode = z["action.release_mode"]
for term, fuel, burn in [a.split(":") for a in sys.argv[2:]]:
    n, k, burn = nid[term], kid[fuel], float(burn)
    print(f"== {term} {fuel} (burn {burn:.0f}/wk)")
    for e in inst.in_edges[n]:
        if k not in E[e].K: continue
        ban = proh[:, e, k]
        first = int(np.argmax(ban)) + 1 if ban.any() else None
        print(f"   in-edge {E[e].id:42s} tau {E[e].tau} u0/burn {E[e].u0/burn:5.2f}  mean u/burn {u[:, e].mean()/burn:5.2f} (min {u[:, e].min()/burn:4.2f})  weeks banned for {fuel}: {int(ban.sum()):3d}" + (f" (first week {first})" if first else ""))
    out = [s for s, (e, kk, lane) in enumerate(inst.action_slots) if kk == k and E[e].tail == n]
    deliv = ex[:, out].sum(1)
    arr = stock[1:T + 1, pos[(n, k)]] - stock[:T, pos[(n, k)]] + deliv
    for a, b in [(0, 13), (13, 26), (26, 52), (52, 78), (78, 104)]:
        print(f"   weeks {a+1:3d}-{b:3d}: arrivals/burn {arr[a:b].mean()/burn:5.2f}   delivered to grid/burn {deliv[a:b].mean()/burn:5.2f}   terminal stock/burn {stock[a:b, pos[(n, k)]].mean()/burn:5.2f}")
# override slots: where queued crude is released to
print("override slots (chokepoint out-edges) for crude: mean released per week")
kc = kid["crude"]
for i, osl in enumerate(inst.override_slots):
    try:
        e, kk = osl[0], osl[1]
    except Exception:
        e, kk = osl.edge, osl.k
    if kk != kc: continue
    print(f"   {i:2d} {E[e].id:40s} u0 {E[e].u0:7.0f} mean u {u[:, e].mean():7.0f} banned wks {int(proh[:, e, kc].sum()):3d}  released/wk {ovq[:, i].mean():8.1f}")

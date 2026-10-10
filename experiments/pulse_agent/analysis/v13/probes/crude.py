"""Where Gulf crude goes in a recorded Full episode: dispatch from the source by lane, the edges into each terminal."""
import sys, json
import numpy as np
from shockbench_flow.hosting.tasks import task_generator
inst, _ = task_generator("full")
N, K, E = inst.nodes, inst.commodities, inst.edges
T = inst.T
z = np.load(sys.argv[1], allow_pickle=True)
kid = {k.id: i for i, k in enumerate(K)}; nid = {n.id: i for i, n in enumerate(N)}
k = kid[sys.argv[2] if len(sys.argv) > 2 else "crude"]
ex = z["obs.last_week.clip.executed"][1:T + 1]; rq = z["obs.last_week.clip.requested"][1:T + 1]
u = z["obs.graph_now.u"][:T]; proh = z["obs.graph_now.prohibited"][:T]; opn = z["obs.graph_now.open"][:T]
kap = z["obs.graph_now.kappa.tb"][:T]; sup = z["obs.graph_now.supply.avail"][:T]
print("chokepoints:", [N[c].id for c in inst.chokepoints])
print("supply slots:", [(N[s.node].id if hasattr(s, 'node') else s) for s in inst.supply_slots][:20] if hasattr(inst, "supply_slots") else "?")
lanes = getattr(inst, "lanes", None)
# slots of this commodity grouped by edge
rows = []
for sl, (e, kk, lane) in enumerate(inst.action_slots):
    if kk != k: continue
    lid = None if lane is None else (lanes[lane].id if lanes is not None and hasattr(lanes[lane], "id") else lane)
    rows.append((sl, E[e].id, lid))
print("slots for", K[k].id)
tot = ex.sum(0)
wk = slice(0, T)
for sl, eid, lid in rows:
    e = [i for i, ed in enumerate(E) if ed.id == eid][0]
    banned = int(proh[:, e, k].sum())
    print(f"  slot {sl:3d} {eid:45s} lane {str(lid):38s} u0 {E[e].u0:8.0f} tau {E[e].tau} | executed/wk mean {ex[:, sl].mean():8.1f} (y1 {ex[:52, sl].mean():7.1f}, y2 {ex[52:, sl].mean():7.1f}) requested mean {rq[:, sl].mean():8.1f} | weeks banned {banned:3d} | mean u {u[:, e].mean():8.0f}")
print("open fraction mean by chokepoint:", {N[c].id: round(float(opn[:, i].mean()), 2) for i, c in enumerate(inst.chokepoints)})
print("kappa.tb mean / first:", {N[c].id: (round(float(kap[:, i].mean())), round(float(kap[0, i]))) for i, c in enumerate(inst.chokepoints)})

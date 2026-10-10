"""Per fab in a recorded Full episode: lots started in whole and in dark weeks of its grid, wafers on hand."""
import sys, json
import numpy as np
from shockbench_flow.hosting.tasks import task_generator
inst, _ = task_generator("full")
N, K, E = inst.nodes, inst.commodities, inst.edges
T = inst.T
z = np.load(sys.argv[1], allow_pickle=True)
chk = set(inst.chokepoints)
slots = [(s.node, s.k) for s in inst.stock_slots if s.node not in chk]
pos = {key: i for i, key in enumerate(slots)}
stock = z["obs.stock.qty"]
wn, wk_, wq, wo = z["obs.wip.node"], z["obs.wip.k"], z["obs.wip.qty"], z["obs.wip.out_week"]
wobs = z["obs.wip.qty.observed"]
shed = z["obs.last_week.shed.qty"][1:T + 1]; ybar = z["obs.graph_now.grid.y_bar"][:T]
cap = z["obs.graph_now.fab.cap_eff"][:T]
grids = list(inst.grids)
fab = N[inst.fabs[0]].fab
print("fab attrs:", [a for a in dir(fab) if not a.startswith("_")])
print("supply avail (first week / mean / min) per supply slot:")
sup = z["obs.graph_now.supply.avail"][:T]
ss = getattr(inst, "supply_slots", None)
for i in range(sup.shape[1]):
    lab = ss[i] if ss is not None else i
    try: lab = (N[lab[0]].id, K[lab[1]].id)
    except Exception: pass
    print("   ", lab, round(float(sup[0, i])), round(float(sup[:, i].mean())), round(float(sup[:, i].min())))

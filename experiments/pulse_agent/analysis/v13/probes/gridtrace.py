"""Per week at one grid of a recorded episode: each fuel's stock at the grid and upstream (weeks of burn), deliveries, shed."""
import sys, json
import numpy as np
from shockbench_flow.hosting.tasks import task_generator
inst, _ = task_generator("full")
N, K, E = inst.nodes, inst.commodities, inst.edges
T = inst.T
ep, gname = sys.argv[1], sys.argv[2]
step = int(sys.argv[3]) if len(sys.argv) > 3 else 1
z = np.load(ep, allow_pickle=True)
chk = set(inst.chokepoints)
slots = [(s.node, s.k) for s in inst.stock_slots if s.node not in chk]
pos = {key: i for i, key in enumerate(slots)}
nid = {n.id: i for i, n in enumerate(N)}
kid = {k.id: i for i, k in enumerate(K)}
g = nid[gname]; gi = list(inst.grids).index(g)
stock = z["obs.stock.qty"]; ex = z["obs.last_week.clip.executed"][1:T + 1]; rq = z["obs.last_week.clip.requested"][1:T + 1]
shed = z["obs.last_week.shed.qty"][1:T + 1, gi]; ybar = z["obs.graph_now.grid.y_bar"][:T, gi]; Gbar = z["obs.graph_now.grid.G_bar"][:T, gi]
u = z["obs.graph_now.u"][:T]; proh = z["obs.graph_now.prohibited"][:T]
grid = N[g].grid
print("grid attrs:", [a for a in dir(grid) if not a.startswith("_")][:40])
fuels = list(grid.fuels)
burn = {k: grid.shares[k] * grid.deliverable for k in fuels}
print(gname, "G", grid.deliverable, "base", grid.base_load, "fuels", {K[k].id: (round(grid.shares[k], 3), round(burn[k])) for k in fuels}, "rationed", None if grid.rationed is None else K[grid.rationed].id, "days_cover", grid.days_cover, "ibar", {K[k].id: v for k, v in dict(grid.ibar).items()} if hasattr(grid.ibar, "items") else grid.ibar)
# upstream nodes per fuel: tails of in-edges carrying it
ins = {k: [e for e in inst.in_edges[g] if k in E[e].K] for k in fuels}
slot_of = {}
for sl, (e, k, lane) in enumerate(inst.action_slots):
    slot_of.setdefault((e, k), []).append(sl)
for k in fuels:
    print("  ", K[k].id, "in-edges:", [(E[e].id, "tau", E[e].tau, "u0/burn", round(E[e].u0 / burn[k], 2)) for e in ins[k]])
hdr = "wk  shed/base |"
for k in fuels:
    hdr += f" {K[k].id[:5]}: grid_stock deliv_exec(req) upstream_stock |"
print(hdr, " [weeks of burn]")
for t in range(0, T, step):
    line = f"{t+1:3d}  {shed[t]/ybar[t]:7.4f}  |"
    for k in fuels:
        gs = stock[t, pos[(g, k)]] / burn[k]
        de = sum(ex[t, s] for e in ins[k] for s in slot_of.get((e, k), ())) / burn[k]
        dr = sum(rq[t, s] for e in ins[k] for s in slot_of.get((e, k), ())) / burn[k]
        up = sum(stock[t, pos[(E[e].tail, k)]] for e in ins[k] if (E[e].tail, k) in pos) / burn[k]
        banned = sum(1 for e in ins[k] if proh[t, e, k])
        line += f" {gs:6.2f} {de:5.2f}({dr:5.2f}) {up:6.2f}{'*'*banned:2s}|"
    print(line)

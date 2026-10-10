"""Lost sales on recorded Full episodes of v12: how much happened while packaged chips sat unshipped one 1-week edge away."""
import sys, json, glob
import numpy as np
from shockbench_flow.hosting.tasks import task_generator

inst, _ = task_generator("full")
K, N, E = inst.commodities, inst.nodes, inst.edges
chk = set(inst.chokepoints)
slots = [(s.node, s.k) for s in inst.stock_slots if s.node not in chk]
pos = {key: i for i, key in enumerate(slots)}
T = inst.T
refs = {r["episode"]: r for r in json.load(open("outputs/recovery_tune/full-e12345-n16/references.json"))}
pol = {r["episode"]: r for r in json.load(open("outputs/recovery_tune/full-e12345-n16/v12_all.json"))["rows"]}
by_tail, edge_slots = {}, {}
for sl, (e, k, lane) in enumerate(inst.action_slots):
    by_tail.setdefault((E[e].tail, k), []).append(sl)
    edge_slots.setdefault(e, []).append(sl)
for ep in sorted(glob.glob(sys.argv[1] + "/record_ep*.npz")):
    z = np.load(ep, allow_pickle=True)
    n = int(ep.split("record_ep")[1].split(".")[0])
    stock = z["obs.stock.qty"]
    dem = z["obs.last_week.sinks.demand"][1:T + 1]
    lost = z["obs.last_week.sinks.lost"][1:T + 1]
    ex = z["obs.last_week.clip.executed"][1:T + 1]
    fc = z["obs.demand_forecast.qty"]
    u = z["obs.graph_now.u"][:T]; proh = z["obs.graph_now.prohibited"][:T]
    J = int(z["reward_cents"].sum())
    tot = strict = bump = 0.0
    first = second = 0.0
    det = {}
    surprise = 0.0  # lost that is no more than (demand - forecast made the week before, h=1)+
    for t in range(1, T):
        pool_s, pool_b, cap_s, cap_b = {}, {}, {}, {}
        for di in sorted(range(len(inst.demands)), key=lambda i: -inst.demands[i].pi):
            d = inst.demands[di]
            l = lost[t, di]
            if l <= 0:
                continue
            v = l * d.pi
            tot += v
            if t < T // 2: first += v
            else: second += v
            sur = max(dem[t, di] - fc[t - 1, di, 1], 0.0)
            surprise += min(l, sur) * d.pi
            cs = cb = 0.0
            for e in inst.in_edges[d.node]:
                ed = E[e]
                if d.k not in ed.K or ed.tau != 1 or ed.tail in chk or proh[t - 1, e, d.k]:
                    continue
                key = (ed.tail, d.k)
                if key not in pos:
                    continue
                left = max(stock[t - 1, pos[key]] - sum(ex[t - 1, s] for s in by_tail.get(key, ())), 0.0)
                pool_s.setdefault(key, left); pool_b.setdefault(key, left)
                all_ex = sum(ex[t - 1, s] for s in edge_slots.get(e, ()))
                own_ex = sum(ex[t - 1, s] for s in edge_slots.get(e, ()) if inst.action_slots[s][1] == d.k)
                cap_s.setdefault(e, max(u[t - 1, e] - all_ex, 0.0)); cap_b.setdefault(e, max(u[t - 1, e] - own_ex, 0.0))
                a = min(cap_s[e], pool_s[key], l - cs); pool_s[key] -= a; cap_s[e] -= a; cs += a
                b = min(cap_b[e], pool_b[key], l - cb); pool_b[key] -= b; cap_b[e] -= b; cb += b
            strict += cs * d.pi; bump += cb * d.pi
            k2 = (N[d.node].id, K[d.k].id)
            x = det.setdefault(k2, [0.0, 0.0, 0.0]); x[0] += v; x[1] += cs * d.pi; x[2] += cb * d.pi
    r = refs[n]
    gap = (pol[n]["J_policy_cents"] - r["J_clairvoyant_cents"]) / 1e11
    att = (r["J_naive_cents"] - r["J_clairvoyant_cents"]) / 1e11
    print(f"ep {n} level {r['stratum']}: gap {gap:.0f} bn of attainable {att:.0f} (score {1-gap/att:.3f}); recorded cost {-J/1e11:.0f} vs cached {pol[n]['J_policy_cents']/1e11:.0f} bn")
    print(f"   lost sales {tot/1e9:.0f} bn (weeks 1-52: {first/1e9:.0f}, 53-104: {second/1e9:.0f}); within the week's demand surprise: {surprise/1e9:.0f} bn")
    print(f"   lost while unshipped packaged stock sat one 1-week edge away: strict spare capacity {strict/1e9:.1f} bn ({strict/1e9/gap:.2f} of gap, {strict/1e9/att:.4f} of attainable); other goods bumped off the edge {bump/1e9:.1f} bn ({bump/1e9/gap:.2f} of gap, {bump/1e9/att:.4f})")
    for k2, x in sorted(det.items(), key=lambda kv: -kv[1][2])[:8]:
        print(f"      {k2[0]:10s} {k2[1]:9s} lost {x[0]/1e9:7.1f}  strict {x[1]/1e9:6.1f}  bump {x[2]/1e9:6.1f}")

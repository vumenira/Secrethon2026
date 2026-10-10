"""Hindsight ceiling of a value-aware pulse: lots made at pulsed grids against the product's unsold stock at the end."""
import fire, numpy as np
from collections import defaultdict
from ana import instance, episodes_of, refs, MEMORY

def main(task="full", entropy=12345, episodes=16, name="r13rec_off", only=""):
    inst, P = instance(task)
    N, K, S = inst.nodes, inst.commodities, inst.stock_slots
    R = refs(task, entropy, episodes)
    prods = sorted({N[f].fab.product for f in inst.fabs})
    pk_of = defaultdict(set)
    for o in inst.osats:
        for a, b in N[o].osat.packages.items():
            pk_of[a].add(b)
    tot = defaultdict(float)
    for n, ep in episodes_of(task, entropy, episodes, name, only):
        rec, trace = ep["records"], {tr["t"]: tr for tr in ep["trace"]}
        chip_slots = ep["chip_slots"]
        line = []
        for p in prods:
            goods = {p} | pk_of[p]
            dos = [do for do, d in enumerate(inst.demands) if d.k in pk_of[p]]
            dem = sum(inst.demands[do].dbar for do in dos)
            end = sum(rec[-1].stock[s] for s in chip_slots if S[s].k in goods)
            start = sum(rec[0].stock[s] for s in chip_slots if S[s].k in goods)
            lowest = min(sum(r.stock[s] for s in chip_slots if S[s].k in goods) for r in rec[12:])
            lost = sum(float(sum(r.lost[do] for do in dos)) for r in rec) / max(sum(float(sum(r.demand[do] for do in dos)) for r in rec), 1.0)
            made = cost = 0.0
            for go, g, fabs, fuels, shares in P.grids:
                ga = N[g].grid
                for fo in fabs:
                    fa = N[inst.fabs[fo]].fab
                    if fa.product != p:
                        continue
                    for t in range(1, inst.T + 1):
                        tr = trace.get(t)
                        if tr is None or t - tr["held_after"].get(go, -10**9) > MEMORY:
                            continue
                        made += float(rec[t - 1].lots_started[fo])
                        cost += float(rec[t - 1].energy[fo]) * ga.voll / 1e9
            unneeded = min(made, lowest)
            share = unneeded / made if made > 0 else 0.0
            line.append(f"{K[p].id[:8]}: lost {lost:4.2f} stock wk-of-demand start {start / dem:4.1f} lowest {lowest / dem:4.1f} end {end / dem:4.1f}; pulsed lots {made / dem:5.1f} wk, at VOLL {cost:6.1f} bn, never needed {share * cost:6.1f} bn")
            tot[K[p].id + " pulsed fab energy bn"] += cost
            tot[K[p].id + " never needed bn"] += share * cost
        ref = R.get(n, {})
        print(f"ep {n:3d} L{ref.get('stratum')} " + " | ".join(line))
    for k, v in tot.items():
        print(f"  {k}: {v:.1f}")

if __name__ == "__main__":
    fire.Fire(main)

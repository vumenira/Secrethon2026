"""Pulsed fab-weeks by how full the fab's own chip store is, and the fab energy drawn there (at VOLL)."""
import fire, numpy as np
from collections import defaultdict
from ana import instance, episodes_of, refs, MEMORY

def main(task="full", entropy=12345, episodes=16, name="r13rec_off", only="", per_episode=False):
    inst, P = instance(task)
    N, K, S = inst.nodes, inst.commodities, inst.stock_slots
    bins = [0.0, 0.25, 0.5, 0.75, 0.9, 1.01]
    tot = defaultdict(float)
    byfab = defaultdict(lambda: defaultdict(float))
    for n, ep in episodes_of(task, entropy, episodes, name, only):
        rec, trace = ep["records"], {tr["t"]: tr for tr in ep["trace"]}
        e_over = e_all = 0.0
        for go, g, fabs, fuels, shares in P.grids:
            ga = N[g].grid
            for fo in fabs:
                f = inst.fabs[fo]
                fa = N[f].fab
                storage = S[inst.slot_index[(f, fa.product)]].storage
                for t in range(1, inst.T + 1):
                    tr = trace.get(t)
                    if tr is None or t - tr["held_after"].get(go, -10**9) > MEMORY or t > P.last[go]:
                        continue
                    fill = tr["grids"][go]["fabs"][fo]["chips"] / storage
                    b = max(i for i, lo in enumerate(bins[:-1]) if fill >= lo)
                    en = float(rec[t - 1].energy[fo]) * ga.voll / 1e9
                    tot[f"weeks fill {bins[b]:.2f}+"] += 1
                    tot[f"bn fill {bins[b]:.2f}+"] += en
                    e_all += en
                    if fill > 0.5:
                        e_over += en
                        byfab[N[f].id]["weeks"] += 1
                        byfab[N[f].id]["bn"] += en
        if per_episode:
            print(f"ep {n:3d}: fab energy at pulsed grids {e_all:6.1f} bn at VOLL, of it with the chip store above half {e_over:6.1f}")
    for k in sorted(tot):
        print(f"  {k}: {tot[k]:.1f}")
    print("  above half, by fab:", {k: (int(v["weeks"]), round(v["bn"], 1)) for k, v in sorted(byfab.items())})

if __name__ == "__main__":
    fire.Fire(main)

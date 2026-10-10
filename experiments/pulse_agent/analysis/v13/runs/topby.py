"""Top-ups of a recorded v12 play by fab: wafers added, and those added where the plan's own lots (wk 3-6) are under 0.9 of capacity."""
import fire, numpy as np
from collections import defaultdict
from ana import instance, episodes_of, MEMORY

def main(task="full", entropy=12345, episodes=16, name="r13rec_off"):
    inst, P = instance(task)
    N, K = inst.nodes, inst.commodities
    agg = defaultdict(lambda: defaultdict(float))
    for n, ep in episodes_of(task, entropy, episodes, name):
        trace = {tr["t"]: tr for tr in ep["trace"]}
        for go, g, fabs, fuels, shares in P.grids:
            for fo in fabs:
                fa = N[inst.fabs[fo]].fab
                for t, tr in trace.items():
                    if tr["plan"] is None:
                        continue
                    fb = tr["grids"][go]["fabs"][fo]
                    cap = max(fb["alpha"] * fb["R"] * fa.cap0, 1.0)
                    added = fb["sent"] - fb["planned"]
                    if added <= 1e-6 * cap:
                        continue
                    H = tr["plan"]["H"]
                    des = float(np.mean(np.maximum(tr["plan"]["p"][fo, 2:min(6, H)], 0.0))) if H > 2 else cap
                    a = agg[(K[fa.product].id, N[inst.fabs[fo]].id)]
                    a["n"] += 1
                    a["wafers"] += added
                    if des < 0.9 * cap:
                        a["n_low"] += 1
                        a["wafers_low"] += added
                        a["nearly none"] += added if des < 0.1 * cap else 0.0
    print(f"{'product':14s} {'fab':20s} top-ups  wafers (M)  where plan < 0.9 cap: top-ups  wafers (M)  of them plan < 0.1 cap (M)")
    tot = defaultdict(float)
    for (prod, fab), a in sorted(agg.items()):
        print(f"{prod:14s} {fab:20s} {a['n']:6.0f} {a['wafers'] / 1e6:10.2f} {a['n_low']:28.0f} {a['wafers_low'] / 1e6:10.2f} {a['nearly none'] / 1e6:10.2f}")
        tot[prod] += a["wafers_low"]
    print({k: round(v / 1e6, 2) for k, v in tot.items()})

if __name__ == "__main__":
    fire.Fire(main)

"""Online surplus signal from the plan: the product's stock at the window's end against now, and a fab's lots in it."""
import fire, numpy as np
from collections import defaultdict
from ana import instance, episodes_of, refs, MEMORY

def main(task="full", entropy=12345, episodes=16, name="r13rec_off", only="", detail=""):
    inst, P = instance(task)
    N, K, S = inst.nodes, inst.commodities, inst.stock_slots
    pk_of = defaultdict(set)
    for o in inst.osats:
        for a, b in N[o].osat.packages.items():
            pk_of[a].add(b)
    for n, ep in episodes_of(task, entropy, episodes, name, only):
        rec, trace = ep["records"], {tr["t"]: tr for tr in ep["trace"]}
        chip_slots = ep["chip_slots"]
        out = []
        for go, g, fabs, fuels, shares in P.grids:
            ga = N[g].grid
            for fo in fabs:
                fa = N[inst.fabs[fo]].fab
                goods = {fa.product} | pk_of[fa.product]
                idx = [i for i, s in enumerate(chip_slots) if S[s].k in goods]
                weeks = fires = 0
                en_all = en_fire = 0.0
                ratios = []
                for t in range(1, inst.T + 1):
                    tr = trace.get(t)
                    if tr is None or tr["plan"] is None or t > P.last[go]:
                        continue
                    if t - tr["held_after"].get(go, -10**9) > MEMORY:
                        continue
                    pl = tr["plan"]
                    H = pl["H"]
                    if H < 14:
                        continue
                    I = np.nansum(pl["I"][idx, :], axis=0)
                    now_stock = sum(rec[t - 2].stock[chip_slots[i]] for i in idx) if t > 1 else I[0]
                    growth = float(I[H - 1] - now_stock)
                    lots = float(np.sum(np.maximum(pl["p"][fo, 1:H], 0.0)))
                    weeks += 1
                    en = float(rec[t - 1].energy[fo]) * ga.voll / 1e9
                    en_all += en
                    ratios.append(growth / max(lots, 1.0))
                    if growth >= lots and lots > 0:
                        fires += 1
                        en_fire += en
                if weeks:
                    out.append(f"{N[inst.fabs[fo]].id[4:]}: {fires}/{weeks} wk, {en_fire:.1f}/{en_all:.1f} bn, median growth/lots {np.median(ratios):+.2f}")
        print(f"ep {n:3d}: " + " | ".join(out))

if __name__ == "__main__":
    fire.Fire(main)

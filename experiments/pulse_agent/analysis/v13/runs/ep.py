"""One recorded episode, one product: week by week, the lots of its fabs, the plan's lots and lost sales, the pulse."""
import sys
import fire, numpy as np
from ana import instance, episodes_of, MEMORY

def main(episode, task="full", entropy=12345, episodes=16, name="r13rec_off", product="chip_mat_raw", grid="grid_cn", first=1, last=104):
    inst, P = instance(task)
    N, K = inst.nodes, inst.commodities
    fabs = [fo for fo, f in enumerate(inst.fabs) if K[N[f].fab.product].id == product]
    pk = {K[b].id for o in inst.osats for a, b in N[o].osat.packages.items() if K[a].id == product}
    dos = [do for do, d in enumerate(inst.demands) if K[d.k].id in pk]
    go = inst.grid_ordinal[inst.node_index[grid]]
    for n, ep in episodes_of(task, entropy, episodes, name, str(episode)):
        rec, trace = ep["records"], {tr["t"]: tr for tr in ep["trace"]}
        chip_slots = ep["chip_slots"]
        S = inst.stock_slots
        raw_slots = [i for i, s in enumerate(chip_slots) if K[S[s].k].id == product]
        pk_slots = [i for i, s in enumerate(chip_slots) if K[S[s].k].id in pk]
        dem = sum(inst.demands[do].dbar for do in dos)
        print(f"episode {n}: product {product}, fabs {[N[inst.fabs[fo]].id for fo in fabs]}, weekly demand {dem:.0f}, total cap {sum(N[inst.fabs[fo]].fab.cap0 for fo in fabs):.0f}")
        print("week pulsed log | lots/cap per fab | plan p(mean wk3-6)/cap per fab | lost sales/demand real | plan U/demand wk1-4, wk5-12, wk13-20 | raw chips stock (wk of demand) | packaged stock (wk of demand) | CN shed GWh, fab energy")
        for t in range(first, min(last, inst.T) + 1):
            tr = trace.get(t)
            r = rec[t - 1]
            if tr is None:
                continue
            pulsed = t - tr["held_after"].get(go, -10**9) <= MEMORY
            logs = ",".join(rule for (_t, g2, rule, _m) in tr["log"] if g2 == go)
            lots = " ".join(f"{r.lots_started[fo] / N[inst.fabs[fo]].fab.cap0:4.2f}" for fo in fabs)
            pl = tr["plan"]
            if pl is not None:
                H = pl["H"]
                des = " ".join(f"{float(np.mean(pl['p'][fo, 2:min(6, H)])) / N[inst.fabs[fo]].fab.cap0 if H > 2 else float('nan'):4.2f}" for fo in fabs)
                U = pl["U"][dos, :]
                u = f"{np.nansum(U[:, :4]) / (4 * dem):4.2f} {np.nansum(U[:, 4:12]) / (8 * dem):4.2f} {np.nansum(U[:, 12:20]) / (8 * dem):4.2f}"
            else:
                des, u = "-", "-"
            lost = float(sum(r.lost[do] for do in dos)) / max(float(sum(r.demand[do] for do in dos)), 1.0)
            raw = sum(r.stock[chip_slots[i]] for i in raw_slots) / dem
            pks = sum(r.stock[chip_slots[i]] for i in pk_slots) / dem
            en = float(sum(r.energy[fo] for fo in inst.grid_fabs[go]))
            print(f"{t:4d} {'P' if pulsed else '.'} {logs:14s} | {lots} | {des} | {lost:4.2f} | {u} | {raw:5.2f} | {pks:5.2f} | {r.shed[go]:6.0f} {en:5.0f}")

if __name__ == "__main__":
    fire.Fire(main)

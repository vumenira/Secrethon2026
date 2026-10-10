"""Chips disposed of at fab stores (the store was full when the lots came out), by fab; and the store's fill over time."""
import fire, numpy as np
from collections import defaultdict
from ana import instance, episodes_of, refs, MEMORY

def main(task="full", entropy=12345, episodes=16, name="r13rec_off", only="", show=""):
    inst, P = instance(task)
    N, K, S = inst.nodes, inst.commodities, inst.stock_slots
    grid_of = {}
    for go, g, fabs, fuels, shares in P.grids:
        for fo in fabs:
            grid_of[fo] = (go, g)
    tot = defaultdict(float)
    for n, ep in episodes_of(task, entropy, episodes, name, only):
        rec, trace = ep["records"], {tr["t"]: tr for tr in ep["trace"]}
        line = []
        for fo, f in enumerate(inst.fabs):
            fa = N[f].fab
            s = inst.slot_index[(f, fa.product)]
            disp = sum(float(r.disposal[s]) for r in rec)
            lots = sum(float(r.lots_started[fo]) for r in rec)
            if fo in grid_of:
                go, g = grid_of[fo]
                voll = N[g].grid.voll
                pulsed = sum(1 for t in range(1, inst.T + 1) if t in trace and t - trace[t]["held_after"].get(go, -10**9) <= MEMORY)
            else:
                voll, pulsed = 0.0, 0
            fee = disp * K[fa.product].disposal_cost / 1e9
            energy = disp * fa.e * voll / 1e9
            tot["chips disposed at fabs, weeks of cap"] += disp / fa.cap0
            tot["their disposal fee bn"] += fee
            tot["their energy at VOLL bn"] += energy
            if pulsed:
                tot["at fabs pulsed some week: weeks of cap"] += disp / fa.cap0
                tot["at fabs pulsed some week: fee + energy bn"] += fee + energy
            if disp > 0.05 * fa.cap0:
                line.append(f"{N[f].id[4:]} {disp / fa.cap0:.1f}wk({fee + energy:.1f}bn,{pulsed}p)")
        # OSAT stores
        od = 0.0
        for s, st in enumerate(S):
            if st.node in inst.osats and K[st.k].id.startswith("chip"):
                od += sum(float(r.disposal[s]) for r in rec) * K[st.k].disposal_cost / 1e9
        tot["disposal fee at OSATs bn"] += od
        print(f"ep {n:3d}: " + " ".join(line) + f" | OSAT disposal fee {od:.1f} bn")
    for k, v in tot.items():
        print(f"  {k}: {v:.1f}")

if __name__ == "__main__":
    fire.Fire(main)

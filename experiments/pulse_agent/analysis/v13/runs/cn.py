"""Per episode, one fab: weeks its grid was pulsed, how full its chip store was then, and the lean_hold weeks."""
import fire, numpy as np
from ana import instance, episodes_of, MEMORY

def main(task="full", entropy=12345, episodes=16, name="r13rec_off", fab="fab_cn_mature_1"):
    inst, P = instance(task)
    N, S = inst.nodes, inst.stock_slots
    f = inst.node_index[fab]
    fo = inst.fabs.index(f)
    fa = N[f].fab
    go = inst.grid_ordinal[fa.grid]
    storage = S[inst.slot_index[(f, fa.product)]].storage
    for n, ep in episodes_of(task, entropy, episodes, name):
        rec, trace = ep["records"], {tr["t"]: tr for tr in ep["trace"]}
        fills, lean, energy = [], 0, 0.0
        for t in range(1, inst.T + 1):
            tr = trace.get(t)
            if tr is None or t > P.last[go]:
                continue
            lean += any(r.startswith("lean") for (_t, g2, r, _m) in tr["log"] if g2 == go)
            if t - tr["held_after"].get(go, -10**9) > MEMORY:
                continue
            fills.append(tr["grids"][go]["fabs"][fo]["chips"] / storage)
            energy += float(rec[t - 1].energy[fo])
        if fills:
            a = np.array(fills)
            print(f"ep {n:3d}: pulsed weeks {len(a):3d}, lean weeks {lean:3d}, store fill mean {a.mean():4.2f}, above half {np.mean(a > 0.5):4.2f}, above 0.9 {np.mean(a > 0.9):4.2f}, fab energy {energy:7.0f} GWh")
        else:
            print(f"ep {n:3d}: never pulsed (lean weeks {lean})")

if __name__ == "__main__":
    fire.Fire(main)

"""Terminal and grid stock of one fuel week by week, v12 against a recorded variant, with what the terminal sends on."""
import pickle
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cmp as C

HERE = Path(__file__).resolve().parent


def main(name, ep, grid, fuel, weeks="1,40"):
    from shockbench_flow.hosting.tasks import task_generator

    inst, _ = task_generator("full")
    N, E, K = inst.nodes, inst.edges, inst.commodities
    root, n = (int(v) for v in str(ep).split(":"))
    a = C.ours(root, n)
    with open(HERE / "data" / f"full-e{root}" / f"{name}_ep{n}.pkl", "rb") as f:
        b = pickle.load(f)
    g = next(i for i, nd in enumerate(N) if nd.id == grid)
    k = next(i for i, c in enumerate(K) if c.id == fuel)
    go = inst.grid_ordinal[g]
    term = next(E[e].tail for e in inst.in_edges[g] if k in E[e].K and E[e].tau == 0 and E[e].tail not in inst.chokepoints)
    sg, st = inst.slot_index[(g, k)], inst.slot_index[(term, k)]
    slot = next(s for s, (e, kk, lane) in enumerate(inst.action_slots) if kk == k and E[e].head == g and E[e].tail == term)
    chk = [(c, inst.slot_index[(c, k)]) for c in inst.chokepoints if (c, k) in inst.slot_index]
    full = N[g].grid.shares[k] * N[g].grid.deliverable
    lo, hi = (int(v) for v in (weeks if isinstance(weeks, (tuple, list)) else str(weeks).split(",")))
    print(f"{grid} {fuel} (full burn {full:.0f}): week | v12: terminal, grid, sent, burnt | {name}: terminal, grid, sent, burnt | queues v12 / {name}: " + " ".join(N[c].id[4:8] for c, _ in chk))
    for t in range(lo - 1, hi):
        row = []
        for d in (a, b):
            row.append(f"{d['stock'][t, st]:6.0f} {d['stock'][t, sg]:6.0f} {d['executed'][t].get(slot, 0.0):6.0f} {d['segment'][t].get((go, k), 0.0):6.0f}")
        qa = " ".join(f"{a['stock'][t, s]:5.0f}" for _, s in chk)
        qb = " ".join(f"{b['stock'][t, s]:5.0f}" for _, s in chk)
        print(f"{t + 1:4d} | {row[0]} | {row[1]} | {qa} / {qb}")


if __name__ == "__main__":
    fire.Fire(main)

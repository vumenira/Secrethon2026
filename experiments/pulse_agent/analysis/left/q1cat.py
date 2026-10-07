"""Which flows of the opening differ between our play and a plan of q1.py, by good and kind of edge.

uv run python q1cat.py --task=small --plan=true --weeks=4
"""

import pickle
from collections import defaultdict

import fire
import numpy as np

import common as C


def kinds(inst):
    chk, sup, grids, fabs, osats = set(inst.chokepoints), set(inst.supply_nodes), set(inst.grids), set(inst.fabs), set(inst.osats)
    sinks = {d.node for d in inst.demands}
    out = {}
    for i, _nd in enumerate(inst.nodes):
        out[i] = "strait" if i in chk else "source" if i in sup else "grid" if i in grids else "fab" if i in fabs else "osat" if i in osats else "sink" if i in sinks else "terminal"
    return out


def main(task="small", plan="true", weeks=4, only=None):
    from shockbench_flow.hosting.tasks import task_generator

    inst, _ = task_generator(task)
    kind = kinds(inst)
    K = [c.id for c in inst.commodities]
    refs = C.references(task)
    ns = sorted(refs) if only is None else [int(v) for v in (only if isinstance(only, (list, tuple)) else [only])]
    keys = pickle.loads((C.HERE / "res" / f"{task}_{ns[0]}_sol.pkl").read_bytes())["keys"]
    xcols = [(j, key) for j, key in enumerate(keys) if key[0] == "x"]
    cat = {}
    for j, key in xcols:
        ed = inst.edges[key[1]]
        cat[j] = (K[key[2]], f"{kind[ed.tail]}->{kind[ed.head]}")
    A, B, L1, used = defaultdict(float), defaultdict(float), defaultdict(float), 0
    per = defaultdict(list)
    for n in ns:
        p = C.HERE / "q1" / f"{task}_{n}.pkl"
        if not p.exists():
            continue
        d = pickle.loads(p.read_bytes())
        if d.get(plan) is None:
            continue
        used += 1
        a, b = d["ours"][:weeks].astype(float), d[plan][:weeks].astype(float)
        ea, eb = defaultdict(float), defaultdict(float)
        for j, key in xcols:
            c = cat[j]
            v = inst.commodities[key[2]].v
            A[c] += a[:, j].sum()
            B[c] += b[:, j].sum()
            L1[c] += np.abs(b[:, j] - a[:, j]).sum() * v
            ea[c] += a[:, j].sum()
            eb[c] += b[:, j].sum()
        for c in ea:
            per[c].append((ea[c], eb[c]))
    tot = sum(L1.values())
    print(f"{task}, {used} episodes, weeks 1-{weeks}: flows of our play against the plan '{plan}' (the near-optimal plan closest to our opening), by good and kind of edge")
    print(f"  {'good':12s} {'edge':20s} {'ours per episode':>18s} {'plan':>14s} {'plan/ours':>10s} {'share of what was moved (by value)':>36s} {'episodes plan<0.9 ours | >1.1 ours':>36s}")
    for c in sorted(L1, key=lambda c: -L1[c]):
        if A[c] + B[c] <= 0:
            continue
        lo = sum(1 for x, y in per[c] if y < 0.9 * x - 1e-9)
        hi = sum(1 for x, y in per[c] if y > 1.1 * x + 1e-9)
        print(f"  {c[0]:12s} {c[1]:20s} {A[c] / used:18,.0f} {B[c] / used:14,.0f} {B[c] / A[c] if A[c] else float('nan'):10.3f} {L1[c] / tot:36.1%} {lo:17d} | {hi:<3d}")
    print(f"  moved in all: {tot / used / 1e9:.2f} bn of goods per episode (customs value)")


if __name__ == "__main__":
    fire.Fire(main)

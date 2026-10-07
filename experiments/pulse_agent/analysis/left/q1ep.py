"""One episode's opening: the flows that differ most between our play and a plan of q1.py, edge by edge.

uv run python q1ep.py --task=small --n=12 --plan=true --weeks=4
"""

import pickle

import fire
import numpy as np

import common as C
import worlds as W


def main(task="small", n=12, plan="true", weeks=4, top=14):
    inst, omega, marks, _fb = C.world(task, int(n))
    node = [nd.id for nd in inst.nodes]
    K = [c.id for c in inst.commodities]
    d = pickle.loads((C.HERE / "q1" / f"{task}_{n}.pkl").read_bytes())
    keys = pickle.loads((C.HERE / "res" / f"{task}_{n}_sol.pkl").read_bytes())["keys"]
    a, b = d["ours"][:weeks].astype(float), d[plan][:weeks].astype(float)
    rows = {}
    for j, key in enumerate(keys):
        if key[0] != "x":
            continue
        e, k = key[1], key[2]
        r = rows.setdefault((e, k), [np.zeros(weeks), np.zeros(weeks)])
        r[0] += a[:, j]
        r[1] += b[:, j]
    out = []
    for (e, k), (x, y) in rows.items():
        v = inst.commodities[k].v
        out.append((np.abs(y - x).sum() * v, e, k, x, y))
    out.sort(key=lambda r: -r[0])
    tot = sum(r[0] for r in out)
    evs = W.events(inst, omega)
    early = [q for q in evs if 0 < q.onset < 12]
    print(f"{task} episode {n}: {len(evs)} events, {sum(q.onset <= 0 for q in evs)} carried in; starting in weeks 1-12: " + "; ".join(W.describe(inst, q) for q in early))
    print(f"flows of weeks 1-{weeks}, ours | plan '{plan}', the {top} largest differences by value (of {tot / 1e9:.2f} bn moved):")
    for val, e, k, x, y in out[:top]:
        ed = inst.edges[e]
        print(f"  {node[ed.tail]:16s}->{node[ed.head]:16s} lead {ed.tau} {K[k]:12s} {val / tot:5.1%}  ours {' '.join(f'{q:9.0f}' for q in x)} | plan {' '.join(f'{q:9.0f}' for q in y)}")


if __name__ == "__main__":
    fire.Fire(main)

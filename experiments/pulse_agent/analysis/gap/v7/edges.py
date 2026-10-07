"""How often the zero-lead edge from a terminal into a grid is cut or prohibited (the risk of holding fuel behind it).

uv run python edges.py --task=small
"""

import sys
from collections import defaultdict
from pathlib import Path

import fire
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402


def main(task="small", entropy=12345):
    refs = C.references(task, entropy)
    acc = defaultdict(lambda: defaultdict(float))
    for n in sorted(refs):
        inst, _omega, marks, _fb = C.world(task, n, entropy)
        inst = inst.at_digest(marks.instance_digest)
        node = [nd.id for nd in inst.nodes]
        chk, supply = set(inst.chokepoints), set(inst.supply_nodes)
        u, Z = np.asarray(marks.u), np.asarray(marks.prohibited)
        u_now = np.asarray(marks.u_now)
        for go, g in enumerate(inst.grids):
            ga = inst.nodes[g].grid
            for e in inst.in_edges[g]:
                ed = inst.edges[e]
                if ed.tau != 0 or ed.tail in chk or ed.tail in supply:
                    continue
                burn = sum(ga.shares[k] * ga.deliverable for k in ga.fuels if k in ed.K)
                key = f"{node[ed.tail]}->{node[g]} (u0 {ed.u0:.0f}, burn of its fuels {burn:.0f})"
                a = acc[key]
                a["weeks"] += inst.T
                a["u below nominal"] += int((u[:, e] < ed.u0 * (1 - 1e-9)).sum())
                a["u below the burn"] += int((u[:, e] < burn).sum())
                a["u zero"] += int((u[:, e] <= 0).sum())
                a["seen at nominal, week below the burn"] += int(((u_now[:, e] >= ed.u0 * (1 - 1e-9)) & (u[:, e] < burn)).sum())
                for k in ga.fuels:
                    if k in ed.K:
                        a[f"prohibited {inst.commodities[k].id}"] += int(Z[:, e, k].sum())
                a["episodes with a cut"] += int((u[:, e] < burn).any())
    print(f"{task} root {entropy}, {len(refs)} episodes")
    for key, a in acc.items():
        print(f"  {key}: " + "; ".join(f"{k} {v:.0f}" for k, v in a.items()))


if __name__ == "__main__":
    fire.Fire(main)

"""Where a fuel that comes straight from a source ran short at a grid, by episode, and what it burnt in slack before.

uv run python nuc.py --task=full
"""

import pickle
import sys
from pathlib import Path

import fire
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402
import describe as Dsc  # noqa: E402


def main(task="full", tag="v6", entropy=12345):
    refs = C.references(task, entropy)
    w = Dsc.weights(task, sorted(refs))
    for n in sorted(refs):
        d = pickle.loads((HERE / "traj" / f"{tag}_{task}_{entropy}_{n}.pkl").read_bytes())
        inst, _omega, marks, _fb = C.world(task, n, entropy)
        inst = inst.at_digest(marks.instance_digest)
        node = [nd.id for nd in inst.nodes]
        chk, supply = set(inst.chokepoints), set(inst.supply_nodes)
        Gbar = np.asarray(marks.G_bar)
        for go, g in enumerate(inst.grids):
            ga = inst.nodes[g].grid
            for k in ga.fuels:
                if k == ga.rationed or any(k in inst.edges[e].K and inst.edges[e].tau == 0 and inst.edges[e].tail not in chk and inst.edges[e].tail not in supply for e in inst.in_edges[g]):
                    continue
                s = inst.slot_index[(g, k)]
                short, weeks, first, slack_burn, shed_then = 0.0, 0, None, 0.0, 0.0
                for ti, r in enumerate(d["records"]):
                    cap = ga.shares[k] * float(Gbar[ti, go])
                    burn = float(r.segment.get((go, k), 0.0))
                    shed = float(r.shed[go])
                    if shed > 1e-9 * float(marks.y_bar[ti, go]):
                        if cap - burn > 1e-6 * cap and float(r.stock[s]) < 1e-6 * cap:  # the stock ran out
                            short += cap - burn
                            shed_then += min(shed, cap - burn)
                            weeks += 1
                            first = first or ti + 1
                    elif first is None:
                        null_cap = ga.shares.get(None, 0.0) * float(Gbar[ti, go])
                        load = float(r.segment.get((go, None), 0.0)) / null_cap if null_cap > 0 else 1.0
                        slack_burn += burn / load * (1 - load) if load > 0 else 0.0  # what full load would have burnt more
                if weeks:
                    print(f"{task} episode {n} level {refs[n]['stratum']} {node[g]} {inst.commodities[k].id}: out of stock in {weeks} shedding weeks from week {first}, {short:,.0f} GWh short ({w[n] * ga.voll * shed_then:.4f} of the score as shed); weeks of full burn the slack saved before that {slack_burn / (ga.shares[k] * ga.deliverable):.2f} ({slack_burn:,.0f} GWh, {w[n] * ga.voll * slack_burn:.4f})")


if __name__ == "__main__":
    fire.Fire(main)

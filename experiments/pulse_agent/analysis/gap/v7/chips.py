"""The goods and the chip routes of an instance: values, disposal fees, storage, freight and penalties.

uv run python chips.py --task=small
"""

import sys
from pathlib import Path

import fire

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402


def main(task="small"):
    inst, _omega, marks, _fb = C.world(task, 0)
    inst = inst.at_digest(marks.instance_digest)
    node = [nd.id for nd in inst.nodes]
    K = [c.id for c in inst.commodities]
    for k, c in enumerate(inst.commodities):
        print(f"good {k} {c.id}: v {c.v:,.0f} disposal {c.disposal_cost:,.0f} pool {c.pool} override {c.override}")
    for d in inst.demands:
        print(f"demand {node[d.node]} {K[d.k]}: dbar {d.dbar:,.0f} pi {d.pi:,.0f} backlog {d.backlog}")
    for o in inst.osats:
        oa = inst.nodes[o].osat
        print(f"osat {node[o]}: thr {oa.thr:,.0f} tau {oa.tau} packages {[(K[a], K[b]) for a, b in oa.packages.items()]}")
        for k in list(oa.packages) + list(oa.packages.values()):
            sl = inst.stock_slots[inst.slot_index[(o, k)]]
            print(f"   store {K[k]}: storage {sl.storage:,.0f} holding {sl.holding} salvage {sl.salvage}")
        for e in inst.in_edges[o]:
            ed = inst.edges[e]
            print(f"   in  {node[ed.tail]} tau {ed.tau} u0 {ed.u0} c0 {ed.c0} goods {[K[k] for k in ed.K]}")
        for e in inst.out_edges[o]:
            ed = inst.edges[e]
            print(f"   out {node[ed.head]} tau {ed.tau} u0 {ed.u0} c0 {ed.c0} goods {[K[k] for k in ed.K]}")
    for f in inst.fabs:
        fa = inst.nodes[f].fab
        sl = inst.stock_slots[inst.slot_index[(f, fa.product)]]
        print(f"fab {node[f]}: cap0 {fa.cap0:,.0f} tau {fa.tau} product {K[fa.product]} store {sl.storage:,.0f} holding {sl.holding} salvage {sl.salvage}; out " + ", ".join(f"{node[inst.edges[e].head]} (tau {inst.edges[e].tau}, u0 {inst.edges[e].u0}, c0 {inst.edges[e].c0})" for e in inst.out_edges[f]))


if __name__ == "__main__":
    fire.Fire(main)

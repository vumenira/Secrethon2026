"""Print the grids of a task's instance: fuels, shares, the fuel-free segment, feeders, storage, fabs.

uv run python inst.py --task=small
"""

import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common as C  # noqa: E402


def main(task="small", n=0):
    inst, omega, marks, _fb = C.world(task, n)
    inst = inst.at_digest(marks.instance_digest)
    K = inst.commodities
    name = lambda v: inst.nodes[v].id  # noqa: E731
    print(f"{task}: T {inst.T}, psi {inst.params.psi}, nodes {len(inst.nodes)}, edges {len(inst.edges)}, slots {len(inst.action_slots)}")
    chk, supply = set(inst.chokepoints), set(inst.supply_nodes)
    for go, g in enumerate(inst.grids):
        ga = inst.nodes[g].grid
        fabs = [fo for fo in inst.grid_fabs[go]]
        draw = sum(inst.nodes[inst.fabs[fo]].fab.cap0 * inst.nodes[inst.fabs[fo]].fab.e for fo in fabs)
        print(
            f"\n{name(g)} ({go}): base {ga.base_load:.0f} deliverable {ga.deliverable:.0f} sum shares {sum(ga.shares.values()):.4f}"
            f" margin {ga.deliverable * sum(ga.shares.values()) - ga.base_load:.1f} fab draw {draw:.1f} voll {ga.voll:.0f} pri {ga.priority}"
            f" rationed {K[ga.rationed].id if ga.rationed is not None else None} null share {ga.shares.get(None, 0.0):.4f}"
        )
        for fo in fabs:
            fa = inst.nodes[inst.fabs[fo]].fab
            s_in = inst.stock_slots[inst.slot_index[(inst.fabs[fo], fa.input)]]
            s_out = inst.stock_slots[inst.slot_index[(inst.fabs[fo], fa.product)]]
            print(f"   fab {name(inst.fabs[fo])}: cap0 {fa.cap0:.0f} e {fa.e:.6f} draw {fa.cap0 * fa.e:.1f} tau {fa.tau} in {K[fa.input].id} storage {s_in.storage:.0f} out storage {s_out.storage:.0f}")
        for k in ga.fuels:
            s = inst.stock_slots[inst.slot_index[(g, k)]]
            burn = ga.shares[k] * ga.deliverable
            thr = inst.params.psi * ga.ibar[k] if k == ga.rationed else 0.0
            i0 = dict(((nd, kk), q) for nd, kk, q in inst.initial_state.stock).get((g, k), float("nan"))
            print(
                f"   fuel {K[k].id}: share {ga.shares[k]:.4f} burn {burn:.1f} ibar {ga.ibar.get(k, float("nan")):.0f} thr {thr:.0f} storage {s.storage:.0f}"
                f" ({s.storage / burn:.1f} wk) init {i0:.0f} ({i0 / burn:.1f} wk) hold {s.holding:.4g} salv {s.salvage:.4g} v {K[k].v:.4g}"
            )
            for e in inst.in_edges[g]:
                ed = inst.edges[e]
                if k not in ed.K:
                    continue
                tail_kind = "chk" if ed.tail in chk else "supply" if ed.tail in supply else "node"
                ts = inst.slot_index.get((ed.tail, k))
                st = inst.stock_slots[ts].storage if ts is not None else float("nan")
                print(f"      in edge {e} from {name(ed.tail)} ({tail_kind}) tau {ed.tau} u0 {ed.u0:.0f} c0 {ed.c0:.4g} tail storage {st:.0f}")


if __name__ == "__main__":
    fire.Fire(main)

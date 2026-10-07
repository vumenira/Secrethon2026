"""Describe a task's instance: goods, grids, demands, fabs, lead times."""

import fire
import numpy as np


def main(task="full"):
    from shockbench_flow.hosting.tasks import task_generator

    inst, _params = task_generator(task)
    K, N, S, E = inst.commodities, inst.nodes, inst.stock_slots, inst.edges
    print(f"{task}: T {inst.T}, nodes {len(N)}, edges {len(E)}, lanes {len(inst.lanes)}, slots {len(S)}, "
          f"action slots {len(inst.action_slots)}, demands {len(inst.demands)}, grids {len(inst.grids)}, "
          f"fabs {len(inst.fabs)}, osats {len(inst.osats)}, chokepoints {len(inst.chokepoints)}, supply {len(inst.supply_nodes)}")
    print("commodities:")
    for k, c in enumerate(K):
        sal = sorted({round(s.salvage, 4) for s in S if s.k == k and s.node not in inst.supply_nodes})
        hold = sorted({round(s.holding, 6) for s in S if s.k == k and s.node not in inst.supply_nodes and s.node not in inst.chokepoint_ordinal})
        print(f"  {k} {c.id:14s} unit {c.unit:10s} pool {c.pool} v {c.v:12.2f} override {c.override} disposal {c.disposal_cost:.2f}"
              f" salvage {sal[:4]}{'...' if len(sal) > 4 else ''} holding/wk {hold[:4]}{'...' if len(hold) > 4 else ''}")
    types = {}
    for n in N:
        types[n.type] = types.get(n.type, 0) + 1
    print("node types:", types)
    print("grids:")
    for go, g in enumerate(inst.grids):
        ga = N[g].grid
        fuels = []
        for k in ga.fuels:
            s = S[inst.slot_index[(g, k)]]
            burn = ga.shares[k] * ga.deliverable
            fuels.append(f"{K[k].id}: share {ga.shares[k]:.2f} burn {burn:.0f}/wk storage {s.storage:.0f} ({s.storage / burn:.1f} wk)"
                         f" ibar {ga.ibar.get(k, 0):.0f} days {ga.days_cover.get(k, 0):.0f} hold {s.holding:.2f} salv {s.salvage:.1f}")
        feeders = []
        for e in inst.in_edges[g]:
            ed = E[e]
            feeders.append(f"{N[ed.tail].id}({N[ed.tail].type},tau{ed.tau},K{list(ed.K)})")
        print(f"  {go} {N[g].id:12s} base {ga.base_load:.0f} deliv {ga.deliverable:.0f} voll {ga.voll:.0f} pri {ga.priority}"
              f" rationed {None if ga.rationed is None else K[ga.rationed].id} unmodelled {ga.shares.get(None, 0):.2f}"
              f" fabs {[N[inst.fabs[f]].id for f in inst.grid_fabs[go]]}")
        for f in fuels:
            print("       ", f)
        print("        in:", "; ".join(feeders))
    print("demands:")
    for do, d in enumerate(inst.demands):
        print(f"  {do} {N[d.node].id:14s} {K[d.k].id:12s} dbar {d.dbar:10.1f} pi {d.pi:12.1f} backlog {d.backlog} pi*dbar/wk {d.pi * d.dbar / 1e9:.3f} bn  v {K[d.k].v:.1f}")
    print("fabs:")
    for fo, f in enumerate(inst.fabs):
        fa = N[f].fab
        print(f"  {fo} {N[f].id:14s} cap0 {fa.cap0:.0f} e {fa.e:.4f} grid {None if fa.grid is None else N[fa.grid].id} cls {fa.cls} in {K[fa.input].id} out {K[fa.product].id} tau {fa.tau} w_scr {fa.w_scr}")
    print("osats:")
    for oo, o in enumerate(inst.osats):
        oa = N[o].osat
        print(f"  {oo} {N[o].id:14s} thr {oa.thr:.0f} tau {oa.tau} packages {[(K[a].id, K[b].id) for a, b in oa.packages.items()]}")
    taus = np.array([e.tau for e in E if not e.coupling])
    print("edge tau histogram:", {int(t): int((taus == t).sum()) for t in np.unique(taus)})
    from shockbench_flow.policies.naive_parts import cached_plan  # noqa: F401
    total_hold = sum(1 for s in S if s.node not in inst.chokepoint_ordinal)
    print("plain slots:", total_hold)
    # storage in weeks of demand at sinks
    print("sink slots:")
    for do, d in enumerate(inst.demands):
        s = S[inst.slot_index[(d.node, d.k)]]
        print(f"  {N[d.node].id:14s} {K[d.k].id:12s} storage {s.storage:.0f} ({s.storage / max(d.dbar, 1e-9):.1f} wk) holding {s.holding:.3f} salvage {s.salvage:.2f}")


if __name__ == "__main__":
    fire.Fire(main)

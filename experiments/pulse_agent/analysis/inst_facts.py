"""Print what the reserve needs to know about a task's instance (scratch, read-only)."""

import sys

from shockbench_flow.hosting.tasks import task_generator


def main(task):
    inst, _params = task_generator(task)
    K, S, E, N = inst.commodities, inst.stock_slots, inst.edges, inst.nodes
    print(f"== {task}: T {inst.T}, nodes {len(N)}, edges {len(E)}, slots {len(S)}, psi {inst.params.psi}")
    print("commodities:")
    for k, c in enumerate(K):
        print(f"  {k} {c.id:12s} pool {c.pool} v {c.v:12.1f} override {c.override} disposal {c.disposal_cost}")
    i0 = {(n, k): q for n, k, q in inst.initial_state.stock}
    supply = set(inst.supply_nodes)
    for go, g in enumerate(inst.grids):
        ga = N[g].grid
        fabs = [N[inst.fabs[fo]].id for fo in inst.grid_fabs[go]]
        es = [N[inst.fabs[fo]].fab.e for fo in inst.grid_fabs[go]]
        print(
            f"grid {go} {N[g].id}: deliverable {ga.deliverable:.1f} base {ga.base_load:.1f} ({ga.base_load / ga.deliverable:.3f})"
            f" voll {ga.voll:.4g} priority {ga.priority} rationed {ga.rationed} fabs {fabs} e {es}"
        )
        print(f"   shares { {('null' if k is None else K[k].id): round(v, 4) for k, v in ga.shares.items()} }")
        for k in ga.fuels:
            s = inst.slot_index[(g, k)]
            st = S[s]
            burn = ga.shares[k] * ga.deliverable
            ins = [
                (E[e].id, N[E[e].tail].id, E[e].tau, E[e].u0, round(E[e].c0, 1))
                for e in range(len(E))
                if E[e].head == g and k in E[e].K
            ]
            outs = [E[e].id for e in range(len(E)) if E[e].tail == g and k in E[e].K]
            print(
                f"   fuel {K[k].id:8s} burn/wk {burn:10.2f} storage {st.storage / burn:6.2f} wk  i0 {i0.get((g, k), 0.0) / burn:6.2f} wk"
                f"  ibar {ga.ibar.get(k, 0.0) / burn:5.2f} wk  holding {st.holding:.2f} salvage {st.salvage:.1f}"
            )
            print(f"      in-edges (id, tail, tau, u0, c0): {ins}")
            print(f"      sum u0 in / burn: {sum(i[3] or 0 for i in ins) / burn:.2f}   out-edges: {outs}")
    print("fabs:")
    for fo, f in enumerate(inst.fabs):
        fa = N[f].fab
        print(f"  {N[f].id}: cap0 {fa.cap0} e {fa.e} grid {None if fa.grid is None else N[fa.grid].id} tau {fa.tau} in {K[fa.input].id} out {K[fa.product].id}")
    print("demands:")
    for d in inst.demands[:40]:
        print(f"  {N[d.node].id} {K[d.k].id}: dbar {d.dbar:.1f} pi {d.pi:.1f} backlog {d.backlog}")
    print("stock slots (non-grid, non-supply, non-chokepoint):")
    chk = set(inst.chokepoints)
    for s, st in enumerate(S):
        if st.node in chk or st.node in supply or N[st.node].grid is not None:
            continue
        print(
            f"  {N[st.node].id:14s} {K[st.k].id:10s} type {N[st.node].type:8s} storage {st.storage:12.1f} i0 {i0.get((st.node, st.k), 0.0):12.1f}"
            f" holding {st.holding:8.2f} salvage {st.salvage:9.1f} v {K[st.k].v:10.1f}"
        )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "small")

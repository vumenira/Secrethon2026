"""Facts of the Full instance for the autopsy (no play): nodes, edges, grids, fabs, demands, lanes."""
import fire


def main(task="full"):
    from shockbench_flow.hosting.tasks import task_generator

    inst, params = task_generator(task)
    N, E, K, S = inst.nodes, inst.edges, inst.commodities, inst.stock_slots
    nid = [n.id for n in N]
    kid = [c.id for c in K]
    print("T", inst.T, "nodes", len(N), "edges", len(E), "commodities", kid)
    print("commodity v:", {c.id: c.v for c in K}, "override:", [c.id for c in K if c.override])
    print("\nNODES")
    for i, n in enumerate(N):
        slots = [(kid[s.k], (None if s.storage is None else round(s.storage)), round(getattr(s, "supply", 0) or 0), s.holding, s.salvage) for s in S if s.node == i]
        print(i, n.id, n.type, getattr(n, "region", ""), "slots(k,storage,supply,hold,salv):", slots)
    print("\nEDGES (e, id, tail->head, tau, u0, K, c0)")
    for e, ed in enumerate(E):
        print(e, ed.id, f"{nid[ed.tail]}->{nid[ed.head]}", ed.tau, ed.u0, [kid[k] for k in ed.K], getattr(ed, "c0", None))
    print("\nGRIDS")
    for go, g in enumerate(inst.grids):
        ga = N[g].grid
        print(go, nid[g], "deliverable", ga.deliverable, "voll", ga.voll, "priority", ga.priority, "rationed", None if ga.rationed is None else kid[ga.rationed],
              "shares", {(kid[k] if k is not None else None): v for k, v in ga.shares.items()}, "ibar", {kid[k]: v for k, v in ga.ibar.items()},
              "fabs", [nid[inst.fabs[fo]] for fo in inst.grid_fabs[go]])
    print("\nFABS")
    for fo, f in enumerate(inst.fabs):
        fa = N[f].fab
        print(fo, nid[f], "cap0", fa.cap0, "e", fa.e, "draw", fa.cap0 * fa.e, "in", kid[fa.input], "out", kid[fa.product], "tau", fa.tau, "grid", None if fa.grid is None else nid[fa.grid])
    print("\nOSATS")
    for oo, o in enumerate(inst.osats):
        oa = N[o].osat
        print(oo, nid[o], "thr", getattr(oa, "thr", None), "tau", oa.tau, {kid[a]: kid[b] for a, b in oa.packages.items()})
    print("\nDEMANDS")
    for do, d in enumerate(inst.demands):
        print(do, nid[d.node], kid[d.k], "pi", d.pi, "backlog", d.backlog)
    print("\nCHOKEPOINTS", [nid[c] for c in inst.chokepoints])
    print("fleet", getattr(inst, "fleet", None))
    print("psi", inst.params.psi)
    print("\nLANES")
    for li, lane in enumerate(inst.lanes):
        print(li, lane)


if __name__ == "__main__":
    fire.Fire(main)


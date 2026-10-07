"""What LNG and crude share on their way to the grids: pools, straits, edges."""

import fire


def main(task="small"):
    from shockbench_flow.hosting.tasks import task_generator

    inst, _ = task_generator(task)
    K = inst.commodities
    print(task, "commodities:", [(c.id, c.pool, "override" if c.override else "") for c in K])
    fuels = [k for k, c in enumerate(K) if c.id in ("lng", "crude", "nucfuel")]
    print("straits:", [(inst.nodes[c].id, [round(v, 0) for v in inst.nodes[c].chokepoint.kappa0]) for c in inst.chokepoints])
    for k in fuels:
        print(f"-- {K[k].id}: edges that carry it (tail -> head, lead, capacity, other goods on the edge)")
        for e, ed in enumerate(inst.edges):
            if k in ed.K and not ed.coupling:
                others = [K[j].id for j in ed.K if j != k]
                print(f"   {inst.nodes[ed.tail].id:16s} -> {inst.nodes[ed.head].id:12s} lead {ed.tau} cap {ed.u0!s:>10s} pool {ed.pool} shares the edge with {others}")
    print("lanes through straits for fuels:")
    for li, ln in enumerate(inst.lanes):
        ks = [K[k].id for k in inst.lane_K[li] if k in fuels]
        if ks:
            print(f"   {ln.id:28s} {ks} edges {[inst.nodes[inst.edges[e].tail].id + '>' + inst.nodes[inst.edges[e].head].id for e in ln.edges]}")
    try:
        print("fleet caps (dup items):", inst.dup_items if hasattr(inst, "dup_items") else None)
    except Exception as err:  # noqa: BLE001
        print("fleet:", err)


if __name__ == "__main__":
    fire.Fire(main)

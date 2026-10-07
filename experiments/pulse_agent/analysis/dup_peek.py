"""Which edges enter the fleet row, and what the wafer edges into fabs look like (slots, lanes, tails)."""

import fire


def main(task="full"):
    from shockbench_flow.dynamics.clip import dup_terms, fleet_caps
    from shockbench_flow.hosting.tasks import task_generator

    inst, _ = task_generator(task)
    N, E, K = inst.nodes, inst.edges, inst.commodities
    terms = dup_terms(inst)
    print(f"{task}: {len(inst.dup_items)} dup items over {len(terms)} edges; fleet caps {fleet_caps(inst)}")
    modes = {}
    for e in terms:
        modes[E[e].mode] = modes.get(E[e].mode, 0) + 1
    print("  modes of the edges in the fleet row:", modes, " all edge modes:", {m: sum(1 for ed in E if ed.mode == m) for m in {ed.mode for ed in E}})
    wafer = inst.commodity_index["wafer"]
    slots = {}
    for s, (e, k, lane) in enumerate(inst.action_slots):
        slots.setdefault((e, k), []).append((s, lane))
    n_direct = n_dup = 0
    for fo, f in enumerate(inst.fabs):
        fa = N[f].fab
        row = []
        for e in inst.in_edges[f]:
            ed = E[e]
            if wafer not in ed.K:
                continue
            kind = "dup" if e in terms else "plain"
            tail = N[ed.tail].type
            ss = slots.get((e, wafer), [])
            row.append(f"{ed.mode}:{tail[:4]}:{kind}:tau{ed.tau}:u{ed.u0 / fa.cap0:.2f}:slots{len(ss)}:K{len(ed.K)}:c{ed.c0:.0f}")
            if tail != "chokepoint":
                n_direct += kind == "plain"
                n_dup += kind == "dup"
        plain = sum(E[e].u0 for e in inst.in_edges[f] if wafer in E[e].K and N[E[e].tail].type != "chokepoint" and e not in terms)
        print(f"  {N[f].id:18s} direct plain capacity {plain / fa.cap0:.2f} of fab capacity; wafer storage {inst.stock_slots[inst.slot_index[(f, fa.input)]].storage / fa.cap0:.1f} wk; chip storage {inst.stock_slots[inst.slot_index[(f, fa.product)]].storage / fa.cap0:.1f} wk | " + " ".join(row))
    print("  direct edges: plain", n_direct, "dup", n_dup)
    lanes_to_fab = sum(1 for ln in inst.lanes if N[E[ln.edges[-1]].head].type == "fab")
    print("  lanes ending at a fab:", lanes_to_fab, "of", len(inst.lanes))


if __name__ == "__main__":
    fire.Fire(main)

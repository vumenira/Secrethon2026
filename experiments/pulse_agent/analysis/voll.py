import sys
from shockbench_flow.hosting.tasks import task_generator
for task in ("full", "small"):
    inst, _ = task_generator(task)
    print(task, {inst.nodes[g].id: (inst.nodes[g].grid.voll, {inst.commodities[k].id: round(s, 3) for k, s in inst.nodes[g].grid.shares.items() if k is not None}, round(inst.nodes[g].grid.shares.get(None, 0.0), 3)) for g in inst.grids})
    for s, slot in enumerate(inst.stock_slots):
        if slot.node in inst.supply_nodes and inst.commodities[slot.k].id in ("lng", "crude"):
            print("   source", inst.nodes[slot.node].id, inst.commodities[slot.k].id, "weekly supply", slot.supply, "storage", slot.storage, "out edges", [(inst.nodes[inst.edges[e].head].id, inst.edges[e].u0) for e in inst.out_edges[slot.node] if slot.k in inst.edges[e].K])

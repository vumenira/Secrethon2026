from shockbench_flow.hosting.tasks import task_generator
inst, _ = task_generator("full")
N, E, K = inst.nodes, inst.edges, inst.commodities
for s, (e, k, lane) in enumerate(inst.action_slots):
    if K[k].id in ("lng", "crude"):
        print(s, E[e].id, K[k].id, None if lane is None else inst.lanes[lane].id)
print(len(inst.action_slots), "override slots", len(inst.override_slots), inst.override_slots[:6])

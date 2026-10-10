from shockbench_flow.hosting.tasks import task_generator
inst, _ = task_generator("full")
K = inst.commodities
for c in inst.chokepoints:
    ch = inst.nodes[c].chokepoint
    print(inst.nodes[c].id, {K[k].id: row for k, row in ch.queue_holding.items()}, "k_c", getattr(ch, "k", None), "mu", getattr(ch, "mu", None))
    print("    ", {f: getattr(ch, f) for f in ch.__dataclass_fields__ if f != "queue_holding"})

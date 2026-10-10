"""Print what was dispatched toward a node in some weeks of a census episode, by slot (no play)."""
import pickle
from pathlib import Path

import fire

HERE = Path(__file__).resolve().parent


def main(root=12345, ep=13, node=18, weeks="30,31,32", name="s13_census", task="full"):
    from shockbench_flow.hosting.tasks import task_generator

    inst, _ = task_generator(task)
    d = pickle.load(open(HERE / name / f"{task}-e{root}" / f"ep{ep}.pkl", "rb"))
    names = [n.id for n in inst.nodes]
    kn = [c.id for c in inst.commodities]
    weeks = [int(w) for w in (weeks.split(",") if isinstance(weeks, str) else weeks)]
    for s, (e, k, lane) in enumerate(inst.action_slots):
        path = [e] if lane is None else list(inst.lanes[lane].edges)
        if inst.edges[path[-1]].head != node:
            continue
        route = names[inst.edges[path[0]].tail] + "".join(">" + names[inst.edges[x].head] for x in path)
        caps = [round(inst.edges[x].u0) for x in path]
        taus = [inst.edges[x].tau for x in path]
        qty = [round(d["executed"][w - 1].get(s, 0.0)) for w in weeks]
        print(f"slot {s:3d} {kn[k]:8s} {route:60s} u0 {caps} tau {taus} sent {qty}")
    for k in range(3):
        if (node, k) in inst.slot_index:
            print(kn[k], "stock at node end of weeks", [round(float(d["stock"][w - 1][inst.slot_index[(node, k)]])) for w in weeks])


if __name__ == "__main__":
    fire.Fire(main)

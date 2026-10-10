"""A fab week by week in a census episode: wafers at hand, lots started, wafers sent toward it, its chip store (no play)."""
import pickle
from pathlib import Path

import fire

HERE = Path(__file__).resolve().parent


def main(root=12345, ep=8, fab=6, first=1, last=60, name="s13_census", task="full"):
    from shockbench_flow.hosting.tasks import task_generator

    inst, _ = task_generator(task)
    d = pickle.load(open(HERE / name / f"{task}-e{root}" / f"ep{ep}.pkl", "rb"))
    names = [n.id for n in inst.nodes]
    f = inst.fabs[fab]
    fa = inst.nodes[f].fab
    k = fa.input
    wslot, cslot = inst.slot_index[(f, k)], inst.slot_index[(f, fa.product)]
    print(names[f], "cap0", round(fa.cap0), "wafer store", round(inst.stock_slots[wslot].storage), "chip store", round(inst.stock_slots[cslot].storage))
    toward = {}
    for s, (e, kk, lane) in enumerate(inst.action_slots):
        path = [e] if lane is None else list(inst.lanes[lane].edges)
        if kk == k and inst.edges[path[-1]].head == f:
            toward[s] = names[inst.edges[path[0]].tail] + ("" if lane is None else "~") + f"(tau {sum(inst.edges[x].tau for x in path)})"
    mats = sorted({inst.edges[e].tail for s in toward for e in [inst.action_slots[s][0]]})
    out_slots = [s for s, (e, kk, lane) in enumerate(inst.action_slots) if kk == fa.product and inst.edges[e].tail == f]
    print("ways in:", toward)
    print("material nodes:", [names[m] for m in mats])
    for t in range(first, last + 1):
        rec_sent = {toward[s]: round(d["executed"][t - 1].get(s, 0.0)) for s in toward if d["executed"][t - 1].get(s, 0.0) > 0.5}
        mat = [round(float(d["stock"][t - 1][inst.slot_index[(m, k)]])) for m in mats]
        chips_out = round(sum(d["executed"][t - 1].get(s, 0.0) for s in out_slots))
        print(t, "lots", round(float(d["lots"][t - 1][fab])), "wafers end", round(float(d["stock"][t - 1][wslot])),
              "chips end", round(float(d["stock"][t - 1][cslot])), "chips out", chips_out, "sent", rec_sent, "mat stock end", mat)


if __name__ == "__main__":
    fire.Fire(main)

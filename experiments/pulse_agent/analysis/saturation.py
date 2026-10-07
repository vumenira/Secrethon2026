"""How full the chip stocks downstream of the fabs are, per product family, per episode (from saved trajectories)."""

from pathlib import Path

import fire
import numpy as np


HERE = Path(__file__).resolve().parent


def main(name="ft_pulse_r4", episodes=16, weeks=(8, 13, 20, 26, 39, 52, 65, 78, 91)):
    from shockbench_flow.hosting.tasks import task_generator

    inst, _ = task_generator("full")
    N, K, S = inst.nodes, inst.commodities, inst.stock_slots
    fam = {}
    for o in inst.osats:
        for raw, pk in N[o].osat.packages.items():
            fam[raw] = fam[pk] = raw
    demand = {(d.node, d.k) for d in inst.demands if d.dbar > 0}
    slots = {}
    for s, sl in enumerate(S):
        if sl.k in fam and sl.storage and sl.node not in inst.chokepoint_ordinal:
            if N[sl.node].type == "sink" and (sl.node, sl.k) not in demand:
                continue
            slots.setdefault(fam[sl.k], []).append(s)
    for f, ss in slots.items():
        print(f"family {K[f].id}: {len(ss)} slots, storage {sum(S[s].storage for s in ss) / 1e6:.1f} M")
    print("share of the family's storage in use at weeks", weeks)
    for n in range(episodes):
        a = np.load(HERE / "gap" / f"{name}_{n}.npz", allow_pickle=True)
        line = f"  episode {n:2d} "
        for f, ss in slots.items():
            cap = sum(S[s].storage for s in ss)
            line += f"| {K[f].id}: " + " ".join(f"{a['stock'][t - 1, ss].sum() / cap:4.2f}" for t in weeks) + " "
        print(line)


if __name__ == "__main__":
    fire.Fire(main)

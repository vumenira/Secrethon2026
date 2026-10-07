"""Chip stocks along the chain and demand served per sink in one episode of one agent (from the saved trajectory)."""

from pathlib import Path

import fire
import numpy as np


HERE = Path(__file__).resolve().parent


def main(name="ft_pulse_r4", episode=14, weeks=(13, 26, 39, 52, 65, 78, 91)):
    from shockbench_flow.hosting.tasks import task_generator

    inst, _ = task_generator("full")
    N, K, S = inst.nodes, inst.commodities, inst.stock_slots
    A = np.load(HERE / "gap" / f"{name}_{episode}.npz", allow_pickle=True)
    O = np.load(HERE / "gap" / f"oracle_{episode}.npz", allow_pickle=True)
    print(f"episode {episode} {name}: stock as a share of storage at weeks {weeks}")
    for s, sl in enumerate(S):
        if sl.k < 4 or sl.node in inst.chokepoint_ordinal or not sl.storage:
            continue
        row = [A["stock"][t - 1, s] / sl.storage for t in weeks]
        if max(row) > 0.5:
            print(f"  {N[sl.node].id:18s} {K[sl.k].id:13s} storage {sl.storage:9.0f}  " + " ".join(f"{x:5.2f}" for x in row))
    print("demand served share per sink and good, by blocks of 13 weeks (agent | clairvoyant):")
    T = A["served"].shape[0]
    for do, d in enumerate(inst.demands):
        if d.dbar <= 0:
            continue
        ra = [A["served"][lo:lo + 13, do].sum() / max(A["demand"][lo:lo + 13, do].sum(), 1e-9) for lo in range(0, T, 13)]
        ro = [O["served"][lo:lo + 13, do].sum() / max(A["demand"][lo:lo + 13, do].sum(), 1e-9) for lo in range(0, T, 13)]
        print(f"  {N[d.node].id:9s} {K[d.k].id:9s} " + " ".join(f"{x:4.2f}" for x in ra) + "  |  " + " ".join(f"{x:4.2f}" for x in ro))
    print("disposal by good (thousand units):", {K[k].id: round(float(sum(A['disposal'][:, s].sum() for s, sl in enumerate(S) if sl.k == k)) / 1e3) for k in range(3, len(K))})


if __name__ == "__main__":
    fire.Fire(main)

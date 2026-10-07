"""Clairvoyant against agent: base load shed per grid and lots per fab in GWh terms, to see whether the clairvoyant plan
runs fabs by shedding base load (which the simulator's base-first rule forbids) or by delivering the fuel."""

from pathlib import Path

import fire
import numpy as np


HERE = Path(__file__).resolve().parent


def main(name="f_e10_w4", episodes=16, first=14):
    from shockbench_flow.hosting.tasks import task_generator

    inst, _ = task_generator("full")
    N = inst.nodes
    tot = {"a_shed": 0.0, "o_shed": 0.0, "o_fab_while_shed": 0.0, "o_fab": 0.0, "a_fab": 0.0}
    per_grid = {}
    for n in range(episodes):
        a = np.load(HERE / "gap" / f"{name}_{n}.npz", allow_pickle=True)
        o = np.load(HERE / "gap" / f"oracle_{n}.npz", allow_pickle=True)
        for go, g in enumerate(inst.grids):
            ga = N[g].grid
            fabs = inst.grid_fabs[go]
            if not fabs:
                continue
            e = np.array([N[inst.fabs[fo]].fab.e for fo in fabs])
            sl = slice(first - 1, None)
            o_shed = o["shed"][sl, go] / ga.voll  # GWh
            a_shed = a["shed"][sl, go] / ga.voll
            o_fab = (o["lots_started"][sl][:, fabs] * e).sum(axis=1)  # GWh at R = 1 (a lower bound of the energy drawn)
            a_fab = (a["lots_started"][sl][:, fabs] * e).sum(axis=1)
            both = np.where(o_shed > 1e-6, o_fab, 0.0)  # fab energy in weeks the clairvoyant plan also sheds base load
            d = per_grid.setdefault(N[g].id, np.zeros(6))
            d += [a_shed.sum(), o_shed.sum(), a_fab.sum(), o_fab.sum(), both.sum(), np.mean(o_shed > 1e-6)]
    print(f"weeks {first}..104, mean per episode, GWh: base load shed (agent, clairvoyant); fab energy (agent, clairvoyant);"
          " clairvoyant fab energy in weeks it sheds base load at that grid (and its share); share of weeks it sheds")
    total = np.zeros(6)
    for gid, d in per_grid.items():
        d = d / episodes
        total += d
        print(f"  {gid:9s} shed {d[0]:9.0f} {d[1]:9.0f}   fab {d[2]:8.0f} {d[3]:8.0f}   while shedding {d[4]:8.0f} ({d[4] / max(d[3], 1e-9):.2f})  weeks {d[5]:.2f}")
    d = total
    print(f"  {'all':9s} shed {d[0]:9.0f} {d[1]:9.0f}   fab {d[2]:8.0f} {d[3]:8.0f}   while shedding {d[4]:8.0f} ({d[4] / max(d[3], 1e-9):.2f})")


if __name__ == "__main__":
    fire.Fire(main)

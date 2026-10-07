"""An estimate without foresight of what trimming the gas burn in slack weeks is worth, from the records alone.

In a week a grid sheds nothing and its fabs draw less than the margin, the simulator loads every segment by the same
factor (load < 1): the segment that needs no fuel gives less than it could and gas burns in its place. Had the gas
segment's availability been lowered by the slack (so the load factor is 1), the gas burnt would have been lower by
slack * (1 - gas share of what was available). Here that gas is banked, week by week, and spent in later weeks the
same grid shed base load, up to the room left in its gas segment and the load shed that week. Nothing of the future is
used: the bank only grows from weeks already played. What it ignores: the rationing rule (the grid's stock has to be
set a week ahead), storage at the terminal, and what the saved gas would do to the on/off calendar.

uv run python trim.py --task=small
"""

from collections import defaultdict

import fire
import numpy as np

import common as C
import describe as Dsc


def main(task="small"):
    refs = C.references(task)
    ns = sorted(refs)
    w = Dsc.weights(task, ns)
    acc = defaultdict(float)
    by_grid = defaultdict(lambda: defaultdict(float))
    for n in ns:
        cal = C.Cal(task, n)
        inst, marks, T = cal.inst, cal.marks, cal.T
        d = C.load_traj(task, n)
        a = d["zpi"].reshape(T, cal.nc)
        z, _ = cal.calendar(d["zpi"])
        v = Dsc.View(cal)
        for gi, (go, fabs, _S) in enumerate(cal.grids):
            g = inst.grids[go]
            ga = inst.nodes[g].grid
            k = ga.rationed
            if k is None or None not in ga.shares:
                continue
            Gbar, ybar = np.asarray(marks.G_bar)[:, go], np.asarray(marks.y_bar)[:, go]
            null_cap = ga.shares[None] * Gbar
            cap = ga.shares[k] * Gbar
            Gn, Gk = v.col(a, ("G", go, None)), v.col(a, ("G", go, k))
            E = sum(v.col(a, ("E", fo)) for fo in fabs)
            y, shed = v.col(a, ("y", go)), v.col(a, ("ysh", go))
            bank = peak = 0.0
            name = v.node[g]
            for t in range(T):
                if z[gi, t] > 0.5:
                    load = Gn[t] / null_cap[t] if null_cap[t] > 0 else 1.0
                    if 0 < load < 1 - 1e-9:
                        need = y[t] + E[t]
                        g_av = need / load
                        slack = g_av - need
                        av_k = Gk[t] / load
                        saved = min(slack * (1.0 - av_k / g_av), Gk[t])
                        bank += saved
                        peak = max(peak, bank)
                        by_grid[name]["gas saved in slack weeks, GWh"] += saved
                        acc["gas saved in slack weeks at VOLL"] += w[n] * ga.voll * saved
                elif shed[t] > 0:
                    use = min(bank, max(cap[t] - Gk[t], 0.0), shed[t])
                    bank -= use
                    by_grid[name]["spent in later shedding weeks, GWh"] += use
                    acc["spent in later shedding weeks at VOLL"] += w[n] * ga.voll * use
            by_grid[name]["left in the bank at the end, GWh"] += bank
            by_grid[name]["largest bank of an episode, GWh"] = max(by_grid[name]["largest bank of an episode, GWh"], peak)
            acc["left in the bank at the end at VOLL"] += w[n] * ga.voll * bank
    print(f"{task}, {len(ns)} episodes, pooled score units:")
    for key, x in acc.items():
        print(f"  {key:44s} {x:.4f}")
    for name, rows in by_grid.items():
        print(f"  {name}: " + "; ".join(f"{key} {x:,.0f}" for key, x in rows.items()))


if __name__ == "__main__":
    fire.Fire(main)

"""Where the shed base load differs between our play and an optimised vector with the same flows (fix_x) or not (cal).

Per grid and week from the records: whether the grid shed (off) or not (on), the load factor of its segments (the
simulator loads every segment pro rata: fuel burnt = available * load, and so is the output of the segment that needs
no fuel), the fabs' energy against their full draw. The same from the optimised vector. Sums are pooled in score units
(shed at VOLL) and in GWh.

uv run python slack.py --task=small --name=fix_x
"""

from collections import defaultdict

import fire
import numpy as np

import common as C
import describe as Dsc


def main(task="small", name="fix_x", only=None):
    refs = C.references(task)
    ns = sorted(refs) if only is None else [int(v) for v in (only if isinstance(only, (list, tuple)) else [only])]
    w = Dsc.weights(task, sorted(refs))
    T = C.Cal(task, ns[0]).T
    q = [0, 4, 8, 13, 26, 39, T] if task == "small" else [0, 8, 16, 26, 52, 78, T]
    P = list(zip(q, q[1:]))
    acc = defaultdict(lambda: np.zeros(len(P)))  # (grid name, quantity) -> per period, score units or GWh
    tot = defaultdict(float)
    for n in ns:
        cal = C.Cal(task, n)
        inst, marks = cal.inst, cal.marks
        d = C.load_traj(task, n)
        a = d["zpi"].reshape(T, cal.nc)
        b, _keys, z = Dsc.load_sol(task, n, name)
        v = Dsc.View(cal)
        for gi, (go, fabs, _S) in enumerate(cal.grids):
            g = inst.grids[go]
            ga = inst.nodes[g].grid
            gname = v.node[g]
            Gbar, ybar = np.asarray(marks.G_bar)[:, go], np.asarray(marks.y_bar)[:, go]
            on = z[gi] > 0.5
            null_cap = ga.shares.get(None, 0.0) * Gbar
            na, nb = v.col(a, ("G", go, None)), v.col(b, ("G", go, None))
            Ea = sum(v.col(a, ("E", fo)) for fo in fabs)
            Eb = sum(v.col(b, ("E", fo)) for fo in fabs)
            sha, shb = v.col(a, ("ysh", go)), v.col(b, ("ysh", go))
            fuel_a = sum(v.col(a, ("G", go, k)) for k in ga.fuels)
            fuel_b = sum(v.col(b, ("G", go, k)) for k in ga.fuels)
            load = np.where(null_cap > 0, na / np.maximum(null_cap, 1e-9), 1.0)
            slack = on & (load < 1 - 1e-6)
            full_draw = cal.emax[gi]
            for pi, (s, e) in enumerate(P):
                sl = slice(s, e)
                o = on[sl]
                acc[(gname, "grid-weeks")][pi] += e - s
                acc[(gname, "on weeks")][pi] += o.sum()
                acc[(gname, "on weeks with slack (load < 1)")][pi] += slack[sl].sum()
                acc[(gname, "on: free segment unused, played GWh")][pi] += (null_cap[sl] - na[sl])[o].sum()
                acc[(gname, "on: free segment unused, optimised GWh")][pi] += (null_cap[sl] - nb[sl])[o].sum()
                acc[(gname, "on: fuel burnt, played GWh")][pi] += fuel_a[sl][o].sum()
                acc[(gname, "on: fuel burnt, optimised GWh")][pi] += fuel_b[sl][o].sum()
                acc[(gname, "off: fuel burnt, played GWh")][pi] += fuel_a[sl][~o].sum()
                acc[(gname, "off: fuel burnt, optimised GWh")][pi] += fuel_b[sl][~o].sum()
                acc[(gname, "on: fab energy, played GWh")][pi] += Ea[sl][o].sum()
                acc[(gname, "on: fab energy, optimised GWh")][pi] += Eb[sl][o].sum()
                acc[(gname, "on: fabs' full draw GWh")][pi] += full_draw[sl][o].sum()
                acc[(gname, "shed, played GWh")][pi] += sha[sl].sum()
                acc[(gname, "shed, optimised GWh")][pi] += shb[sl].sum()
                acc[("all", "shed saved, score units")][pi] += w[n] * ga.voll * (sha[sl] - shb[sl]).sum()
                acc[("all", "free segment unused in on weeks at VOLL, played, score units")][pi] += w[n] * ga.voll * (null_cap[sl] - na[sl])[o].sum()
                acc[("all", "free segment unused in on weeks at VOLL, optimised, score units")][pi] += w[n] * ga.voll * (null_cap[sl] - nb[sl])[o].sum()
                acc[("all", "shed at VOLL, played, score units")][pi] += w[n] * ga.voll * sha[sl].sum()
    print(f"{task}, {len(ns)} episodes, played against {name}; periods (weeks): " + ", ".join(f"{s + 1}-{e}" for s, e in P))
    for (gname, what), vals in acc.items():
        if gname == "all":
            print(f"  {what:66s} " + " ".join(f"{x:9.4f}" for x in vals) + f"   total {vals.sum():.4f}")
    for (gname, what), vals in acc.items():
        if gname != "all":
            print(f"  {gname:8s} {what:42s} " + " ".join(f"{x:11,.0f}" for x in vals) + f"   total {vals.sum():12,.0f}")


if __name__ == "__main__":
    fire.Fire(main)

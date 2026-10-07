"""One grid of one episode week by week, played | optimised: what limited the fabs in the weeks the grid did not shed.

uv run python gridweek.py --task=small --n=25 --grid=grid_tw --name=cal
"""

import fire
import numpy as np

import common as C
import describe as Dsc


def main(task="small", n=0, grid="grid_tw", name="cal", first=1, last=None):
    cal = C.Cal(task, int(n))
    inst, marks, T = cal.inst, cal.marks, cal.T
    d = C.load_traj(task, int(n))
    a = d["zpi"].reshape(T, cal.nc)
    b, _keys, z = Dsc.load_sol(task, int(n), name)
    v = Dsc.View(cal)
    go = next(i for i, g in enumerate(inst.grids) if v.node[g] == grid)
    gi = next(i for i, (g2, _f, _s) in enumerate(cal.grids) if g2 == go)
    g = inst.grids[go]
    ga = inst.nodes[g].grid
    fabs = cal.grids[gi][1]
    Gbar, ybar = np.asarray(marks.G_bar)[:, go], np.asarray(marks.y_bar)[:, go]
    null_cap = ga.shares.get(None, 0.0) * Gbar
    feeders = {}
    for k in ga.fuels:
        for e in inst.in_edges[g]:
            ed = inst.edges[e]
            if k in ed.K and ed.tau == 0 and (ed.tail, k) in inst.slot_index:
                feeders[k] = (e, inst.slot_index[(ed.tail, k)])
    print(f"{task} episode {n}, {grid}: base load {ybar[0]:.0f}, deliverable {Gbar[0]:.0f}, the fabs' full draw {cal.emax[gi][0]:.0f} GWh a week; played | optimised ({name})")
    head = f"{'wk':>3s} on {'Gbar':>6s} {'shed':>11s} {'load':>5s} {'fab E':>9s} {'draw':>5s}"
    for fo in fabs:
        head += f" | {v.node[inst.fabs[fo]][4:14]:>10s} lots {'':>4s} {'wafers':>6s} {'cap':>4s}"
    for k in ga.fuels:
        head += f" | {v.K[k][:5]:>5s} burn {'':>5s} {'cap':>5s} {'at grid':>11s} {'at term':>11s} {'delivered':>11s}"
    print(head + "   (thousands for lots and wafers)")
    for t in range(int(first) - 1, int(last) if last else T):
        on = z[gi, t] > 0.5
        Ea = sum(v.col(a, ("E", fo))[t] for fo in fabs)
        Eb = sum(v.col(b, ("E", fo))[t] for fo in fabs)
        na = v.col(a, ("G", go, None))[t]
        load = na / null_cap[t] if null_cap[t] > 0 else float("nan")
        draw = 0.0
        line = ""
        for fo in fabs:
            f = inst.fabs[fo]
            fa = inst.nodes[f].fab
            s = inst.slot_index[(f, fa.input)]
            la, lb = v.col(a, ("p", fo))[t], v.col(b, ("p", fo))[t]
            wa = v.col(a, ("I", s))[t] + la  # wafers on hand before this week's lots
            cap = float(np.asarray(marks.alpha_bar)[t, fo] * np.asarray(marks.R)[t, fo] * fa.cap0)
            R = float(np.asarray(marks.R)[t, fo])
            draw += fa.e * min(cap, wa) / R if R > 0 else 0.0
            line += f" | {la / 1e3:10.0f}|{lb / 1e3:<4.0f} {wa / 1e3:6.0f} {cap / 1e3:4.0f}"
        out = f"{t + 1:3d} {'#' if on else '.':>2s} {Gbar[t]:6.0f} {v.col(a, ('ysh', go))[t]:5.0f}|{v.col(b, ('ysh', go))[t]:<5.0f} {load:5.2f} {Ea:4.0f}|{Eb:<4.0f} {draw:5.0f}" + line
        for k in ga.fuels:
            s = inst.slot_index[(g, k)]
            Ga, Gb = v.col(a, ("G", go, k))[t], v.col(b, ("G", go, k))[t]
            out += f" | {Ga:5.0f}|{Gb:<5.0f} {ga.shares[k] * Gbar[t]:5.0f} {v.col(a, ('I', s))[t]:5.0f}|{v.col(b, ('I', s))[t]:<5.0f}"
            if k in feeders:
                e, ts = feeders[k]
                out += f" {v.col(a, ('I', ts))[t]:5.0f}|{v.col(b, ('I', ts))[t]:<5.0f} {v.flow(a, e, k)[t]:5.0f}|{v.flow(b, e, k)[t]:<5.0f}"
            else:
                out += f" {'':11s} {'':11s}"
        print(out)


if __name__ == "__main__":
    fire.Fire(main)

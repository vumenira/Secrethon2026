"""Per episode: the pooled saving of an optimised vector by cost part; and fuel burnt by fuel in on and off weeks.

uv run python byepisode.py parts --task=small --name=cal
uv run python byepisode.py fuels --task=small --name=fix_x
"""

from collections import defaultdict

import fire
import numpy as np

import common as C
import describe as Dsc


def parts(task="small", name="cal", top=14):
    refs = C.references(task)
    ns = sorted(refs)
    w = Dsc.weights(task, ns)
    rows = []
    for n in ns:
        cal = C.Cal(task, n)
        d = C.load_traj(task, n)
        b, _keys, _z = Dsc.load_sol(task, n, name)
        pa, sa = Dsc.cost_parts(cal.model, d["zpi"])
        pb, sb = Dsc.cost_parts(cal.model, b.ravel())
        g = {p: w[n] * (pa[p].sum() - pb[p].sum()) for p in Dsc.PARTS}
        g["salvage"] = w[n] * (sb - sa)
        T = cal.T
        early = w[n] * (pa["shortage"][: T // 2].sum() - pb["shortage"][: T // 2].sum())
        rows.append((sum(g.values()), n, refs[n]["stratum"], g, early))
    tot = sum(r[0] for r in rows)
    print(f"{task}: pooled saving of {name} over played {tot:.4f}; by episode (pooled score units): total, unserved demand (of which the first half of the episode), shed, the rest")
    for key, label in (("shortage", "unserved demand"), ("shed", "shed load")):
        rows.sort(key=lambda r: -r[3][key])
        print(f"  the {top} largest in {label}:")
        cum = 0.0
        allp = sum(r[3][key] for r in rows)
        for total, n, s, g, early in rows[:top]:
            cum += g[key]
            rest = total - g["shortage"] - g["shed"]
            print(f"    episode {n:2d} level {s}: total {total:+.4f}  unserved {g['shortage']:+.4f} (first half {early:+.4f})  shed {g['shed']:+.4f}  rest {rest:+.4f}   cumulative share of the part {cum / allp:5.0%}")
        neg = sum(r[3][key] for r in rows if r[3][key] < 0)
        print(f"    all episodes {allp:+.4f}; episodes where the optimised plan pays more of it: {sum(r[3][key] < -1e-6 for r in rows)} (sum {neg:+.4f})")


def fuels(task="small", name="fix_x"):
    refs = C.references(task)
    ns = sorted(refs)
    w = Dsc.weights(task, ns)
    acc = defaultdict(lambda: np.zeros(6))  # (grid, fuel) -> on played, on optimised, off played, off optimised, end played, end optimised
    val = defaultdict(lambda: np.zeros(2))  # (grid, fuel) -> burn saved in on weeks, extra burn in off weeks, at VOLL, pooled
    for n in ns:
        cal = C.Cal(task, n)
        inst = cal.inst
        d = C.load_traj(task, n)
        a = d["zpi"].reshape(cal.T, cal.nc)
        b, _keys, z = Dsc.load_sol(task, n, name)
        v = Dsc.View(cal)
        for gi, (go, _fabs, _S) in enumerate(cal.grids):
            g = inst.grids[go]
            ga = inst.nodes[g].grid
            on = z[gi] > 0.5
            for k in list(ga.fuels) + [None]:
                Ga, Gb = v.col(a, ("G", go, k)), v.col(b, ("G", go, k))
                key = (v.node[g], "free segment" if k is None else v.K[k])
                acc[key][:4] += [Ga[on].sum(), Gb[on].sum(), Ga[~on].sum(), Gb[~on].sum()]
                if k is not None:
                    s = inst.slot_index[(g, k)]
                    acc[key][4:] += [v.col(a, ("I", s))[-1], v.col(b, ("I", s))[-1]]
                val[key] += [w[n] * ga.voll * (Ga[on].sum() - Gb[on].sum()), w[n] * ga.voll * (Gb[~on].sum() - Ga[~on].sum())]
    print(f"{task}, 64 episodes summed, GWh: output by segment in on weeks and off weeks, played | optimised ({name}); stock at the grid at the end")
    print(f"  {'grid':8s} {'segment':13s} {'on: played':>12s} {'optimised':>12s} {'off: played':>12s} {'optimised':>12s} {'end: played':>12s} {'optimised':>12s} | at VOLL, pooled: less in on weeks, more in off weeks")
    for key in sorted(acc):
        x = acc[key]
        print(f"  {key[0]:8s} {key[1]:13s} {x[0]:12,.0f} {x[1]:12,.0f} {x[2]:12,.0f} {x[3]:12,.0f} {x[4]:12,.0f} {x[5]:12,.0f} | {val[key][0]:+.4f} {val[key][1]:+.4f}")
    tot = defaultdict(lambda: np.zeros(2))
    for key, x in val.items():
        tot[key[1]] += x
    for seg, x in sorted(tot.items()):
        print(f"  all grids {seg:13s} less in on weeks {x[0]:+.4f}, more in off weeks {x[1]:+.4f}")


if __name__ == "__main__":
    fire.Fire({"parts": parts, "fuels": fuels})

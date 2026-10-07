"""Chips made and not sold: disposed above storage or left at the episode's end, by node, played | optimised.

uv run python waste.py --task=small --name=cal
"""

from collections import defaultdict

import fire
import numpy as np

import common as C
import describe as Dsc


def main(task="small", name="cal", top=12):
    refs = C.references(task)
    ns = sorted(refs)
    w = Dsc.weights(task, ns)
    T = C.Cal(task, ns[0]).T
    q = [0, 4, 8, 13, 26, 39, T] if task == "small" else [0, 8, 16, 26, 52, 78, T]
    P = list(zip(q, q[1:]))
    disp = defaultdict(lambda: [np.zeros(len(P)), np.zeros(len(P))])  # (node, good) -> per period, played and optimised
    end = defaultdict(lambda: [0.0, 0.0])
    per = []
    for n in ns:
        cal = C.Cal(task, n)
        inst = cal.inst
        d = C.load_traj(task, n)
        a = d["zpi"].reshape(T, cal.nc)
        b, _keys, _z = Dsc.load_sol(task, n, name)
        v = Dsc.View(cal)
        chips = {k for k, c in enumerate(inst.commodities) if c.id.startswith("chip")}
        pi_of = {}
        for dm in inst.demands:
            pi_of[dm.k] = max(pi_of.get(dm.k, 0.0), dm.pi)
        for o in inst.osats:
            for raw, pk in inst.nodes[o].osat.packages.items():
                pi_of[raw] = pi_of.get(pk, 0.0)
        lost_a = lost_b = 0.0  # chips disposed or left, at the shortage penalty of their packaged good
        for s, slot in enumerate(inst.stock_slots):
            if slot.k not in chips:
                continue
            jO, jI = cal.tmpl.get(("O", s)), cal.tmpl.get(("I", s))
            key = (v.node[slot.node], v.K[slot.k])
            if jO is not None:
                for pi, (s0, e0) in enumerate(P):
                    disp[key][0][pi] += a[s0:e0, jO].sum()
                    disp[key][1][pi] += b[s0:e0, jO].sum()
                lost_a += a[:, jO].sum() * pi_of.get(slot.k, 0.0)
                lost_b += b[:, jO].sum() * pi_of.get(slot.k, 0.0)
            if jI is not None:
                end[key][0] += a[-1, jI]
                end[key][1] += b[-1, jI]
                lost_a += a[-1, jI] * pi_of.get(slot.k, 0.0)
                lost_b += b[-1, jI] * pi_of.get(slot.k, 0.0)
        r = refs[n]
        D = (r["J_naive_cents"] - r["J_oracle_cents"]) / 100
        Ja = Dsc.cost_parts(cal.model, a.ravel())
        Jb = Dsc.cost_parts(cal.model, b.ravel())
        short_gain = (Ja[0]["shortage"].sum() - Jb[0]["shortage"].sum())
        shed_gain = (Ja[0]["shed"].sum() - Jb[0]["shed"].sum())
        per.append((w[n] * lost_a, n, r["stratum"], lost_a / 1e9, lost_b / 1e9, w[n] * lost_b, w[n] * short_gain, w[n] * shed_gain, D / 1e9))
    print(f"{task}, 64 episodes summed: chips disposed above storage, played | optimised ({name}); periods (weeks): " + ", ".join(f"{s + 1}-{e}" for s, e in P))
    for key in sorted(disp):
        x, y = disp[key]
        if x.sum() + y.sum() > 1:
            print(f"  {key[0]:18s} {key[1]:13s} " + " ".join(f"{p / 1e6:6.2f}|{o / 1e6:6.2f}" for p, o in zip(x, y)) + f"   total {x.sum() / 1e6:8.2f} | {y.sum() / 1e6:8.2f} M")
    print("chips left at the episode's end, played | optimised (M):")
    for key in sorted(end):
        x, y = end[key]
        if x + y > 1e4:
            print(f"  {key[0]:18s} {key[1]:13s} {x / 1e6:8.2f} | {y / 1e6:8.2f}")
    tot_a, tot_b = sum(p[0] for p in per), sum(p[5] for p in per)
    print(f"chips disposed or left, at the shortage penalty of their packaged good, pooled score units: played {tot_a:.4f}, optimised {tot_b:.4f}")
    per.sort(reverse=True)
    print(f"the {top} episodes with the most (pooled share; bn USD played | optimised; the optimised plan's saving in unserved demand and in shed load, pooled):")
    for share, n, s, la, lb, _wb, sg, hg, D in per[:top]:
        print(f"  episode {n:2d} level {s}: {share:.4f}  {la:8.1f} | {lb:8.1f} bn   shortage saved {sg:+.4f}  shed saved {hg:+.4f}   (gap {D:.0f} bn)")


if __name__ == "__main__":
    fire.Fire(main)

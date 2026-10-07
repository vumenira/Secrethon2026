"""Two pooled checks on Small.

spells: packaged chips disposed at OSATs in our play, by how long the overflow had already lasted (a store that has
        been overflowing for more weeks than raw chips take to arrive and be packaged is fed by chips dispatched
        after the overflow was visible).
variant: the cost parts of one variant's saving over fix_x (every flow as played), solved here.

uv run python pieces.py spells --task=small
uv run python pieces.py variant --task=small --name=freex_nucfuel+
"""

from collections import defaultdict

import fire
import numpy as np

import common as C
import describe as Dsc
import solve as S


def spells(task="small"):
    refs = C.references(task)
    ns = sorted(refs)
    by_week = defaultdict(float)  # weeks since the overflow began (1 = its first week) -> packaged chips disposed
    by_osat = defaultdict(lambda: defaultdict(float))
    lead = {}
    total = 0.0
    longest = defaultdict(int)
    for n in ns:
        cal = C.Cal(task, n)
        inst = cal.inst
        d = C.load_traj(task, n)
        a = d["zpi"].reshape(cal.T, cal.nc)
        v = Dsc.View(cal)
        for o in inst.osats:
            oa = inst.nodes[o].osat
            # the slowest direct way a raw chip takes from a fab to this OSAT, plus the packaging time
            ways = [inst.edges[e].tau for e in inst.in_edges[o] if any(k in inst.edges[e].K for k in oa.packages)]
            lead[v.node[o]] = (max(ways) if ways else 0) + oa.tau
            for _kr, kp in oa.packages.items():
                s = inst.slot_index[(o, kp)]
                O = v.col(a, ("O", s))
                run = 0
                for t in range(cal.T):
                    if O[t] > 1e-6:
                        run += 1
                        by_week[run] += O[t]
                        by_osat[v.node[o]][min(run, 12)] += O[t]
                        total += O[t]
                        longest[v.node[o]] = max(longest[v.node[o]], run)
                    else:
                        run = 0
    print(f"{task}: packaged chips disposed at OSATs in our play, {total / 1e6:.1f} M over {len(ns)} episodes, by the week of the overflow spell they fell in")
    cum = 0.0
    for wk in sorted(by_week):
        cum += by_week[wk]
        if wk <= 16 or wk % 4 == 0:
            print(f"  week {wk:2d} of a spell: {by_week[wk] / 1e6:6.2f} M  (cumulative {cum / total:5.1%})")
    print(f"  the slowest direct way of a raw chip into each OSAT plus its packaging time, weeks: {lead}; the longest spell: {dict(longest)}")
    for name, rows in by_osat.items():
        late = sum(q for wk, q in rows.items() if wk > lead[name] + 1)
        tot = sum(rows.values())
        print(f"  {name}: {tot / 1e6:.2f} M disposed; after week {lead[name] + 1} of a spell (chips dispatched once the overflow could be seen): {late / 1e6:.2f} M ({late / tot:.0%})")


def variant(task="small", name="freex_nucfuel+", base="fix_x"):
    refs = C.references(task)
    ns = sorted(refs)
    w = Dsc.weights(task, ns)
    acc = defaultdict(float)
    status = defaultdict(int)
    skipped, used = [], []
    for n in ns:
        cal = C.Cal(task, n)
        d = C.load_traj(task, n)
        zpi = np.concatenate([d["zpi"], np.zeros(cal.nz)])
        z, _ = cal.calendar(d["zpi"])
        zpi[cal.nn :] = z.ravel()
        V = S.variants(cal, task, "groups")
        out = {}
        for nm in (base, name):
            J, st, _sec, x = cal.solve(zpi, z, V[nm], fresh=True)
            if x is None:  # as solve.py does: once more, cold with the wider band
                J, st2, _sec, x = cal.solve(zpi, z, V[nm], band=1e-7, fresh=True)
                st = f"{st}; wider band: {st2}"
            status[st] += 1
            if x is None:
                skipped.append((n, nm, st))
                break
            out[nm] = Dsc.cost_parts(cal.model, x[: cal.nn])
        if len(out) < 2:
            continue
        used.append(n)
        (pa, sa), (pb, sb) = out[base], out[name]
        for p in Dsc.PARTS:
            acc[p] += w[n] * (pa[p].sum() - pb[p].sum())
        acc["salvage credit"] += w[n] * (sb - sa)
    print(f"{task}, {len(used)} of {len(ns)} episodes: {name} against {base} by cost part, score units (positive: {name} pays less); solve statuses {dict(status)}")
    print(f"  episodes left out because a solve gave no point (their share is missing from every line below): {skipped}")
    for p, x in acc.items():
        print(f"  {p:14s} {x:+.4f}")
    print(f"  {'sum':14s} {sum(acc.values()):+.4f}")


if __name__ == "__main__":
    fire.Fire({"spells": spells, "variant": variant})

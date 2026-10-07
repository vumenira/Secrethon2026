"""What an optimised plan does that our play does not: cost parts, chips served, lots, fuel and flows, side by side.

uv run python describe.py parts --task=small --name=cal              # the difference by cost part, pooled
uv run python describe.py episode --task=small --n=17 --name=cal     # one episode in detail
uv run python describe.py pooled --task=small --name=freex_fuel+     # physical differences summed over episodes
"""

import pickle
from collections import defaultdict

import fire
import numpy as np

import common as C


PARTS = ("freight", "war_risk", "tariff", "holding", "queue_holding", "shortage", "disposal", "shed")


def cost_parts(model, z):
    """{part: (T,) USD} and the salvage credit of a column vector."""
    from shockbench_flow.oracle.lp import lp_costs

    weekly, salvage = lp_costs(model, np.asarray(z, dtype=float))
    out = {p: np.array([getattr(w, p) for w in weekly]) for p in PARTS}
    return out, float(salvage)


def load_sol(task, n, name="cal"):
    if name.startswith("close"):
        d = pickle.loads((C.HERE / "res" / f"{task}_{n}_{name}.pkl").read_bytes())
        return d["x"], d["keys"], d["z"]
    d = pickle.loads((C.HERE / "res" / f"{task}_{n}_sol.pkl").read_bytes())
    return d["sols"][name], d["keys"], d["z"]


def weights(task, ns):
    """Per episode, the factor that turns USD into pooled score units on the episodes ``ns``."""
    from shockbench_flow.scoring.rss import STRATUM_WEIGHTS

    refs = C.references(task)
    counts = {s: sum(refs[n]["stratum"] == s for n in ns) for s in (1, 2, 3, 4)}
    den = sum(STRATUM_WEIGHTS[s - 1] * np.mean([refs[n]["J_naive_cents"] - refs[n]["J_oracle_cents"] for n in ns if refs[n]["stratum"] == s]) for s in counts if counts[s]) / 100
    return {n: STRATUM_WEIGHTS[refs[n]["stratum"] - 1] / counts[refs[n]["stratum"]] / den for n in ns}


def parts(task="small", names="cal,fix_x,freex_fuel+,freex_chip+,freex_mat+", only=None):
    """The pooled difference played to each optimised vector by cost part and by period, in score units."""
    from shockbench_flow.oracle.lp import build_lp

    names = list(names) if isinstance(names, (list, tuple)) else [v for v in str(names).split(",") if v]
    refs = C.references(task)
    ns = sorted(refs) if only is None else [int(v) for v in (only if isinstance(only, (list, tuple)) else [only])]
    w = weights(task, sorted(refs))
    T = {"small": 52, "full": 104}[task]
    acc = {name: {p: np.zeros(T) for p in PARTS} for name in names}
    sal = {name: 0.0 for name in names}
    for n in ns:
        inst, omega, marks, _fb = C.world(task, n)
        model = build_lp(inst, marks)
        d = C.load_traj(task, n)
        a, sa = cost_parts(model, d["zpi"])
        for name in names:
            x, _keys, _z = load_sol(task, n, name)
            b, sb = cost_parts(model, x.ravel())
            for p in PARTS:
                acc[name][p] += w[n] * (a[p] - b[p])
            sal[name] += w[n] * (sb - sa)
    edges = [0, T // 4, T // 2, 3 * T // 4, T]
    for name in names:
        total = sum(v.sum() for v in acc[name].values()) + sal[name]
        print(f"{task}, {len(ns)} episodes: played minus optimised ({name}) by cost part, score units (positive: the optimised plan pays less)")
        print(f"  {'part':14s} {'total':>8s} " + " ".join(f"{f'wk {a + 1}-{b}':>10s}" for a, b in zip(edges, edges[1:])))
        for p in PARTS:
            print(f"  {p:14s} {acc[name][p].sum():8.4f} " + " ".join(f"{acc[name][p][a:b].sum():10.4f}" for a, b in zip(edges, edges[1:])))
        print(f"  {'salvage credit':14s} {sal[name]:8.4f}")
        print(f"  {'sum':14s} {total:8.4f}")


class View:
    """One episode's columns by name: a (T, nc) array read through the model's template."""

    def __init__(self, cal):
        self.cal, self.inst, self.tmpl = cal, cal.inst, cal.tmpl
        inst = cal.inst
        self.K = [c.id for c in inst.commodities]
        self.node = [nd.id for nd in inst.nodes]
        self.flows = defaultdict(list)  # (edge, k) -> template columns (its lanes)
        for key, j in cal.tmpl.items():
            if key[0] == "x":
                self.flows[(key[1], key[2])].append(j)

    def col(self, a, key):
        j = self.tmpl.get(key)
        return np.zeros(a.shape[0]) if j is None else a[:, j]

    def flow(self, a, e, k):
        return a[:, self.flows[(e, k)]].sum(axis=1) if (e, k) in self.flows else np.zeros(a.shape[0])

    def edge_name(self, e):
        ed = self.inst.edges[e]
        return f"{self.node[ed.tail]}->{self.node[ed.head]}"


def spans(mask):
    """The runs of True in a 1-D mask as "a-b" in weeks from 1."""
    out, start = [], None
    for i, m in enumerate(list(mask) + [False]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append(f"{start + 1}-{i}" if i - start > 1 else f"{start + 1}")
            start = None
    return ",".join(out)


def events(task="small", n=0):
    """What the scenario did to the network, by element: the weeks impaired and the lowest share of nominal."""
    cal = C.Cal(task, int(n))
    inst, marks = cal.inst, cal.marks
    node = [nd.id for nd in inst.nodes]
    K = [c.id for c in inst.commodities]
    print(f"{task} episode {n}: impairments (weeks; lowest share of nominal)")
    o = np.asarray(marks.o)
    for ci, c in enumerate(inst.chokepoints):
        if (o[:, ci] < 1 - 1e-9).any():
            print(f"  strait {node[c]:14s} open below 1 in weeks {spans(o[:, ci] < 1 - 1e-9)}; lowest {o[:, ci].min():.2f}; shut in {spans(o[:, ci] <= 1e-9) or 'none'}")
    u = np.asarray(marks.u)
    for e, ed in enumerate(inst.edges):
        if ed.coupling or not ed.u0 or not np.isfinite(ed.u0):
            continue
        cut = u[:, e] < ed.u0 * (1 - 1e-9)
        if cut.any():
            print(f"  edge {node[ed.tail]}->{node[ed.head]} (lead {ed.tau}, goods {[K[k] for k in ed.K]}) capacity cut in weeks {spans(cut)}; lowest {u[:, e].min() / ed.u0:.2f}")
    pr = np.asarray(marks.prohibited)
    for e, ed in enumerate(inst.edges):
        for k in ed.K:
            if pr[:, e, k].any():
                print(f"  prohibited {node[ed.tail]}->{node[ed.head]} {K[k]} in weeks {spans(pr[:, e, k])}")
    tar = np.asarray(marks.tariff)
    for e, ed in enumerate(inst.edges):
        for k in ed.K:
            if (tar[:, e, k] > 0).any():
                print(f"  tariff {node[ed.tail]}->{node[ed.head]} {K[k]} in weeks {spans(tar[:, e, k] > 0)}; highest {tar[:, e, k].max():.2f}")
    G = np.asarray(marks.G_bar)
    for go, g in enumerate(inst.grids):
        nom = inst.nodes[g].grid.deliverable
        if (G[:, go] < nom * (1 - 1e-9)).any():
            print(f"  grid {node[g]} power below nominal in weeks {spans(G[:, go] < nom * (1 - 1e-9))}; lowest {G[:, go].min() / nom:.2f}")
    R, al = np.asarray(marks.R), np.asarray(marks.alpha_bar)
    for fo, f in enumerate(inst.fabs):
        if (R[:, fo] < 1 - 1e-9).any() or (al[:, fo] < 1 - 1e-9).any():
            print(f"  fab {node[f]} restoration below 1 in weeks {spans(R[:, fo] < 1 - 1e-9) or 'none'} (lowest {R[:, fo].min():.2f}); capacity factor below 1 in {spans(al[:, fo] < 1 - 1e-9) or 'none'} (lowest {al[:, fo].min():.2f})")
    for h in marks.fab_hits:
        print(f"  fab hit: {node[inst.fabs[h.fab]]} week {h.onset_week} severity {h.severity:.2f}")
    Ro = np.asarray(marks.R_osat)
    for oo, o_ in enumerate(inst.osats):
        if (Ro[:, oo] < 1 - 1e-9).any():
            print(f"  osat {node[o_]} restoration below 1 in weeks {spans(Ro[:, oo] < 1 - 1e-9)}; lowest {Ro[:, oo].min():.2f}")
    sup = np.asarray(marks.supply)
    for s, slot in enumerate(inst.stock_slots):
        if slot.node in set(inst.supply_nodes) and sup[:, s].max() > 0 and (sup[:, s] < sup[:, s].max() * (1 - 1e-9)).any():
            print(f"  supply {node[slot.node]} {K[slot.k]} below its highest in weeks {spans(sup[:, s] < sup[:, s].max() * (1 - 1e-9))}; lowest {sup[:, s].min() / sup[:, s].max():.2f}")


def periods(T):
    q = [0, T // 4, T // 2, 3 * T // 4, T]
    return list(zip(q, q[1:]))


def fmt(v):
    return f"{v:12,.0f}"


def episode(task="small", n=0, name="cal", top=25):
    """One episode: played against the optimised vector ``name``."""
    refs = C.references(task)[int(n)]
    cal = C.Cal(task, int(n))
    inst, marks, T = cal.inst, cal.marks, cal.T
    d = C.load_traj(task, int(n))
    a = d["zpi"].reshape(T, cal.nc)
    b, _keys, z = load_sol(task, int(n), name)
    v = View(cal)
    D = (refs["J_naive_cents"] - refs["J_oracle_cents"]) / 100
    pa, sa = cost_parts(cal.model, a.ravel())
    pb, sb = cost_parts(cal.model, b.ravel())
    Ja = sum(x.sum() for x in pa.values()) - sa
    Jb = sum(x.sum() for x in pb.values()) - sb
    sc = lambda J: (refs["J_naive_cents"] / 100 - J) / D  # noqa: E731
    print(f"{task} episode {n} level {refs['stratum']}: played {sc(Ja):.4f}, {name} {sc(Jb):.4f}; the episode's gap {D / 1e9:.1f} bn")
    print("cost parts, bn USD (played, optimised, saving as a share of the gap), by quarter of the episode:")
    P = periods(T)
    for p in PARTS:
        if abs(pa[p].sum()) + abs(pb[p].sum()) < 1e6:
            continue
        print(f"  {p:14s} {pa[p].sum() / 1e9:10.2f} {pb[p].sum() / 1e9:10.2f} {(pa[p].sum() - pb[p].sum()) / D:+8.4f} | " + " ".join(f"{(pa[p][s:e].sum() - pb[p][s:e].sum()) / D:+8.4f}" for s, e in P))
    print(f"  {'salvage':14s} {sa / 1e9:10.2f} {sb / 1e9:10.2f} {(sb - sa) / D:+8.4f}")
    # grids
    print("grids: weeks the fabs could run (z = 1) of the episode; fab energy, base load shed, fuel burnt (GWh): played | optimised")
    for gi, (go, fabs, _S) in enumerate(cal.grids):
        g = inst.grids[go]
        ga = inst.nodes[g].grid
        on = z[gi] > 0.5
        Ea = sum(v.col(a, ("E", fo)) for fo in fabs)
        Eb = sum(v.col(b, ("E", fo)) for fo in fabs)
        sha, shb = v.col(a, ("ysh", go)), v.col(b, ("ysh", go))
        print(f"  {v.node[g]}: on {int(on.sum())}/{T}; fab energy {Ea.sum():9.0f} | {Eb.sum():9.0f} (ceiling in on weeks {cal.emax[gi][on].sum():9.0f}); shed {sha.sum():10.0f} | {shb.sum():10.0f}")
        print(f"      on weeks: {''.join('#' if o else '.' for o in on)}")
        print(f"      fab energy by quarter: played {' '.join(f'{Ea[s:e].sum():8.0f}' for s, e in P)} | optimised {' '.join(f'{Eb[s:e].sum():8.0f}' for s, e in P)}")
        for k in ga.fuels:
            Ga, Gb = v.col(a, ("G", go, k)), v.col(b, ("G", go, k))
            s = inst.slot_index[(g, k)]
            Ia, Ib = v.col(a, ("I", s)), v.col(b, ("I", s))
            print(
                f"      {v.K[k]:8s} burnt {Ga.sum():10.0f} | {Gb.sum():10.0f}; in on weeks {Ga[on].sum():10.0f} | {Gb[on].sum():10.0f}; in off weeks {Ga[~on].sum():10.0f} | {Gb[~on].sum():10.0f};"
                f" mean stock at the grid {Ia.mean():9.0f} | {Ib.mean():9.0f}; left at the end {Ia[-1]:8.0f} | {Ib[-1]:8.0f}"
            )
    # fabs
    print("fabs: lots started by quarter, played | optimised; wafers left at the fab at the end")
    for fo, f in enumerate(inst.fabs):
        fa = inst.nodes[f].fab
        la, lb = v.col(a, ("p", fo)), v.col(b, ("p", fo))
        s = inst.slot_index[(f, fa.input)]
        cap = (np.asarray(marks.alpha_bar)[:, fo] * np.asarray(marks.R)[:, fo] * fa.cap0).sum()
        print(f"  {v.node[f]:18s} {' '.join(fmt(la[s0:e0].sum()) for s0, e0 in P)} | {' '.join(fmt(lb[s0:e0].sum()) for s0, e0 in P)}  total {fmt(la.sum())} | {fmt(lb.sum())} of capacity {fmt(cap)}; wafers left {fmt(v.col(a, ('I', s))[-1])} | {fmt(v.col(b, ('I', s))[-1])}")
    # demand
    print("demand served (share of demand), by quarter: played | optimised; unserved value saved, bn")
    for do, dm in enumerate(inst.demands):
        dem = np.asarray(marks.demand)[:, do]
        if dem.sum() <= 0:
            continue
        Da, Db = v.col(a, ("D", do)), v.col(b, ("D", do))
        print(
            f"  {v.node[dm.node]:8s} {v.K[dm.k]:9s} {' '.join(f'{Da[s:e].sum() / max(dem[s:e].sum(), 1):6.2f}' for s, e in P)} | {' '.join(f'{Db[s:e].sum() / max(dem[s:e].sum(), 1):6.2f}' for s, e in P)}"
            f"   total {Da.sum() / dem.sum():.3f} | {Db.sum() / dem.sum():.3f}  saved {(Db.sum() - Da.sum()) * dm.pi / 1e9:8.2f} bn ({(Db.sum() - Da.sum()) * dm.pi / D:+.4f})"
        )
    # flows
    print(f"flows by edge and good, the {top} largest differences by customs value: total played | optimised; by quarter (optimised minus played)")
    rows = []
    for (e, k), _cols in v.flows.items():
        fa_, fb_ = v.flow(a, e, k), v.flow(b, e, k)
        diff = np.abs(fb_ - fa_).sum() * inst.commodities[k].v
        if diff > 0:
            rows.append((diff, e, k, fa_, fb_))
    rows.sort(key=lambda r: -r[0])
    for diff, e, k, fa_, fb_ in rows[:top]:
        ed = inst.edges[e]
        print(f"  {v.edge_name(e):32s} lead {ed.tau} {v.K[k]:12s} {fmt(fa_.sum())} | {fmt(fb_.sum())}  ({(fb_.sum() - fa_.sum()) / max(fa_.sum(), 1):+6.1%})  " + " ".join(f"{fb_[s:e2].sum() - fa_[s:e2].sum():+11,.0f}" for s, e2 in P))
    # stocks at the end and lifts
    print("lifted at the sources: played | optimised")
    lift = defaultdict(lambda: [0.0, 0.0])
    for s, slot in enumerate(inst.stock_slots):
        if ("lift", s) in cal.tmpl:
            lift[(v.node[slot.node], v.K[slot.k])][0] += v.col(a, ("lift", s)).sum()
            lift[(v.node[slot.node], v.K[slot.k])][1] += v.col(b, ("lift", s)).sum()
    for (node, good), (x1, x2) in sorted(lift.items(), key=lambda kv: kv[0][1]):
        if x1 + x2 > 0:
            print(f"  {node:18s} {good:10s} {fmt(x1)} | {fmt(x2)}")
    print("stock at the episode's end by good (customs value, bn): played | optimised")
    end = defaultdict(lambda: [0.0, 0.0])
    for s, slot in enumerate(inst.stock_slots):
        key = ("I", s)
        if key in cal.tmpl:
            end[v.K[slot.k]][0] += a[-1, cal.tmpl[key]] * inst.commodities[slot.k].v
            end[v.K[slot.k]][1] += b[-1, cal.tmpl[key]] * inst.commodities[slot.k].v
    for good, (x1, x2) in end.items():
        print(f"  {good:12s} {x1 / 1e9:10.2f} | {x2 / 1e9:10.2f}")


def pooled(task="small", name="cal", only=None):
    """Physical quantities summed over episodes, played | optimised, by period: lots, chips served, goods left over."""
    refs = C.references(task)
    ns = sorted(refs) if only is None else [int(v) for v in (only if isinstance(only, (list, tuple)) else [only])]
    w = weights(task, sorted(refs))
    T = C.Cal(task, ns[0]).T
    q = [0, 4, 8, 13, 26, 39, T] if task == "small" else [0, 8, 16, 26, 52, 78, T]
    P = list(zip(q, q[1:]))
    A, B = defaultdict(lambda: np.zeros(len(P))), defaultdict(lambda: np.zeros(len(P)))
    endA, endB = defaultdict(float), defaultdict(float)
    for n in ns:
        cal = C.Cal(task, n)
        inst, marks = cal.inst, cal.marks
        d = C.load_traj(task, n)
        a = d["zpi"].reshape(T, cal.nc)
        b, _keys, _z = load_sol(task, n, name)
        v = View(cal)
        chk = set(inst.chokepoints)
        kind = {}
        for i, nd in enumerate(inst.nodes):
            kind[i] = "strait" if i in chk else ("fab" if i in inst.fabs else "osat" if i in inst.osats else "grid" if i in inst.grids else "supply" if i in inst.supply_nodes else "sink" if any(dm.node == i for dm in inst.demands) else "other")
        for pi, (s, e) in enumerate(P):
            sl = slice(s, e)
            for fo, f in enumerate(inst.fabs):
                fa = inst.nodes[f].fab
                A[("lots", v.node[f])][pi] += v.col(a, ("p", fo))[sl].sum()
                B[("lots", v.node[f])][pi] += v.col(b, ("p", fo))[sl].sum()
                cap = (np.asarray(marks.alpha_bar)[sl, fo] * np.asarray(marks.R)[sl, fo] * fa.cap0).sum()
                A[("capacity", v.node[f])][pi] += cap
                B[("capacity", v.node[f])][pi] += cap
            for do, dm in enumerate(inst.demands):
                dem = np.asarray(marks.demand)[sl, do].sum()
                key = ("served", v.K[dm.k])
                A[key][pi] += v.col(a, ("D", do))[sl].sum()
                B[key][pi] += v.col(b, ("D", do))[sl].sum()
                A[("demand", v.K[dm.k])][pi] += dem
                B[("demand", v.K[dm.k])][pi] += dem
                A[("unserved, score units", v.K[dm.k])][pi] += w[n] * dm.pi * (dem - v.col(a, ("D", do))[sl].sum())
                B[("unserved, score units", v.K[dm.k])][pi] += w[n] * dm.pi * (dem - v.col(b, ("D", do))[sl].sum())
            for oo, o in enumerate(inst.osats):
                for _kr, kp in inst.nodes[o].osat.packages.items():
                    A[("packaged", v.K[kp])][pi] += v.col(a, ("xi", oo, kp))[sl].sum()
                    B[("packaged", v.K[kp])][pi] += v.col(b, ("xi", oo, kp))[sl].sum()
            for s_, slot in enumerate(inst.stock_slots):
                if ("O", s_) in cal.tmpl:
                    A[("disposed", v.K[slot.k])][pi] += v.col(a, ("O", s_))[sl].sum()
                    B[("disposed", v.K[slot.k])][pi] += v.col(b, ("O", s_))[sl].sum()
                if ("lift", s_) in cal.tmpl:
                    A[("lifted", v.K[slot.k])][pi] += v.col(a, ("lift", s_))[sl].sum()
                    B[("lifted", v.K[slot.k])][pi] += v.col(b, ("lift", s_))[sl].sum()
        for s_, slot in enumerate(inst.stock_slots):
            j = cal.tmpl.get(("I", s_))
            if j is not None:
                endA[(v.K[slot.k], kind[slot.node])] += a[-1, j]
                endB[(v.K[slot.k], kind[slot.node])] += b[-1, j]
        for key, j in cal.tmpl.items():
            if key[0] == "Q":
                endA[(v.K[key[2]], "strait")] += a[-1, j]
                endB[(v.K[key[2]], "strait")] += b[-1, j]
    print(f"{task}, {len(ns)} episodes summed, played | optimised ({name}); periods (weeks): " + ", ".join(f"{s + 1}-{e}" for s, e in P))
    for key in sorted(A):
        if key[0] in ("capacity", "demand"):
            print(f"  {key[0]:22s} {key[1]:18s} " + " ".join(f"{x:13,.0f}" for x in A[key]) + f"   total {A[key].sum():15,.0f}")
            continue
        if key[0].endswith("score units"):
            print(f"  {key[0]:22s} {key[1]:18s} " + " ".join(f"{x:6.4f}|{y:6.4f}" for x, y in zip(A[key], B[key])) + f"   total {A[key].sum():.4f} | {B[key].sum():.4f}")
            continue
        print(f"  {key[0]:22s} {key[1]:18s} " + " ".join(f"{y / x if x else float('nan'):6.3f}" for x, y in zip(A[key], B[key])) + f"   optimised/played; totals {A[key].sum():15,.0f} | {B[key].sum():15,.0f}")
    print("  left at the episode's end (units), played | optimised:")
    for key in sorted(endA):
        if endA[key] + endB[key] > 1:
            print(f"    {key[0]:14s} at {key[1]:8s} {endA[key]:15,.0f} | {endB[key]:15,.0f}")


if __name__ == "__main__":
    fire.Fire({"parts": parts, "episode": episode, "pooled": pooled, "events": events})

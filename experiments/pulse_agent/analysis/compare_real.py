"""What the realizable clairvoyant (milp_real.py) does that our agent (trace_v2.py) does not, on chosen episodes.

Sums over the chosen episodes: per grid the weeks its fabs ran, fab energy and base load shed; per grid and fuel the
fuel burnt; wafers lifted and lots; per strait the tanker throughput used by good against its capacity; the edges
that carry both LNG and crude.

uv run python compare_real.py --task=small --top=8
"""

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import fire
import numpy as np


HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import analyze_v2 as A  # noqa: E402
import milp_real as M  # noqa: E402


def main(task="small", entropy=12345, top=8, only=None, tag="v2", res=""):
    from shockbench_flow.hosting.tasks import task_generator

    inst, _ = task_generator(task)
    st = A.static(inst)
    K = inst.commodities
    name_k = {k: c.id for k, c in enumerate(K)}
    results = json.loads((HERE / f"milp_{task}{res}.json").read_text())
    total = {"small": 64, "full": 16}[task]
    refs = {r["episode"]: r for r in json.loads((A.ROOT / "outputs" / "recovery_tune" / f"{task}-e{entropy}-n{total}" / "references.json").read_text())}
    rows = []
    for key, res in results.items():
        n = int(key)
        path = HERE / "trace" / f"{task}_{tag}" / f"{n}.pkl"
        if res.get("score") is None or not path.exists() or not (M.OUT / f"{task}_{n}.pkl").exists():
            continue
        d = pickle.loads(path.read_bytes())
        r = refs[n]
        gap = (r["J_naive_cents"] - r["J_oracle_cents"]) / 100
        ours = (r["J_naive_cents"] / 100 - d["played"]["J"]) / gap
        bnd = (r["J_naive_cents"] / 100 - res["bound"]) / gap if res.get("bound") is not None else float("nan")
        rows.append((res["score"] - ours, n, ours, res["score"], bnd, gap))
    rows.sort(reverse=True)
    if only is not None:
        only = [only] if isinstance(only, int) else [int(v) for v in only]
        rows = [r for r in rows if r[1] in only]
    else:
        rows = rows[:top]
    print(f"{task}: the {len(rows)} episodes compared (realizable incumbent minus ours, episode, ours, incumbent, bound):")
    for diff, n, ours, inc, bnd, _gap in rows:
        print(f"   episode {n:2d} level {refs[n]['stratum']}: ours {ours:.3f}  realizable {inc:.3f} (bound {bnd:.3f})  difference {diff:+.3f}")
    G = defaultdict(lambda: defaultdict(float))
    F = defaultdict(lambda: defaultdict(float))
    S = defaultdict(lambda: defaultdict(float))
    Ed = defaultdict(lambda: defaultdict(float))
    W = defaultdict(float)
    for _diff, n, *_ in rows:
        d = pickle.loads((HERE / "trace" / f"{task}_{tag}" / f"{n}.pkl").read_bytes())
        m = pickle.loads((M.OUT / f"{task}_{n}.pkl").read_bytes())
        _inst, _omega, marks, _fb = M.world(task, entropy, n)
        p, plan = d["played"], d["plan"]
        colm = {key: j for j, key in enumerate(m["columns"])}
        colp = {key: j for j, key in enumerate(plan["columns"])}
        xm, xp = m["x"], plan["x"]
        T = xm.shape[0]
        zmap = {go: m["z"][gi] for gi, (go, _fabs, _S) in enumerate(m["grids"])}
        for go in st:
            name = st[go]["id"]
            gw = A.grid_weeks(inst, st, d, go)
            fabs = st[go]["fabs"]
            row = G[name]
            live = gw["Ecap"] > A.EPS
            row["weeks"] += live.sum()
            row["shed_o"] += p["shed"][:, go].sum()
            row["shed_m"] += xm[:, colm[("ysh", go)]].sum()
            row["shed_p"] += xp[:, colp[("ysh", go)]].sum()
            if fabs:
                Em = sum(xm[:, colm[("E", fo)]] for fo in fabs)
                Eo = sum(p["energy"][:, fo] for fo in fabs)
                Ep = sum(xp[:, colp[("E", fo)]] for fo in fabs)
                cap = gw["Ecap"]
                row["E_o"] += Eo[live].sum()
                row["E_m"] += Em[live].sum()
                row["E_p"] += Ep[live].sum()
                row["cap"] += cap.sum()
                row["whole_o"] += (gw["cls"] == "whole").sum()
                row["on_m"] += ((Em > 1e-6) & live).sum()
                row["full_m"] += ((Em >= 0.98 * cap) & live).sum()
                row["noshed_m"] += ((zmap[go] > 0.5) & live).sum() if go in zmap else 0
                row["noshed_o"] += ((p["shed"][:, go] <= 1e-9 * gw["ybar"]) & live).sum()
                row["mix_p"] += ((Ep > 1e-6) & (xp[:, colp[("ysh", go)]] > 1e-6 * gw["ybar"]) & live).sum()
            for k in st[go]["fuels"]:
                fr = F[(name, name_k[k])]
                bo = np.array([seg.get((go, k), 0.0) for seg in p["segment"]])
                bm = xm[:, colm[("G", go, k)]]
                fr["o"] += bo.sum()
                fr["m"] += bm.sum()
                fr["p"] += xp[:, colp[("G", go, k)]].sum()
                fr["full"] += gw["fuels"][k]["full"].sum()
                if fabs:
                    onm = Em > 1e-6
                    fr["m_on"] += bm[onm].sum()
                    fr["o_on"] += bo[Eo > 1e-6].sum()
        for s, slot in enumerate(inst.stock_slots):
            if ("lift", s) in colm:
                W[(name_k[slot.k], "o")] += p["lift"][:, s].sum()
                W[(name_k[slot.k], "m")] += xm[:, colm[("lift", s)]].sum()
                W[(name_k[slot.k], "p")] += xp[:, colp[("lift", s)]].sum()
        for fo in range(len(inst.fabs)):
            W[("lots", "o")] += p["lots"][:, fo].sum()
            W[("lots", "m")] += xm[:, colm[("p", fo)]].sum()
            W[("lots", "p")] += xp[:, colp[("p", fo)]].sum()
        # flows by edge and good, week by week: ours, the realizable plan's, the clairvoyant plan's
        flow = {"o": defaultdict(lambda: np.zeros(T)), "m": defaultdict(lambda: np.zeros(T)), "p": defaultdict(lambda: np.zeros(T))}
        for t, x in enumerate(p["x"]):
            for (e, k, _lane), q in x.items():
                flow["o"][(e, k)][t] += q
        for key, j in colm.items():
            if key[0] == "x":
                flow["m"][(key[1], key[2])] += xm[:, j]
        for key, j in colp.items():
            if key[0] == "x":
                flow["p"][(key[1], key[2])] += xp[:, j]
        kappa = np.asarray(marks.kappa)[:, :, 0]  # the tanker pool
        u = np.asarray(marks.u)
        tb = [k for k, c in enumerate(K) if c.pool == "tb"]
        for ci, c in enumerate(inst.chokepoints):
            if kappa[:, ci].sum() <= 0:
                continue
            row = S[inst.nodes[c].id]
            row["cap"] += kappa[:, ci].sum()
            for who in ("o", "m", "p"):
                used = np.zeros(T)
                for e in inst.out_edges[c]:
                    for k in tb:
                        if (e, k) in flow[who]:
                            used += flow[who][(e, k)]
                            row[f"{who}_{name_k[k]}"] += flow[who][(e, k)].sum()
                row[f"{who}_full"] += (used >= kappa[:, ci] * (1 - 1e-6)).sum()
                row[f"{who}_used"] += used.sum()
            row["weeks"] += T
        lng, crude = next(k for k, c in enumerate(K) if c.id == "lng"), next(k for k, c in enumerate(K) if c.id == "crude")
        for e, ed in enumerate(inst.edges):
            if ed.coupling or ed.tau == 0 or not (lng in ed.K or crude in ed.K):
                continue
            row = Ed[(f"{inst.nodes[ed.tail].id}->{inst.nodes[ed.head].id}", ed.tau, "both" if (lng in ed.K and crude in ed.K) else ("lng" if lng in ed.K else "crude"))]
            cap = np.where(np.isfinite(u[:, e]), u[:, e], 0.0)
            row["cap"] += cap.sum()
            row["nominal"] += (ed.u0 or 0.0) * T
            for who in ("o", "m", "p"):
                tot = np.zeros(T)
                for k in (lng, crude):
                    if (e, k) in flow[who]:
                        tot += flow[who][(e, k)]
                        row[f"{who}_{name_k[k]}"] += flow[who][(e, k)].sum()
                row[f"{who}_full"] += ((tot >= cap * (1 - 1e-6)) & (cap > 0)).sum()
            row["weeks"] += T
    print("\nA. Grids: sellable weeks; weeks the fabs ran (ours: whole; realizable: any fab energy, at capacity, no shed); fab energy; base load shed (GWh)")
    print(f"   {'grid':9s} {'weeks':>5s} | {'whole, ours':>11s} {'no shed, ours':>13s} | realizable: {'fabs on':>7s} {'at cap':>6s} {'no shed':>7s} | fab energy {'ours':>8s} {'realiz.':>8s} {'clairv.':>8s} | shed {'ours':>9s} {'realiz.':>9s} {'clairv.':>9s} | clairv. mixes")
    for go in st:
        name = st[go]["id"]
        r = G[name]
        print(
            f"   {name:9s} {r['weeks']:5.0f} | {r['whole_o']:11.0f} {r['noshed_o']:13.0f} |             {r['on_m']:7.0f} {r['full_m']:6.0f} {r['noshed_m']:7.0f} |"
            f"            {r['E_o']:8.0f} {r['E_m']:8.0f} {r['E_p']:8.0f} |      {r['shed_o']:9.0f} {r['shed_m']:9.0f} {r['shed_p']:9.0f} | {r['mix_p']:6.0f}"
        )
    print("\nB. Fuel burnt by grid (GWh): ours, the realizable plan, the clairvoyant plan; share of the segment's full output; share burnt in weeks the fabs ran")
    print(f"   {'grid':9s} {'fuel':8s} {'ours':>10s} {'realiz.':>10s} {'clairv.':>10s} | realiz./ours {'':2s} | of full: {'ours':>5s} {'realiz.':>7s} | in fab weeks: {'ours':>5s} {'realiz.':>7s}")
    for go in st:
        name = st[go]["id"]
        for (gname, fuel), r in F.items():
            if gname == name:
                print(
                    f"   {name:9s} {fuel:8s} {r['o']:10.0f} {r['m']:10.0f} {r['p']:10.0f} | {r['m'] / r['o'] if r['o'] else float('nan'):12.3f}    |          {r['o'] / r['full']:5.2f} {r['m'] / r['full']:7.2f} |"
                    f"               {r['o_on'] / r['o'] if r['o'] else float('nan'):5.2f} {r['m_on'] / r['m'] if r['m'] else float('nan'):7.2f}"
                )
    print("\nC. Lifted at the sources and lots started: ours, realizable, clairvoyant")
    for good in sorted({k for k, _ in W}):
        if W[(good, "o")] or W[(good, "m")]:
            print(f"   {good:10s} {W[(good, 'o')]:14,.0f} {W[(good, 'm')]:14,.0f} {W[(good, 'p')]:14,.0f}   realizable/ours {W[(good, 'm')] / W[(good, 'o')] if W[(good, 'o')] else float('nan'):.3f}")
    print("\nD. Straits, tanker pool: throughput capacity, used (share), weeks at capacity (share), and what passed by good: ours | realizable | clairvoyant")
    for name, r in S.items():
        line = f"   {name:12s} cap {r['cap']:10.0f}"
        for who, label in (("o", "ours"), ("m", "realiz."), ("p", "clairv.")):
            line += f" | {label}: used {r[who + '_used'] / r['cap']:.2f} full {r[who + '_full'] / r['weeks']:.2f} lng {r[who + '_lng']:9.0f} crude {r[who + '_crude']:8.0f}"
        print(line)
    print("\nE. Edges that carry LNG or crude (lead > 0): capacity as a share of nominal; weeks at capacity; LNG and crude carried: ours | realizable | clairvoyant")
    for (name, tau, kind), r in sorted(Ed.items(), key=lambda kv: (kv[0][2] != "both", kv[0][0])):
        if r["o_lng"] + r["o_crude"] + r["m_lng"] + r["m_crude"] < 1:
            continue
        line = f"   {name:28s} lead {tau} {kind:5s} cap/nominal {r['cap'] / r['nominal'] if r['nominal'] else float('nan'):.2f}"
        for who, label in (("o", "ours"), ("m", "realiz."), ("p", "clairv.")):
            line += f" | {label}: full {r[who + '_full'] / r['weeks']:.2f} lng {r[who + '_lng']:9.0f} crude {r[who + '_crude']:8.0f}"
        print(line)


if __name__ == "__main__":
    fire.Fire(main)

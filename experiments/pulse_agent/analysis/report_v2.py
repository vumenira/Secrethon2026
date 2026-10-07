"""The four tables of the pulse analysis from the dumps of trace_v2.py (bounds, causes, logistics, demand side).

Energy in GWh, money in bn USD; "of the gap" is money over the summed naive-to-clairvoyant gap of the traced episodes
(all episodes alike). Money for fab energy: per episode and chip family, the shortage cost we pay above the clairvoyant
plan is shared out over the lots we start fewer than the plan, fab by fab; a grid's missing fab energy is worth its
fabs' share of that. So the steps of table 1 add up to the shortage cost of the chips we make fewer lots for.

uv run python report_v2.py --task=full
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

EPS = A.EPS
LEVELS = ("B0", "B1", "B2", "B3", "B4")
STEPS = (
    ("out of reach: our fuel, held ideally, cannot run the plan's lots", "plan", "B0"),
    ("a short fuel has no terminal (cannot be held)", "B0", "B1"),
    ("a short fuel is refused by the pulse (payback)", "B1", "B2"),
    ("the gas charge at each start of a run", "B2", "B3"),
    ("terminal storage: held fuel overflows", "B3", "B4"),
    ("the pulse's rules and timing, stranded fuel, wafers", "B4", "got"),
)


def money(inst, d):
    """Per fab the worth of a lot we did not start (USD), and the episode's cost above the plan by kind."""
    p, plan = d["played"], d["plan"]
    col = {key: j for j, key in enumerate(plan["columns"])}
    x = plan["x"]
    packaged = defaultdict(set)
    for o in inst.osats:
        for raw, pk in inst.nodes[o].osat.packages.items():
            packaged[raw].add(pk)
    D = len(inst.demands)
    short_o, short_p, served_o, served_p = (np.zeros(D) for _ in range(4))
    for do, dm in enumerate(inst.demands):
        if dm.backlog:
            short_o[do], short_p[do] = dm.pi * p["backlog"][:, do].sum(), dm.pi * x[:, col[("B", do)]].sum()
        else:
            short_o[do], short_p[do] = dm.pi * p["lost"][:, do].sum(), dm.pi * x[:, col[("U", do)]].sum()
        served_o[do], served_p[do] = p["served"][:, do].sum(), x[:, col[("D", do)]].sum()
    lots_o = p["lots"].sum(axis=0)
    lots_p = np.array([x[:, col[("p", fo)]].sum() for fo in range(len(inst.fabs))])
    fam = defaultdict(list)
    for fo, f in enumerate(inst.fabs):
        fa = inst.nodes[f].fab
        fam[frozenset(packaged[fa.product] | {fa.product})].append(fo)
    v = np.zeros(len(inst.fabs))
    rows, in_family = [], set()
    for goods, fabs in fam.items():
        ds = [do for do, dm in enumerate(inst.demands) if dm.k in goods]
        in_family.update(ds)
        dshort = short_o[ds].sum() - short_p[ds].sum()
        miss = np.maximum(lots_p[fabs] - lots_o[fabs], 0.0).sum()
        if dshort > 0 and miss > 0:
            v[fabs] = dshort / miss
        name = "+".join(sorted(inst.commodities[k].id for k in goods if any(inst.demands[do].k == k for do in ds)) or sorted(inst.commodities[k].id for k in goods))
        rows.append(
            {
                "family": name,
                "lots": lots_o[fabs].sum(),
                "lots_plan": lots_p[fabs].sum(),
                "served": served_o[ds].sum(),
                "served_plan": served_p[ds].sum(),
                "dshort": dshort,
                "explained": dshort if (dshort > 0 and miss > 0) else 0.0,
            }
        )
    rest = [do for do in range(D) if do not in in_family]
    cost_o = {k: sum(c[k] for c in p["costs"]) for k in p["costs"][0]}
    cost_p = {k: sum(c[k] for c in plan["costs"]) for k in plan["costs"][0]}
    kinds = {
        "chips we start fewer lots for": sum(r["explained"] for r in rows),
        "chips, other (lots not fewer, or fewer lots but no dearer)": sum(r["dshort"] - r["explained"] for r in rows),
        "other goods short": short_o[rest].sum() - short_p[rest].sum(),
        "base load shed": cost_o["shed"] - cost_p["shed"],
        "freight, holding, tariffs, war risk, disposal": sum(cost_o[k] - cost_p[k] for k in cost_o if k not in ("shortage", "shed")),
        "terminal credit": -(d["played"]["salvage"] - plan["salvage"]),
    }
    return v, lots_o, lots_p, rows, kinds


def bounds_of(task, tag, n):
    """The bounds of every grid with fabs in one traced episode (the slow part: one worker per episode)."""
    from shockbench_flow.hosting.tasks import task_generator

    inst, _ = task_generator(task)
    st = A.static(inst)
    d = pickle.loads((HERE / "trace" / f"{task}_{tag}" / f"{n}.pkl").read_bytes())
    accepted = {row[0]: {k for k, val in row[3].items() if val[2] and val[3]} for row in d["pulse"]["grids"]}
    out = {}
    for go in st:
        if not st[go]["fabs"]:
            continue
        gw = A.grid_weeks(inst, st, d, go)
        every = set(gw["fuels"])
        terminal = {k for k, fu in gw["fuels"].items() if fu["terminal"]}
        acc = accepted.get(go, set())
        target = float(gw["Eplan"].sum())
        b = {
            "B0": A.bound(gw, every, cap=target),
            "B1": A.bound(gw, terminal, cap=target),
            "B2": A.bound(gw, acc, cap=target),
            "B3": A.bound(gw, acc, charge=True, cap=target),
            "B4": A.bound(gw, acc, charge=True, tank=True, cap=target),
        }
        for a, c in zip(LEVELS[:-1], LEVELS[1:]):  # the bounds only tighten
            b[c] = min(b[c], b[a])
        more = {k: max(A.bound(gw, acc | {k}, charge=True, tank=True, cap=target), b["B4"]) for k in every - acc}
        out[go] = {"b": b, "with": more}
    return n, out


def main(task="full", tag="v2", entropy=12345, episodes=None, n_jobs=2):
    from joblib import Parallel, delayed
    from shockbench_flow.hosting.tasks import task_generator

    inst, _ = task_generator(task)
    st = A.static(inst)
    episodes = episodes or {"full": 16, "small": 64}[task]
    run = A.ROOT / "outputs" / "recovery_tune" / f"{task}-e{entropy}-n{episodes}"
    refs = {r["episode"]: r for r in json.loads((run / "references.json").read_text())}
    dumps = {}
    for path in sorted((HERE / "trace" / f"{task}_{tag}").glob("*.pkl"), key=lambda q: int(q.stem)):
        dumps[int(path.stem)] = pickle.loads(path.read_bytes())
    pre = dict(Parallel(n_jobs=n_jobs)(delayed(bounds_of)(task, tag, n) for n in dumps))
    gap = sum((refs[n]["J_naive_cents"] - refs[n]["J_oracle_cents"]) / 100 for n in dumps)
    ours = sum(d["played"]["J"] for d in dumps.values())
    planJ = sum(d["plan"]["J"] for d in dumps.values())
    print(f"{task}: {len(dumps)} episodes {sorted(dumps)}")
    print(f"naive-to-clairvoyant gap {gap / 1e9:,.0f} bn; played above the clairvoyant plan by {(ours - planJ) / 1e9:,.0f} bn = {(ours - planJ) / gap:.3f} of the gap")
    name_k = {k: inst.commodities[k].id for k in range(len(inst.commodities))}
    order = [st[go]["id"] for go in st if st[go]["fabs"]]
    tot = defaultdict(lambda: defaultdict(float))
    marg = defaultdict(lambda: defaultdict(float))
    states = defaultdict(lambda: defaultdict(float))
    fuel_tab = defaultdict(lambda: defaultdict(float))
    kinds_tot = defaultdict(float)
    fam_tot = defaultdict(lambda: defaultdict(float))
    lift = defaultdict(lambda: defaultdict(float))
    grid_energy = defaultdict(lambda: defaultdict(float))
    for n, d in dumps.items():
        v_fab, lots_o, lots_p, fam_rows, kinds = money(inst, d)
        for k, val in kinds.items():
            kinds_tot[k] += val
        for r in fam_rows:
            for c in ("lots", "lots_plan", "served", "served_plan", "dshort", "explained"):
                fam_tot[r["family"]][c] += r[c]
        plan = d["plan"]
        col = {key: j for j, key in enumerate(plan["columns"])}
        p = d["played"]
        for s, slot in enumerate(inst.stock_slots):  # what was lifted at the sources, by good
            if ("lift", s) in col:
                lift[name_k[slot.k]]["ours"] += p["lift"][:, s].sum()
                lift[name_k[slot.k]]["plan"] += plan["x"][:, col[("lift", s)]].sum()
        accepted = {row[0]: {k for k, val in row[3].items() if val[2] and val[3]} for row in d["pulse"]["grids"]}
        why = defaultdict(dict)
        for r in d["pulse"]["trace"]:
            why[r["go"]][r["t"]] = r
        for go in st:
            name = st[go]["id"]
            ge = grid_energy[name]
            ge["shed"] += p["shed"][:, go].sum()
            ge["shed_plan"] += plan["x"][:, col[("ysh", go)]].sum()
            ge["base"] += d["marks"]["y_bar"][:, go].sum()
            ge["fabs"] = len(st[go]["fabs"])
            gw = A.grid_weeks(inst, st, d, go)
            for k, fu in gw["fuels"].items():
                ft = fuel_tab[(name, name_k[k])]
                live = gw["Ecap"] > EPS
                last = int(np.flatnonzero(live)[-1]) if live.any() else gw["T"] - 1
                ft["need"] += fu["whole_burn"][: last + 1].sum()
                ft["arrived"] += fu["Ig"][0] + fu["It"][0] + (fu["a_direct"] + fu["a_term"])[: last + 1].sum()
                ft["burn"] += fu["burn"].sum()
                ft["burn_plan"] += fu["burn_plan"].sum()
                ft["burn_live"] += fu["burn"][live].sum()
                ft["burn_whole"] += fu["burn"][live & (gw["cls"] == "whole")].sum()
                ft["full"] += fu["full"].sum()
                ft["kind"] = "accepted" if k in accepted.get(go, set()) else ("refused" if fu["terminal"] else "no terminal")
                if live.any() and k in accepted.get(go, set()):
                    ft["left"] += fu["Ig"][last + 1] + fu["It"][last + 1]
            if not st[go]["fabs"]:
                continue
            every = set(gw["fuels"])
            terminal = {k for k, fu in gw["fuels"].items() if fu["terminal"]}
            acc = accepted.get(go, set())
            target = gw["Eplan"].sum()
            b = {"plan": target, **pre[n][go]["b"], "got": min(gw["E"].sum(), target)}
            for a, c in zip(LEVELS[:-1], LEVELS[1:]):  # the bounds only tighten
                b[c] = min(b[c], b[a])
            usd = sum(max(lots_p[fo] - lots_o[fo], 0.0) * v_fab[fo] for fo in st[go]["fabs"])
            miss = b["plan"] - b["got"]
            v = usd / miss if miss > EPS else 0.0
            row = tot[name]
            live = gw["Ecap"] > EPS
            row["weeks"] += live.sum()
            for c in ("whole", "partly", "none", "no wafers"):
                row[c] += (gw["cls"] == c).sum()
            row["cap"] += gw["Ecap"].sum()
            row["got_all"] += gw["E"].sum()
            row["usd"] += usd
            for c, val in b.items():
                row[c] += val
            for _label, a, c in STEPS:
                row["usd_" + a + c] += (b[a] - b[c]) * v
            for k in every - acc:  # holding this fuel too, the rest as the pulse has it
                kind = "no terminal" if k not in terminal else "refused (payback)"
                gain = pre[n][go]["with"][k] - b["B4"]
                marg[(name, name_k[k], kind)]["gwh"] += max(gain, 0.0)
                marg[(name, name_k[k], kind)]["usd"] += max(gain, 0.0) * v
            for t in range(gw["T"]):
                if not live[t]:
                    continue
                r = why[go].get(t + 1)
                state = "last week" if r is None else r["why"]
                if state == "unfixable":
                    state = "cannot hold: " + ", ".join(sorted(f"{name_k[k]} {c}" for k, c in r["short"].items() if c != "holdable"))
                elif state == "off":
                    waiting = [name_k[k] for k in r["short"]]
                    state = "off: holding " + "+".join(sorted(waiting))
                key = (name, state)
                states[key]["weeks"] += 1
                states[key]["cap"] += gw["Ecap"][t]
                states[key]["got"] += gw["E"][t]
                states[key]["plan"] += gw["Eplan"][t]
    P = defaultdict(float)
    print("\n1. Per grid with fabs, summed over the traced episodes. Weeks: grid-weeks in which a lot could still be sold.")
    print("   Fab energy, GWh: at capacity; in the clairvoyant plan; bounds on our arriving fuel (capped at the plan); got.")
    print(f"{'grid':9s} {'weeks':>5s} {'whole':>5s} {'partly':>6s} {'none':>5s} {'nowaf':>5s} | {'capacity':>8s} {'plan':>7s} | {'ideal':>7s} {'terminal':>8s} {'accepted':>8s} {'+charge':>8s} {'+tank':>7s} | {'got':>7s} {'got/plan':>8s} {'got/+tank':>9s}")
    for name in order:
        r = tot[name]
        print(
            f"{name:9s} {r['weeks']:5.0f} {r['whole']:5.0f} {r['partly']:6.0f} {r['none']:5.0f} {r['no wafers']:5.0f} | {r['cap']:8.0f} {r['plan']:7.0f} |"
            f" {r['B0']:7.0f} {r['B1']:8.0f} {r['B2']:8.0f} {r['B3']:8.0f} {r['B4']:7.0f} | {r['got_all']:7.0f} {r['got_all'] / r['plan'] if r['plan'] else float('nan'):8.2f} {r['got'] / r['B4'] if r['B4'] else float('nan'):9.2f}"
        )
        for c, val in r.items():
            P[c] += val
    print(
        f"{'pooled':9s} {P['weeks']:5.0f} {P['whole']:5.0f} {P['partly']:6.0f} {P['none']:5.0f} {P['no wafers']:5.0f} | {P['cap']:8.0f} {P['plan']:7.0f} |"
        f" {P['B0']:7.0f} {P['B1']:8.0f} {P['B2']:8.0f} {P['B3']:8.0f} {P['B4']:7.0f} | {P['got_all']:7.0f} {P['got_all'] / P['plan']:8.2f} {P['got'] / P['B4']:9.2f}"
    )
    print(f"\n   The cost above the plan, {(ours - planJ) / 1e9:,.0f} bn, by kind:")
    for k, val in sorted(kinds_tot.items(), key=lambda kv: -kv[1]):
        print(f"     {k:60s} {val / 1e9:9,.0f} bn  {val / gap:+.3f} of the gap")
    print("\n   The first kind (chips we start fewer lots for), step by step from the plan's fab energy down to ours:")
    for label, a, c in STEPS:
        usd = P["usd_" + a + c]
        print(f"     {label:60s} {P[a] - P[c]:8.0f} GWh {usd / 1e9:8,.0f} bn  {usd / gap:+.3f} of the gap")
    print("\n   By grid: GWh each step takes (plan ... got) and the money of each step, bn:")
    print(f"   {'grid':9s} {'plan':>7s} {'reach':>7s} {'no term':>7s} {'refused':>7s} {'charge':>7s} {'tank':>7s} {'rules':>7s} {'got':>7s} | bn: {'reach':>6s} {'noterm':>6s} {'refus':>6s} {'charge':>6s} {'tank':>6s} {'rules':>6s}")
    for name in order:
        r = tot[name]
        line = f"   {name:9s} {r['plan']:7.0f}" + "".join(f" {r[a] - r[c]:7.0f}" for _l, a, c in STEPS) + f" {r['got']:7.0f} |    "
        line += "".join(f" {r['usd_' + a + c] / 1e9:6.1f}" for _l, a, c in STEPS)
        print(line)
    print("\n2a. A fuel that is not held: what holding it too would add (the rest as the pulse has it)")
    for (name, fuel, kind), m in sorted(marg.items(), key=lambda kv: -kv[1]["usd"]):
        if m["gwh"] > 1.0:
            print(f"     {name:9s} {fuel:8s} {kind:18s} {m['gwh']:8.0f} GWh {m['usd'] / 1e9:7.1f} bn  {m['usd'] / gap:+.4f} of the gap")
    print("\n2b. Sellable grid-weeks by the pulse's own state: fab energy at capacity, in the plan, got (GWh)")
    agg = defaultdict(lambda: defaultdict(float))
    for (name, state), val in states.items():
        base = state.split(":")[0] if state.startswith("off") else state
        for c, x in val.items():
            agg[base][c] += x
    for state, val in sorted(agg.items(), key=lambda kv: -kv[1]["weeks"]):
        print(f"     {state:46s} weeks {val['weeks']:5.0f}  capacity {val['cap']:8.0f}  plan {val['plan']:8.0f}  got {val['got']:8.0f}")
    print("    by grid (the largest by weeks):")
    for (name, state), val in sorted(states.items(), key=lambda kv: -kv[1]["weeks"])[:26]:
        print(f"     {name:9s} {state:46s} weeks {val['weeks']:5.0f}  capacity {val['cap']:8.0f}  plan {val['plan']:8.0f}  got {val['got']:8.0f}")
    print("\n3. Fuel by grid. arrived/need: fuel that reached grid and terminal by the last sellable week over what whole weeks at")
    print("   capacity burn; whole: share of the burn in sellable weeks that fell in whole weeks; burn over the episode, ours and the plan's.")
    print(f"   {'grid':9s} {'fuel':8s} {'pulse':12s} {'arrived/need':>12s} {'whole':>6s} {'left':>8s} {'burn ours':>10s} {'burn plan':>10s} {'plan/ours':>9s} {'plan-ours':>10s} {'ours/full':>9s}")
    for go in st:
        name = st[go]["id"]
        for (gname, fuel), ft in sorted(fuel_tab.items()):
            if gname != name:
                continue
            print(
                f"   {name:9s} {fuel:8s} {ft['kind']:12s} {ft['arrived'] / ft['need'] if ft['need'] else float('nan'):12.2f} {ft['burn_whole'] / ft['burn_live'] if ft['burn_live'] else float('nan'):6.2f} {ft['left']:8.0f}"
                f" {ft['burn']:10.0f} {ft['burn_plan']:10.0f} {ft['burn_plan'] / ft['burn'] if ft['burn'] else float('nan'):9.3f} {ft['burn_plan'] - ft['burn']:10.0f} {ft['burn'] / ft['full']:9.2f}"
            )
    print("\n   Base load shed by grid (GWh), ours and the plan's, as a share of base load:")
    for go in st:
        ge = grid_energy[st[go]["id"]]
        print(f"     {st[go]['id']:9s} fabs {ge['fabs']:1.0f}  ours {ge['shed']:9.0f} ({ge['shed'] / ge['base']:.3f})  plan {ge['shed_plan']:9.0f} ({ge['shed_plan'] / ge['base']:.3f})  plan - ours {ge['shed_plan'] - ge['shed']:+9.0f}")
    print("\n   Lifted at the sources over the episodes, ours and the plan's:")
    for good, val in sorted(lift.items(), key=lambda kv: -kv[1]["plan"]):
        if val["plan"] > 0 or val["ours"] > 0:
            print(f"     {good:10s} ours {val['ours']:14,.0f}  plan {val['plan']:14,.0f}  plan/ours {val['plan'] / val['ours'] if val['ours'] else float('nan'):.3f}")
    print("\n4. Chips by family: lots started and chips served, ours over the plan's; the shortage cost above the plan")
    for name, val in sorted(fam_tot.items(), key=lambda kv: -kv[1]["dshort"]):
        print(
            f"     {name:28s} lots {val['lots'] / val['lots_plan'] if val['lots_plan'] else float('nan'):5.2f}  served {val['served'] / val['served_plan'] if val['served_plan'] else float('nan'):5.2f}"
            f"  shortage above the plan {val['dshort'] / 1e9:8,.0f} bn ({val['dshort'] / gap:+.3f} of the gap)"
        )
    out = {"tot": {k: dict(val) for k, val in tot.items()}, "pooled": dict(P), "gap": gap, "kinds": dict(kinds_tot)}
    (HERE / f"report_{task}.json").write_text(json.dumps(out, default=float))


if __name__ == "__main__":
    fire.Fire(main)

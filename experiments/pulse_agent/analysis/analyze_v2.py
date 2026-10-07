"""The pulse planner against what its fuel allowed, per grid: bounds, losses by cause, logistics, the demand side.

Reads the dumps of trace_v2.py. Energy is in the simulator's units (GWh), money in bn USD, and "of the gap" is money
over the summed naive-to-clairvoyant gap of the traced episodes (all episodes alike, not weighted by harm level).

uv run python analyze_v2.py --task=full
"""

import json
import pickle
from collections import defaultdict
from pathlib import Path

import fire
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


ROOT = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026")
HERE = Path(__file__).parent
EPS = 1e-6
FILL = 0.9  # pulse_fill: the share of a terminal's storage up to which fuel is held


def static(inst):
    """Per grid: its fabs, fuels (share, threshold, terminals), and what a lot is worth at each fab."""
    packaged = defaultdict(set)
    for o in inst.osats:
        for raw, pk in inst.nodes[o].osat.packages.items():
            packaged[raw].add(pk)
    chk, supply = set(inst.chokepoints), set(inst.supply_nodes)
    out = {}
    for go, g in enumerate(inst.grids):
        ga = inst.nodes[g].grid
        fabs = [fo for fo in inst.grid_fabs[go] if inst.nodes[inst.fabs[fo]].fab.e > 0]
        fuels = {}
        for k in ga.fuels:
            feeders = []
            for e in inst.in_edges[g]:
                ed = inst.edges[e]
                if k in ed.K and ed.tau == 0 and ed.tail not in chk and ed.tail not in supply:
                    if (ed.tail, k) in inst.slot_index:
                        feeders.append((e, ed.tail))
            fuels[k] = {
                "share": ga.shares[k],
                "threshold": inst.params.psi * ga.ibar[k] if k == ga.rationed else 0.0,
                "feeders": feeders,
                "slot": inst.slot_index[(g, k)],
                "room": sum(inst.stock_slots[inst.slot_index[(tail, k)]].storage for _e, tail in feeders),
            }
        worth = {}
        for fo in fabs:
            fa = inst.nodes[inst.fabs[fo]].fab
            goods = packaged[fa.product] | {fa.product}
            worth[fo] = max((d.pi for d in inst.demands if d.k in goods), default=0.0)
        out[go] = {"node": g, "id": inst.nodes[g].id, "fabs": fabs, "fuels": fuels, "S": sum(ga.shares.values()), "voll": ga.voll, "worth": worth}
    return out


def grid_weeks(inst, st, d, go):
    """Everything about one grid over one played episode, week by week (arrays of length T), and the plan beside it."""
    p, marks, i0 = d["played"], d["marks"], d["i0"]
    T = p["stock"].shape[0]
    info = st[go]
    lead = d["pulse"]["lead"] if d["pulse"] else {}
    ybar, Gbar, S = marks["y_bar"][:, go], marks["G_bar"][:, go], info["S"]
    weeks = np.arange(1, T + 1)
    plan = d.get("plan")
    col = {key: j for j, key in enumerate(plan["columns"])} if plan else {}
    out = {"T": T, "ybar": ybar, "Gbar": Gbar}
    z = lambda: np.zeros(T)  # noqa: E731
    Ecap, Ewant, E, Eplan, lots, lots_plan, lots_cap, usd_cap, usd, usd_plan = (z() for _ in range(10))
    for fo in info["fabs"]:
        f = inst.fabs[fo]
        fa = inst.nodes[f].fab
        useful = weeks + lead.get(fo, 0.0) <= T  # a lot started now can still be sold (pulse_tail 0)
        R, alpha = marks["R"][:, fo], marks["alpha_bar"][:, fo]
        s_in = inst.slot_index[(f, fa.input)]
        cap = alpha * R * fa.cap0
        wafers = p["stock"][:, s_in] + p["lots"][:, fo] + p["disposal"][:, s_in]
        phat = np.minimum(cap, wafers)
        ok = (R > 0) & useful
        net = info["worth"][fo] - info["voll"] * fa.e  # a lot's chip less the base load its energy would serve
        Ecap += np.where(ok, fa.e * alpha * fa.cap0, 0.0)
        Ewant += np.where(ok, fa.e * phat / np.where(R > 0, R, 1.0), 0.0)
        E += np.where(useful, p["energy"][:, fo], 0.0)
        lots += np.where(useful, p["lots"][:, fo], 0.0)
        lots_cap += np.where(ok, cap, 0.0)
        usd_cap += np.where(ok, cap * net, 0.0)
        usd += np.where(useful, p["lots"][:, fo] * net, 0.0)
        if plan:
            Eplan += np.where(useful, plan["x"][:, col[("E", fo)]], 0.0)
            lots_plan += np.where(useful, plan["x"][:, col[("p", fo)]], 0.0)
            usd_plan += np.where(useful, plan["x"][:, col[("p", fo)]] * net, 0.0)
    head = np.maximum(Gbar * S - ybar, 0.0)
    Ecap, Ewant = np.minimum(Ecap, head), np.minimum(Ewant, head)
    out.update(Ecap=Ecap, Ewant=Ewant, E=E, Eplan=Eplan, lots=lots, lots_plan=lots_plan, lots_cap=lots_cap, shed=p["shed"][:, go])
    out["shed_plan"] = plan["x"][:, col[("ysh", go)]] if plan else z()
    out["v"] = usd_cap.sum() / Ecap.sum() if Ecap.sum() > EPS else 0.0  # USD per GWh of fab energy at this grid
    out["usd"], out["usd_lots_plan"] = usd.sum(), usd_plan.sum()
    cls = np.full(T, "idle", dtype=object)
    live = Ecap > EPS
    cls[live & (E >= Ewant * (1 - 1e-6)) & (Ewant > EPS)] = "whole"
    cls[live & (E < Ewant * (1 - 1e-6)) & (E > EPS)] = "partly"
    cls[live & (E <= EPS) & (Ewant > EPS)] = "none"
    cls[live & (Ewant <= EPS)] = "no wafers"
    out["cls"] = cls
    fuels = {}
    for k, fu in info["fuels"].items():
        sg = fu["slot"]
        burn = np.array([seg.get((go, k), 0.0) for seg in p["segment"]])
        Ig = np.concatenate([[i0[sg]], p["stock"][:, sg]])
        terms = [inst.slot_index[(tail, k)] for _e, tail in fu["feeders"]]
        It = np.concatenate([[sum(i0[s] for s in terms)], p["stock"][:, terms].sum(axis=1)]) if terms else np.zeros(T + 1)
        deliver, outflow = z(), z()
        tails = {tail for _e, tail in fu["feeders"]}
        feed_edges = {e for e, _tail in fu["feeders"]}
        for t, x in enumerate(p["x"]):
            for (e, kk, _lane), q in x.items():
                if kk != k:
                    continue
                if e in feed_edges:
                    deliver[t] += q
                if inst.edges[e].tail in tails:
                    outflow[t] += q
        disp_t = p["disposal"][:, terms].sum(axis=1) if terms else z()
        a_term = np.diff(It) + outflow + disp_t - (outflow - deliver)  # net of what the terminal sent elsewhere
        a_direct = np.diff(Ig) + burn + p["disposal"][:, sg] - deliver  # what reached the grid by any other way
        ration = np.minimum(1.0, Ig[:-1] / fu["threshold"]) if fu["threshold"] > 0 else np.ones(T)
        on_hand = Ig[:-1] + a_direct + deliver
        fuels[k] = {
            "burn": burn,
            "burn_plan": plan["x"][:, col[("G", go, k)]] if plan else z(),
            "Ig": Ig,
            "It": It,
            "deliver": deliver,
            "a_term": a_term,
            "a_direct": a_direct,
            "ration": ration,
            "avail": np.minimum(fu["share"] * Gbar * ration, on_hand),  # what the segment could give as played
            "full": fu["share"] * Gbar,
            "whole_burn": fu["share"] * (ybar + Ecap) / S,  # the fuel a whole week burns
            "base_burn": fu["share"] * ybar / S,
            "per_E": fu["share"] / S,  # the fuel a unit of fab energy burns on top
            "terminal": bool(fu["feeders"]),
            "threshold": fu["threshold"],
            "cap": FILL * fu["room"],
        }
    out["fuels"] = fuels
    return out


def bound(gw, held, charge=False, tank=False, cap=None):
    """The most fab energy over the episode when the fuels in ``held`` are burnt only in weeks the fabs run.

    A week is on (base load served whole, the fabs get up to their draw at capacity) or off (none of a held fuel is
    burnt). Held fuel is used as it arrived (a terminal's arrivals from the week after). The other fuels are taken as
    played: their segments' shortfall in a week comes off the fabs' energy first. ``charge``: each start of a run of on
    weeks costs the rationed fuel its rationing threshold (burnt at the rationed rate after the run, with no fab
    energy). ``tank``: a held fuel beyond ``FILL`` of its terminals' storage is delivered and burnt in an off week.
    """
    T, Ecap = gw["T"], gw["Ecap"].copy()
    for k, fu in gw["fuels"].items():
        if k not in held:
            Ecap = np.maximum(Ecap - np.maximum(fu["full"] - fu["avail"], 0.0), 0.0)
    held = [k for k in held if k in gw["fuels"]]
    best = float(Ecap.sum()) if cap is None else min(float(Ecap.sum()), float(cap))
    if not held or best <= EPS:
        return best
    # the quick way: every sellable week on, the fabs at their draw (scaled down to the cap), is it within the fuel?
    scale = best / float(Ecap.sum())
    fits = True
    for k in held:
        fu = gw["fuels"][k]
        usable = fu["Ig"][0] + fu["It"][0] + np.cumsum(fu["a_direct"]) + np.concatenate([[0.0], np.cumsum(fu["a_term"])[:-1]])
        use = np.cumsum(np.where(Ecap > EPS, fu["base_burn"], 0.0) + fu["per_E"] * Ecap * scale) + (fu["threshold"] if charge else 0.0)
        if np.any(use > usable * (1 + 1e-9) + 1e-6):
            fits = False
            break
    if fits:
        return best
    n_spill = len(held) if tank else 0
    nv = 3 * T + n_spill * T  # on, E, start, spill per held fuel
    c = np.zeros(nv)
    c[T : 2 * T] = -1.0
    rows, lo, hi = [], [], []

    def add(r, low, high):
        rows.append(r)
        lo.append(low)
        hi.append(high)

    for t in range(T):
        r = np.zeros(nv)  # E_t <= Ecap_t on_t
        r[t], r[T + t] = -Ecap[t], 1.0
        add(r, -np.inf, 0.0)
        r = np.zeros(nv)  # start_t >= on_t - on_{t-1}
        r[2 * T + t], r[t] = 1.0, -1.0
        if t:
            r[t - 1] = 1.0
        add(r, 0.0, np.inf)
    for i, k in enumerate(held):
        fu = gw["fuels"][k]
        a_dir, a_term = fu["a_direct"], fu["a_term"]
        start0 = fu["Ig"][0] + fu["It"][0]
        usable = start0 + np.cumsum(a_dir) + np.concatenate([[0.0], np.cumsum(a_term)[:-1]])  # by week t
        total = start0 + np.cumsum(a_dir + a_term)  # in the catchment by the end of week t
        thr = fu["threshold"] if charge else 0.0
        for t in range(T):
            use = np.zeros(nv)
            use[: t + 1] = fu["base_burn"][: t + 1]
            use[T : T + t + 1] = fu["per_E"]
            use[2 * T : 2 * T + t + 1] = thr
            r = use.copy()
            if tank:  # what was spilled before this week is gone too
                r[3 * T + i * T : 3 * T + i * T + t] = 1.0
            add(r, -np.inf, max(usable[t], 0.0) * (1 + 1e-9) + 1e-6)
            if tank:  # the catchment's stock at the end of the week fits the tank
                r = -use
                r[3 * T + i * T : 3 * T + i * T + t + 1] = -1.0
                add(r, -np.inf, fu["cap"] + fu["threshold"] + fu["full"][t] - total[t] + 1e-6)
    if cap is not None:  # no more than the cap: the search stops once it is reached
        r = np.zeros(nv)
        r[T : 2 * T] = 1.0
        add(r, -np.inf, float(cap))
    ub = np.concatenate([(Ecap > EPS).astype(float), Ecap, np.ones(T), np.full(n_spill * T, np.inf)])
    res = milp(
        c,
        constraints=LinearConstraint(np.array(rows), np.array(lo), np.array(hi)),
        integrality=np.concatenate([np.ones(T), np.zeros(nv - T)]),
        bounds=Bounds(np.zeros(nv), ub),
        options={"time_limit": 4.0, "mip_rel_gap": 2e-3},
    )
    return float(res.x[T : 2 * T].sum()) if res.x is not None else float("nan")


def main(task="full", tag="v2", entropy=12345, episodes=None):
    from shockbench_flow.hosting.tasks import task_generator

    inst, _ = task_generator(task)
    st = static(inst)
    episodes = episodes or {"full": 16, "small": 64}[task]
    run = ROOT / "outputs" / "recovery_tune" / f"{task}-e{entropy}-n{episodes}"
    refs = {r["episode"]: r for r in json.loads((run / "references.json").read_text())}
    dumps = {}
    for path in sorted((HERE / "trace" / f"{task}_{tag}").glob("*.pkl"), key=lambda q: int(q.stem)):
        dumps[int(path.stem)] = pickle.loads(path.read_bytes())
    gap = sum((refs[n]["J_naive_cents"] - refs[n]["J_oracle_cents"]) / 100 for n in dumps)
    ours = sum(d["played"]["J"] for d in dumps.values())
    planJ = sum(d["plan"]["J"] for d in dumps.values())
    left = (ours - planJ) / gap
    print(f"{task}: {len(dumps)} episodes {sorted(dumps)}")
    print(f"naive-to-clairvoyant gap {gap / 1e9:,.0f} bn; played above the clairvoyant plan by {(ours - planJ) / 1e9:,.0f} bn = {left:.3f} of the gap")
    name_k = {k: inst.commodities[k].id for k in range(len(inst.commodities))}
    tot = defaultdict(lambda: defaultdict(float))
    marg = defaultdict(float)
    states = defaultdict(lambda: defaultdict(float))
    fuel_tab = defaultdict(lambda: defaultdict(float))
    stranded = defaultdict(float)
    levels = ("B0", "B1", "B2", "B3", "B4")
    for n, d in dumps.items():
        accepted = {row[0]: {k for k, v in row[3].items() if v[2] and v[3]} for row in d["pulse"]["grids"]}
        why = defaultdict(dict)
        for r in d["pulse"]["trace"]:
            why[r["go"]][r["t"]] = r
        for go in st:
            if not st[go]["fabs"]:
                continue
            gw = grid_weeks(inst, st, d, go)
            name = st[go]["id"]
            every = set(gw["fuels"])
            terminal = {k for k, fu in gw["fuels"].items() if fu["terminal"]}
            acc = accepted.get(go, set())
            target = gw["Eplan"].sum()
            b = {
                "B0": bound(gw, every),
                "B1": bound(gw, terminal),
                "B2": bound(gw, acc),
                "B3": bound(gw, acc, charge=True),
                "B4": bound(gw, acc, charge=True, tank=True),
            }
            row = tot[name]
            live = gw["Ecap"] > EPS
            row["weeks"] += live.sum()
            for c in ("whole", "partly", "none", "no wafers"):
                row[c] += (gw["cls"] == c).sum()
            row["cap"] += gw["Ecap"].sum()
            row["plan"] += target
            row["got"] += gw["E"].sum()
            row["got_c"] += min(gw["E"].sum(), target)
            row["usd_cap"] += gw["v"] * gw["Ecap"].sum()
            row["lots"] += gw["lots"].sum()
            row["lots_plan"] += gw["lots_plan"].sum()
            row["usd"] += gw["usd"]
            row["usd_lots_plan"] += gw["usd_lots_plan"]
            for lv in levels:
                row[lv] += b[lv]
                row[lv + "_c"] += min(b[lv], target)  # no more than the clairvoyant plan gives this grid's fabs
            row["wafers"] += np.maximum(gw["Ecap"] - gw["E"], 0.0)[gw["cls"] == "whole"].sum()
            # the fuels that are not held: what holding each would add (the others as the pulse has them)
            for k in every - acc:
                kind = "no terminal" if k not in terminal else "refused (payback)"
                gain = min(bound(gw, acc | {k}, charge=True, tank=True), target) - min(b["B4"], target)
                marg[(name, name_k[k], kind)] += gain
            # the weeks by the pulse's own state, and the fab energy short of capacity and of the plan in them
            for t in range(gw["T"]):
                if not live[t]:
                    continue
                r = why[go].get(t + 1)
                state = "last week" if r is None else r["why"]
                if state == "unfixable":
                    state = "cannot hold: " + ", ".join(sorted(f"{name_k[k]} {c}" for k, c in r["short"].items() if c != "holdable"))
                key = (name, state)
                states[key]["weeks"] += 1
                states[key]["short_cap"] += gw["Ecap"][t] - gw["E"][t]
                states[key]["short_plan"] += gw["Eplan"][t] - gw["E"][t]
                states[key]["got"] += gw["E"][t]
            last = int(np.flatnonzero(live)[-1]) if live.any() else None
            for k, fu in gw["fuels"].items():
                ft = fuel_tab[(name, name_k[k])]
                ft["need"] += fu["whole_burn"][live].sum()
                ft["arrived"] += fu["Ig"][0] + fu["It"][0] + (fu["a_direct"] + fu["a_term"])[: (last + 1 if last is not None else 0)].sum()
                ft["burn"] += fu["burn"].sum()
                ft["burn_plan"] += fu["burn_plan"].sum()
                ft["burn_useful"] += fu["burn"][live].sum()
                ft["burn_whole"] += fu["burn"][live & (gw["cls"] == "whole")].sum()
                ft["terminal"] = fu["terminal"]
                ft["accepted"] = k in acc
                if last is not None and k in acc:
                    left_k = fu["Ig"][last + 1] + fu["It"][last + 1]
                    wb = fu["whole_burn"][last]
                    stranded[name] += min(left_k / wb, 1e9) * gw["Ecap"][last] if wb > 0 else 0.0
    order = [st[go]["id"] for go in st if st[go]["fabs"]]
    P = defaultdict(float)
    print("\n1. Per grid with fabs, summed over the traced episodes. Weeks are grid-weeks in which a lot could still be sold.")
    print("   Fab energy in GWh: at capacity; in the clairvoyant plan; the bounds on our arriving fuel; what the fabs got.")
    print(f"{'grid':9s} {'weeks':>5s} {'whole':>5s} {'partly':>6s} {'none':>5s} {'nowaf':>5s} | {'capacity':>8s} {'plan':>7s} | {'ideal':>7s} {'terminal':>8s} {'accepted':>8s} {'+charge':>8s} {'+tank':>7s} | {'got':>7s} {'got/plan':>8s} {'got/+tank':>9s}")
    for name in order:
        row = tot[name]
        print(
            f"{name:9s} {row['weeks']:5.0f} {row['whole']:5.0f} {row['partly']:6.0f} {row['none']:5.0f} {row['no wafers']:5.0f} | {row['cap']:8.0f} {row['plan']:7.0f} |"
            f" {row['B0_c']:7.0f} {row['B1_c']:8.0f} {row['B2_c']:8.0f} {row['B3_c']:8.0f} {row['B4_c']:7.0f} | {row['got']:7.0f} {row['got'] / row['plan'] if row['plan'] else float('nan'):8.2f} {row['got_c'] / row['B4_c'] if row['B4_c'] else float('nan'):9.2f}"
        )
        for c in ("weeks", "whole", "partly", "none", "no wafers", "cap", "plan", "got", "got_c", "usd", "usd_lots_plan", "lots", "lots_plan", "wafers", "usd_cap") + tuple(lv + "_c" for lv in levels) + levels:
            P[c] += row[c]
        row["v"] = row["usd_cap"] / row["cap"] if row["cap"] else 0.0
        for c in ("plan", "got_c") + tuple(lv + "_c" for lv in levels):
            P["val_" + c] += row[c] * row["v"]
    print(
        f"{'pooled':9s} {P['weeks']:5.0f} {P['whole']:5.0f} {P['partly']:6.0f} {P['none']:5.0f} {P['no wafers']:5.0f} | {P['cap']:8.0f} {P['plan']:7.0f} |"
        f" {P['B0_c']:7.0f} {P['B1_c']:8.0f} {P['B2_c']:8.0f} {P['B3_c']:8.0f} {P['B4_c']:7.0f} | {P['got']:7.0f} {P['got'] / P['plan']:8.2f} {P['got_c'] / P['B4_c']:9.2f}"
    )
    print(f"   (bounds are capped at the plan's energy per grid and episode; uncapped, ideal holding allows {P['B0']:.0f} GWh pooled)")
    # money: the played cost above the plan, shared out by the fab energy missing, each grid's energy at its own worth
    miss = P["val_plan"] - P["val_got_c"]
    scale = (ours - planJ) / miss if miss > 0 else float("nan")
    print(f"\n   In money. Fab energy valued at each grid's chips: the plan's fabs are worth {P['val_plan'] / 1e9:,.0f} bn more than idle fabs, ours {P['val_got_c'] / 1e9:,.0f} bn.")
    print(f"   The played cost above the plan is {scale:.2f} of that difference; the steps below are scaled by it to add up to {left:.3f} of the gap.")
    steps = [
        ("out of reach: our fuel, held ideally, cannot run the plan's lots", "val_plan", "val_B0_c"),
        ("short fuels with no terminal (cannot be held)", "val_B0_c", "val_B1_c"),
        ("short fuels the pulse refuses (payback)", "val_B1_c", "val_B2_c"),
        ("the gas charge at each start of a run", "val_B2_c", "val_B3_c"),
        ("terminal storage (held fuel that overflows)", "val_B3_c", "val_B4_c"),
        ("the pulse's own rules and timing, wafers, stranded fuel", "val_B4_c", "val_got_c"),
    ]
    for label, a, b_ in steps:
        usd = (P[a] - P[b_]) * scale
        print(f"   {label:66s} {usd / 1e9:8,.0f} bn  {usd / gap:6.3f} of the gap")
    print("\n   By grid, the same steps in GWh of fab energy (plan, then what each step takes, then got):")
    print(f"   {'grid':9s} {'plan':>7s} {'reach':>7s} {'no term':>7s} {'refused':>7s} {'charge':>7s} {'tank':>7s} {'rules':>7s} {'got':>7s}  USD m per GWh")
    for name in order:
        r = tot[name]
        print(
            f"   {name:9s} {r['plan']:7.0f} {r['plan'] - r['B0_c']:7.0f} {r['B0_c'] - r['B1_c']:7.0f} {r['B1_c'] - r['B2_c']:7.0f} {r['B2_c'] - r['B3_c']:7.0f}"
            f" {r['B3_c'] - r['B4_c']:7.0f} {r['B4_c'] - r['got_c']:7.0f} {r['got_c']:7.0f}  {r['v'] * scale / 1e6:8.1f}"
        )
    print("\n2a. Fuels not held: what holding each one alone would add (others as the pulse has them; GWh, scaled money, of the gap)")
    for (name, fuel, kind), gain in sorted(marg.items(), key=lambda kv: -kv[1] * tot[kv[0][0]]["v"]):
        if gain > 1.0:
            usd = gain * tot[name]["v"] * scale
            print(f"   {name:9s} {fuel:8s} {kind:18s} {gain:8.0f} GWh  {usd / 1e9:7.1f} bn  {usd / gap:.4f}")
    print("\n2b. Grid-weeks by the pulse's own state: fab energy got, short of capacity, short of the plan (GWh)")
    agg = defaultdict(lambda: defaultdict(float))
    for (name, state), v in states.items():
        for c, x in v.items():
            agg[state][c] += x
        agg[state]["usd_short_plan"] += v["short_plan"] * tot[name]["v"] * scale
    for state, v in sorted(agg.items(), key=lambda kv: -kv[1]["short_cap"]):
        print(f"   {state:44s} weeks {v['weeks']:5.0f}  got {v['got']:7.0f}  short of capacity {v['short_cap']:7.0f}  of the plan {v['short_plan']:7.0f} ({v['usd_short_plan'] / gap:+.3f} of the gap)")
    print("   by grid, the states that are short of the plan by most:")
    for (name, state), v in sorted(states.items(), key=lambda kv: -kv[1]["short_plan"] * tot[kv[0][0]]["v"])[:14]:
        print(f"   {name:9s} {state:44s} weeks {v['weeks']:5.0f}  short of the plan {v['short_plan']:7.0f} GWh ({v['short_plan'] * tot[name]['v'] * scale / gap:+.4f} of the gap)")
    print(f"\n2c. Fab energy at capacity not drawn in whole weeks (wafers, restoration): {P['wafers']:.0f} GWh pooled")
    print("    Held fuel left in tank and grid when the last sellable week ends, in whole weeks' worth of fab energy (GWh):")
    for name in order:
        print(f"      {name:9s} {stranded[name]:8.0f} GWh ({stranded[name] * tot[name]['v'] * scale / gap:.4f} of the gap)")
    print("\n3. Fuel by grid: arrived for the sellable weeks against what whole weeks at capacity need; burnt in whole weeks; the plan's burn")
    print(f"   {'grid':9s} {'fuel':8s} {'held?':10s} {'arrived/need':>12s} {'burnt whole / burnt':>20s} {'burn, ours':>11s} {'burn, plan':>11s} {'plan/ours':>9s}")
    for (name, fuel), ft in sorted(fuel_tab.items(), key=lambda kv: (order.index(kv[0][0]), kv[0][1])):
        kind = "accepted" if ft["accepted"] else ("refused" if ft["terminal"] else "no term.")
        print(
            f"   {name:9s} {fuel:8s} {kind:10s} {ft['arrived'] / ft['need'] if ft['need'] else float('nan'):12.2f} {ft['burn_whole'] / ft['burn_useful'] if ft['burn_useful'] else float('nan'):20.2f}"
            f" {ft['burn']:11.0f} {ft['burn_plan']:11.0f} {ft['burn_plan'] / ft['burn'] if ft['burn'] else float('nan'):9.3f}"
        )
    print(f"\n   lots started, ours over the plan's (sellable weeks): {P['lots'] / P['lots_plan']:.3f}")
    with open(HERE / f"analysis_{task}.json", "w") as f:
        json.dump({"tot": {k: dict(v) for k, v in tot.items()}, "pooled": dict(P), "gap": gap, "left": left, "scale": scale}, f)


if __name__ == "__main__":
    fire.Fire(main)

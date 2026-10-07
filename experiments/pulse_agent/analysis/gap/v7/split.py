"""Where the fuel-free energy still unused in weeks without shedding sits, from the records of a play (rec.py).

Per grid and week t without shedding: the segment that needs no fuel gave ``share * G * load``; what it left unused,
``share * G * (1 - load)``, burnt as fuel instead. It is split by grid, by the fuel that burnt (pro rata to each fuel's
burn: the rationed fuel, a fuel with a terminal that is not rationed, a fuel straight from a source), and by where the
gas trim of agents/mpc_pulse_rec_v6 left the grid the week before (the ration of week t is set by the stock week t-1
closes on): the label ``rec.why`` noted in week t-1.

What a part could be worth at most: the analyst's bank (gap/trim.py), per fuel. Had that fuel alone been throttled to
take the week's slack, it would have burnt less by ``saved``; that is banked under the week's label and spent in later
weeks the same grid shed base load, up to the room left in that fuel's segment and the load shed. Nothing of the
future is used; storage, the rationing rule and what the saved fuel does to the calendar are ignored.

uv run python split.py --task=small
"""

import pickle
import sys
from collections import defaultdict
from pathlib import Path

import fire
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402
import describe as Dsc  # noqa: E402

ORDER = ["week1", "sheds", "on_hand_limits", "no_terminal", "no_reach", "cover", "needs_all", "none_sent", "as_low", "fill", "trim", "trim_fill", "no_rationed", "no_rule", "no_note"]
MEANING = {
    "week1": "week 1: the ration is set by the opening stock",
    "sheds": "the week before shed base load (no trim in a shedding week)",
    "on_hand_limits": "the week before, the gas on hand already limited the segment",
    "no_terminal": "gas has no terminal at this grid (it comes from a source or by pipe)",
    "no_reach": "the terminal's edge was prohibited or carried no flow of the planner",
    "cover": "the cover gate: grid and terminal held over ration_trim_cover weeks",
    "needs_all": "next week was forecast to need the whole gas segment",
    "none_sent": "nothing was delivered and the stock still closed above the target",
    "as_low": "the delivery was already at or under the trim's level",
    "fill": "the terminal was full: it had to pass the gas on",
    "trim": "trimmed the week before; the slack is what the forecast of the need was too high by",
    "trim_fill": "trimmed, but only down to what the full terminal had to pass on",
    "no_rationed": "no rationed fuel observed", "no_rule": "no rationing rule or no fuel-free segment", "no_note": "no note (a week without flows)",
}


def main(task="small", tag="v6", entropy=12345, only=None, detail=False):
    refs = C.references(task, entropy)
    ns = sorted(refs) if only is None else [int(v) for v in (only if isinstance(only, (list, tuple)) else [only])]
    w = Dsc.weights(task, sorted(refs))
    acc = defaultdict(lambda: defaultdict(float))  # (grid, label) -> quantity -> sum
    tot = defaultdict(float)
    short = defaultdict(lambda: defaultdict(float))  # grid -> what was short in its shedding weeks
    cause = defaultdict(float)  # (label, cause) -> unused energy of weeks after the trim's arithmetic ran, score units
    names = {}
    for n in ns:
        d = pickle.loads((HERE / "traj" / f"{tag}_{task}_{entropy}_{n}.pkl").read_bytes())
        inst, _omega, marks, _fb = C.world(task, n, entropy)
        inst = inst.at_digest(marks.instance_digest)
        node = [nd.id for nd in inst.nodes]
        chk, supply = set(inst.chokepoints), set(inst.supply_nodes)
        Gbar, ybar = np.asarray(marks.G_bar), np.asarray(marks.y_bar)
        recs, notes = d["records"], d["why"]
        for go, g in enumerate(inst.grids):
            ga = inst.nodes[g].grid
            fabs = [fo for fo in inst.grid_fabs[go] if inst.nodes[inst.fabs[fo]].fab.e > 0]
            null_share = ga.shares.get(None, 0.0)
            if ga.priority != "base_first" or not fabs or null_share <= 0:
                continue
            gname = names.setdefault(go, node[g])
            kind = {}  # fuel -> "gas" (rationed), "held" (not rationed, a zero-lead feeder that is a plain node), "source"
            for k in ga.fuels:
                feeder = any(k in inst.edges[e].K and inst.edges[e].tau == 0 and inst.edges[e].tail not in chk and inst.edges[e].tail not in supply for e in inst.in_edges[g])
                kind[k] = "gas" if k == ga.rationed else ("held" if feeder else "source")
            bank = {c: defaultdict(float) for c in ("gas", "held")}  # fuel class -> label -> GWh banked
            wv = w[n] * ga.voll
            for ti, r in enumerate(recs):
                t = ti + 1
                Gb, yb = float(Gbar[ti, go]), float(ybar[ti, go])
                shed = float(r.shed[go])
                E = sum(float(r.energy[fo]) for fo in fabs)
                y = float(r.served_load[go])
                burn = {k: float(r.segment.get((go, k), 0.0)) for k in ga.fuels}
                null_cap, null_out = null_share * Gb, float(r.segment.get((go, None), 0.0))
                note = notes.get(t - 1, {}).get(go) if t > 1 else None
                label = "week1" if t == 1 else (note[0] if note else "no_note")
                key = (gname, label)
                tot["grid-weeks"] += 1
                if shed > 1e-9 * max(yb, 1.0):  # a shedding week: the banks are spent
                    tot["shedding grid-weeks"] += 1
                    short[gname]["weeks"] += 1
                    short[gname]["shed GWh"] += shed
                    short[gname]["shed"] += wv * shed
                    for c in ("gas", "held", "source"):
                        room = sum(max(ga.shares[k] * Gb - burn[k], 0.0) for k in ga.fuels if kind[k] == c)
                        short[gname][f"{c} short GWh"] += room
                        short[gname][f"weeks {c} short"] += room > 1e-6 * Gb
                    short[gname]["grid power down GWh"] += max(ga.deliverable - Gb, 0.0)
                    for c in bank:
                        room = sum(max(ga.shares[k] * Gb - burn[k], 0.0) for k in ga.fuels if kind[k] == c)
                        have = sum(bank[c].values())
                        use = min(have, room, shed)
                        if use > 0:
                            for lab in list(bank[c]):
                                part = use * bank[c][lab] / have
                                bank[c][lab] -= part
                                acc[(gname, lab)][f"{c} spent"] += wv * part
                                acc[(gname, lab)][f"{c} spent GWh"] += part
                    continue
                acc[key]["on weeks"] += 1
                if null_cap <= 0:
                    continue
                load = null_out / null_cap
                unused = null_cap - null_out
                if unused <= 1e-6 * null_cap:
                    continue
                need = y + E
                g_av = need / load if load > 0 else 0.0
                slack = g_av - need
                acc[key]["slack weeks"] += 1
                acc[key]["unused GWh"] += unused
                acc[key]["unused"] += wv * unused
                fuel = sum(burn.values())
                for k in ga.fuels:
                    acc[key][f"burnt as {kind[k]}"] += wv * unused * burn[k] / fuel if fuel > 0 else 0.0
                if note and note[1].get("pulsed"):
                    acc[key]["unused, pulsed grid"] += wv * unused
                if note and "close" in note[1] and slack > 0:  # a week after the trim's arithmetic ran: why slack is left
                    i = note[1]
                    r_k = ga.rationed
                    closed = float(recs[ti - 1].stock[inst.slot_index[(g, r_k)]])
                    thr = i["threshold"]
                    above = ga.shares[r_k] * Gb * (min(1.0, closed / thr) - min(1.0, i["close"] / thr))  # ration over the target
                    fab = max(i["need"] - i["y"] - E, 0.0)  # the fabs' draw forecast over the draw they had
                    f_above = min(1.0, max(above, 0.0) / slack)
                    f_fab = min(1.0 - f_above, fab / slack)
                    queued = 0.0  # the part of the forecast draw that rests on wafers a strait had yet to pass on
                    for fb in i.get("fabs", {}).values():
                        left = fb["wafers"] - i["rho"] * fb["lots"] + fb["soon"] + sum(fb["sent1"].values())
                        queued += fb["e"] * (min(fb["cap"], left) - min(fb["cap"], left - fb["strait"])) / fb["R"]
                    cause[(label, "stock closed above the target")] += wv * unused * f_above
                    cause[(label, "fab draw forecast too high")] += wv * unused * f_fab
                    cause[(label, "fab draw too high: wafers waiting at a strait")] += wv * unused * min(f_fab, queued / slack)
                    cause[(label, "other (grid power, base load, other fuels)")] += wv * unused * (1.0 - f_above - f_fab)
                    cause[(label, "of it in the last 6 weeks")] += wv * unused * (t > len(recs) - 6)
                    cause[(label, "of it delivered nothing (stock too high already)")] += wv * unused * f_above * (i["new"] <= 0)
                for c in bank:  # that class alone throttled by the slack, as far as its segment goes
                    av = sum(burn[k] / load for k in ga.fuels if kind[k] == c) if load > 0 else 0.0
                    cut = min(av, slack)
                    if cut <= 0:
                        continue
                    load2 = need / (g_av - cut)
                    saved = av * load - (av - cut) * load2
                    bank[c][label] += saved
                    acc[key][f"{c} saved"] += wv * saved
                    acc[key][f"{c} saved GWh"] += saved
                    if c == "held":  # the gas that burns in its place
                        gas_av = sum(burn[k] / load for k in ga.fuels if kind[k] == "gas")
                        acc[key]["held: more gas burnt"] += wv * gas_av * (load2 - load)
                if detail and note and label in ("trim", "needs_all") and unused > 0.02 * null_cap:
                    i = note[1]
                    print(f"  ep {n} {gname} wk {t} {label}: unused {unused:.0f} load {load:.3f}; forecast need {i.get('need', 0):.0f} others {i.get('others', 0):.0f} needed {i.get('needed', 0):.0f} cap {i.get('cap', 0):.0f}; realised y+E {need:.0f} (E {E:.0f}) G {Gb:.0f} vs seen {i['G']:.0f}")
            for c in bank:
                for lab, left in bank[c].items():
                    acc[(gname, lab)][f"{c} left"] += wv * left
    cols = ["on weeks", "slack weeks", "unused GWh", "unused", "burnt as gas", "burnt as held", "burnt as source", "unused, pulsed grid", "gas saved", "gas spent", "held saved", "held spent", "held: more gas burnt"]
    head = f"{'grid':9s} {'left the trim at':15s} " + " ".join(f"{c[:13]:>13s}" for c in cols)
    print(f"{task} root {entropy}, play {tag}, {len(ns)} episodes; {tot['grid-weeks']:.0f} grid-weeks at grids with fabs, {tot['shedding grid-weeks']:.0f} shedding. Score units (USD at VOLL, pooled) unless GWh or weeks.")
    print(head)
    fmt = lambda c, x: f"{x:13,.0f}" if c in ("on weeks", "slack weeks", "unused GWh") else f"{x:13.4f}"  # noqa: E731
    by_label, by_grid, total = defaultdict(lambda: defaultdict(float)), defaultdict(lambda: defaultdict(float)), defaultdict(float)
    for gname in sorted({g for g, _l in acc}):
        for label in ORDER:
            row = acc.get((gname, label))
            if not row:
                continue
            print(f"{gname:9s} {label:15s} " + " ".join(fmt(c, row.get(c, 0.0)) for c in cols))
            for c in cols:
                by_label[label][c] += row.get(c, 0.0)
                by_grid[gname][c] += row.get(c, 0.0)
                total[c] += row.get(c, 0.0)
    print("by grid")
    for gname, row in by_grid.items():
        print(f"{gname:9s} {'':15s} " + " ".join(fmt(c, row.get(c, 0.0)) for c in cols))
    print("by label")
    for label in ORDER:
        if label in by_label:
            print(f"{'all':9s} {label:15s} " + " ".join(fmt(c, by_label[label].get(c, 0.0)) for c in cols) + f"   {MEANING[label]}")
    print(f"{'all':9s} {'all':15s} " + " ".join(fmt(c, total.get(c, 0.0)) for c in cols))
    print("weeks after the trim's arithmetic ran: why slack is left (unused energy, score units)")
    for (label, what), x in sorted(cause.items()):
        print(f"  {label:10s} {what:52s} {x:.4f}")
    print("shedding weeks: what was short (a segment's cap less its burn; GWh summed over episodes)")
    scols = ["weeks", "shed GWh", "shed", "gas short GWh", "weeks gas short", "held short GWh", "weeks held short", "source short GWh", "weeks source short", "grid power down GWh"]
    print(f"{'grid':9s} " + " ".join(f"{c[:18]:>18s}" for c in scols))
    for gname, row in short.items():
        print(f"{gname:9s} " + " ".join(f"{row.get(c, 0.0):18.4f}" if c == "shed" else f"{row.get(c, 0.0):18,.0f}" for c in scols))


if __name__ == "__main__":
    fire.Fire(main)

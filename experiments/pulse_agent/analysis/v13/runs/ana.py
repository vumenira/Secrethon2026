"""Size the three misfires on recorded plays (rec.py). Reads only; plays nothing.

uv run python <this> ends|runs|waf|plan --task=full --entropy=12345 --episodes=16 [--name=r13rec_off] [--only=3]
"""

import gzip
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import fire
import numpy as np


ROOT = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026")
OUT = ROOT / "outputs" / "recovery_tune"
AG = ROOT / "agents" / "mpc_pulse_rec_v13r"
MEMORY = 4.0


def instance(task):
    sys.path.insert(0, str(AG))
    import agent  # noqa: F401
    import pulse
    from sbflow.policies.naive_parts import cached_plan
    from sbflow.policies.registry import PolicyContext

    raw = json.loads((AG / "instances" / f"{task}.json").read_text(encoding="utf-8"))
    inst, _plan = cached_plan({"instance": raw}, PolicyContext().fq_quantile)
    return inst, pulse.Pulse(inst, agent.PARAMS)


def episodes_of(task, entropy, episodes, name, only=""):
    d = OUT / "r13" / "rec" / f"{task}-e{entropy}-n{episodes}" / name
    chosen = None if only == "" else {int(v) for v in (only if isinstance(only, (tuple, list)) else str(only).split(","))}
    for p in sorted(d.glob("ep*.pkl.gz"), key=lambda p: int(p.stem.split(".")[0][2:])):
        n = int(p.stem.split(".")[0][2:])
        if chosen is not None and n not in chosen:
            continue
        with gzip.open(p, "rb") as fh:
            yield n, pickle.load(fh)


def refs(task, entropy, episodes):
    p = OUT / f"{task}-e{entropy}-n{episodes}" / "references.json"
    return {r["episode"]: r for r in json.loads(p.read_text())} if p.exists() else {}


def ends(task="full", entropy=12345, episodes=16, name="r13rec_off", only="", show=40):
    """schedule_ends: the runs it ended at a grid that was not pulsed, and what followed."""
    inst, P = instance(task)
    N, K = inst.nodes, inst.commodities
    total = defaultdict(float)
    shown = 0
    for n, ep in episodes_of(task, entropy, episodes, name, only):
        rec, trace = ep["records"], {tr["t"]: tr for tr in ep["trace"]}
        per = defaultdict(list)
        for t, tr in trace.items():
            for (_t, go, rule, moved) in tr["log"]:
                if rule != "lapse":
                    continue
                pulsed = t - tr["held"].get(go, -10**9) <= MEMORY
                per[go].append((t, pulsed, moved))
        for go, events in per.items():
            g = inst.grids[go]
            ga = N[g].grid
            fabs = [fo for fo in inst.grid_fabs[go] if N[inst.fabs[fo]].fab.e > 0]
            r = ga.rationed
            share = ga.shares[r]
            thr = inst.params.psi * ga.ibar[r]
            for t, pulsed, moved in events:
                total["lapses"] += 1
                if pulsed:
                    total["lapses at a pulsed grid"] += 1
                    continue
                total["lapses by schedule_ends"] += 1
                tr = trace[t]
                gr = tr["grids"][go]
                fu = gr["fuels"][r]
                burn = share * gr["G"]
                back = [u for u in range(max(1, t - 3), t + 1) if u in trace]
                infl = np.mean([trace[u]["grids"][go]["fuels"][r]["inflow"] for u in back])
                burnt = np.mean([rec[u - 1].segment.get((go, r), 0.0) for u in back])
                tot_now = fu["tank"] + fu["here"]
                first = trace[back[0]]["grids"][go]["fuels"][r]
                tot_then = first["tank"] + first["here"]
                # what followed: the fabs' energy and the shed load of the next weeks
                nxt = []
                for u in range(t, min(t + 5, inst.T) + 1):
                    ru = rec[u - 1]
                    en = float(sum(ru.energy[fo] for fo in fabs))
                    wafers = sum(min(trace[u]["grids"][go]["fabs"][fo]["alpha"] * trace[u]["grids"][go]["fabs"][fo]["R"] * N[inst.fabs[fo]].fab.cap0,
                                     trace[u]["grids"][go]["fabs"][fo]["wafers"] + trace[u]["grids"][go]["fabs"][fo]["warr"]) * N[inst.fabs[fo]].fab.e
                                 / max(trace[u]["grids"][go]["fabs"][fo]["R"], 1e-9) for fo in fabs) if u in trace else 0.0
                    nxt.append((en, wafers, float(ru.shed[go]), ru.segment.get((go, r), 0.0) / burn))
                dark = float(rec[t].shed[go]) * ga.voll / 1e9 if t < inst.T else 0.0
                lostfab = max(nxt[1][1] - nxt[1][0], 0.0) if len(nxt) > 1 else 0.0
                worth = P.gain[go] / P.full_draw[go] + ga.voll
                down = infl < burnt * 0.98
                total["ends: terminal running down" if down else "ends: terminal NOT running down"] += 1
                total["ends: bn shed next week" + (" (running down)" if down else " (not running down)")] += dark
                total["ends: bn fab energy lost next week" + (" (running down)" if down else " (not running down)")] += lostfab * worth / 1e9
                if shown < show:
                    shown += 1
                    want_now = nxt[0][1]
                    print(
                        f"ep {n:3d} wk {t:3d} {N[g].id:8s} here {fu['here'] / burn:5.2f} arr {fu['arr'] / burn:4.2f} tank {fu['tank'] / burn:5.2f} "
                        f"inflow {fu['inflow'] / burn:4.2f} rate {tr['rate'].get((go, r), 0) / burn:4.2f} thr {thr / burn:4.2f} plan {fu['planned'] / burn:4.2f} sent {fu['sent'] / burn:4.2f} | "
                        f"4wk inflow {infl / burn:4.2f} burnt {burnt / burn:4.2f} stock {tot_then / burn:5.2f}->{tot_now / burn:5.2f} {'DOWN' if down else 'flat'} | "
                        f"want/draw {want_now / P.full_draw[go]:4.2f} | next (fab got/want, shed/burn, seg): "
                        + " ".join(f"[{a / max(b, 1e-9):4.2f} {c / burn:4.2f} {s:4.2f}]" for a, b, c, s in nxt)
                    )
    print("---- totals")
    for k, v in sorted(total.items()):
        print(f"  {k}: {v:.1f}")


def runs(task="full", entropy=12345, episodes=16, name="r13rec_off", only="", show=60):
    """schedule_runs: weeks a charge was possible for one whole week and waited for a longer run."""
    inst, P = instance(task)
    N = inst.nodes
    total = defaultdict(float)
    shown = 0
    for n, ep in episodes_of(task, entropy, episodes, name, only):
        rec, trace = ep["records"], {tr["t"]: tr for tr in ep["trace"]}
        for go, g, fabs, fuels, shares in P.grids:
            ga = N[g].grid
            r = ga.rationed
            if r not in fuels:
                continue
            share, thr, feeders, worth = fuels[r]
            if not feeders or not worth or thr <= share * ga.deliverable:
                continue
            room = sum(st for _e, _t, st, _s in feeders)
            spell = 0
            for t in range(1, inst.T + 1):
                tr = trace.get(t)
                if tr is None:
                    continue
                gr = tr["grids"][go]
                fu = gr["fuels"][r]
                burn = share * gr["G"]
                if burn <= 0:
                    continue
                on_hand = fu["here"] + fu["arr"]
                ration = min(1.0, fu["here"] / thr)
                pulsed = t - tr["held_after"].get(go, -10**9) <= 0  # held this week
                ahead = max(P.last[go] - t, 0)
                charge = max(thr - on_hand + burn * ration, 0.0)
                rate = min(max(fu["inflow"], tr["rate"].get((go, r), 0.0)), burn)
                nn = 1 + int(max(P.fill * room - thr, 0.0) // burn)
                nn = min(nn, int(max(ahead, 1)))
                full = fu["tank"] + fu["inflow"] >= P.fill * room
                one = charge <= fu["tank"] + 1e-6 and fu["tank"] - charge + fu["inflow"] >= 0.0
                more = P.run_needs(burn, thr, rate, nn) if nn > 1 else 0.0
                waits = pulsed and ration < 1 and one and not full and nn > 1 and fu["tank"] - charge + fu["inflow"] < more - 1e-6 and fu["sent"] < charge - 1e-6
                if waits:
                    spell += 1
                    total["weeks a charge waited for a longer run"] += 1
                    need_w = (more - (fu["tank"] - charge + fu["inflow"])) / max(rate, 1e-9)
                    total["of them, the run more than 12 weeks away"] += need_w > 12
                    if shown < show:
                        shown += 1
                        en = float(sum(rec[t - 1].energy[fo] for fo in fabs))
                        print(f"ep {n:3d} wk {t:3d} {N[g].id:8s} spell {spell:2d} tank {fu['tank'] / burn:5.2f} inflow {fu['inflow'] / burn:4.2f} rate {rate / burn:4.2f} here {fu['here'] / burn:4.2f} "
                              f"charge {charge / burn:4.2f} n {nn} needs {more / burn:5.2f} weeks to gather {need_w:6.1f} ahead {ahead} fab energy {en:5.0f}")
                else:
                    if spell:
                        total["spells"] += 1
                        total["longest spell"] = max(total["longest spell"], spell)
                    spell = 0
    print("---- totals")
    for k, v in sorted(total.items()):
        print(f"  {k}: {v:.1f}")


def waf(task="full", entropy=12345, episodes=16, name="r13rec_off", only="", show=0, lo=2, hi=5):
    """The top-up: wafers added, against the lots the plan itself wants at the fab; the energy the fabs then drew."""
    inst, P = instance(task)
    N, K = inst.nodes, inst.commodities
    packaged = defaultdict(set)
    for o in inst.osats:
        for raw, pk in N[o].osat.packages.items():
            packaged[raw].add(pk)
    R = refs(task, entropy, episodes)
    grand = defaultdict(float)
    for n, ep in episodes_of(task, entropy, episodes, name, only):
        rec, trace = ep["records"], {tr["t"]: tr for tr in ep["trace"]}
        tot = defaultdict(float)
        for go, g, fabs, fuels, shares in P.grids:
            ga = N[g].grid
            for fo in fabs:
                f = inst.fabs[fo]
                fa = N[f].fab
                goods = packaged[fa.product]
                dos = [do for do, d in enumerate(inst.demands) if d.k in goods]
                for t in range(1, inst.T + 1):
                    tr = trace.get(t)
                    if tr is None or tr["plan"] is None:
                        continue
                    fb = tr["grids"][go]["fabs"][fo]
                    cap = fb["alpha"] * fb["R"] * fa.cap0
                    pulsed = t - tr["held_after"].get(go, -10**9) <= MEMORY
                    pl = tr["plan"]
                    H = pl["H"]
                    span = pl["p"][fo, lo : min(hi, H)]
                    des = float(np.mean(span)) if span.size else float("nan")
                    U = float(np.nansum(pl["U"][dos, :]))
                    added = fb["sent"] - fb["planned"]
                    en = float(rec[t - 1].energy[fo])
                    lots = float(rec[t - 1].lots_started[fo])
                    if not pulsed:
                        continue
                    tot["pulsed fab-weeks"] += 1
                    tot["fab energy at pulsed grids, bn at VOLL"] += en * ga.voll / 1e9
                    low = cap > 0 and des == des and des < 0.9 * cap
                    if low:
                        tot["pulsed fab-weeks, plan wants < 0.9 cap"] += 1
                        tot["fab energy there, bn at VOLL"] += en * ga.voll / 1e9
                        tot["lots there beyond the plan's own weekly lots, bn at VOLL"] += max(lots - des, 0.0) * fa.e / max(fb["R"], 1e-9) * ga.voll / 1e9
                    if U <= 1e-6 * sum(inst.demands[do].dbar for do in dos):
                        tot["pulsed fab-weeks, product not short in plan"] += 1
                        tot["fab energy, product not short in plan, bn at VOLL"] += en * ga.voll / 1e9
                    if added > 1e-6 * max(cap, 1.0):
                        tot["top-ups"] += 1
                        tot["wafers topped up (weeks of capacity)"] += added / max(cap, 1.0)
                        if low:
                            tot["top-ups where plan wants < 0.9 cap"] += 1
                            tot["wafers topped up there (weeks of capacity)"] += added / max(cap, 1.0)
                        if show and grand["shown"] < show:
                            grand["shown"] += 1
                            print(f"ep {n:3d} wk {t:3d} {N[f].id:18s} cap {cap:8.0f} wafers {fb['wafers'] / max(cap, 1):4.2f} planned {fb['planned'] / max(cap, 1):4.2f} added {added / max(cap, 1):4.2f} "
                                  f"plan p/cap {' '.join(f'{v / max(cap, 1):4.2f}' for v in pl['p'][fo, :8])} U {U:9.0f} chips {fb['chips'] / fa.cap0:4.2f}")
        ref = R.get(n, {})
        print(f"ep {n:3d} level {ref.get('stratum')} J {ep['row']['J_policy_cents'] / 1e11:8.1f}: " + "; ".join(f"{k} {v:.1f}" for k, v in tot.items()))
        for k, v in tot.items():
            grand[k] += v
    print("---- totals")
    for k, v in grand.items():
        print(f"  {k}: {v:.1f}")


def fabs(task="full", entropy=12345, episodes=16, name="r13rec_off", only="", lo=2, hi=6):
    """Per fab, over the weeks its grid is pulsed: the lots started against the lots the plan itself wanted."""
    inst, P = instance(task)
    N, K = inst.nodes, inst.commodities
    agg = defaultdict(lambda: defaultdict(float))
    for n, ep in episodes_of(task, entropy, episodes, name, only):
        rec, trace = ep["records"], {tr["t"]: tr for tr in ep["trace"]}
        for go, g, fabs_, fuels, shares in P.grids:
            ga = N[g].grid
            for fo in fabs_:
                f = inst.fabs[fo]
                fa = N[f].fab
                lots = des = added = weeks = 0.0
                for t in range(1, inst.T + 1):
                    tr = trace.get(t)
                    if tr is None or tr["plan"] is None or t > P.last[go]:
                        continue
                    if t - tr["held_after"].get(go, -10**9) > MEMORY:
                        continue
                    fb = tr["grids"][go]["fabs"][fo]
                    cap = max(fb["alpha"] * fb["R"] * fa.cap0, 1.0)
                    span = tr["plan"]["p"][fo, lo : min(hi, tr["plan"]["H"])]
                    if not span.size:
                        continue
                    weeks += 1
                    lots += float(rec[t - 1].lots_started[fo]) / cap
                    des += float(np.mean(span)) / cap
                    added += (fb["sent"] - fb["planned"]) / cap
                a = agg[N[f].id]
                a["weeks"] += weeks
                a["lots"] += lots
                a["des"] += des
                a["added"] += added
                a["excess"] += max(lots - des, 0.0)
                a["excess_bn"] += max(lots - des, 0.0) * fa.cap0 * fa.e * ga.voll / 1e9
                a["short"] += max(des - lots, 0.0)
    print(f"{'fab':20s} pulsed weeks | lots started | plan's lots | topped up | excess over plan (weeks of cap, bn of base load at VOLL) | short of plan")
    tot = 0.0
    for name_, a in sorted(agg.items()):
        tot += a["excess_bn"]
        print(f"{name_:20s} {a['weeks']:7.0f} {a['lots']:8.1f} {a['des']:8.1f} {a['added']:7.1f} {a['excess']:7.1f} {a['excess_bn']:7.1f} {a['short']:7.1f}")
    print(f"excess in all: {tot:.1f} bn over the set")


if __name__ == "__main__":
    fire.Fire({"ends": ends, "runs": runs, "waf": waf, "fabs": fabs})

"""Per grid and episode: what the week-1 plan expected, what the agent's own play did afterwards, and (Small) what the
analyst's one-grid hold of weeks 2 to 4 changed (left/playhold_small_2_4_0.9_<grid>.json, against u_v6).

uv run python look.py grid --task=small --tag=base
"""

import json
import pickle
from pathlib import Path

import fire
import numpy as np

import play as P


LEFT = Path(__file__).resolve().parent.parent / "left"


def features(task, tag, n, inst):
    d = pickle.loads(P.path(task, tag, n).read_bytes())
    slot_of = {(s.node, s.k): i for i, s in enumerate(inst.stock_slots)}
    chk, supply = set(inst.chokepoints), set(inst.supply_nodes)
    plan = d["plans"][1]
    out = {}
    for go, g in enumerate(inst.grids):
        ga = inst.nodes[g].grid
        fabs = [fo for fo in inst.grid_fabs[go] if inst.nodes[inst.fabs[fo]].fab.e > 0]
        if not fabs or ga.priority != "base_first":
            continue
        draw = sum(inst.nodes[inst.fabs[fo]].fab.cap0 * inst.nodes[inst.fabs[fo]].fab.e for fo in fabs)
        row = {"draw": draw}
        for k in ga.fuels:
            if (go, k) in plan["short"]:
                sh, cap = plan["short"][(go, k)], plan["cap"][(go, k)]
                row[f"short_{k}"] = float(sh[3:12].sum() / max(cap[3:12].sum(), 1e-9))
                first = np.flatnonzero(sh > 0.1 * cap)
                row[f"first_{k}"] = int(first[0]) + 1 if first.size else 0
            feeders = [e for e in inst.in_edges[g] if k in inst.edges[e].K and inst.edges[e].tau == 0 and inst.edges[e].tail not in chk and inst.edges[e].tail not in supply]
            row[f"term_{k}"] = bool(feeders)
        y = d["y_bar"][:, go]
        shed = d["shed"][:, go] > 1e-6 * y
        row["shed_5_12"] = int(shed[4:12].sum())
        row["shed_5_20"] = int(shed[4:20].sum())
        row["shed_2_4"] = int(shed[1:4].sum())
        E = d["energy"][:, fabs].sum(axis=1) / draw
        row["E_2_4"] = float(E[1:4].mean())
        row["E_5_12"] = float(E[4:12].mean())
        row["E_13_30"] = float(E[12:30].mean())
        whole = ~shed
        row["E_whole_5_20"] = float(E[4:20][whole[4:20]].mean()) if whole[4:20].any() else float("nan")
        fill = []
        for fo in fabs:
            f = inst.fabs[fo]
            fa = inst.nodes[f].fab
            st = inst.stock_slots[slot_of[(f, fa.product)]].storage
            fill.append(float(d["stock"][0:4, slot_of[(f, fa.product)]].mean() / st) if st else float("nan"))
        row["chips_1_4"] = max(fill)
        row["planp_2_5"] = float(np.mean([plan["p"][fo][1:5].mean() / max(plan["pcap"][fo][1:5].mean(), 1e-9) for fo in fabs]))
        row["planp_6_12"] = float(np.mean([plan["p"][fo][5:12].mean() / max(plan["pcap"][fo][5:12].mean(), 1e-9) for fo in fabs]))
        out[go] = row
    return out, d


def grid(task="small", tag="base", trig=0.1):
    from shockbench_flow.hosting.tasks import task_generator

    inst, _params = task_generator(task)
    node, K = P.names(inst)
    refs = P.references(task)
    v6 = P.cached(task, "u_v6") if task == "small" else {}
    rows = {}
    for n in sorted(refs):
        if P.path(task, tag, n).exists():
            rows[n], _d = features(task, tag, n, inst)
    gos = sorted({go for r in rows.values() for go in r})
    for go in gos:
        g = inst.grids[go]
        ga = inst.nodes[g].grid
        r = ga.rationed
        name = node[g]
        hold = None
        f = LEFT / f"playhold_small_2_4_0.9_{name[5:]}.json"
        if task == "small" and f.exists():
            hold = {int(k): v for k, v in json.loads(f.read_text())["J"].items()}
        others = [k for k in ga.fuels if k != r]
        print(f"{name}: rationed {K[r]}; terminal for " + ", ".join(f"{K[k]} {rows[min(rows)][go][f'term_{k}']}" for k in ga.fuels) + f"; fabs' draw {rows[min(rows)][go]['draw']:.0f}")
        print("   ep L | gas short4-12 first | " + " ".join(f"{K[k][:5]} short first |" for k in others) + " shed wk2-4 5-12 5-20 | E/draw wk2-4 5-12 13-30 whole5-20 | chips1-4 planp2-5 6-12 | hold d")
        fired, quiet = [], []
        for n, per in rows.items():
            x = per[go]
            D = refs[n]["J_naive_cents"] - refs[n]["J_oracle_cents"]
            dd = (v6[n] - hold[n]) / D if hold and n in hold else float("nan")
            s = x.get(f"short_{r}", float("nan"))
            (fired if s >= trig else quiet).append((n, x, dd))
            if s >= trig or x["shed_5_12"] >= 2:
                oth = " ".join(f"{x.get(f'short_{k}', float('nan')):10.2f} {x.get(f'first_{k}', 0):5d} |" for k in others)
                print(f"   {n:2d} {refs[n]['stratum']} | {s:13.2f} {x.get(f'first_{r}', 0):5d} | {oth} {x['shed_2_4']:9d} {x['shed_5_12']:4d} {x['shed_5_20']:4d} | {x['E_2_4']:11.2f} {x['E_5_12']:4.2f} {x['E_13_30']:5.2f} {x['E_whole_5_20']:9.2f} | {x['chips_1_4']:8.2f} {x['planp_2_5']:8.2f} {x['planp_6_12']:4.2f} | {dd:+.4f}")
        ok = sum(1 for _n, x, _d in fired if x["shed_5_12"] >= 2)
        miss = sum(1 for _n, x, _d in quiet if x["shed_5_12"] >= 2)
        print(f"   trigger {trig}: fired in {len(fired)} episodes, of them {ok} shed in 2 or more of weeks 5-12; not fired but shed: {miss}")
        if hold:
            dd = np.array([d for _n, _x, d in fired])
            print(f"   the analyst's one-grid hold, fired episodes: mean {np.nanmean(dd):+.4f}, better {int((dd > 1e-6).sum())}, worse {int((dd < -1e-6).sum())}")


if __name__ == "__main__":
    fire.Fire({"grid": grid})

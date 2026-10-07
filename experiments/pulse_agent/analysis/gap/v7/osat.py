"""Packaged chips disposed of at OSATs in a recorded play, and the raw chips sent toward a store that was already full.

uv run python osat.py --task=small                  the totals by OSAT and good
uv run python osat.py --task=small --n=8 --osat=osat_my --good=chip_le     one store week by week
"""

import pickle
import sys
from collections import defaultdict
from pathlib import Path

import fire

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402
import describe as Dsc  # noqa: E402


def ends(inst):
    """action slot -> the node its goods end at (the edge's head, or the last edge of its lane)."""
    return [inst.edges[e if lane is None else inst.lanes[lane].edges[-1]].head for e, _k, lane in inst.action_slots]


def main(task="small", tag="v6", entropy=12345, n=None, osat=None, good=None, top=10):
    refs = C.references(task, entropy)
    ns = sorted(refs) if n is None else [int(n)]
    w = Dsc.weights(task, sorted(refs))
    acc = defaultdict(lambda: defaultdict(float))
    per = []
    for ep in ns:
        d = pickle.loads((HERE / "traj" / f"{tag}_{task}_{entropy}_{ep}.pkl").read_bytes())
        inst, _omega, marks, _fb = C.world(task, ep, entropy)
        inst = inst.at_digest(marks.instance_digest)
        node = [nd.id for nd in inst.nodes]
        K = [c.id for c in inst.commodities]
        end = ends(inst)
        recs = d["records"]
        pi = {}
        for dm in inst.demands:
            pi[dm.k] = max(pi.get(dm.k, 0.0), dm.pi)
        lost_ep = 0.0
        for oi, o in enumerate(inst.osats):
            oa = inst.nodes[o].osat
            for raw, pk in sorted(oa.packages.items()):
                s_pk, s_raw = inst.slot_index[(o, pk)], inst.slot_index[(o, raw)]
                slot = inst.stock_slots[s_pk]
                toward = [s for s, (_e, k, _lane) in enumerate(inst.action_slots) if k == raw and end[s] == o]
                out_edges = [e for e in inst.out_edges[o] if pk in inst.edges[e].K]
                key = (node[o], K[pk])
                a = acc[key]
                a["storage"] = slot.storage
                a["thr"] = oa.thr
                a["tau"] = oa.tau
                show = n is not None and (osat is None or node[o] == osat) and (good is None or K[pk] == good)
                if show:
                    print(f"{task} episode {ep} {node[o]} {K[pk]} (from {K[raw]}): storage {slot.storage:,.0f} thr {oa.thr:,.0f} tau {oa.tau}; disposal fee {slot.disposal if hasattr(slot, 'disposal') else float('nan')}; outlets " + ", ".join(f"{node[inst.edges[e].head]} (tau {inst.edges[e].tau})" for e in out_edges))
                    print(f"{'wk':>3s} {'R':>5s} {'raw in':>9s} {'raw stock':>9s} {'started':>9s} {'out of wip':>10s} {'stock':>10s} {'left':>9s} {'disposed':>9s} {'raw sent':>9s}  (thousands)")
                full_before = False
                for ti, r in enumerate(recs):
                    disposed = float(r.disposal[s_pk])
                    left = sum(q for (e, k, _lane), q in r.x.items() if k == pk and e in out_edges)
                    sent = sum(float(r.executed.get(s, 0.0)) for s in toward)
                    started = float(r.packaged.get((oi, pk), 0.0))
                    stock = float(r.stock[s_pk])
                    a["disposed"] += disposed
                    a["disposed, score"] += w[ep] * disposed * pi.get(pk, 0.0)
                    lost_ep += disposed * pi.get(pk, 0.0)
                    a["raw sent"] += sent
                    if full_before:  # the store was full when the week began: the overflow was visible
                        a["disposed, store full at the week's start"] += disposed
                        a["raw sent, store full at the week's start"] += sent
                        a["weeks full"] += 1
                    if show:
                        prev = recs[ti - 1].stock if ti else None
                        arrived = float(r.stock[s_raw]) + started - (float(prev[s_raw]) if ti else float("nan"))
                        matured = stock + disposed + left - (float(prev[s_pk]) if ti else float("nan"))
                        print(f"{ti + 1:3d} {float(marks.R_osat[ti, oi]):5.2f} {arrived / 1e3:9.0f} {float(r.stock[s_raw]) / 1e3:9.0f} {started / 1e3:9.0f} {matured / 1e3:10.0f} {stock / 1e3:10.0f} {left / 1e3:9.0f} {disposed / 1e3:9.0f} {sent / 1e3:9.0f}{'  full' if full_before else ''}")
                    full_before = stock >= slot.storage * (1 - 1e-9)
        per.append((w[ep] * lost_ep, ep, refs[ep]["stratum"]))
    if n is None:
        print(f"{task} root {entropy}, play {tag}, {len(ns)} episodes: packaged chips at OSATs (millions; score = at the shortage penalty, pooled)")
        cols = ["storage", "thr", "disposed", "disposed, store full at the week's start", "raw sent", "raw sent, store full at the week's start", "weeks full"]
        for key, a in sorted(acc.items()):
            print(f"  {key[0]:10s} {key[1]:10s} " + "; ".join(f"{c} {a[c] / (1 if c == 'weeks full' else 1e6):,.2f}" for c in cols) + f"; disposed at the penalty {a['disposed, score']:.4f}")
        print(f"  all: disposed {sum(a['disposed'] for a in acc.values()) / 1e6:,.2f} M, {sum(a['disposed, score'] for a in acc.values()):.4f} at the penalty; with the store full at the week's start {sum(a[cols[3]] for a in acc.values()) / 1e6:,.2f} M; raw sent then {sum(a[cols[5]] for a in acc.values()) / 1e6:,.2f} M")
        per.sort(reverse=True)
        print("  most: " + ", ".join(f"ep {ep} (level {s}) {x:.4f}" for x, ep, s in per[: int(top)]))


if __name__ == "__main__":
    fire.Fire(main)

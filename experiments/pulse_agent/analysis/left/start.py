"""The starting position of a task: stock, shipments under way and work in process, against weekly rates.

uv run python start.py --task=small
"""

from collections import defaultdict

import fire
import numpy as np


def main(task="small"):
    from shockbench_flow.hosting.tasks import task_generator

    inst, _params = task_generator(task)
    node = [nd.id for nd in inst.nodes]
    K = [c.id for c in inst.commodities]
    ist = inst.initial_state
    stock = defaultdict(float)
    for nd, k, q in ist.stock:
        stock[(nd, k)] += q
    arr = defaultdict(lambda: defaultdict(float))  # (head, k) -> week -> qty
    for s in ist.pipeline:
        arr[(inst.edges[s.edge].head, s.k)][s.arrival_week] += s.qty
    print(f"{task}: T {inst.T}; fabs")
    for fo, f in enumerate(inst.fabs):
        fa = inst.nodes[f].fab
        wip = defaultdict(float)
        for w in ist.fab_wip:
            if w.node == f:
                wip[w.out_week] += w.qty
        a = arr[(f, fa.input)]
        print(
            f"  {node[f]:18s} cap0 {fa.cap0:9.0f} tau {fa.tau} e {fa.e}; wafers at the fab {stock[(f, fa.input)]:10.0f} ({stock[(f, fa.input)] / fa.cap0:.2f} wk);"
            f" arriving wk1..8 {' '.join(f'{a.get(w, 0) / fa.cap0:.2f}' for w in range(1, 9))} (weeks of capacity);"
            f" raw chips at the fab {stock[(f, fa.product)]:9.0f}; wip out wk1..8 {' '.join(f'{wip.get(w, 0) / fa.cap0:.2f}' for w in range(1, 9))}"
        )
        outs = [(e, inst.edges[e]) for e in inst.out_edges[f]]
        ins = [(e, inst.edges[e]) for e in inst.in_edges[f]]
        print(f"      in: {[(node[ed.tail], ed.tau, round(ed.u0 or 0), [K[k] for k in ed.K]) for _e, ed in ins]}")
        print(f"      out: {[(node[ed.head], ed.tau, round(ed.u0 or 0), [K[k] for k in ed.K]) for _e, ed in outs]}")
    print("grids and their terminals (fuel in weeks of full-rate burn)")
    for go, g in enumerate(inst.grids):
        ga = inst.nodes[g].grid
        for k in ga.fuels:
            burn = ga.shares[k] * ga.deliverable
            line = f"  {node[g]:8s} {K[k]:8s} burn {burn:8.0f}; at the grid {stock[(g, k)] / burn:5.2f} wk"
            if k == ga.rationed:
                line += f" (ration threshold {inst.params.psi * ga.ibar[k] / burn:.2f} wk)"
            a = arr[(g, k)]
            if a:
                line += f"; arriving at the grid wk1..8 {' '.join(f'{a.get(w, 0) / burn:.2f}' for w in range(1, 9))}"
            for e in inst.in_edges[g]:
                ed = inst.edges[e]
                if k in ed.K:
                    tl = ed.tail
                    a = arr[(tl, k)]
                    line += f"; {node[tl]} (lead {ed.tau}) holds {stock[(tl, k)] / burn:5.2f} wk, arriving wk1..10 {' '.join(f'{a.get(w, 0) / burn:.2f}' for w in range(1, 11))}"
            print(line)
    print("osats")
    for oo, o in enumerate(inst.osats):
        oa = inst.nodes[o].osat
        for kr, kp in oa.packages.items():
            wip = defaultdict(float)
            for w in ist.osat_wip:
                if w.node == o and w.k == kp:
                    wip[w.out_week] += w.qty
            a = arr[(o, kr)]
            print(f"  {node[o]:14s} thr {oa.thr:9.0f} tau {oa.tau}; {K[kr]} at hand {stock[(o, kr)]:10.0f}, arriving wk1..8 {' '.join(f'{a.get(w, 0):9.0f}' for w in range(1, 9))}; {K[kp]} at hand {stock[(o, kp)]:10.0f}, wip out {dict(wip)}")
    print("sinks")
    for do, d in enumerate(inst.demands):
        a = arr[(d.node, d.k)]
        db = d.dbar if d.dbar > 0 else 1.0
        print(f"  {node[d.node]:10s} {K[d.k]:9s} dbar {d.dbar:9.0f} pi {d.pi:9.0f} backlog {d.backlog}; at hand {stock[(d.node, d.k)]:10.0f} ({stock[(d.node, d.k)] / db:.2f} wk); arriving wk1..8 {' '.join(f'{a.get(w, 0) / db:.2f}' for w in range(1, 9))}")
        print(f"      in: {[(node[inst.edges[e].tail], inst.edges[e].tau, round(inst.edges[e].u0 or 0)) for e in inst.in_edges[d.node] if d.k in inst.edges[e].K]}")
    print("queues at straits (initial lots), by good")
    q = defaultdict(float)
    for lt in ist.queue_lots:
        q[(node[lt.chokepoint], K[lt.k])] += lt.qty
    for key, v in sorted(q.items()):
        print(f"  {key[0]:14s} {key[1]:10s} {v:12.0f}")
    print("goods: value v, salvage examples")
    for k, c in enumerate(inst.commodities):
        print(f"  {c.id:12s} v {c.v:10.1f} disposal {c.disposal_cost:8.1f} pool {c.pool} override {c.override}")


if __name__ == "__main__":
    fire.Fire(main)

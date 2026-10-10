"""Print the grids with fabs of an instance as the pulse sees them (no play)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agents" / "mpc_pulse_rec_v13s"))
import json

import fire


def main(task="full"):
    from shockbench_flow.hosting.tasks import task_generator

    inst, params = task_generator(task)
    import pulse  # the agent's
    import agent as ag  # noqa: F401  (PARAMS)

    p = pulse.Pulse(inst, ag.PARAMS)
    names = [n.name if hasattr(n, "name") else str(i) for i, n in enumerate(inst.nodes)]
    kn = [c.name if hasattr(c, "name") else str(i) for i, c in enumerate(inst.commodities)]
    print("T", inst.T, "psi", inst.params.psi, "grids", len(inst.grids), "fabs", len(inst.fabs))
    for go, g, fabs, fuels, shares in p.grids:
        ga = inst.nodes[g].grid
        print(f"\ngrid {go} {names[g]}: deliverable {ga.deliverable:.0f} y_bar {getattr(ga, 'ybar', None)} voll {ga.voll:.3g} "
              f"shares {shares:.3f} rationed {kn[ga.rationed] if ga.rationed is not None else None} "
              f"full_draw {p.full_draw[go]:.1f} gain {p.gain[go]/1e9:.2f} bn last {p.last[go]}")
        for k, (share, thr, feeders, worth) in fuels.items():
            burn = share * ga.deliverable
            pipes = p.pipes.get((go, k), [])
            fs = [(names[t], round(st), [inst.edges[e].u0 for e in [e]][0]) for e, t, st, ss in feeders]
            ps = [(names[src], round(inst.stock_slots[slot].storage), round(inst.stock_slots[slot].supply), inst.edges[e].u0) for e, src, slot, ss in pipes]
            src = [(names[inst.edges[e].tail], inst.edges[e].tau, inst.edges[e].u0) for e in inst.in_edges[g] if k in inst.edges[e].K]
            print(f"  fuel {kn[k]:10s} share {share:.3f} burn {burn:8.1f} thr {thr:8.1f} ({thr/burn if burn else 0:.2f} wk) worth {worth} "
                  f"lean {p.lean_worth.get((go,k))} pipew {p.pipe_worth.get((go,k))} small {k in p.small[go]}")
            print(f"       feeders {fs} pipes {ps}\n       in-edges {src}")
        for fo in fabs:
            f = inst.fabs[fo]
            fa = inst.nodes[f].fab
            pis = {kn[d.k]: d.pi for d in inst.demands}
            print(f"  fab {fo} {names[f]}: cap0 {fa.cap0:.0f} e {fa.e:.4g} draw {fa.cap0*fa.e:.1f} product {kn[fa.product]} tau {fa.tau} lead {p.lead.get(fo)}")
    print("\ndemands:", sorted({(kn[d.k], d.pi) for d in inst.demands}))


if __name__ == "__main__":
    fire.Fire(main)

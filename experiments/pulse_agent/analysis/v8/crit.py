"""The grid criterion's evidence, from the recorded play of the rule switched off (tag base): per grid whose gas the
pulse can hold, the episodes whose week-1 plan leaves the gas segment short, split by whether the plan also leaves
another fuel short that can take all the fabs' energy; and what the fabs then got in the agent's own play.

uv run python crit.py --task=small
"""

import pickle

import fire
import numpy as np

import play as P


def main(task="small", tag="base", trig=0.1):
    from shockbench_flow.hosting.tasks import task_generator

    inst, _params = task_generator(task)
    node, K = P.names(inst)
    refs = P.references(task)
    chk, supply = set(inst.chokepoints), set(inst.supply_nodes)
    psi = inst.params.psi
    packaged = {}
    for o in inst.osats:
        for raw, pk in inst.nodes[o].osat.packages.items():
            packaged.setdefault(raw, set()).add(pk)
    for go, g in enumerate(inst.grids):
        ga = inst.nodes[g].grid
        fabs = [fo for fo in inst.grid_fabs[go] if inst.nodes[inst.fabs[fo]].fab.e > 0]
        if not fabs or ga.priority != "base_first" or ga.rationed is None:
            continue
        r = ga.rationed
        draw = sum(inst.nodes[inst.fabs[fo]].fab.cap0 * inst.nodes[inst.fabs[fo]].fab.e for fo in fabs)
        feeders = [e for e in inst.in_edges[g] if r in inst.edges[e].K and inst.edges[e].tau == 0 and inst.edges[e].tail not in chk and inst.edges[e].tail not in supply]
        gain = 0.0
        for fo in fabs:
            fa = inst.nodes[inst.fabs[fo]].fab
            goods = packaged.get(fa.product, set()) | {fa.product}
            gain += fa.cap0 * (max((d.pi for d in inst.demands if d.k in goods), default=0.0) - ga.voll * fa.e)
        tank = (psi * ga.ibar[r] + ga.shares[r] * ga.deliverable) * ga.voll
        small = [k for k in ga.fuels if ga.shares[k] * ga.deliverable < draw]
        big = [k for k in ga.fuels if k != r and k not in small]
        week = ga.shares[r] * ga.deliverable * ga.voll
        print(
            f"{node[g]}: gas burn {ga.shares[r] * ga.deliverable:.0f}/wk, fabs' draw {draw:.0f}; a terminal for gas: {bool(feeders)}; a whole week's gain {gain / 1e9:.1f} bn,"
            f" a held week's base load {week / 1e9:.1f} bn, a full tank {tank / 1e9:.1f} bn = {tank / gain if gain > 0 else float('inf'):.1f} whole weeks (payback 12: {'held' if gain > 0 and tank <= 12 * gain else 'not held'});"
            f" small fuels {[K[k] for k in small]}; fuels that can stop the fabs besides gas {[K[k] for k in big]}"
        )
        if not feeders or not (gain > 0 and tank <= 12 * gain):
            continue
        rows = {"alone": [], "with another": []}
        for n in sorted(refs):
            p = P.path(task, tag, n)
            if not p.exists():
                continue
            d = pickle.loads(p.read_bytes())
            plan = d["plans"][1]

            def share(k):
                sh, cap = plan["short"][(go, k)], plan["cap"][(go, k)]
                return float(sh[3:12].sum() / max(cap[3:12].sum(), 1e-9))

            if share(r) < trig:
                continue
            other = max((share(k) for k in big), default=0.0)
            y = d["y_bar"][:, go]
            shed = d["shed"][:, go] > 1e-6 * y
            E = d["energy"][:, fabs].sum(axis=1) / draw
            T = len(E)
            rows["alone" if other < trig else "with another"].append((n, int((~shed[4:20]).sum()), float(E[4:20].sum()), float(E[1:4].mean()), int(shed[4:12].sum()), float(E[4:T].sum())))
        for label, rr in rows.items():
            if not rr:
                print(f"   gas short in the week-1 plan, {label}: 0 episodes")
                continue
            whole = np.array([x[1] for x in rr])
            E = np.array([x[2] for x in rr])
            print(
                f"   gas short in the week-1 plan, {label}: {len(rr)} episodes; weeks 5-20 without shedding in the agent's play: mean {whole.mean():.1f} of 16,"
                f" none in {int((whole == 0).sum())} episodes; fab energy in weeks 5-20: mean {E.mean():.2f} full weeks, none in {int((E < 1e-6).sum())} episodes;"
                f" shed in 2 or more of weeks 5-12 in {sum(1 for x in rr if x[4] >= 2)}; fabs' draw in weeks 2-4: mean {np.mean([x[3] for x in rr]):.2f} of full"
            )
            print("      episodes: " + " ".join(str(x[0]) for x in rr))


if __name__ == "__main__":
    fire.Fire(main)

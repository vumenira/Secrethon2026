"""Which grids the opening hold acted at, per episode, against the episode's change in score (recorded plays).

uv run python who.py --task=full --tag=h24 --base=base
"""

import pickle
from collections import defaultdict

import fire
import numpy as np

import delta as D
import play as P


def main(task="full", tag="h24", base="base", detail=True):
    from shockbench_flow.hosting.tasks import task_generator

    inst, _params = task_generator(task)
    node, _K = P.names(inst)
    refs = P.references(task)
    w = D.weights(refs)
    by = defaultdict(list)
    per_grid = defaultdict(list)
    for n in sorted(refs):
        pa, pb = P.path(task, tag, n), P.path(task, base, n)
        if not pa.exists() or not pb.exists():
            continue
        a, b = pickle.loads(pa.read_bytes()), pickle.loads(pb.read_bytes())
        gap = refs[n]["J_naive_cents"] - refs[n]["J_oracle_cents"]
        d = b["J_cents"] - a["J_cents"]
        held = defaultdict(list)
        for entry in a["pulse_log"]:
            if entry[2] == "open":
                held[node[inst.grids[entry[1]]][5:]].append(entry[0])
        info = []
        for g, weeks in sorted(held.items()):
            go = next(i for i, x in enumerate(inst.grids) if node[x][5:] == g)
            ga = inst.nodes[inst.grids[go]].grid
            fabs = [fo for fo in inst.grid_fabs[go] if inst.nodes[inst.fabs[fo]].fab.e > 0]
            draw = sum(inst.nodes[inst.fabs[fo]].fab.cap0 * inst.nodes[inst.fabs[fo]].fab.e for fo in fabs)
            r = ga.rationed
            plan = a["plans"][weeks[0]]
            sh, cap = plan["short"][(go, r)], plan["cap"][(go, r)]
            trig = float(sh[3:12].sum() / max(cap[3:12].sum(), 1e-9))
            E = b["energy"][:, fabs].sum(axis=1) / draw  # the base play's fab energy in the weeks held
            shed_b = b["shed"][:, go] > 1e-6 * b["y_bar"][:, go]
            info.append(f"{g} wk{','.join(map(str, weeks))} (plan short {trig:.2f}; without the hold fabs drew {np.mean([E[t - 1] for t in weeks]):.2f} then, shed in {int(shed_b[4:12].sum())} of wk5-12, {int(shed_b[4:20].sum())} of wk5-20)")
        key = "+".join(sorted(held)) or "none"
        by[key].append((n, d / gap, d * w[n]))
        for g in held:
            per_grid[g].append(d * w[n])
        if detail:
            print(f"  {n:2d} L{refs[n]['stratum']} {d / gap:+.4f} ({d / 1e11:+7.1f} bn, pooled {1e4 * d * w[n]:+6.2f}e-4): " + ("; ".join(info) or "no hold"))
    print(f"{task} tag {tag} against {base}: by the grids held")
    for key, rows in sorted(by.items()):
        x = np.array([r[2] for r in rows])
        print(f"  {key:10s}: {len(rows):2d} episodes, pooled {x.sum():+.4f}; better {int((x > 0).sum())}, worse {int((x < 0).sum())}; episodes {[r[0] for r in rows]}")


if __name__ == "__main__":
    fire.Fire(main)

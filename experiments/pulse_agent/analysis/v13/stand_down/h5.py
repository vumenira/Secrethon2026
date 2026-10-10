"""H5 per episode: what serving the high-value fab first would add in the weeks a grid's fabs are partly powered."""
from collections import defaultdict

import fire

from analyse import GRID, load


def main(name="s13_census", sets="full:12345:16,full:2026:16,full:777:32", task="full", top=12):
    from shockbench_flow.hosting.tasks import task_generator

    inst, _params = task_generator(task)
    docs, att = load(name, sets)
    packaged = defaultdict(set)
    for o in inst.osats:
        for raw, pk in inst.nodes[o].osat.packages.items():
            packaged[raw].add(pk)
    val, en = {}, {}
    for go, g in enumerate(inst.grids):
        ga = inst.nodes[g].grid
        for fo in inst.grid_fabs[go]:
            fa = inst.nodes[inst.fabs[fo]].fab
            goods = packaged[fa.product] | {fa.product}
            val[fo] = (max((d.pi for d in inst.demands if d.k in goods), default=0.0) - ga.voll * fa.e) / fa.e
            en[fo] = fa.e
    rows = []
    by_code = defaultdict(float)
    for d in docs:
        per = defaultdict(lambda: [0.0, 0])
        for t, go, code, row in d["census"]:
            if t > d["last"][go]:
                continue
            ehat, egot, vals = [], [], []
            for fo, fab in row["fabs"].items():
                if fab["R"] > 0 and fab["sold"]:
                    ehat.append(min(fab["cap"], fab["wafers"]) * en[fo] / fab["R"])
                    egot.append(float(d["energy"][t - 1][fo]))
                    vals.append(val[fo] * fab["R"])
            if len(ehat) > 1 and sum(egot) > 1e-9 and sum(egot) < sum(ehat) * (1 - 1e-6):
                left, best = sum(egot), 0.0
                for v, eh in sorted(zip(vals, ehat), reverse=True):
                    take = min(left, eh)
                    best += take * v
                    left -= take
                bn = (best - sum(v * e for v, e in zip(vals, egot))) / 1e9
                per[go][0] += bn
                per[go][1] += 1
                by_code[(GRID[go], code)] += d["w"] * bn
        for go, (bn, n) in per.items():
            rows.append((round(d["w"] * bn, 2), round(bn, 1), n, GRID[go], d["set"], d["episode"], d["ref"]["stratum"]))
    print("weighted bn, plain bn, weeks, grid, set, episode, level")
    for r in sorted(rows, reverse=True)[:top]:
        print("  ", *r)
    print("total weighted", round(sum(r[0] for r in rows), 2), "bn =", round(sum(r[0] for r in rows) / att, 4))
    print("by grid and what the pulse did:", {k: round(v, 2) for k, v in sorted(by_code.items(), key=lambda kv: -kv[1])[:10]})


if __name__ == "__main__":
    fire.Fire(main)

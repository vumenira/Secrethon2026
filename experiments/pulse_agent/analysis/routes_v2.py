"""Where a grid's short fuel comes from, and whether the way in was full.

Per grid and fuel, over the traced episodes: the edges that bring the fuel to the grid's terminal (or to the grid),
their flow against their capacity in the weeks the fuel's segment could not run in full, and the same in the plan.
Then, per fuel, the routes two levels up (the tails of those edges), to see which grids draw on the same nodes.
"""

import pickle
import sys
from collections import defaultdict
from pathlib import Path

import fire
import numpy as np


HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import analyze_v2 as A  # noqa: E402


def main(task="full", tag="v2", fuels="lng,crude"):
    from shockbench_flow.hosting.tasks import task_generator

    inst, _ = task_generator(task)
    st = A.static(inst)
    want = set(fuels.split(",")) if isinstance(fuels, str) else set(fuels)
    name_k = {k: inst.commodities[k].id for k in range(len(inst.commodities))}
    paths = sorted((HERE / "trace" / f"{task}_{tag}").glob("*.pkl"), key=lambda q: int(q.stem))
    use = defaultdict(lambda: defaultdict(float))
    for path in paths:
        d = pickle.loads(path.read_bytes())
        p, plan, u = d["played"], d["plan"], d["marks"]["u"]
        col = {key: j for j, key in enumerate(plan["columns"])}
        T = u.shape[0]
        flow = defaultdict(lambda: np.zeros(T))  # (edge, k) -> played flow per week (all commodities share u)
        allflow = defaultdict(lambda: np.zeros(T))
        for t, x in enumerate(p["x"]):
            for (e, k, _lane), q in x.items():
                flow[(e, k)][t] += q
                allflow[e][t] += q
        pflow = defaultdict(lambda: np.zeros(T))
        pall = defaultdict(lambda: np.zeros(T))
        for key, j in col.items():
            if key[0] == "x":
                pflow[(key[1], key[2])] += plan["x"][:, j]
                pall[key[1]] += plan["x"][:, j]
        for go in st:
            gw = A.grid_weeks(inst, st, d, go)
            g = st[go]["node"]
            for k, fu in gw["fuels"].items():
                if name_k[k] not in want:
                    continue
                short = fu["avail"] < fu["full"] * (1 - 1e-6)  # weeks the segment could not run in full, as played
                heads = {tail for _e, tail in st[go]["fuels"][k]["feeders"]} or {g}
                for e, ed in enumerate(inst.edges):
                    if ed.head in heads and k in ed.K and ed.tail not in heads:
                        key = (st[go]["id"], name_k[k], f"{inst.nodes[ed.tail].id}->{inst.nodes[ed.head].id}", ed.tau)
                        row = use[key]
                        cap = np.where(np.isfinite(u[:, e]), u[:, e], 0.0)
                        nominal = ed.u0 if ed.u0 is not None else float("nan")
                        row["weeks"] += T
                        row["short_weeks"] += short.sum()
                        row["flow"] += flow[(e, k)].sum()
                        row["flow_short"] += flow[(e, k)][short].sum()
                        row["cap_short"] += cap[short].sum()
                        row["all_short"] += allflow[e][short].sum()
                        row["plan"] += pflow[(e, k)].sum()
                        row["plan_short"] += pflow[(e, k)][short].sum()
                        row["nominal_short"] += nominal * short.sum()
                        row["full_short"] += ((allflow[e] >= cap * (1 - 1e-6)) & short & (cap > 0)).sum()
                        row["cut_short"] += ((cap < nominal * (1 - 1e-6)) & short).sum()
                        row["shut_short"] += ((cap <= 1e-9) & short).sum()
    print(f"{task}: {len(paths)} episodes. Per grid and fuel, the edges that bring it in; 'short' = weeks its segment could not run in full.")
    print(f"{'grid':9s} {'fuel':6s} {'edge':34s} {'lead':>4s} {'short wk':>8s} | in short weeks: {'flow/cap':>8s} {'cap/nominal':>11s} {'edge full':>9s} {'edge cut':>8s} {'shut':>6s} | {'flow, ours':>11s} {'plan':>11s}")
    for key, r in sorted(use.items()):
        if r["short_weeks"] < 1 or (r["flow"] < 1 and r["plan"] < 1):
            continue
        print(
            f"{key[0]:9s} {key[1]:6s} {key[2]:34s} {key[3]:4d} {r['short_weeks']:8.0f} |                 {r['all_short'] / r['cap_short'] if r['cap_short'] else float('nan'):8.2f}"
            f" {r['cap_short'] / r['nominal_short'] if r['nominal_short'] else float('nan'):11.2f} {r['full_short'] / r['short_weeks']:9.2f} {r['cut_short'] / r['short_weeks']:8.2f} {r['shut_short'] / r['short_weeks']:6.2f} |"
            f" {r['flow']:11.0f} {r['plan']:11.0f}"
        )


if __name__ == "__main__":
    fire.Fire(main)

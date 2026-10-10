"""The duty gap: per episode and grid, the whole weeks made against the whole weeks its scarcest fuel allowed.

uv run python outputs/s13/duty.py --name=s13_census --sets="full:12345:16,full:2026:16,full:777:32"

No play; it reads the census pickles. A whole week burns a week's burn of every fuel, so a grid can have no more
whole weeks than the weeks of burn of its scarcest fuel that were there to burn: what its segment gave over the
sellable weeks plus what was left at the grid and its terminal. The gap to the weeks the fabs did get their whole
draw is fuel burnt for base load in weeks that were not whole. It is an upper bound: a tank is finite, a rationed
fuel wastes part of a charge, a fuel without a terminal cannot wait, and not every chip is needed.
"""
from collections import defaultdict

import fire
import numpy as np

from analyse import FUEL, GRID, load


def main(name="s13_census", sets="full:12345:16", task="full", top=25, base=None):
    from shockbench_flow.hosting.tasks import task_generator

    inst, _params = task_generator(task)
    docs, att = load(name, sets)
    feeders = defaultdict(list)  # (grid ordinal, fuel) -> the nodes with a zero-lead edge into the grid (terminals)
    for e, ed in enumerate(inst.edges):
        if ed.tau == 0 and ed.head in inst.grids:
            for k in ed.K:
                feeders[(inst.grids.index(ed.head), k)].append(ed.tail)
    rows, tot = [], defaultdict(lambda: np.zeros(3))
    leaks = defaultdict(lambda: np.zeros(3))  # (grid, scarcest fuel, what the pulse did) -> weeks, weeks of burn, bn
    cases = defaultdict(list)
    for d in docs:
        census = defaultdict(dict)
        for t, go, code, row in d["census"]:
            census[go][t] = (code, row)
        for go, weeks in census.items():
            last = d["last"][go]
            burnt = defaultdict(float)
            dark = defaultdict(lambda: defaultdict(lambda: np.zeros(2)))  # fuel -> code -> weeks, burnt beside the fabs
            whole = ready = 0.0  # in weeks of the fabs' full draw: what they got, what their wafers and power allowed
            n = 0
            codes = defaultdict(int)
            full = d["full_draw"][go]
            for t, (code, row) in sorted(weeks.items()):
                if t > last:
                    continue
                n += 1
                codes[code] += 1
                ehat = egot = 0.0
                for fo, fab in row["fabs"].items():
                    if fab["R"] > 0 and fab["sold"]:
                        e = inst.nodes[inst.fabs[fo]].fab.e / fab["R"]
                        ehat += min(fab["cap"], fab["wafers"]) * e
                        egot += float(d["lots"][t - 1][fo]) * e
                powered = row["power"] - row["y"] > 1e-9 * max(row["y"], 1.0)
                ready += min(ehat, max(row["power"] - row["y"], 0.0)) / full if powered else 0.0
                whole += egot / full
                served = min(egot / ehat, 1.0) if ehat > 1e-6 * full else 0.0  # the share of their draw the fabs got
                sub = code
                if code == "off":
                    sub = "off:" + (("pipe" if row["piping"] else "") + ("lean" if row["leaning"] else "") or "tank")
                elif code == "dead":
                    sub = "dead:" + "+".join(sorted({v[0] for v in row["blocked"].values()}))
                elif code == "idle":
                    sub = "idle:shock" if not powered else "idle:nowafers"
                for k, f in row["fuels"].items():
                    if not f["small"] and f["burn"] > 0:
                        gave = float(d["segment"][t - 1].get((go, k), 0.0)) / f["burn"]
                        burnt[k] += gave
                        if gave * (1 - served) > 1e-6:
                            dark[k][sub] += (1, gave * (1 - served))
            if not burnt:
                continue
            g = inst.grids[go]
            left = {}
            for k in burnt:
                burn = inst.nodes[g].grid.shares[k] * inst.nodes[g].grid.deliverable
                nodes = [g, *feeders[(go, k)]]
                left[k] = sum(float(d["stock"][min(last, len(d["stock"])) - 1][inst.slot_index[(nd, k)]]) for nd in nodes) / burn
            able = {k: burnt[k] + left[k] for k in burnt}
            k_min = min(able, key=able.get)
            # the wafers that were there to process: the lots started and what the fabs still held at the end
            wafers = 0.0
            for fo in inst.grid_fabs[go]:
                fa = inst.nodes[inst.fabs[fo]].fab
                held = float(d["stock"][min(last, len(d["stock"])) - 1][inst.slot_index[(inst.fabs[fo], fa.input)]])
                wafers += fa.e * (float(d["lots"][:last, fo].sum()) + held) / full
            best = min(able[k_min], ready, wafers)
            gap = max(best - whole, 0.0)
            bn = gap * d["gain"][go] / 1e9
            rows.append((d["w"] * bn, bn, d["set"], d["episode"], d["ref"]["stratum"], GRID[go], FUEL[k_min], n, round(ready, 1),
                         round(whole, 1), round(best, 1), round(wafers, 1), {FUEL[k]: round(v, 1) for k, v in able.items()}, dict(codes)))
            tot[(GRID[go], FUEL[k_min])] += (d["w"] * bn, gap, 1 if gap > 2 else 0)
            if able[k_min] < n - 2:  # the fuel was short of the weeks: what of it burnt in weeks not whole is a loss
                for sub, v in dark[k_min].items():
                    leaks[(GRID[go], FUEL[k_min], sub)] += (v[0], v[1], d["w"] * v[1] * d["gain"][go] / 1e9)
                    cases[(GRID[go], FUEL[k_min], sub)].append((round(v[1], 1), d["set"], d["episode"]))
    print("\nthe scarcest fuel burnt in weeks the fabs did not get their draw (weeks, weeks of burn, weighted bn, score, largest cases)")
    for key, v in sorted(leaks.items(), key=lambda kv: -kv[1][2])[:top]:
        worst = sorted(cases[key], reverse=True)[:3]
        print(f"  {key[0]:4s} {key[1]:6s} {key[2]:20s} {int(v[0]):5d} wk {v[1]:7.1f} weeks of burn {v[2]:8.2f} bn = {v[2] / att:.4f}  {worst}")
    all_ = sum(v[2] for v in leaks.values())
    print(f"  all {all_:8.2f} bn = {all_ / att:.4f}")
    print(f"{len(docs)} episodes; 0.001 of score = {att / 1000:.2f} bn (weighted)")
    print("\nthe duty gap by grid and scarcest fuel: weighted bn per episode, as score, weeks in all, (episode, grid) pairs over 2 weeks")
    for key, v in sorted(tot.items(), key=lambda kv: -kv[1][0]):
        print(f"  {key[0]:4s} {key[1]:6s} {v[0]:8.2f} bn = {v[0] / att:.4f}   {v[1]:7.1f} weeks   {int(v[2])} cases")
    all_ = sum(v[0] for v in tot.values())
    print(f"  all         {all_:8.2f} bn = {all_ / att:.4f}")
    print("\nthe largest (weighted bn, plain bn, set, episode, level, grid, scarcest fuel, weeks, ready, whole, allowed, wafer weeks, fuel weeks, codes):")
    for r in sorted(rows, reverse=True)[:top]:
        print("  ", round(r[0], 2), round(r[1], 1), *r[2:])


if __name__ == "__main__":
    fire.Fire(main)

"""H4: weeks a grid the pulse has given up burns a fuel that another, pulsed grid is short of (no play).

A grid is "given up" in a week the pulse stood down for a fuel that cannot be held or gathered (dead). Its other
fuels keep their shortfall premium in the plan. Counted: the weeks such a grid burnt a fuel k that was not the one
blocking it while another grid was held for want of the same fuel k, and what the second grid's fabs would have
gained had the first grid's burn of k gone to it instead (at most the second grid's shortfall of that week, as a
share of its burn, times its gain per whole week). An upper bound: it takes the two grids to draw on one supply.
"""
from collections import defaultdict

import fire

from analyse import FUEL, GRID, load


def main(name="s13_census", sets="full:12345:16,full:2026:16,full:777:32", top=12):
    docs, att = load(name, sets)
    tot = defaultdict(lambda: [0, 0.0])
    per_ep = defaultdict(float)
    for d in docs:
        weeks = defaultdict(dict)
        for t, go, code, row in d["census"]:
            weeks[t][go] = (code, row)
        for t, grids in weeks.items():
            dead = {go: row for go, (code, row) in grids.items() if code == "dead" and t <= d["last"][go]}
            off = {go: row for go, (code, row) in grids.items() if code == "off" and t <= d["last"][go]}
            for go, row in off.items():
                for k in row["short"]:
                    f = row["fuels"][k]
                    if f["small"] or f["burn"] <= 0:
                        continue
                    lacking = f["burn"] - float(d["segment"][t - 1].get((go, k), 0.0))  # what the week misses of k
                    for g2, r2 in dead.items():
                        if k in r2["blocked"] or k not in r2["fuels"]:
                            continue
                        burnt = float(d["segment"][t - 1].get((g2, k), 0.0))
                        moved = min(burnt, lacking)
                        if moved <= 0:
                            continue
                        bn = moved / f["burn"] * d["gain"][go] / 1e9
                        key = (GRID[g2], GRID[go], FUEL[k])
                        tot[key][0] += 1
                        tot[key][1] += d["w"] * bn
                        per_ep[(d["set"], d["episode"], d["ref"]["stratum"])] += bn
                        lacking -= moved
    print("given-up grid, grid held, fuel: weeks, weighted bn, score")
    for key, (n, bn) in sorted(tot.items(), key=lambda kv: -kv[1][1])[:top]:
        print(f"  {key[0]:4s} -> {key[1]:4s} {key[2]:6s} {n:5d} wk {bn:7.2f} bn = {bn / att:.4f}")
    all_ = sum(v[1] for v in tot.values())
    print(f"  all {all_:.2f} bn = {all_ / att:.4f}")
    print("largest episodes (plain bn):", [(k, round(v, 1)) for k, v in sorted(per_ep.items(), key=lambda kv: -kv[1])[:8]])


if __name__ == "__main__":
    fire.Fire(main)

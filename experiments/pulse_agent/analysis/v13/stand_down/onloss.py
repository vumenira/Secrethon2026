"""Weeks the pulse called whole in which the fabs still missed part of their draw: which segment was short (no play)."""
from collections import defaultdict

import fire
import numpy as np

from analyse import FUEL, GRID, load


def main(name="s13_census", sets="full:12345:16,full:2026:16,full:777:32", task="full", top=14):
    from shockbench_flow.hosting.tasks import task_generator

    inst, _params = task_generator(task)
    docs, att = load(name, sets)
    out = defaultdict(lambda: np.zeros(3))
    cases = defaultdict(list)
    for d in docs:
        per = defaultdict(float)
        for t, go, code, row in d["census"]:
            if code not in ("on", "lapse") or t > d["last"][go]:
                continue
            want = got = 0.0
            for fo, fab in row["fabs"].items():
                if fab["R"] > 0 and fab["sold"]:
                    e = inst.nodes[inst.fabs[fo]].fab.e / fab["R"]
                    want += min(fab["cap"], fab["wafers"]) * e
                    got += float(d["lots"][t - 1][fo]) * e
            miss = want - got
            if miss <= 0.01 * max(want, 1e-9):
                continue
            # the chips lost, at the grid's gain per unit of its fabs' draw
            bn = miss * d["gain"][go] / d["full_draw"][go] / 1e9
            short = []
            for k, f in row["fuels"].items():
                gave = float(d["segment"][t - 1].get((go, k), 0.0))
                if gave < f["burn"] * (1 - 1e-6):
                    kind = "small" if f["small"] else ("rationed" if f["threshold"] > 0 else "other")
                    short.append(f"{FUEL[k]}:{kind}" + ("" if f["reach"] else ":noway"))
            null = float(d["segment"][t - 1].get((go, None), 0.0))
            tag = "+".join(sorted(short)) or ("load" if null < row["power"] - sum(f["burn"] for f in row["fuels"].values()) - 1e-6 else "none")
            key = (GRID[go], "pulsed" if row.get("pulsed") else "unpulsed", tag)
            out[key] += (1, d["w"] * bn, miss / d["full_draw"][go])
            per[key] += bn
        for key, bn in per.items():
            cases[key].append((round(bn, 1), d["set"], d["episode"], d["ref"]["stratum"]))
    print(f"{len(docs)} episodes; 0.001 of score = {att / 1000:.2f} bn")
    print("grid, pulsed or not, the segments that gave less than their share: weeks, weighted bn, score, weeks of draw missed, largest cases")
    for key, v in sorted(out.items(), key=lambda kv: -kv[1][1])[:top]:
        print(f"  {key[0]:4s} {key[1]:9s} {key[2]:34s} {int(v[0]):5d} wk {v[1]:7.2f} bn = {v[1] / att:.4f}  {v[2]:6.1f}  {sorted(cases[key], reverse=True)[:3]}")
    tot = sum(v[1] for v in out.values())
    print(f"  all {tot:.2f} bn = {tot / att:.4f}")


if __name__ == "__main__":
    fire.Fire(main)

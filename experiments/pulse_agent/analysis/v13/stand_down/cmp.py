"""Compare two recorded plays of one episode week by week: cost parts, and per grid the codes and lots (no play)."""
import pickle
from pathlib import Path

import fire
import numpy as np

HERE = Path(__file__).resolve().parent
GRID = ["tw", "kr", "jp", "cn", "us", "eu", "sea", "in"]


def main(root=2026, ep=8, a="s13_trace_ends", b="s13_census", go=None, first=1, last=104, task="full"):
    da = pickle.load(open(HERE / a / f"{task}-e{root}" / f"ep{ep}.pkl", "rb"))
    db = pickle.load(open(HERE / b / f"{task}-e{root}" / f"ep{ep}.pkl", "rb"))
    print("J difference a - b, bn:", (da["J_cents"] - db["J_cents"]) / 1e11, "salvage a - b:", (da["salvage"] - db["salvage"]) / 1e9)
    parts = sorted(da["costs"][0])
    tot = {k: sum(c[k] for c in da["costs"]) - sum(c[k] for c in db["costs"]) for k in parts}
    print("cost parts a - b, bn:", {k: round(v / 1e9, 2) for k, v in tot.items()})
    ca = {(t, g): (code, row) for t, g, code, row in da["census"]}
    cb = {(t, g): (code, row) for t, g, code, row in db["census"]}
    logs = {}
    for t, g, rule, moved in da["log"]:
        if rule == "start":
            logs[(t, g)] = moved
    print("starts in a:", {k: {f: round(v / 1e9, 2) for f, v in m.items()} for k, m in logs.items()})
    cum = 0.0
    for t in range(first, min(last, len(da["costs"])) + 1):
        dc = (sum(da["costs"][t - 1].values()) - sum(db["costs"][t - 1].values())) / 1e9
        cum += dc
        line = []
        for g in sorted({g for (tt, g) in ca if tt == t}):
            if go is not None and g != go:
                continue
            xa, xb = ca.get((t, g)), cb.get((t, g))
            if xa is None or xb is None:
                continue
            la = sum(float(da["lots"][t - 1][fo]) for fo in xa[1]["fabs"])
            lb = sum(float(db["lots"][t - 1][fo]) for fo in xb[1]["fabs"])
            if xa[0] != xb[0] or abs(la - lb) > 1 or abs(da["shed"][t - 1][g] - db["shed"][t - 1][g]) > 1:
                line.append(f"{GRID[g]}: {xa[0]}/{xb[0]} lots {round(la)}/{round(lb)} shed {round(float(da['shed'][t - 1][g]))}/{round(float(db['shed'][t - 1][g]))}")
        if line or abs(dc) > 0.05:
            print(t, "cost a-b", round(dc, 2), "cum", round(cum, 2), "|", "; ".join(line))


if __name__ == "__main__":
    fire.Fire(main)

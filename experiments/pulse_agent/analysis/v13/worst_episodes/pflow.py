"""A plan's flows on chosen edges by commodity against the week's capacity, by quarter (no play, no solve).

uv run python outputs/w13a/pflow.py --ep="12345:12" --edges="40,48,50,51,52,85,86,87" --plan_name=b2
"""
import sys
from pathlib import Path

import fire
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cmp as C
from fuel import marks_of


def q(a, step=13):
    return " ".join(f"{a[i:i + step].mean():6.0f}" for i in range(0, len(a), step))


def main(ep, edges, plan_name="b2"):
    from shockbench_flow.hosting.tasks import task_generator

    inst, _ = task_generator("full")
    N, E, K = inst.nodes, inst.edges, inst.commodities
    root, n = (int(v) for v in str(ep).split(":"))
    P, m = C.plan(root, n, plan_name), marks_of(root, n)
    x, cols = P["x"], P["columns"]
    for e in (int(v) for v in (edges if isinstance(edges, (tuple, list)) else str(edges).split(","))):
        print(f"edge {e} {E[e].id}: capacity {q(m['u'][:, e])}")
        for k in E[e].K:
            js = [j for j, key in enumerate(cols) if key[0] == "x" and key[1] == e and key[2] == k]
            if js:
                flow = x[:, js].sum(axis=1)
                banned = int(m["prohibited"][:, e, k].sum())
                print(f"      {K[k].id:6s} plan {q(flow)}   banned weeks {banned}")


if __name__ == "__main__":
    fire.Fire(main)

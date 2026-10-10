"""Fuel waiting at straits and held at terminals and grids, ours against a plan, by 8-week block (no play, no solve)."""
import pickle
import sys
from pathlib import Path

import fire
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cmp as C


def b(a, step=8):
    return " ".join(f"{a[i:i + step].mean():6.0f}" for i in range(0, len(a), step))


def main(eps, plan_name="bf"):
    from shockbench_flow.hosting.tasks import task_generator

    inst, _ = task_generator("full")
    N, K, S = inst.nodes, inst.commodities, inst.stock_slots
    for item in str(eps).split(","):
        root, n = (int(v) for v in item.split(":"))
        d, P = C.ours(root, n), C.plan(root, n, plan_name)
        x, col = P["x"], P["col"]
        print(f"\n===== root {root} episode {n}: stock by 8-week block, ours then {plan_name}")
        for k in (0, 1):
            for c in inst.chokepoints:
                s = inst.slot_index.get((c, k))
                if s is None:
                    continue
                keys = [key for key in P["columns"] if key[0] == "Q" and key[1] == c and key[2] == k]
                plan_q = sum(x[:, col[key]] for key in keys)
                print(f"  {N[c].id:12s} {K[k].id:5s} ours {b(d['stock'][:, s])}\n  {'':18s} plan {b(plan_q)}")
            for node in list(inst.supply_nodes) + [i for i, nd in enumerate(N) if nd.type in ('terminal', 'grid')]:
                s = inst.slot_index.get((node, k))
                if s is None or ("I", s) not in col:
                    continue
                print(f"  {N[node].id:12s} {K[k].id:5s} ours {b(d['stock'][:, s])}\n  {'':18s} plan {b(x[:, col[('I', s)]])}   storage {S[s].storage:.0f}")


if __name__ == "__main__":
    fire.Fire(main)

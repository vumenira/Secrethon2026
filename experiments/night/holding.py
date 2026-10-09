"""Holding cost by stock slot, ours against the clairvoyant plan's, from gap.py's files (its "stock_sum").

uv run python experiments/night/holding.py --agent=mpc_pulse_rec_v9 --task=full --entropy=12345
"""

import json
from pathlib import Path

import fire
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def main(agent="mpc_pulse_rec_v9", task="full", entropy=12345, top=20):
    from shockbench_flow.hosting.tasks import task_generator

    inst, _ = task_generator(task)
    chk = set(inst.chokepoints)
    out = ROOT / "outputs" / "gap" / f"{task}-e{entropy}" / agent
    eps = [json.loads(p.read_text()) for p in sorted(out.glob("*.json"), key=lambda p: int(p.stem))]
    h = np.array([0.0 if sl.node in chk else sl.holding for sl in inst.stock_slots])
    so = np.sum([e["ours"]["stock_sum"] for e in eps], axis=0) * h
    sc = np.sum([e["oracle"]["stock_sum"] for e in eps], axis=0) * h
    print(f"{task} root {entropy}, {len(eps)} episodes: holding ours {so.sum() / 1e9:,.0f} bn, clairvoyant {sc.sum() / 1e9:,.0f} bn")
    kinds = {}
    for s, sl in enumerate(inst.stock_slots):
        node = inst.nodes[sl.node]
        role = next((r for r in ("grid", "fab", "osat") if getattr(node, r, None) is not None), None)
        role = role or ("supply" if sl.node in inst.supply_nodes else "store")
        key = (role, sl.k)
        kinds.setdefault(key, [0.0, 0.0])
        kinds[key][0] += so[s]
        kinds[key][1] += sc[s]
    print("by kind of node and good (bn USD): ours, clairvoyant, gap")
    for (role, k), (a, b) in sorted(kinds.items(), key=lambda kv: -(kv[1][0] - kv[1][1])):
        if abs(a - b) > 1e9:
            print(f"  {role:7s} good {k}: {a / 1e9:8,.1f} {b / 1e9:8,.1f} {(a - b) / 1e9:+8,.1f}")
    print(f"the {top} slots with the largest gap:")
    for s in np.argsort(-(so - sc))[:top]:
        sl = inst.stock_slots[s]
        print(f"  slot {s} node {sl.node} good {sl.k} storage {sl.storage}: {so[s] / 1e9:8,.1f} {sc[s] / 1e9:8,.1f} {(so[s] - sc[s]) / 1e9:+8,.1f}")


if __name__ == "__main__":
    fire.Fire(main)

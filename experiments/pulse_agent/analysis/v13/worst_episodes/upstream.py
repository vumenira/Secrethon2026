"""Spells in which a source's route to its first strait, or the source itself, is cut (marks only; no play).

Per episode: for each fuel source that ships into a strait, the weeks its edge into the strait carries under 0.7 of
nominal (capacity cut or the fuel prohibited) or its availability is under 0.7 of nominal.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fuel import marks_of  # noqa: E402
from straits import spells  # noqa: E402

from shockbench_flow.hosting.tasks import task_generator

ROOT = Path(__file__).resolve().parents[2]
SETS = {12345: 16, 2026: 16, 777: 32}
inst, _ = task_generator("full")
N, E, K = inst.nodes, inst.edges, inst.commodities
ways = []
for e, ed in enumerate(E):
    if ed.tail in inst.supply_nodes and ed.head in inst.chokepoints:
        for k in ed.K:
            if K[k].override:
                ways.append((e, k, inst.slot_index[(ed.tail, k)]))
count = {}
for root, n_eps in SETS.items():
    for n in range(n_eps):
        m = marks_of(root, n)
        line = []
        for e, k, slot in ways:
            cap = np.where(m["prohibited"][:, e, k], 0.0, m["u"][:, e]) / E[e].u0
            sup = m["supply"][:, slot] / max(inst.stock_slots[slot].supply if hasattr(inst.stock_slots[slot], "supply") else m["supply"][:, slot].max(), 1e-9)
            for what, a in (("route", cap), ("supply", sup / max(sup.max(), 1e-9))):
                sp = [s for s in spells(a, 0.7) if s[1] - s[0] + 1 >= 4]
                if sp:
                    line.append(f"{N[E[e].tail].id[4:]} {what} " + ",".join(f"{a_}-{b_}@{v:.2f}" for a_, b_, v in sp))
                    for a_, b_, v in sp:
                        key = (N[E[e].tail].id[4:], what, "late" if a_ > 8 else "from the start")
                        count[key] = count.get(key, 0) + 1
        print(f"{root} ep {n:2d}: " + "; ".join(line), flush=True)
print("spells of 4 weeks or more by source, kind and onset:")
for key in sorted(count):
    print("  ", key, count[key])

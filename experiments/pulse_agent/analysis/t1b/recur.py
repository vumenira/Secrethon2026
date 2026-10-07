"""After an element recovers: how soon is it impaired again (clustering), against the unconditional rate.

uv run python recur.py spells_big.json
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


data = json.loads((Path(__file__).parent / sys.argv[1]).read_text())
T = data["meta"]["T"]
rows = data["rows"]
episodes = sorted({r["episode"] for r in rows})
n_el = {"closure": len(data["meta"]["chokepoints"]), "grid": len(data["meta"]["grids"])}
for kind in ("closure", "grid"):
    spells = defaultdict(list)
    for r in rows:
        if r["kind"] == kind:
            spells[(r["episode"], r["element"])].append((r["first"], r["last"], r))
    starts = sum(1 for v in spells.values() for a, b, r in v if a > 1)
    weeks_nominal = len(episodes) * n_el[kind] * T - sum(b - a + 1 for v in spells.values() for a, b, r in v)
    base = starts / weeks_nominal  # new spells per nominal element-week
    print(f"\n{kind}: {len(episodes)} episodes, {sum(len(v) for v in spells.values())} spells, {starts} with a seen start; "
          f"a nominal element starts a spell with probability {base:.4f} per week ({base * 52:.2f} per year)")
    for label, pick in (
        ("all", lambda r: True),
        ("weather" if kind == "closure" else "one shock", lambda r: (all(e["type"] == "weather_closure" for e in r["events"]) if kind == "closure" else len(r["events"]) == 1)),
        ("militarised" if kind == "closure" else "several shocks", lambda r: (any(e["type"] == "militarised_closure" for e in r["events"]) if kind == "closure" else len(r["events"]) > 1)),
    ):
        again = {k: [0, 0] for k in (1, 2, 4, 8, 13)}
        for v in spells.values():
            v.sort()
            for i, (a, b, r) in enumerate(v):
                if not pick(r):
                    continue
                nxt = v[i + 1][0] if i + 1 < len(v) else None
                for k in again:
                    if b + k > T:
                        continue
                    again[k][1] += 1
                    again[k][0] += int(nxt is not None and nxt <= b + k)
        cells = "  ".join(f"within {k} wk {a / n:.3f} (base {1 - (1 - base) ** k:.3f}, n {n})" for k, (a, n) in again.items() if n)
        print(f"  after a spell ends [{label}]: impaired again {cells}")

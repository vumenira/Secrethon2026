"""Strait spells of the 64 used Full episodes: per episode and strait, the weeks its open share is below 0.9
(no play, no solve: the scenario's marks)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fuel import marks_of  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SETS = {12345: 16, 2026: 16, 777: 32}
NAMES = ["hormuz", "malacca", "suez", "cape", "taiwan", "panama", "turkish"]


def spells(a, level=0.9):
    out, start = [], None
    for t, v in enumerate(a):
        if v < level and start is None:
            start = t
        if v >= level and start is not None:
            out.append((start + 1, t, float(a[start:t].mean())))
            start = None
    if start is not None:
        out.append((start + 1, len(a), float(a[start:].mean())))
    return out


for root, N in SETS.items():
    refs = json.loads((ROOT / "outputs" / "recovery_tune" / f"full-e{root}-n{N}" / "references.json").read_text())
    for r in refs:
        m = marks_of(root, r["episode"])
        line = []
        for c, name in enumerate(NAMES):
            sp = [s for s in spells(m["o"][:, c]) if s[1] - s[0] + 1 >= 4]
            if sp:
                line.append(f"{name} " + ",".join(f"{a}-{b}@{v:.2f}" for a, b, v in sp))
        print(f"{root} ep {r['episode']:2d} L{r['stratum']}: " + "; ".join(line), flush=True)

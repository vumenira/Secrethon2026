"""What the true future of the capacity cuts looks like, by cause: do they end, ease or worsen inside the window?"""

import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def main():
    data = pickle.loads((Path(__file__).parent / "capcells.pkl").read_bytes())
    rows = data["rows"]
    by = defaultdict(Counter)
    for r in rows:
        group = r["cls"] if r["cls"] != "kept" else "kept:" + r["causes"]
        c = by[group]
        c["cells"] += 1
        window = min(20, r["left"])
        now = np.array(r["now"])  # the instant values of weeks w .. w + 20 (as far as the episode goes)
        fut = now[1 : window + 1] if len(now) > 1 else np.array([])
        if r["remaining"] <= window:
            c["back to nominal inside the window"] += 1
        if fut.size and fut.min() < r["share"] - 1e-9:
            c["worsens inside the window"] += 1
        if fut.size and fut.max() > r["share"] + 1e-9:
            c["eases inside the window"] += 1
        if fut.size and (np.abs(fut - r["share"]) < 1e-9).all():
            c["unchanged all window"] += 1
        if r["until_change"] == 1:
            c["changes next week"] += 1
            c["  of which eases"] += r["next_share"] > r["share"]
    for group, c in sorted(by.items(), key=lambda kv: -kv[1]["cells"]):
        n = c["cells"]
        print(f"{group}: {n} cells")
        for k, v in c.items():
            if k != "cells":
                print(f"     {k:36s} {v:7d}  {v / n:6.1%}")


if __name__ == "__main__":
    main()

"""Walk the mild weeks of the census and say when mild_start's account would have started a pulse (no play)."""
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse import FUEL, GRID, load


def main(name="s13_census", sets="full:12345:16,full:2026:16,full:777:32", task="full", voll=4.13e6):
    docs, att = load(name, sets)
    tot = 0
    for d in docs:
        owed = defaultdict(float)
        lastmild = {}
        for t, go, code, row in sorted(d["census"], key=lambda r: (r[0], r[1])):
            if code in ("on", "lapse") or row.get("pulsed"):
                owed.pop(go, None)
            if code != "mild":
                continue
            worth = d["gain"][go] / d["full_draw"][go] + voll
            lack = row["y"] + row["target"] - row["able"]
            owed[go] += lack * worth
            carried = 0.0
            for k, f in row["fuels"].items():
                gives = min(f["burn"] * f["ration"], f["on_hand"] + f["feed"])
                if not f["small"] and gives < f["burn"] * (1 - 1e-9):
                    carried += gives
            dark = row["target"] * worth + max(carried - (row["able"] - row["y"]), 0.0) * voll
            left = max(d["last"][go] - t, 0)
            start = min(owed[go], left * lack * worth) >= dark
            tot += start
            if start or t - lastmild.get(go, -99) > 1:
                print(d["set"], d["episode"], "lvl", d["ref"]["stratum"], GRID[go], "week", t, "lack", round(lack, 1), "of target", round(row["target"], 1),
                      "owed bn", round(owed[go] / 1e9, 2), "dark bn", round(dark / 1e9, 2), "left", left, "START" if start else "")
            lastmild[go] = t
    print("starts:", tot)


if __name__ == "__main__":
    fire.Fire(main)

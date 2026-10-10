"""Walk the mild weeks of the census and count where mild_ends would hold (no play)."""
import sys
from collections import defaultdict
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse import FUEL, GRID, load


def main(name="s13_census", sets="full:12345:16,full:2026:16,full:777:32", partial=0.5, voll=4.13e6):
    docs, att = load(name, sets)
    n = 0
    tot = wtot = 0.0
    per = defaultdict(int)
    for d in docs:
        for t, go, code, row in d["census"]:
            if code != "mild" or d["last"][go] - t < 2:
                continue
            nxt, swap, hit = row["able"], True, False
            for k, f in row["fuels"].items():
                gives = min(f["burn"] * f["ration"], f["on_hand"] + f["feed"])
                if f["small"] or gives >= f["burn"] * (1 - 1e-9):
                    continue
                if f["threshold"] > 0 or not f["reach"] or not f["worth"]:
                    swap = False
                    break
                hit = True
                later = max(f["tank"] - f["feed"], 0.0) + f["inflow"]
                nxt += min(f["burn"], f["on_hand"] + f["feed"] - gives + later) - gives
                swap = f["on_hand"] - min(f["burn"], f["on_hand"]) + f["tank"] + f["inflow"] >= f["burn"] * (1 - 1e-9)
                if not swap:
                    break
            if hit and swap and nxt - row["y"] < partial * row["target"]:
                n += 1
                lack = row["y"] + row["target"] - row["able"]
                bn = lack * d["gain"][go] / d["full_draw"][go] / 1e9
                tot += bn
                wtot += d["w"] * bn
                per[(d["set"], d["episode"], GRID[go])] += 1
    print("weeks held:", n, "plain bn (the lack at the grid's gain):", round(tot, 1), "weighted", round(wtot, 2), "=", round(wtot / att, 5))
    print(dict(per))


if __name__ == "__main__":
    fire.Fire(main)

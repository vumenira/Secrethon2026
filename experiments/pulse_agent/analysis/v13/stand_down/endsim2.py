"""After each week mild_ends would hold: what v12's play did at that grid in the two weeks after (no play)."""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse import GRID, load

docs, att = load("s13_census", "full:12345:16,full:2026:16,full:777:32")
after = Counter()
rows = []
for d in docs:
    codes = {(t, go): (code, row) for t, go, code, row in d["census"]}
    for (t, go), (code, row) in sorted(codes.items()):
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
        if hit and swap and nxt - row["y"] < 0.5 * row["target"]:
            n1 = codes.get((t + 1, go), ("-", {}))
            n2 = codes.get((t + 2, go), ("-", {}))
            def kind(c):
                code, r = c
                if code == "idle":
                    return "shock" if r["power"] - r["y"] <= 1e-9 * max(r["y"], 1.0) else "nowafers"
                return code
            # the week after as the simulator ran it: did the grid's power fall within the week?
            null_now = float(d["segment"][t - 1].get((go, None), 0.0))
            null_next = float(d["segment"][t].get((go, None), 0.0)) if t < len(d["segment"]) else null_now
            fell = null_next < null_now * (1 - 1e-6)
            after[(kind(n1), kind(n2), "power fell in the week after" if fell else "")] += 1
            rows.append((d["set"], d["episode"], GRID[go], t, kind(n1), kind(n2), fell, round(row["want"] / max(d["full_draw"][go], 1e-9), 2)))
for k, v in after.most_common():
    print(v, k)
print([r for r in rows if r[6]])

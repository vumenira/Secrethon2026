"""Calibration of the forecast's still-on probability against the spells of the tuning root.

For every week an element is shown impaired: the agent's P(still impaired h weeks later) against whether it was.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).parent
sys.path.insert(0, r"C:\Users\mykha\Claude code projects\Secrethon2026\agents\mpc_recovery")
import recovery as R  # noqa: E402
from sbflow.disruption.params import FlagshipMarkLaws  # noqa: E402


laws = dict(FlagshipMarkLaws().duration)
parts = {name: R.components([(1.0, law)]) for name, law in laws.items()}
data = json.loads((HERE / (sys.argv[1] if len(sys.argv) > 1 else "spells.json")).read_text())
T = data["meta"]["T"]
HS = (1, 2, 3, 4, 6, 8, 13, 20)

acc = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0]))  # group -> h -> [sum forecast, sum truth, n]
for r in data["rows"]:
    if r["kind"] not in ("closure", "grid"):
        continue
    first, last = r["first"], r["last"]
    seen = first > 1
    for i, v in enumerate(r["values"]):
        t = first + i
        if r["kind"] == "closure":
            wr = r["wr"][i]
            if wr == 0:
                event = "weather_closure"
            else:
                event = "militarised_closure" if v > 1e-6 else None
            label = {"weather_closure": "weather", "militarised_closure": "militarised", None: "shut+wr (kept)"}[event]
        else:
            event, label = "energy_shock", "grid"
        age = t - first
        group = f"{label:16s} {'start seen' if seen else 'carried in':10s} age {'0' if age == 0 else '1-3' if age < 4 else '4-12' if age < 13 else '13+'}"
        p = np.ones(len(HS)) if event is None else R.still_on(parts[event], float(age), seen, np.array(HS, dtype=float))
        for j, h in enumerate(HS):
            if t + h > T:
                continue
            a = acc[group][h]
            a[0] += float(p[j])
            a[1] += float(t + h <= last)
            a[2] += 1
            b = acc[f"{label:16s} {'start seen' if seen else 'carried in':10s} all ages"][h]
            b[0] += float(p[j])
            b[1] += float(t + h <= last)
            b[2] += 1

print("P(still impaired h weeks later): forecast / observed (weeks counted)")
print(f"{'group':46s}" + "".join(f"{'h=' + str(h):>18s}" for h in HS))
for group in sorted(acc):
    cells = []
    for h in HS:
        f, o, n = acc[group][h]
        cells.append(f"{f / n:.2f}/{o / n:.2f} ({n:4d})" if n else " " * 16)
    print(f"{group:46s}" + "".join(f"{c:>18s}" for c in cells))

# the remaining life's quantiles the quantile mode would use, by age
print("\nweeks until P(still on) falls to 1 - q (the quantile mode's end), by event, start seen / carried in, age")
r = np.arange(0, 400.25, 0.25)
for event in ("weather_closure", "militarised_closure", "energy_shock"):
    for seen in (True, False):
        for age in (0, 1, 2, 4, 8, 16, 32):
            p = R.still_on(parts[event], float(age), seen, r)
            ends = []
            for q in (0.25, 0.5, 0.75, 0.9):
                past = np.flatnonzero(p <= 1 - q)
                ends.append(r[past[0]] if past.size else np.inf)
            mean_share = [float(np.trapezoid(p[h * 4 : h * 4 + 5], dx=0.25)) for h in (0, 1, 2, 4, 8, 19)]
            print(f"  {event:20s} {'seen' if seen else 'carried'} age {age:2d}: q25/50/75/90 end at {ends} weeks; mean share of weeks 0,1,2,4,8,19: {np.round(mean_share, 2)}")

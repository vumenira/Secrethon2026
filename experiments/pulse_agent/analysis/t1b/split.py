"""Cached costs split by what each episode shows: where a variant's saving over the base comes from.

uv run python split.py base variant [variant ...]
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).parent
RUN = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026\outputs\recovery_tune\small-e12345-n64")
data = json.loads((HERE / "spells.json").read_text())
T = data["meta"]["T"]
refs = json.loads((RUN / "references.json").read_text())
gap = np.array([r["J_naive_cents"] - r["J_oracle_cents"] for r in refs], dtype=float) / 1e11


def cost(name):
    return np.array([r["J_policy_cents"] for r in json.loads((RUN / f"{name}.json").read_text())["rows"]], dtype=float) / 1e11


tags = defaultdict(set)
for r in data["rows"]:
    n = r["episode"]
    if r["kind"] == "closure":
        types = {("W" if e["type"] == "weather_closure" else "D" if e["derived"] else "M") for e in r["events"]}
        for x in types:
            tags[n].add(x)
        if "M" in types:
            tags[n].add("M1" if r["first"] == 1 else "Mseen")
            if r["last"] < T:
                tags[n].add("Mends")
        if len(r["events"]) > 1:
            tags[n].add("multi")
    elif r["kind"] == "grid":
        tags[n].add("G")
        if len(r["events"]) > 1:
            tags[n].add("Gmulti")
        if r["first"] == 1:
            tags[n].add("G1")

base = cost(sys.argv[1])
groups = {
    "all": lambda t: True,
    "no closure shown": lambda t: not ({"W", "M", "D"} & t),
    "weather only": lambda t: "W" in t and not ({"M", "D"} & t),
    "militarised, none ends in episode": lambda t: "M" in t and "Mends" not in t,
    "militarised, one ends in episode": lambda t: "Mends" in t,
    "militarised at week 1": lambda t: "M1" in t,
    "militarised, only seen starts": lambda t: "Mseen" in t and "M1" not in t,
    "overlapping closures": lambda t: "multi" in t,
    "derived closure": lambda t: "D" in t,
    "grid shortage shown": lambda t: "G" in t,
    "no grid shortage": lambda t: "G" not in t,
    "grid, overlapping shocks": lambda t: "Gmulti" in t,
    "grid at week 1": lambda t: "G1" in t,
}
for name in sys.argv[2:]:
    save = base - cost(name)  # bn USD the variant saves over the base
    print(f"\n{name} against {sys.argv[1]}: saved {save.sum():+.1f} bn over {len(save)} episodes; mean saving / mean gap = {save.sum() / gap.sum():+.5f}")
    for label, test in groups.items():
        idx = [n for n in range(len(save)) if test(tags[n])]
        if not idx:
            continue
        s = save[idx]
        print(
            f"  {label:36s} {len(idx):3d} episodes  saved {s.sum():+8.1f}  per episode {s.mean():+6.2f}"
            f"  better/worse {int((s > 1e-6).sum())}/{int((s < -1e-6).sum())}  worst {s.min():+.1f} best {s.max():+.1f}"
        )
    order = np.argsort(save)
    print("  worst episodes:", [(int(n), round(float(save[n]), 1), "".join(sorted(tags[n]))) for n in order[:6]])
    print("  best episodes:", [(int(n), round(float(save[n]), 1), "".join(sorted(tags[n]))) for n in order[::-1][:6]])

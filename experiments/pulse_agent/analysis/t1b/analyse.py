"""Read spells.json: what the agent's classification sees against the events behind it."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


import sys
data = json.loads((Path(__file__).parent / (sys.argv[1] if len(sys.argv) > 1 else "spells.json")).read_text())
rows = data["rows"]
T = data["meta"]["T"]
closures = [r for r in rows if r["kind"] == "closure"]
grids = [r for r in rows if r["kind"] == "grid"]
events = [r for r in rows if r["kind"] == "event"]

print("events overlapping the episode, by type:", Counter(e["type"] for e in events))
print("  with onset in the episode:", Counter(e["type"] for e in events if e["onset"] > 0))
print("  carried in (onset <= 0, still on at 0):", Counter(e["type"] for e in events if e["onset"] <= 0))
print("  derived strait closures (sev 1 militarised):", sum(e["type"] == "militarised_closure" and e["sev"] >= 1 for e in events))
print("  militarised closure targets:", Counter(e["target"] for e in events if e["type"] == "militarised_closure"))
print("  energy shock target kinds:", Counter(e["target_kind"] for e in events if e["type"] == "energy_shock"))


def cls(v, w):
    shut = v <= 1e-6
    return ("shut" if shut else "partial") + ("+wr" if w > 0 else "")


def truth(r, week):
    """The event types active on the strait at the instant of ``week`` (week - 1)."""
    s = week - 1
    act = [e for e in r["events"] if e["onset"] <= s < e["end"]]
    kinds = []
    for e in act:
        if e["type"] == "weather_closure":
            kinds.append("W")
        elif e["derived"]:
            kinds.append("D")
        else:
            kinds.append("M")
    war = any(w["onset"] <= s < w["end"] for w in r["wars"])
    return "".join(sorted(kinds)) + ("|war" if war else "")


print("\n== closures: strait-weeks shown impaired, the agent's class against the events active ==")
table = defaultdict(Counter)
for r in closures:
    for i, (v, w) in enumerate(zip(r["values"], r["wr"])):
        table[cls(v, w)][truth(r, r["first"] + i)] += 1
for k, c in table.items():
    print(f"  {k:12s} {sum(c.values()):5d} weeks: {dict(c)}")

print("\n== closure spells ==")
print("  spells:", len(closures), "carried in (week 1):", sum(r["first"] == 1 for r in closures), "run to the end:", sum(r["last"] == T for r in closures))
print("  episodes with a spell at week 1:", len({r["episode"] for r in closures if r["first"] == 1}))
by = defaultdict(list)
for r in closures:
    first = cls(r["values"][0], r["wr"][0])
    by[(first, "carried" if r["first"] == 1 else "seen")].append(r)
for key, rs in sorted(by.items()):
    lens = [r["last"] - r["first"] + 1 for r in rs]
    cens = sum(r["last"] == T for r in rs)
    changes = sum(len(set(r["values"])) > 1 for r in rs)
    multi = sum(len(r["events"]) > 1 for r in rs)
    print(f"  first class {key}: {len(rs)} spells, weeks shown {sorted(lens)}, censored {cens}, value changes in {changes}, >1 event in {multi}")

print("\n== spells whose shown open fraction changes (value path, events) ==")
for r in closures:
    if len(set(r["values"])) > 1:
        path = [r["values"][0]]
        for v in r["values"][1:]:
            if v != path[-1]:
                path.append(v)
        ev = [(e["type"][:3] + ("D" if e["derived"] else ""), e["onset"], e["end"], e["sev"]) for e in r["events"]]
        print(f"  ep {r['episode']} {r['name']} weeks {r['first']}-{r['last']} path {path} after {r['after']} events {ev}")

print("\n== weather closures: how they show ==")
weather = [e for e in events if e["type"] == "weather_closure" and e["onset"] > 0]
shown = 0
for e in weather:
    # shown in week t iff onset <= t - 1 < end for an integer t
    weeks = [t for t in range(1, T + 1) if e["onset"] <= t - 1 < e["end"]]
    shown += bool(weeks)
print(f"  in-episode weather closures {len(weather)}, shown in at least one week {shown}")
print("  durations (weeks) of in-episode weather closures:", np.round(np.quantile([e['end'] - e['onset'] for e in weather], [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]), 2))

print("\n== militarised closures (not derived): durations of those with onset in the episode ==")
mil = [e for e in events if e["type"] == "militarised_closure" and e["sev"] < 1]
ins = [e for e in mil if e["onset"] > 0]
print("  count", len(ins), "durations:", sorted(round(e["end"] - e["onset"], 1) for e in ins))
print("  carried-in count", len(mil) - len(ins), "remaining life at 0:", sorted(round(e["end"], 1) for e in mil if e["onset"] <= 0))
print("  carried-in ages at 0:", sorted(round(-e["onset"], 1) for e in mil if e["onset"] <= 0))
print("  severities:", sorted(e["sev"] for e in mil))
der = [e for e in events if e["type"] == "militarised_closure" and e["sev"] >= 1]
print("  derived strait closures:", [(e["episode"], e["target"], e["onset"], e["end"]) for e in der])

print("\n== grid spells ==")
print("  spells:", len(grids), "carried in:", sum(r["first"] == 1 for r in grids), "to the end:", sum(r["last"] == T for r in grids))
print("  spells with no energy shock behind them:", sum(not r["events"] for r in grids))
print("  spells with more than one shock:", sum(len(r["events"]) > 1 for r in grids), "value changes in", sum(len(set(r["values"])) > 1 for r in grids))
print("  weeks shown per spell (seen start):", sorted(r["last"] - r["first"] + 1 for r in grids if r["first"] > 1))
print("  weeks shown per spell (carried):", sorted(r["last"] - r["first"] + 1 for r in grids if r["first"] == 1))
print("  by grid:", Counter(r["name"] for r in grids))
shocks = [e for e in events if e["type"] == "energy_shock"]
print("  energy shocks overlapping episodes:", len(shocks), "in episode:", sum(e["onset"] > 0 for e in shocks))
inside = [e for e in shocks if e["onset"] > 0]
never = sum(not any(e["onset"] <= t - 1 < e["end"] for t in range(1, T + 1)) for e in inside)
print("  in-episode shocks never shown at a week's instant:", never)
print("  durations (weeks):", np.round(np.quantile([e['end'] - e['onset'] for e in inside], [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]), 2))
print("  severities:", np.round(np.quantile([e['sev'] for e in shocks], [0, 0.25, 0.5, 0.75, 1.0]), 2))
for r in grids:
    if len(r["events"]) != 1 or len(set(r["values"])) > 1:
        print(f"  ep {r['episode']} {r['name']} weeks {r['first']}-{r['last']} values {r['values']} events {[(e['onset'], e['end'], e['sev']) for e in r['events']]}")

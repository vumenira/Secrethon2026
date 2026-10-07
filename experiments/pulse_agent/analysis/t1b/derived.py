"""Derived strait closures and regional conflicts on the sampled scenarios: durations against the law."""

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.special import ndtr

data = json.loads((Path(__file__).parent / sys.argv[1]).read_text())
ev = [r for r in data["rows"] if r["kind"] == "event"]
der = [e for e in ev if e["type"] == "militarised_closure" and e["sev"] >= 1.0]
ins = [e for e in der if e["onset"] > 0]
d = np.array([e["end"] - e["onset"] for e in ins])
print(f"derived closures overlapping an episode: {len(der)}, with onset in the episode: {len(ins)}")
print("  durations of those with onset in the episode:", np.round(np.sort(d), 1))
print(f"  share lasting exactly 52 weeks: {np.mean(np.abs(d - 52.0) < 1e-6):.2f}; the conflict law gives P(T <= 52 wk) = {ndtr((math.log(364.0) - 5.531) / 1.453):.2f}")
con = [e for e in ev if e["type"] == "regional_conflict" and e["onset"] > 0]
c = np.array([e["end"] - e["onset"] for e in con])
print(f"regional conflicts with onset in the episode: {len(con)}; share <= 52 wk {np.mean(c <= 52):.2f}; median {np.median(c):.1f} wk (law {math.exp(5.531) / 7:.1f})")
by = {}
for e in con:
    by.setdefault((e["region"], e["cp"]), []).append(e["end"] - e["onset"])
regions = data["meta"]["regions"]
for (m, cp), v in sorted(by.items(), key=lambda kv: -len(kv[1]))[:12]:
    print(f"    region {regions[m]} counterpart {regions[cp] if cp >= 0 else None}: {len(v)} conflicts, share <= 52 wk {np.mean(np.array(v) <= 52):.2f}, median {np.median(v):.1f}")
# episodes with a derived closure and how many
eps = {}
for e in der:
    eps.setdefault(e["episode"], []).append((e["onset"], e["end"]))
print(f"episodes with a derived closure: {len(eps)} of {len({r['episode'] for r in data['rows']})}; with several: {sum(len(v) > 1 for v in eps.values())}")

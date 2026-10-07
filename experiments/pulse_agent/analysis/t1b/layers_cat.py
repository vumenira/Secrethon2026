"""Forecast errors of the single and the layered booking by what the layers hold (closures only).

uv run python layers_cat.py spells_big.json
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np


HERE = Path(__file__).parent
sys.path.insert(0, r"C:\Users\mykha\Claude code projects\Secrethon2026\agents\rec_rule")
import recovery as R  # noqa: E402

from shockbench_flow.hosting.tasks import task_generator  # noqa: E402

inst, _params = task_generator("small")
PARAMS = {"kinds": ["closure", "grid"], "mode": "mean", "quantile": 0.5, "caution": 1.0, "rules": {}}
data = json.loads((HERE / sys.argv[1]).read_text())
T = data["meta"]["T"]
rows = data["rows"]
HS = (1, 2, 4, 8, 13, 20)
rec = R.Recovery(inst, PARAMS)
u0 = np.array([np.inf if e.u0 is None else e.u0 for e in inst.edges])

series = defaultdict(lambda: np.ones(T + 1))
for r in rows:
    if r["kind"] == "closure":
        series[(r["episode"], r["element"])][r["first"] : r["last"] + 1] = r["values"]

err = {m: defaultdict(lambda: defaultdict(list)) for m in ("old", "new")}
tell = Counter()
for r in rows:
    if r["kind"] != "closure":
        continue
    el = r["element"]
    full = series[(r["episode"], el)]
    layers, first, seen0 = None, r["first"], r["first"] > 1
    for i, v in enumerate(r["values"]):
        t = first + i
        wr = r["wr"][i]
        u = u0.copy()
        if r.get("dyad") is not None:
            for e in rec.dyads.get(el, ()):
                u[e] = r["dyad"][i] * u0[e]
        memory = SimpleNamespace(values={"war_risk": np.full(len(inst.chokepoints), wr), "u": u})
        layers = rec.relayer(layers, "closure", el, v, t, (t > 1) if layers is None else True, memory)
        s = t - 1
        act = [e for e in r["events"] if e["onset"] <= s < e["end"]]
        true_d = any(e.get("derived") for e in act)
        true_w = any(e["type"] == "weather_closure" for e in act)
        mine = [layer[1] for layer in layers]
        if v <= R.NEAR and wr > 0:  # shut under a war-risk class: what the agent says against what it is
            tell[("says " + ("derived" if None in mine else "weather"), "is " + ("derived" if true_d else "weather" if true_w else "other"))] += 1
        if None in mine:
            cat = "a derived closure, start seen" if any(l[1] is None and l[3] for l in layers) else "a derived closure, carried in"
        elif "weather_closure" in mine and wr > 0:
            cat = "weather under a war-risk class"
        elif len(layers) > 1:
            cat = "several militarised layers"
        else:
            cat = None
        hs = np.array(HS, dtype=float)
        event = "weather_closure" if wr == 0 else ("militarised_closure" if v > R.NEAR else None)
        p_old = np.ones(len(HS)) if event is None else R.still_on(rec.parts[event], float(t - first), seen0, hs)
        old = 1.0 - (1.0 - v) * p_old
        new = np.ones(len(HS))
        for factor, ev, f0, sn in layers:
            new = new * (1.0 - (1.0 - factor) * (1.0 if ev is None else R.still_on(rec.parts[ev], float(t - f0), sn, hs)))
        if cat is None:
            continue
        for j, h in enumerate(HS):
            if t + h <= T:
                err["old"][cat][h].append(old[j] - full[t + h])
                err["new"][cat][h].append(new[j] - full[t + h])

print("a strait shut under a war-risk class, strait-weeks:", dict(tell))
print("forecast of the open share h weeks ahead: bias / mean abs error / rmse, single | layered")
for cat in sorted(err["old"]):
    print(" ", cat)
    for h in HS:
        a, b = np.array(err["old"][cat][h]), np.array(err["new"][cat][h])
        if len(a):
            print(f"    h={h:2d} n={len(a):5d}  single {a.mean():+.3f} / {np.abs(a).mean():.3f} / {np.sqrt((a**2).mean()):.3f} | layered {b.mean():+.3f} / {np.abs(b).mean():.3f} / {np.sqrt((b**2).mean()):.3f}")

"""The layered booking of agents/rec_rule against the events behind each spell, and both models' forecast errors.

uv run python layers_test.py spells_big.json [first_episode last_episode]
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
lo, hi = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 10**9)
T = data["meta"]["T"]
rows = [r for r in data["rows"] if lo <= r["episode"] <= hi]
HS = (1, 2, 3, 4, 6, 8, 13, 20)
rec = R.Recovery(inst, PARAMS)
u0 = np.array([np.inf if e.u0 is None else e.u0 for e in inst.edges])


def truth_types(r, week):
    s = week - 1
    out = []
    for e in r["events"]:
        if e["onset"] <= s < e["end"]:
            out.append("weather_closure" if e["type"] == "weather_closure" else "strait_closure" if e.get("derived") else e["type"])
    return sorted(out)


# the full series per (kind, episode, element)
series = defaultdict(lambda: np.ones(T + 1))
for r in rows:
    if r["kind"] in ("closure", "grid"):
        series[(r["kind"], r["episode"], r["element"])][r["first"] : r["last"] + 1] = r["values"]

agree, confusion = Counter(), Counter()
err = {m: defaultdict(lambda: defaultdict(list)) for m in ("old", "new")}
changed = Counter()
for r in rows:
    if r["kind"] not in ("closure", "grid"):
        continue
    kind, el = r["kind"], r["element"]
    full = series[(kind, r["episode"], el)]
    layers, first, seen0 = None, r["first"], r["first"] > 1
    for i, v in enumerate(r["values"]):
        t = first + i
        wr = r["wr"][i] if kind == "closure" else 0
        u = u0.copy()
        if kind == "closure" and r.get("dyad") is not None:
            for e in rec.dyads.get(el, ()):
                u[e] = r["dyad"][i] * u0[e]
        memory = SimpleNamespace(values={"war_risk": np.full(len(inst.chokepoints), wr), "u": u})
        seen = (t > 1) if layers is None else True
        layers = rec.relayer(layers, kind, el, v, t, seen, memory)
        mine = sorted("strait_closure" if layer[1] is None else layer[1] for layer in layers)
        true = truth_types(r, t)
        agree[(kind, mine == true)] += 1
        if mine != true:
            confusion[(kind, tuple(mine), tuple(true))] += 1
        # forecasts of the share at the instants h weeks ahead, in "mean" mode
        hs = np.array(HS, dtype=float)
        if kind == "closure":
            event = "weather_closure" if wr == 0 else ("militarised_closure" if v > R.NEAR else None)
        else:
            event = "energy_shock"
        p_old = np.ones(len(HS)) if event is None else R.still_on(rec.parts[event], float(t - first), seen0, hs)
        old = 1.0 - (1.0 - v) * p_old
        new = np.ones(len(HS))
        for factor, ev, f0, sn in layers:
            new = new * (1.0 - (1.0 - factor) * (1.0 if ev is None else R.still_on(rec.parts[ev], float(t - f0), sn, hs)))
        differ = bool(np.max(np.abs(new - old)) > 1e-9)
        changed[(kind, differ)] += 1
        for j, h in enumerate(HS):
            if t + h > T:
                continue
            true_v = full[t + h]
            for m, f in (("old", old[j]), ("new", new[j])):
                err[m][(kind, "all")][h].append(f - true_v)
                if differ:
                    err[m][(kind, "weeks the two models differ")][h].append(f - true_v)

print("layer types against the events in force, weeks:", dict(agree))
for key, n in confusion.most_common(12):
    print("   ", key, n)
print("weeks where the layered forecast differs from the single one:", dict(changed))
print("\nforecast of the share h weeks ahead: bias (forecast - truth) / mean abs error, old | new")
for key in sorted(err["old"]):
    print(" ", key)
    for h in HS:
        a, b = np.array(err["old"][key][h]), np.array(err["new"][key][h])
        if len(a):
            print(f"    h={h:2d} n={len(a):5d}  old {a.mean():+.3f} / {np.abs(a).mean():.3f} rmse {np.sqrt((a**2).mean()):.3f} | new {b.mean():+.3f} / {np.abs(b).mean():.3f} rmse {np.sqrt((b**2).mean()):.3f}")

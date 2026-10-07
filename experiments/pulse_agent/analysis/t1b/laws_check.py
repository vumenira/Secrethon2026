"""The duration laws the Small generator really uses, and the tuning root's in-episode durations against them."""

import json
from pathlib import Path

import numpy as np
from scipy import stats


from shockbench_flow.hosting.tasks import task_generator

inst, params = task_generator("small")
print("laws class:", type(params.laws).__name__, "burn-in", params.burn_in)
for name, law in params.laws.duration:
    print("  duration", name, law.kind, law.params)
for name, law in params.laws.severity:
    print("  severity", name, law.kind, law.params)
print("  strait closures:", params.laws.strait_closures, params.laws.strait_closure_weeks, params.laws.strait_closure_severity)
print("  poisson:", params.poisson.weather_per_chokepoint_year, params.poisson.strike_per_region_year)
print("  hawkes gamma", params.hawkes.gamma, "gamma_M_max", params.hawkes.gamma_M_max, "beta_inv", params.hawkes.beta_inv, "r_cross", params.hawkes.r_cross)
print("  types M:", params.types.militarised)
a, b = "CN", "TW"
ids = [r if isinstance(r, str) else r.id for r in inst.regions]
ra, rb = ids.index(a), ids.index(b)
between = [
    (e, edge.id, edge.u0, edge.mode)
    for e, edge in enumerate(inst.edges)
    if not edge.coupling and {inst.nodes[edge.tail].region, inst.nodes[edge.head].region} == {ra, rb}
]
print("  edges between CN and TW:", between)
print("  grids:", [(inst.nodes[g].id, inst.nodes[g].region) for g in inst.grids])

data = json.loads((Path(__file__).parent / "spells.json").read_text())
ev = [r for r in data["rows"] if r["kind"] == "event" and r["onset"] > 0]
laws = dict(params.laws.duration)
for name in ("weather_closure", "energy_shock", "militarised_closure"):
    d = np.array([e["end"] - e["onset"] for e in ev if e["type"] == name and e["sev"] < 1.0 or (e["type"] == name and name != "militarised_closure")])
    u = np.linspace(0.0005, 0.9995, 2000)
    q = np.array([laws[name].ppf(np.array([x] * laws[name].uniforms) if laws[name].uniforms > 1 else x) for x in u]) if laws[name].uniforms == 1 else None
    if q is None:
        print(name, "n", len(d), "(a mixture: no direct quantile check)", "median", np.median(d))
        continue
    cdf = lambda x: np.interp(x, q, u)  # noqa: E731
    ks = stats.kstest(d, cdf)
    print(f"{name}: n {len(d)}, sample median {np.median(d):.2f} wk against the law's {laws[name].ppf(0.5):.2f}; "
          f"share over the law's 75 % point {np.mean(d > laws[name].ppf(0.75)):.2f}, over its 90 % point {np.mean(d > laws[name].ppf(0.9)):.2f}; KS p {ks.pvalue:.3f}")

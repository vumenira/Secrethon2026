"""The planned still-on share of the first window weeks under each rule, from the agent's own code."""

import sys

import numpy as np

sys.path.insert(0, r"C:\Users\mykha\Claude code projects\Secrethon2026\agents\rec_rule")
import recovery as R  # noqa: E402

from shockbench_flow.hosting.tasks import task_generator  # noqa: E402

inst, _ = task_generator("small")
for label, rules in (
    ("mean", {}),
    ("quantile 0.5", {"closure": {"mode": "quantile", "quantile": 0.5}, "grid": {"mode": "quantile", "quantile": 0.5}}),
    ("quantile 0.25", {"closure": {"mode": "quantile", "quantile": 0.25}, "grid": {"mode": "quantile", "quantile": 0.25}}),
    ("quantile 0.1", {"closure": {"mode": "quantile", "quantile": 0.1}, "grid": {"mode": "quantile", "quantile": 0.1}}),
):
    rec = R.Recovery(inst, {"kinds": ["closure", "grid"], "mode": "mean", "quantile": 0.5, "caution": 1.0, "rules": rules})
    print(label)
    for kind, event in (("closure", "weather_closure"), ("grid", "energy_shock"), ("closure", "militarised_closure")):
        for seen in (True, False):
            for age in (0, 1, 3, 8):
                s = rec.share(kind, event, age, seen, 20)
                print(f"   {event:20s} {'start seen' if seen else 'carried in'} age {age}: weeks 0-5 {np.round(s[:6], 2)} week 10 {s[10]:.2f} week 19 {s[19]:.2f}")

"""The slowdown and stoppage duration laws as the generator draws them, and where this root's strikes fall in them."""
import math
import pickle
from pathlib import Path

import numpy as np
from scipy.special import ndtr
from shockbench_flow.disruption.params import PoissonParams

p = PoissonParams()
for name in ("stoppage_duration", "slowdown_duration"):
    law = getattr(p, name)
    print(name, law.kind, law.params, "ppf 0.1/0.5/0.9/0.99:", [round(float(law.ppf(q)), 2) for q in (0.1, 0.5, 0.9, 0.99)])
data = pickle.loads((Path(__file__).parent / "caparrays.pkl").read_bytes())
stop, slow = [], []
for w in data:
    for onset, dur, sev, target in w["strikes"]:
        (stop if abs(sev - 0.93) < 1e-9 else slow).append((onset, dur))
for name, xs, (mu, sg) in (("stoppage", stop, (0.48, 0.67)), ("slowdown", slow, (1.93, 0.77))):
    d = np.array([x[1] for x in xs])
    on = np.array([x[0] for x in xs])
    u = ndtr((np.log(d) - mu) / sg)
    inside = on >= 0
    print(f"{name}: {len(d)} strikes, {int(inside.sum())} with onset inside the episode; mean of the law's CDF at the drawn durations:"
          f" all {u.mean():.3f}, onset inside {u[inside].mean():.3f} (0.5 expected, s.e. {0.2887 / math.sqrt(max(inside.sum(), 1)):.3f});"
          f" over the law's 90 % point: {int((u > 0.9).sum())} of {len(d)}")

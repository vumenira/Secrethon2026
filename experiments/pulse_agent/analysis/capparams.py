"""The task's generator parameters for strikes against the defaults the agent ships, and the strikes drawn."""

import math
import pickle
from pathlib import Path

import numpy as np


def main():
    from shockbench_flow.disruption.params import PoissonParams
    from shockbench_flow.hosting.tasks import task_generator

    for task in ("small", "full"):
        inst, params = task_generator(task)
        print(task, "poisson equals the defaults:", params.poisson == PoissonParams(), "| regions", len(inst.regions), "T", inst.T, "burn_in", params.burn_in)
        if params.poisson != PoissonParams():
            print("   ", params.poisson)
    data = pickle.loads((Path(__file__).parent / "capcells.pkl").read_bytes())
    stop, slow = [], []
    for p in data["per_episode"]:
        for onset, dur, sev, kind, target in p["strikes"]:
            (stop if abs(sev - 0.93) < 1e-9 else slow).append((onset, dur, sev))
    for name, xs, (mu, sg) in (("stoppage", stop, (0.48, 0.67)), ("slowdown", slow, (1.93, 0.77))):
        d = np.array([x[1] for x in xs])
        print(f"{name}: {len(xs)} strikes; duration weeks median {np.median(d):.2f} mean {d.mean():.2f} max {d.max():.1f};"
              f" law median {math.exp(mu):.2f} mean {math.exp(mu + sg * sg / 2):.2f}; log-mean {np.log(d).mean():.2f} log-sd {np.log(d).std():.2f}")
    print("slowdowns (onset, duration, severity):", [(round(a, 1), round(b, 1), round(c, 2)) for a, b, c in sorted(slow)])


if __name__ == "__main__":
    main()

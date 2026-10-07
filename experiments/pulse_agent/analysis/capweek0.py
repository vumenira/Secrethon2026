"""This week's true capacity on edges the agent sees under a strike, against what each rule would request."""

import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026")
sys.path.insert(0, str(REPO / "agents" / "rec_capacity"))


def main():
    from recovery import Edges, components, still_on
    from sbflow.disruption.params import PoissonParams
    from sbflow.marks import MarkParams

    poisson = PoissonParams()
    parts = {
        "port_stoppage": components([(1.0, poisson.stoppage_duration)]),
        "port_slowdown": components([(1.0, poisson.slowdown_duration)]),
    }
    data = pickle.loads((Path(__file__).parent / "caparrays.pkl").read_bytes())
    w4 = np.array([1.0, 4.0, 2.0, 4.0, 1.0]) / 12.0
    r = np.arange(5) / 4.0
    stats = defaultdict(lambda: defaultdict(float))
    for w in data:
        edges = Edges(poisson, MarkParams())
        T, E = w["share"].shape
        for e in np.flatnonzero(w["finite"]):
            book = None
            for t in range(T):
                s = float(w["share"][t, e])
                event, level = edges.read(int(e), s)
                if event is None:
                    book = None
                    continue
                if book is None or book[2] != event:
                    book = (t, t > 0, event)
                age = t - book[0]
                p = still_on(parts[event], float(age), book[1], r)
                share0 = float(w4 @ p)  # the expected share of this week the strike is on
                true = float(w["avg"][t, e])  # the week's true average share of nominal capacity
                cut = level - s
                st = stats[event]
                st["cells"] += 1
                st["ends this week"] += true > s + 1e-9
                st["cut (share of nominal)"] += cut
                st["true capacity back this week / cut"] += (true - s) / cut
                st["mean rule plans back / cut"] += 1.0 - share0
                # what a request sized to each planned capacity can use of the capacity that is really back
                st["used by the mean rule / cut"] += min(true - s, cut * (1.0 - share0)) / cut
                st["used by open / cut"] += (min(true, level) - s) / cut
                st["asked above the true capacity by the mean rule / cut"] += max(s + cut * (1.0 - share0) - true, 0.0) / cut
                st["asked above the true capacity by open / cut"] += max(level - true, 0.0) / cut
    for event, st in stats.items():
        n = st["cells"]
        print(f"{event}: {int(n)} cells")
        for k, v in st.items():
            if k != "cells":
                print(f"     {k:55s} {v / n:6.3f} per cell")


if __name__ == "__main__":
    main()

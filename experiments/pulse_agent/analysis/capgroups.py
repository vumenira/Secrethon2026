"""Savings of cached variants by the kind of strike cells an episode holds (rec_capacity's reading of edges)."""

import json
import pickle
import sys
from pathlib import Path

import numpy as np


REPO = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026")
RUN = REPO / "outputs" / "recovery_tune" / "small-e12345-n64"
sys.path.insert(0, str(REPO / "agents" / "rec_capacity"))


def costs(name):
    rows = sorted(json.loads((RUN / f"{name}.json").read_text())["rows"], key=lambda r: r["episode"])
    return np.array([r["J_policy_cents"] for r in rows], dtype=float)


def main(*names, rows=False):
    from recovery import Edges
    from sbflow.disruption.params import PoissonParams
    from sbflow.marks import MarkParams

    data = pickle.loads((Path(__file__).parent / "caparrays.pkl").read_bytes())
    refs = sorted(json.loads((RUN / "references.json").read_text()), key=lambda r: r["episode"])
    gap = np.array([(r["J_naive_cents"] - r["J_oracle_cents"]) / 1e11 for r in refs])
    counts = []
    for w in data:
        edges = Edges(PoissonParams(), MarkParams())
        T, E = w["share"].shape
        c = {"port_stoppage": 0, "port_slowdown": 0, "over": 0}
        for e in np.flatnonzero(w["finite"]):
            for t in range(T):
                event, level = edges.read(int(e), float(w["share"][t, e]))
                if event is not None:
                    c[event] += 1
                    c["over"] += level < 1.0 - 1e-9
        counts.append(c)
    stop = np.array([c["port_stoppage"] for c in counts])
    slow = np.array([c["port_slowdown"] for c in counts])
    groups = {
        "no strike cell": (stop == 0) & (slow == 0),
        "stoppage cells only": (stop > 0) & (slow == 0),
        "slowdown cells only": (stop == 0) & (slow > 0),
        "both": (stop > 0) & (slow > 0),
    }
    base = costs("mpc_det_h20")
    gains = {n: (base - costs(n)) / 1e11 for n in names}
    print(f"{'group':22s} {'episodes':>8s} {'stop':>6s} {'slow':>6s}  " + " ".join(f"{n:>18s}" for n in names))
    for label, m in groups.items():
        print(f"{label:22s} {int(m.sum()):8d} {int(stop[m].sum()):6d} {int(slow[m].sum()):6d}  "
              + " ".join(f"{gains[n][m].sum():+10.1f} ({int((gains[n][m] > 1e-6).sum()):2d}/{int((gains[n][m] < -1e-6).sum()):2d})" for n in names))
    print(f"{'all':22s} {len(base):8d} {int(stop.sum()):6d} {int(slow.sum()):6d}  " + " ".join(f"{gains[n].sum():+10.1f} ({int((gains[n] > 1e-6).sum()):2d}/{int((gains[n] < -1e-6).sum()):2d})" for n in names))
    if rows:
        print("\nepisode gap stop slow over  " + " ".join(names))
        order = np.argsort(-(stop + slow))
        for i in order:
            if stop[i] + slow[i] == 0:
                continue
            print(f"{i:4d} {gap[i]:7.1f} {stop[i]:4d} {slow[i]:4d} {counts[i]['over']:4d}  " + " ".join(f"{gains[n][i]:+9.2f}" for n in names))


if __name__ == "__main__":
    import fire

    fire.Fire(main)

"""Calibration of the capacity forecast against ground truth, from capcells.pkl (capdiag.py)."""

import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


REPO = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026")
sys.path.insert(0, str(REPO / "agents" / "mpc_recovery"))
H = 20


def main():
    from recovery import components, still_on
    from sbflow.disruption.params import PoissonParams

    poisson = PoissonParams()
    parts = {
        "port_stoppage": components([(1.0, poisson.stoppage_duration)]),
        "port_slowdown": components([(1.0, poisson.slowdown_duration)]),
    }
    data = pickle.loads((Path(__file__).parent / "capcells.pkl").read_bytes())
    rows = data["rows"]
    print("episodes with a strike:", sum(bool(p["strikes"]) for p in data["per_episode"]), "of", len(data["per_episode"]))
    print("strikes:", sum(len(p["strikes"]) for p in data["per_episode"]))
    print("episodes with a cell classed as a strike:", len({r["episode"] for r in rows if r["cls"] != "kept"}))
    print("episodes with any impaired edge:", len({r["episode"] for r in rows}))
    edges = data["per_episode"][0]["edges"]
    print("edges with a finite capacity:", edges)

    # 1. the remaining life: forecast against truth, by class and true cause
    groups = defaultdict(list)
    for r in rows:
        if r["cls"] == "kept":
            continue
        groups[(r["cls"], r["causes"], r["seen"])].append(r)
    r_grid = np.arange(0, H * 4 + 1) / 4.0
    for key, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        fc = np.zeros(H)
        tr = np.zeros(H)
        cnt = np.zeros(H)
        rem_true, rem_fc = [], []
        for r in rs:
            age = r["week"] - r["first"]
            p = still_on(parts[r["cls"]], float(age), r["seen"], r_grid)
            w = np.array([1.0, 4.0, 2.0, 4.0, 1.0]) / 12.0
            share = np.array([float(w @ p[h * 4 : (h + 1) * 4 + 1]) for h in range(H)])
            avg = np.array(r["avg"])
            n = len(avg)
            true_share = (1.0 - avg) / (1.0 - r["share"])  # the share of the observed cut still on in week h
            fc[:n] += share[:n]
            tr[:n] += true_share
            cnt[:n] += 1
            rem_true.append(min(r["remaining"], r["left"]))
            rem_fc.append(float(np.trapezoid(p, r_grid)))
        cnt = np.maximum(cnt, 1)
        print(f"\n{key}: {len(rs)} cells in {len({r['episode'] for r in rs})} episodes, "
              f"{len({(r['episode'], r['edge'], r['first']) for r in rs})} spells")
        print("   true remaining weeks (capped by the episode's end): mean %.1f median %.1f" % (np.mean(rem_true), np.median(rem_true)))
        print("   forecast remaining weeks (within 20):                mean %.1f" % np.mean(rem_fc))
        print("   h        " + " ".join(f"{h:5d}" for h in (0, 1, 2, 3, 4, 6, 8, 12, 16, 19)))
        print("   forecast " + " ".join(f"{fc[h] / cnt[h]:5.2f}" for h in (0, 1, 2, 3, 4, 6, 8, 12, 16, 19)))
        print("   truth    " + " ".join(f"{tr[h] / cnt[h]:5.2f}" for h in (0, 1, 2, 3, 4, 6, 8, 12, 16, 19)))

    # 2. the share values of cells classed as a strike whose cause is not a strike alone
    odd = Counter()
    for r in rows:
        if r["cls"] != "kept" and r["causes"] != "strike":
            odd[(r["cls"], r["causes"], round(r["share"], 4))] += 1
    print("\nshares of cells classed as a strike with another cause:")
    for k, v in sorted(odd.items(), key=lambda kv: -kv[1])[:30]:
        print("  ", k, v)
    kept = Counter()
    for r in rows:
        if r["cls"] == "kept":
            kept[(r["causes"], round(r["share"], 4))] += 1
    print("\nshares of cells kept:")
    for k, v in sorted(kept.items(), key=lambda kv: -kv[1])[:30]:
        print("  ", k, v)


if __name__ == "__main__":
    main()

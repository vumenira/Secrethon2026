"""Our play of an episode (the census pickles of v12, equal to v12_all to the cent), grid by grid: no play, no solve.

uv run python outputs/w13a/ours.py --eps="777:26,2026:3"
"""
import json
import pickle
from pathlib import Path

import fire
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SETS = {12345: 16, 2026: 16, 777: 32}
W = {1: 0.50, 2: 0.30, 3: 0.15, 4: 0.05}


def refs(root):
    return {r["episode"]: r for r in json.loads((ROOT / "outputs" / "recovery_tune" / f"full-e{root}-n{SETS[root]}" / "references.json").read_text())}


def load(root, n):
    with open(ROOT / "outputs" / "s13" / "s13_census" / f"full-e{root}" / f"ep{n}.pkl", "rb") as f:
        return pickle.load(f)


def blocks(a, step=13):
    return " ".join(f"{a[i:i + step].mean():4.2f}" for i in range(0, len(a), step))


def main(eps, weeks=False):
    from shockbench_flow.hosting.tasks import task_generator

    inst, _ = task_generator("full")
    N, K = inst.nodes, inst.commodities
    kid = [c.id for c in K]
    for item in str(eps).split(","):
        root, n = (int(v) for v in item.split(":"))
        d, ref = load(root, n), refs(root)[n]
        J, Jn, Jo = d["J_cents"] / 1e11, ref["J_naive_cents"] / 1e11, ref["J_oracle_cents"] / 1e11
        costs = {c: sum(w[c] for w in d["costs"]) / 1e9 for c in d["costs"][0]}
        print(f"\n===== root {root} episode {n} level {ref['stratum']}: ours {J:,.0f} naive {Jn:,.0f} clairvoyant {Jo:,.0f} bn; score {(Jn - J) / (Jn - Jo):.3f}; shortfall {J - Jo:,.0f}")
        print("  our costs bn:", {c: round(v) for c, v in costs.items()})
        T = len(d["segment"])
        for go, g in enumerate(inst.grids):
            ga = N[g].grid
            fabs = [fo for fo in inst.grid_fabs[go] if N[inst.fabs[fo]].fab.e > 0]
            draw = sum(N[inst.fabs[fo]].fab.e * N[inst.fabs[fo]].fab.cap0 for fo in fabs)
            E = d["energy"][:, fabs].sum(axis=1) if fabs else np.zeros(T)
            shed = d["shed"][:, go]
            line = f"  {N[g].id:9s} shed {shed.sum() * ga.voll / 1e9:7,.0f} bn"
            if fabs:
                line += f" | fab energy share {E.sum() / (draw * T):.2f} by quarter: {blocks(E / draw)}"
            print(line)
            for k in ga.fuels:
                seg = np.array([w.get((go, k), 0.0) for w in d["segment"]])
                full = ga.shares[k] * ga.deliverable
                print(f"      {kid[k]:8s} burn {full:7.0f}/wk segment share {seg.sum() / (full * T):.2f} by quarter: {blocks(seg / full)}")
            for fo in fabs:
                fa = N[inst.fabs[fo]].fab
                lots = d["lots"][:, fo]
                print(f"      {N[inst.fabs[fo]].id:18s} lots/cap {lots.sum() / (fa.cap0 * T):.2f} by quarter: {blocks(lots / fa.cap0)}")
        print("  lost sales, bn (share of demand lost):")
        for do, dm in enumerate(inst.demands):
            lost, dem = d["lost"][:, do], d["demand"][:, do]
            print(f"      {N[dm.node].id:9s} {kid[dm.k]:9s} {lost.sum() * dm.pi / 1e9:7,.0f} ({lost.sum() / dem.sum():.2f}) by quarter: {blocks(lost / np.maximum(dem, 1e-9))}")


if __name__ == "__main__":
    fire.Fire(main)

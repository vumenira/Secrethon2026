"""Where an agent's cost exceeds the clairvoyant plan's on Full root 12345: by category, node, and part of the episode.

Every number is a share of the naive-to-clairvoyant gap, weighted by harm level as the score is, so the shares of all
categories (less the difference in end credit) add up to 1 - score.
"""

import json
from pathlib import Path

import fire
import numpy as np


HERE = Path(__file__).resolve().parent
REFS = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026\outputs\recovery_tune\full-e12345-n16\references.json")
W = {1: 0.50, 2: 0.30, 3: 0.15, 4: 0.05}
COMPS = ("freight", "war_risk", "tariff", "holding", "queue_holding", "shortage", "disposal", "shed")
BUCKETS = ((0, 26), (26, 52), (52, 78), (78, 104))


def main(name="f_e10_w4", other="", episodes=16, detail=True):
    from shockbench_flow.hosting.tasks import task_generator

    inst, _ = task_generator("full")
    N, K, S = inst.nodes, inst.commodities, inst.stock_slots
    refs = json.loads(REFS.read_text())[:episodes]
    strata = np.array([r["stratum"] for r in refs])
    w = np.array([W[s] / (strata == s).sum() for s in strata])  # weight of each episode in the pooled means
    gap = np.array([r["J_naive_cents"] - r["J_oracle_cents"] for r in refs], dtype=float) / 100
    den = float((w * gap).sum())
    A = [np.load(HERE / "gap" / f"{name}_{n}.npz", allow_pickle=True) for n in range(episodes)]
    O = [np.load(HERE / "gap" / f"oracle_{n}.npz", allow_pickle=True) for n in range(episodes)]
    B = [np.load(HERE / "gap" / f"{other}_{n}.npz", allow_pickle=True) for n in range(episodes)] if other else None

    def pooled(values):  # values: per-episode numbers (USD) -> share of the gap
        return float((w * np.asarray(values, dtype=float)).sum()) / den

    Ja = np.array([float(int(a["J_cents"])) / 100 for a in A])
    Jo = np.array([float(int(o["J_cents"])) / 100 for o in O])
    Jo_ref = np.array([r["J_oracle_cents"] / 100 for r in refs])
    print(f"{name}: score {1 - pooled(Ja - Jo_ref):.4f}; clairvoyant re-solve differs from the reference by {np.abs(Jo - Jo_ref).max():.0f} USD at most")
    print(f"gap naive - clairvoyant, bn USD per episode (pooled) {den / 1e9:.1f}; per level "
          + " ".join(f"{gap[strata == s].mean() / 1e9:.0f}" for s in (1, 2, 3, 4)))
    print("\nshare of the gap by cost category (agent - clairvoyant), whole episode and by quarter of the episode:")
    total = 0.0
    for ci, c in enumerate(COMPS):
        d = [a["comps"][:, ci].sum() - o["comps"][:, ci].sum() for a, o in zip(A, O)]
        by = [pooled([a["comps"][lo:hi, ci].sum() - o["comps"][lo:hi, ci].sum() for a, o in zip(A, O)]) for lo, hi in BUCKETS]
        lev = [float(np.mean([d[i] for i in range(episodes) if strata[i] == s])) / 1e9 for s in (1, 2, 3, 4)]
        ag = pooled([a["comps"][:, ci].sum() for a in A]) * den / 1e9
        orc = pooled([o["comps"][:, ci].sum() for o in O]) * den / 1e9
        total += pooled(d)
        print(f"  {c:14s} {pooled(d):+.4f}   quarters " + " ".join(f"{b:+.4f}" for b in by)
              + f"   agent {ag:8.1f} bn, clairvoyant {orc:8.1f} bn;  by level (bn) " + " ".join(f"{x:+7.1f}" for x in lev))
    sal = pooled([float(o["salvage"]) - float(a["salvage"]) for a, o in zip(A, O)])
    print(f"  {'end credit':14s} {sal:+.4f}   (clairvoyant's credit minus the agent's)")
    print(f"  sum {total + sal:+.4f} = 1 - score")
    if not detail:
        return
    print("\npower shed by grid (share of the gap; agent - clairvoyant), by quarter; agent bn / clairvoyant bn:")
    for go, g in enumerate(inst.grids):
        d = [a["shed"][:, go].sum() - o["shed"][:, go].sum() for a, o in zip(A, O)]
        by = [pooled([a["shed"][lo:hi, go].sum() - o["shed"][lo:hi, go].sum() for a, o in zip(A, O)]) for lo, hi in BUCKETS]
        ag = pooled([a["shed"][:, go].sum() for a in A]) * den / 1e9
        orc = pooled([o["shed"][:, go].sum() for o in O]) * den / 1e9
        print(f"  {N[g].id:10s} {pooled(d):+.4f}   quarters " + " ".join(f"{b:+.4f}" for b in by) + f"   {ag:8.1f} / {orc:8.1f}")
    print("\nunserved demand by sink and good (share of the gap), by quarter; agent bn / clairvoyant bn:")
    for do, dm in enumerate(inst.demands):
        d = [a["short"][:, do].sum() - o["short"][:, do].sum() for a, o in zip(A, O)]
        by = [pooled([a["short"][lo:hi, do].sum() - o["short"][lo:hi, do].sum() for a, o in zip(A, O)]) for lo, hi in BUCKETS]
        ag = pooled([a["short"][:, do].sum() for a in A]) * den / 1e9
        orc = pooled([o["short"][:, do].sum() for o in O]) * den / 1e9
        print(f"  {N[dm.node].id:9s} {K[dm.k].id:9s} {pooled(d):+.4f}   quarters " + " ".join(f"{b:+.4f}" for b in by) + f"   {ag:8.1f} / {orc:8.1f}")
    for good in sorted({dm.k for dm in inst.demands}):
        idx = [do for do, dm in enumerate(inst.demands) if dm.k == good]
        d = [a["short"][:, idx].sum() - o["short"][:, idx].sum() for a, o in zip(A, O)]
        print(f"  all sinks {K[good].id:9s} {pooled(d):+.4f}")
    print("\nholding by good (share of the gap):")
    for k, com in enumerate(K):
        idx = [s for s, sl in enumerate(S) if sl.k == k and sl.node not in inst.chokepoint_ordinal]
        h = np.array([S[s].holding for s in idx])
        d = [(a["stock"][:, idx] * h).sum() - (np.nan_to_num(o["stock"][:, idx]) * h).sum() for a, o in zip(A, O)]
        ag = pooled([(a["stock"][:, idx] * h).sum() for a in A]) * den / 1e9
        orc = pooled([(np.nan_to_num(o["stock"][:, idx]) * h).sum() for o in O]) * den / 1e9
        print(f"  {com.id:13s} {pooled(d):+.4f}   {ag:7.1f} / {orc:7.1f} bn")
    print("\nfuel at grids, weeks of full-rate burn, mean over episodes at weeks 1, 13, 26, 52, 78, 104 (agent | clairvoyant):")
    for go, g in enumerate(inst.grids):
        ga = N[g].grid
        for k in ga.fuels:
            s = inst.slot_index[(g, k)]
            burn = ga.shares[k] * ga.deliverable
            ag = [np.mean([a["stock"][t, s] for a in A]) / burn for t in (0, 12, 25, 51, 77, 103)]
            orc = [np.mean([o["stock"][t, s] for o in O]) / burn for t in (0, 12, 25, 51, 77, 103)]
            print(f"  {N[g].id:9s} {K[k].id:8s} " + " ".join(f"{x:6.1f}" for x in ag) + "  |  " + " ".join(f"{x:6.1f}" for x in orc))
    if B is not None:
        print(f"\n{name} minus {other}, share of the gap by category:")
        for ci, c in enumerate(COMPS):
            print(f"  {c:14s} {pooled([a['comps'][:, ci].sum() - b['comps'][:, ci].sum() for a, b in zip(A, B)]):+.4f}")
        print(f"  end credit     {pooled([float(b['salvage']) - float(a['salvage']) for a, b in zip(A, B)]):+.4f}")


if __name__ == "__main__":
    fire.Fire(main)

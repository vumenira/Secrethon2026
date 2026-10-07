"""Lots started per fab (share of nominal capacity), wafers on hand, chips made and delivered: agent against clairvoyant."""

import json
from pathlib import Path

import fire
import numpy as np


HERE = Path(__file__).resolve().parent
BUCKETS = ((0, 13), (13, 26), (26, 52), (52, 78), (78, 104))


def main(name="f_e10_w4", episodes=16, per_episode=False):
    from shockbench_flow.hosting.tasks import task_generator

    inst, _ = task_generator("full")
    N, K, S = inst.nodes, inst.commodities, inst.stock_slots
    A = [np.load(HERE / "gap" / f"{name}_{n}.npz", allow_pickle=True) for n in range(episodes)]
    O = [np.load(HERE / "gap" / f"oracle_{n}.npz", allow_pickle=True) for n in range(episodes)]
    print("lots started as a share of nominal capacity, mean over episodes, weeks 1-13, 14-26, 27-52, 53-78, 79-104 (agent | clairvoyant); wafers on hand at the fab in weeks of capacity (agent | clairvoyant)")
    for fo, f in enumerate(inst.fabs):
        fa = N[f].fab
        ag = [np.mean([a["lots_started"][lo:hi, fo].mean() for a in A]) / fa.cap0 for lo, hi in BUCKETS]
        orc = [np.mean([o["lots_started"][lo:hi, fo].mean() for o in O]) / fa.cap0 for lo, hi in BUCKETS]
        s = inst.slot_index[(f, fa.input)]
        wa = [np.mean([a["stock"][lo:hi, s].mean() for a in A]) / fa.cap0 for lo, hi in BUCKETS]
        wo = [np.mean([np.nan_to_num(o["stock"][lo:hi, s]).mean() for o in O]) / fa.cap0 for lo, hi in BUCKETS]
        print(f"  {N[f].id:18s} {N[fa.grid].id:9s} " + " ".join(f"{x:5.2f}" for x in ag) + "  |  " + " ".join(f"{x:5.2f}" for x in orc)
              + "    wafers " + " ".join(f"{x:5.2f}" for x in wa) + "  |  " + " ".join(f"{x:5.2f}" for x in wo))
    for cls, good in (("leading+memory", "chip_le_raw"), ("mature", "chip_mat_raw")):
        k = inst.commodity_index[good]
        idx = [fo for fo, f in enumerate(inst.fabs) if N[f].fab.product == k]
        cap = sum(N[inst.fabs[fo]].fab.cap0 for fo in idx)
        ag = [np.mean([a["lots_started"][lo:hi][:, idx].sum(axis=1).mean() for a in A]) / cap for lo, hi in BUCKETS]
        orc = [np.mean([o["lots_started"][lo:hi][:, idx].sum(axis=1).mean() for o in O]) / cap for lo, hi in BUCKETS]
        print(f"  all {cls:15s} cap {cap:9.0f}/wk  " + " ".join(f"{x:5.3f}" for x in ag) + "  |  " + " ".join(f"{x:5.3f}" for x in orc))
    print("\ndemand served as a share of demand, by good:")
    for good in ("chip_le", "chip_mat"):
        k = inst.commodity_index[good]
        idx = [do for do, d in enumerate(inst.demands) if d.k == k]
        ag = [np.mean([a["served"][lo:hi][:, idx].sum() / a["demand"][lo:hi][:, idx].sum() for a in A]) for lo, hi in BUCKETS]
        orc = [np.mean([o["served"][lo:hi][:, idx].sum() / a["demand"][lo:hi][:, idx].sum() for a, o in zip(A, O)]) for lo, hi in BUCKETS]
        dem = np.mean([a["demand"][:, idx].sum(axis=1).mean() for a in A])
        print(f"  {good:9s} demand {dem:9.0f}/wk  " + " ".join(f"{x:5.3f}" for x in ag) + "  |  " + " ".join(f"{x:5.3f}" for x in orc))
    print("\nstock by good and node type at weeks 13, 26, 52, 78, 104 in thousands (agent | clairvoyant):")
    for k, com in enumerate(K):
        if k < 3:
            continue
        for typ in ("material", "fab", "osat", "sink", "terminal"):
            idx = [s for s, sl in enumerate(S) if sl.k == k and N[sl.node].type == typ]
            if not idx:
                continue
            ag = [np.mean([a["stock"][t, idx].sum() for a in A]) / 1e3 for t in (12, 25, 51, 77, 103)]
            orc = [np.mean([np.nan_to_num(o["stock"][t, idx]).sum() for o in O]) / 1e3 for t in (12, 25, 51, 77, 103)]
            if max(ag + orc) > 0.5:
                print(f"  {com.id:13s} {typ:9s} " + " ".join(f"{x:8.0f}" for x in ag) + "  |  " + " ".join(f"{x:8.0f}" for x in orc))
    print("\ndisposal by good, thousands of units per episode (agent | clairvoyant):")
    for k, com in enumerate(K):
        idx = [s for s, sl in enumerate(S) if sl.k == k]
        print(f"  {com.id:13s} {np.mean([a['disposal'][:, idx].sum() for a in A]) / 1e3:10.0f} | {np.mean([o['disposal'][:, idx].sum() for o in O]) / 1e3:10.0f}")
    if per_episode:
        refs = json.loads(Path(r"C:\Users\mykha\Claude code projects\Secrethon2026\outputs\recovery_tune\full-e12345-n16\references.json").read_text())
        print("\nper episode: level, score, shortage gap share, shed gap share")
        for n in range(episodes):
            r = refs[n]
            gap = (r["J_naive_cents"] - r["J_oracle_cents"]) / 100
            a, o = A[n], O[n]
            print(f"  {n:2d} L{r['stratum']} score {(r['J_naive_cents'] / 100 - float(int(a['J_cents'])) / 100) / gap:.3f}"
                  f" shortage {(a['comps'][:, 5].sum() - o['comps'][:, 5].sum()) / gap:+.3f} shed {(a['comps'][:, 7].sum() - o['comps'][:, 7].sum()) / gap:+.3f}"
                  f" other {(np.delete(a['comps'], [5, 7], axis=1).sum() - np.delete(o['comps'], [5, 7], axis=1).sum()) / gap:+.3f}")


if __name__ == "__main__":
    fire.Fire(main)

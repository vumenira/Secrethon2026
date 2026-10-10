"""A variant's recorded play against v12's of the same episode: costs, grids, straits (no play, no solve).

uv run python outputs/w13a/two.py --name=a13_s8 --ep="12345:7"
"""
import pickle
import sys
from pathlib import Path

import fire
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cmp as C

HERE = Path(__file__).resolve().parent


def q(a, step=13, fmt="{:5.2f}"):
    return " ".join(fmt.format(a[i:i + step].mean()) for i in range(0, len(a), step))


def main(name, ep, weekly=None):
    from shockbench_flow.hosting.tasks import task_generator

    inst, _ = task_generator("full")
    N, K = inst.nodes, inst.commodities
    kid = [c.id for c in K]
    root, n = (int(v) for v in str(ep).split(":"))
    a = C.ours(root, n)
    with open(HERE / "data" / f"full-e{root}" / f"{name}_ep{n}.pkl", "rb") as f:
        b = pickle.load(f)
    T = len(a["costs"])
    print(f"== {root}/{n}: v12 {a['J_cents'] / 1e11:,.1f} bn, {name} {b['J_cents'] / 1e11:,.1f} bn: {name} saves {(a['J_cents'] - b['J_cents']) / 1e11:+,.1f} bn; errors {b['errors'][:1]}")
    print("  cost parts, bn (v12 | variant | saved): " + "; ".join(
        f"{c} {sum(w[c] for w in a['costs']) / 1e9:,.1f} | {sum(w[c] for w in b['costs']) / 1e9:,.1f} | {(sum(w[c] for w in a['costs']) - sum(w[c] for w in b['costs'])) / 1e9:+,.1f}"
        for c in a["costs"][0]) + f"; salvage {a['salvage'] / 1e9:,.1f} | {b['salvage'] / 1e9:,.1f}")
    saved = np.array([sum(w.values()) for w in a["costs"]]) - np.array([sum(w.values()) for w in b["costs"]])
    print("  saved by 8-week block, bn: " + " ".join(f"{saved[i:i + 8].sum() / 1e9:+6.1f}" for i in range(0, T, 8)))
    for go, g in enumerate(inst.grids):
        ga = N[g].grid
        fabs = [fo for fo in inst.grid_fabs[go] if N[inst.fabs[fo]].fab.e > 0]
        draw = sum(N[inst.fabs[fo]].fab.e * N[inst.fabs[fo]].fab.cap0 for fo in fabs)
        sa, sb = a["shed"][:, go] * ga.voll / 1e9, b["shed"][:, go] * ga.voll / 1e9
        line = f"  {N[g].id:9s} shed {sa.sum():7,.1f} | {sb.sum():7,.1f} ({sa.sum() - sb.sum():+6.1f})"
        if fabs:
            Ea, Eb = a["energy"][:, fabs].sum(axis=1) / draw, b["energy"][:, fabs].sum(axis=1) / draw
            line += f"  fab energy {Ea.mean():.3f} | {Eb.mean():.3f}  [{q(Ea)} | {q(Eb)}]"
        print(line)
        for k in ga.fuels:
            full = ga.shares[k] * ga.deliverable
            xa = np.array([w.get((go, k), 0.0) for w in a["segment"]]) / full
            xb = np.array([w.get((go, k), 0.0) for w in b["segment"]]) / full
            if abs(xa.mean() - xb.mean()) > 0.002:
                print(f"      {kid[k]:8s} segment {xa.mean():.3f} | {xb.mean():.3f}  [{q(xa)} | {q(xb)}]")
    print("  lost sales saved by product, bn:", {kid[k]: round(sum(((a["lost"][:, do] - b["lost"][:, do]).sum()) * dm.pi for do, dm in enumerate(inst.demands) if dm.k == k) / 1e9, 1) for k in sorted({dm.k for dm in inst.demands})})
    print("  fuel waiting at straits by 8-week block (v12 | variant):")
    for k in (0, 1):
        for c in inst.chokepoints:
            s = inst.slot_index.get((c, k))
            if s is None or (a["stock"][:, s].max() < 1 and b["stock"][:, s].max() < 1):
                continue
            print(f"      {N[c].id:12s} {kid[k]:5s} {q(a['stock'][:, s], 8, '{:6.0f}')}\n      {'':18s} {q(b['stock'][:, s], 8, '{:6.0f}')}")
    if weekly:
        g = next(i for i, nd in enumerate(N) if nd.id == weekly)
        go = inst.grid_ordinal[g]
        fabs = [fo for fo in inst.grid_fabs[go] if N[inst.fabs[fo]].fab.e > 0]
        draw = sum(N[inst.fabs[fo]].fab.e * N[inst.fabs[fo]].fab.cap0 for fo in fabs)
        ga = N[g].grid
        print(f"  {weekly} week by week: fab energy share and each fuel's segment share (v12 | variant)")
        for t in range(T):
            ea, eb = a["energy"][t, fabs].sum() / draw, b["energy"][t, fabs].sum() / draw
            fa = " ".join(f"{a['segment'][t].get((go, k), 0.0) / (ga.shares[k] * ga.deliverable):4.2f}" for k in ga.fuels)
            fb = " ".join(f"{b['segment'][t].get((go, k), 0.0) / (ga.shares[k] * ga.deliverable):4.2f}" for k in ga.fuels)
            print(f"   {t + 1:3d}  {ea:4.2f} {fa} | {eb:4.2f} {fb}" + ("   <" if abs(ea - eb) > 0.05 else ""))


if __name__ == "__main__":
    fire.Fire(main)

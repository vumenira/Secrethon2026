"""Our play against a clairvoyant plan (bf: the base-first hull bound; f1, f2: with our burn or calendar kept).

uv run python outputs/w13a/cmp.py --eps="2026:3" --plan=bf
"""
import json
import pickle
from pathlib import Path

import fire
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SETS = {12345: 16, 2026: 16, 777: 32}


def refs(root):
    return {r["episode"]: r for r in json.loads((ROOT / "outputs" / "recovery_tune" / f"full-e{root}-n{SETS[root]}" / "references.json").read_text())}


def ours(root, n):
    with open(ROOT / "outputs" / "s13" / "s13_census" / f"full-e{root}" / f"ep{n}.pkl", "rb") as f:
        return pickle.load(f)


def plan(root, n, name="bf"):
    folder = HERE / "data" / f"full-e{root}"
    p = folder / f"ep{n}_{name}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    with open(folder / "columns.pkl", "rb") as f:
        columns = pickle.load(f)
    return {"x": z["x"], "J": float(z["J"]), "col": {key: j for j, key in enumerate(columns)}, "columns": columns}


def q(a, step=26, fmt="{:5.2f}"):
    return " ".join(fmt.format(a[i:i + step].mean()) for i in range(0, len(a), step))


def main(eps, plan_name="bf", detail=True):
    from shockbench_flow.hosting.tasks import task_generator

    inst, _ = task_generator("full")
    N, K = inst.nodes, inst.commodities
    kid = [c.id for c in K]
    for item in str(eps).split(","):
        root, n = (int(v) for v in item.split(":"))
        d, ref, P = ours(root, n), refs(root)[n], plan(root, n, plan_name)
        if P is None:
            print(f"root {root} episode {n}: no {plan_name} plan yet")
            continue
        x, col = P["x"], P["col"]
        J, Jn, Jo = d["J_cents"] / 1e11, ref["J_naive_cents"] / 1e11, ref["J_oracle_cents"] / 1e11
        T = x.shape[0]
        print(f"\n===== root {root} episode {n} level {ref['stratum']}: ours {J:,.0f}, {plan_name} {P['J'] / 1e9:,.0f}, clairvoyant {Jo:,.0f}, naive {Jn:,.0f} bn;"
              f" shortfall {J - Jo:,.0f} = closed {P['J'] / 1e9 - Jo:,.0f} + open {J - P['J'] / 1e9:,.0f}")
        o_short = sum(w["shortage"] for w in d["costs"]) / 1e9
        o_shed = sum(w["shed"] for w in d["costs"]) / 1e9
        p_short = sum(float(x[:, col[("U", do)]].sum()) * dm.pi for do, dm in enumerate(inst.demands) if ("U", do) in col) / 1e9
        p_shed = sum(float(x[:, col[("ysh", go)]].sum()) * N[g].grid.voll for go, g in enumerate(inst.grids)) / 1e9
        print(f"  shortage ours {o_short:,.0f} plan {p_short:,.0f} ({o_short - p_short:+,.0f}); shed ours {o_shed:,.0f} plan {p_shed:,.0f} ({o_shed - p_shed:+,.0f});"
              f" other {J - o_short - o_shed:,.0f} / {P['J'] / 1e9 - p_short - p_shed:,.0f}")
        if not detail:
            continue
        print("  per grid: (ours | plan) by half-year")
        for go, g in enumerate(inst.grids):
            ga = N[g].grid
            fabs = [fo for fo in inst.grid_fabs[go] if N[inst.fabs[fo]].fab.e > 0]
            draw = sum(N[inst.fabs[fo]].fab.e * N[inst.fabs[fo]].fab.cap0 for fo in fabs)
            shed_o, shed_p = d["shed"][:, go] * ga.voll / 1e9, x[:, col[("ysh", go)]] * ga.voll / 1e9
            line = f"  {N[g].id:9s} shed bn {shed_o.sum():6,.0f} | {shed_p.sum():6,.0f}"
            if fabs:
                Eo = d["energy"][:, fabs].sum(axis=1) / draw
                Ep = sum(x[:, col[("E", fo)]] for fo in fabs) / draw
                line += f"   fab energy {Eo.mean():.2f} | {Ep.mean():.2f}   [{q(Eo)} | {q(Ep)}]"
            print(line)
            for k in ga.fuels:
                full = ga.shares[k] * ga.deliverable
                so = np.array([w.get((go, k), 0.0) for w in d["segment"]]) / full
                sp = x[:, col[("G", go, k)]] / full
                print(f"      {kid[k]:8s} {full:6.0f}/wk  segment {so.mean():.2f} | {sp.mean():.2f}   [{q(so)} | {q(sp)}]")
            for fo in fabs:
                fa = N[inst.fabs[fo]].fab
                lo, lp = d["lots"][:, fo] / fa.cap0, x[:, col[("p", fo)]] / fa.cap0
                print(f"      {N[inst.fabs[fo]].id:18s} lots {lo.mean():.2f} | {lp.mean():.2f}   [{q(lo)} | {q(lp)}]")
        print("  lost sales bn (ours | plan):")
        for do, dm in enumerate(inst.demands):
            lo = d["lost"][:, do] * dm.pi / 1e9
            lp = x[:, col[("U", do)]] * dm.pi / 1e9
            print(f"      {N[dm.node].id:9s} {kid[dm.k]:9s} {lo.sum():6,.0f} | {lp.sum():6,.0f}  ({lo.sum() - lp.sum():+6,.0f})   [{q(lo, fmt='{:5.1f}')} | {q(lp, fmt='{:5.1f}')}]")


if __name__ == "__main__":
    fire.Fire(main)

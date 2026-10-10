"""Where a plan makes more chips than our play, grid by grid, and how: more of the scarcest fuel, or more of the
fabs' draw out of the same fuel (no play, no solve).

uv run python outputs/w13a/attr.py --eps="777:30,12345:12" --plan_name=b2
Per grid: the lots the plan starts beyond ours, valued at the product's shortage penalty (an upper estimate: a chip
is not always the thing that is short), split by
  fuel   the share of the scarcest fuel's full burn the plan burns beyond ours (the least segment share over the
         fuels that can stop the fabs), times the plan's fab energy per unit of that share;
  duty   the plan's fab energy per unit of that share beyond ours, at our share;
  mix    the rest: the same fab energy on other fabs of the grid, or lots that energy did not limit (wafers).
"""
import sys
from pathlib import Path

import fire
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cmp as C


def main(eps, plan_name="b2", quiet=False):
    from shockbench_flow.hosting.tasks import task_generator

    inst, _ = task_generator("full")
    N, K = inst.nodes, inst.commodities
    packaged = {}
    for o in inst.osats:
        packaged.update(N[o].osat.packages)
    pi = {}
    for d in inst.demands:
        pi[d.k] = max(pi.get(d.k, 0.0), d.pi)
    total = {}
    for item in str(eps).split(","):
        root, n = (int(v) for v in item.split(":"))
        d, P, ref = C.ours(root, n), C.plan(root, n, plan_name), C.refs(root)[n]
        if P is None:
            continue
        x, col = P["x"], P["col"]
        J = d["J_cents"] / 1e11
        o_short = sum(w["shortage"] for w in d["costs"]) / 1e9
        p_short = sum(float(x[:, col[("U", do)]].sum()) * dm.pi for do, dm in enumerate(inst.demands)) / 1e9
        o_shed = sum(w["shed"] for w in d["costs"]) / 1e9
        p_shed = sum(float(x[:, col[("ysh", go)]].sum()) * N[g].grid.voll for go, g in enumerate(inst.grids)) / 1e9
        if not quiet:
            print(f"\n== {root}/{n} L{ref['stratum']}: open {J - P['J'] / 1e9:,.0f} bn = shortage {o_short - p_short:+,.0f}, shed {o_shed - p_shed:+,.0f}, other {J - P['J'] / 1e9 - (o_short - p_short) - (o_shed - p_shed):+,.0f}")
        for go, g in enumerate(inst.grids):
            ga = N[g].grid
            fabs = [fo for fo in inst.grid_fabs[go] if N[inst.fabs[fo]].fab.e > 0]
            shed = (float(d["shed"][:, go].sum()) - float(x[:, col[("ysh", go)]].sum())) * ga.voll / 1e9
            key = N[g].id
            acc = total.setdefault(key, np.zeros(5))
            acc[4] += shed
            if not fabs:
                if not quiet and abs(shed) > 5:
                    print(f"   {key:9s} shed {shed:+6.0f}")
                continue
            draw = sum(N[inst.fabs[fo]].fab.e * N[inst.fabs[fo]].fab.cap0 for fo in fabs)
            big = [k for k in ga.fuels if ga.shares[k] * ga.deliverable >= draw]
            T = x.shape[0]
            Fo = min(sum(w.get((go, k), 0.0) for w in d["segment"]) / (ga.shares[k] * ga.deliverable * T) for k in big)
            Fp = min(float(x[:, col[("G", go, k)]].sum()) / (ga.shares[k] * ga.deliverable * T) for k in big)
            Eo = float(d["energy"][:, fabs].sum()) / (draw * T)
            Ep = float(sum(x[:, col[("E", fo)]].sum() for fo in fabs)) / (draw * T)
            value = 0.0  # the plan's lots beyond ours at the penalty of their chip
            per_energy = 0.0  # value of the grid's fabs per unit of draw share, all fabs at capacity for the episode
            for fo in fabs:
                fa = N[inst.fabs[fo]].fab
                worth = pi[packaged.get(fa.product, fa.product)]
                value += (float(x[:, col[("p", fo)]].sum()) - float(d["lots"][:, fo].sum())) * worth / 1e9
                per_energy += fa.cap0 * T * worth / 1e9
            eta_o, eta_p = (Eo / Fo if Fo > 1e-9 else 0.0), (Ep / Fp if Fp > 1e-9 else 0.0)
            fuel = (Fp - Fo) * eta_p * per_energy
            duty = Fo * (eta_p - eta_o) * per_energy
            mix = value - fuel - duty
            acc[:4] += (value, fuel, duty, mix)
            if not quiet:
                print(f"   {key:9s} lots {value:+6.0f} bn = fuel {fuel:+6.0f} (scarcest fuel {Fo:.2f} | {Fp:.2f}) + duty {duty:+6.0f} (fab energy {Eo:.2f} | {Ep:.2f}) + mix {mix:+6.0f};  shed {shed:+6.0f}")
    print("\nTOTAL over the episodes listed (bn): lots, of which fuel, duty, mix; shed")
    for key, a in total.items():
        print(f"   {key:9s} {a[0]:+7.0f} = {a[1]:+7.0f} {a[2]:+7.0f} {a[3]:+7.0f};  shed {a[4]:+7.0f}")
    s = sum(total.values())
    print(f"   {'all':9s} {s[0]:+7.0f} = {s[1]:+7.0f} {s[2]:+7.0f} {s[3]:+7.0f};  shed {s[4]:+7.0f}")


if __name__ == "__main__":
    fire.Fire(main)

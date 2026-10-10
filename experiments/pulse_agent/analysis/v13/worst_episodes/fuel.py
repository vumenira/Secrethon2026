"""Fuel dispatches of our play, slot by slot, against what the routes could carry (no play, no solve).

uv run python outputs/w13a/fuel.py --eps="777:26" --fuels="lng,crude"
Per slot: executed per week by quarter; the lane's capacity (the least over its edges of the week's u, 0 where the
fuel is prohibited on an edge that week; strait throughput not counted) by quarter.
"""
import pickle
from pathlib import Path

import fire
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def marks_of(root, n):
    p = HERE / "data" / f"full-e{root}" / f"ep{n}_marks.npz"
    if p.exists():
        return dict(np.load(p))
    from shockbench_flow.evaluation.cache import default_cache_dir
    from shockbench_flow.policies.naive_fq import REPLICATIONS
    from shockbench_flow_agent.scoring import _world

    inst, omega, marks, fallback = _world("full", root, n, REPLICATIONS, str(default_cache_dir()))
    fields = ("u", "c", "o", "kappa", "supply", "G_bar", "y_bar", "R", "alpha_bar", "sigma_scr", "R_osat", "demand",
              "prohibited", "tariff", "u_now", "o_now", "kappa_now", "supply_now", "G_bar_now", "y_bar_now", "R_now")
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **{f: np.asarray(getattr(marks, f)) for f in fields})
    return dict(np.load(p))


def q(a, step=13):
    return " ".join(f"{a[i:i + step].mean():6.0f}" for i in range(0, len(a), step))


def main(eps, fuels="lng,crude", last=69):
    from shockbench_flow.hosting.tasks import task_generator

    inst, _ = task_generator("full")
    N, E, K = inst.nodes, inst.edges, inst.commodities
    kid = [c.id for c in K]
    want = set(str(fuels).split(","))
    for item in str(eps).split(","):
        root, n = (int(v) for v in item.split(":"))
        with open(ROOT / "outputs" / "s13" / "s13_census" / f"full-e{root}" / f"ep{n}.pkl", "rb") as f:
            d = pickle.load(f)
        m = marks_of(root, n)
        T = len(d["executed"])
        print(f"\n===== root {root} episode {n}: straits open by quarter (hormuz malacca suez cape taiwan panama turkish)")
        for c in range(m["o"].shape[1]):
            print(f"   {N[inst.chokepoints[c]].id:12s} open {' '.join(f'{m['o'][i:i + 13, c].mean():4.2f}' for i in range(0, T, 13))}"
                  f"  kappa tb {q(m['kappa'][:, c, 0])}")
        src = None
        for s, (e, k, lane) in enumerate(inst.action_slots):
            if kid[k] not in want or s > last:
                continue
            if E[e].tail != src:
                src = E[e].tail
                slot = inst.slot_index.get((src, k))
                if src in inst.supply_nodes and slot is not None:
                    print(f" {N[src].id} {kid[k]}: availability {q(m['supply'][:, slot])}  stock {q(d['stock'][:, slot])}")
                else:
                    print(f" {N[src].id}")
            edges = inst.lanes[lane].edges if lane is not None else (e,)
            cap = np.min([np.where(m["prohibited"][:, x, k], 0.0, m["u"][:, x]) for x in edges], axis=0)
            banned = np.any([m["prohibited"][:, x, k] for x in edges], axis=0)
            done = np.array([w.get(s, 0.0) for w in d["executed"]])
            name = inst.lanes[lane].id if lane is not None else E[e].id
            print(f"   {s:3d} {name:38s} {kid[k]:5s} sent {q(done)} | cap {q(cap)} | banned weeks {int(banned.sum()):3d}")


if __name__ == "__main__":
    fire.Fire(main)

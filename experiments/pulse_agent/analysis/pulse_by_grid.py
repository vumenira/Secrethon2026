"""Per grid: what the pulse costs in base load shed and what its fabs gain in lots (priced at the shortage penalty),
pooled over the episodes as the score pools them, as shares of the naive-to-clairvoyant gap."""

import json
from pathlib import Path

import fire
import numpy as np


HERE = Path(__file__).resolve().parent
REFS = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026\outputs\recovery_tune\full-e12345-n16\references.json")
W = {1: 0.50, 2: 0.30, 3: 0.15, 4: 0.05}


def main(a="ft_pulse_r4", b="f_e10_w4", episodes=16, per_episode=False):
    from shockbench_flow.hosting.tasks import task_generator

    inst, _ = task_generator("full")
    N, K = inst.nodes, inst.commodities
    refs = json.loads(REFS.read_text())[:episodes]
    strata = np.array([r["stratum"] for r in refs])
    w = np.array([W[s] / (strata == s).sum() for s in strata])
    gap = np.array([r["J_naive_cents"] - r["J_oracle_cents"] for r in refs], dtype=float) / 100
    den = float((w * gap).sum())
    A = [np.load(HERE / "gap" / f"{a}_{n}.npz", allow_pickle=True) for n in range(episodes)]
    B = [np.load(HERE / "gap" / f"{b}_{n}.npz", allow_pickle=True) for n in range(episodes)]
    worth = {}
    for fo, f in enumerate(inst.fabs):
        fa = N[f].fab
        goods = {fa.product} | {pk for o in inst.osats for raw, pk in N[o].osat.packages.items() if raw == fa.product}
        worth[fo] = max(d.pi for d in inst.demands if d.k in goods)
    print(f"{a} minus {b}: share of the gap; shed (cost), lots at the shortage penalty (an upper bound of the gain), episodes where shed rose by more than the lots are worth")
    tot_s = tot_l = 0.0
    for go, g in enumerate(inst.grids):
        fabs = inst.grid_fabs[go]
        ds = np.array([x["shed"][:, go].sum() - y["shed"][:, go].sum() for x, y in zip(A, B)])
        dl = np.array([sum((x["lots_started"][:, fo].sum() - y["lots_started"][:, fo].sum()) * worth[fo] for fo in fabs) for x, y in zip(A, B)])
        bad = [n for n in range(episodes) if ds[n] > dl[n] + 1e9]
        tot_s += float((w * ds).sum()) / den
        tot_l += float((w * dl).sum()) / den
        print(f"  {N[g].id:9s} shed {float((w * ds).sum()) / den:+.4f}  lots {float((w * dl).sum()) / den:+.4f}  losing episodes {bad}")
        if per_episode:
            print("     per episode (bn): " + " ".join(f"{n}:{ds[n] / 1e9:+.0f}/{dl[n] / 1e9:+.0f}" for n in range(episodes)))
    print(f"  all       shed {tot_s:+.4f}  lots {tot_l:+.4f}")
    dj = np.array([float(int(y["J_cents"])) - float(int(x["J_cents"])) for x, y in zip(A, B)]) / 100
    print(f"  score difference {float((w * dj).sum()) / den:+.4f}; unserved demand {float((w * np.array([x['short'].sum() - y['short'].sum() for x, y in zip(A, B)])).sum()) / den:+.4f}")


if __name__ == "__main__":
    fire.Fire(main)

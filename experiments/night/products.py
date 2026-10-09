"""The chip gap by product and by period: demand not served, ours against the clairvoyant plan's, from gap.py's files.

uv run python experiments/night/products.py --agent=mpc_pulse_rec_v9 --task=small --entropy=12345
"""

import json
from pathlib import Path

import fire
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def main(agent="mpc_pulse_rec_v9", task="small", entropy=12345):
    from shockbench_flow.hosting.tasks import task_generator

    inst, _ = task_generator(task)
    out = ROOT / "outputs" / "gap" / f"{task}-e{entropy}" / agent
    eps = [json.loads(p.read_text()) for p in sorted(out.glob("*.json"), key=lambda p: int(p.stem))]
    pi = np.array([d.pi for d in inst.demands])
    ks = sorted({d.k for d in inst.demands})
    T = len(eps[0]["ours"]["lost"])
    lo = np.array([e["ours"]["lost"] for e in eps])  # (n, T, D)
    lc = np.array([e["oracle"]["lost"] for e in eps])
    dem = np.array([e["ours"]["demand"] for e in eps])
    print(f"{task} root {entropy}, {len(eps)} episodes: chips not served, ours against the clairvoyant plan's")
    for k in ks:
        idx = [i for i, d in enumerate(inst.demands) if d.k == k]
        o, c, d = lo[:, :, idx].sum(), lc[:, :, idx].sum(), dem[:, :, idx].sum()
        usd = ((lo[:, :, idx] - lc[:, :, idx]) * pi[idx]).sum()
        print(
            f"  product {k} (price {pi[idx].mean():,.0f}): demand {d / 1e6:,.1f}M, lost ours {o / 1e6:,.1f}M"
            f" ({o / d:.3f}), clairvoyant {c / 1e6:,.1f}M ({c / d:.3f}); gap {usd / 1e9:+,.0f} bn USD"
        )
    print("gap by 4-week period and product (bn USD)")
    for a in range(0, T, 4):
        row = []
        for k in ks:
            idx = [i for i, d in enumerate(inst.demands) if d.k == k]
            row.append(((lo[:, a : a + 4, idx] - lc[:, a : a + 4, idx]) * pi[idx]).sum() / 1e9)
        print(f"  weeks {a + 1:3d}-{min(T, a + 4):3d}: " + "  ".join(f"k{k} {v:+7,.0f}" for k, v in zip(ks, row)))
    print("gap by market (bn USD)")
    for i, d in enumerate(inst.demands):
        g = ((lo[:, :, i] - lc[:, :, i]) * pi[i]).sum() / 1e9
        print(f"  demand {i} (node {d.node}, product {d.k}): {g:+,.0f}")


if __name__ == "__main__":
    fire.Fire(main)

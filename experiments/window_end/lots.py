"""Why our fabs start fewer lots than the clairvoyant plan's: each fab-week's binding limit (capacity, wafers on hand,
energy), from the episodes gap.py saved.

uv run python experiments/window_end/lots.py --agent=mpc_pulse_rec_v8_on --task=small --entropy=12345
"""

import json
from pathlib import Path

import fire
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def main(agent="mpc_pulse_rec_v8_on", task="small", entropy=12345):
    from shockbench_flow.hosting.tasks import task_generator

    inst, _ = task_generator(task)
    fabs = [inst.nodes[f].fab for f in inst.fabs]
    cap0 = np.array([fa.cap0 for fa in fabs])
    e = np.array([fa.e for fa in fabs])
    grid = [inst.grids.index(fa.grid) for fa in fabs]
    out = ROOT / "outputs" / "gap" / f"{task}-e{entropy}" / agent
    eps = [json.loads(p.read_text()) for p in sorted(out.glob("*.json"), key=lambda p: int(p.stem))]
    F = len(fabs)
    causes = ("capacity", "wafers", "energy")
    lost = {c: np.zeros(F) for c in causes}  # lots under capacity, by the binding limit
    weeks = {c: np.zeros(F) for c in causes}
    o_lost = np.zeros(F)
    energy_ours, energy_orc, lots_ours, lots_orc = np.zeros(F), np.zeros(F), np.zeros(F), np.zeros(F)
    zero_energy_weeks, zero_energy_full_wafers = np.zeros(F), np.zeros(F)
    for ep in eps:
        cap = np.array(ep["cap"]) * cap0  # (T, F)
        L = np.array(ep["ours"]["lots"])
        E = np.array(ep["ours"]["energy"])
        W = np.array(ep["ours"]["wafers"]) + L  # wafers on hand when lots start
        Lo = np.array(ep["oracle"]["lots"])
        Eo = np.array(ep["oracle"]["energy"])
        energy_ours += E.sum(axis=0)
        energy_orc += Eo.sum(axis=0)
        lots_ours += L.sum(axis=0)
        lots_orc += Lo.sum(axis=0)
        o_lost += np.maximum(cap - Lo, 0).sum(axis=0)
        limits = np.stack([cap, W, E / e], axis=-1)  # (T, F, 3)
        which = limits.argmin(axis=-1)
        gap = np.maximum(cap - L, 0)
        for i, c in enumerate(causes):
            m = which == i
            lost[c] += (gap * m).sum(axis=0)
            weeks[c] += (m & (gap > 0.01 * cap)).sum(axis=0)
        z = E < 0.01 * e * cap
        zero_energy_weeks += z.sum(axis=0)
        zero_energy_full_wafers += (z & (W >= 0.99 * cap)).sum(axis=0)
    n = len(eps)
    print(f"{task} root {entropy}, {n} episodes; per fab (M lots unless marked)")
    heads = ("fab", "grid", "ours", "clairv", "E ours/clairv", "under cap: capacity", "wafers", "energy")
    heads += ("clairv under cap", "zero-energy weeks/ep", "of them wafers full")
    widths = (3, 4, 7, 7, 13, 19, 7, 7, 16, 21, 19)
    print(" ".join(f"{h:>{w}s}" for h, w in zip(heads, widths)))
    for f in range(F):
        print(
            f"{f:3d} {grid[f]:4d} {lots_ours[f] / 1e6:7.1f} {lots_orc[f] / 1e6:7.1f}"
            f" {energy_ours[f] / max(energy_orc[f], 1e-9):13.3f} {lost['capacity'][f] / 1e6:19.1f}"
            f" {lost['wafers'][f] / 1e6:7.1f} {lost['energy'][f] / 1e6:7.1f} {o_lost[f] / 1e6:16.1f}"
            f" {zero_energy_weeks[f] / n:21.1f} {zero_energy_full_wafers[f] / n:19.1f}"
        )
    print("weeks (per episode) a fab ran under 99 % of capacity, by binding limit:")
    for c in causes:
        print(f"  {c:9s} " + " ".join(f"{weeks[c][f] / n:5.1f}" for f in range(F)))


if __name__ == "__main__":
    fire.Fire(main)

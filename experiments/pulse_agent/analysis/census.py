"""Events per episode on the 64 episodes of root 12345: active at week 1, onsets in the episode, by type."""

import json
from collections import Counter
from pathlib import Path

import fire
import numpy as np


ROOT = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026")


def main(task="small", entropy=12345, episodes=64):
    from shockbench_flow.evaluation.cache import default_cache_dir
    from shockbench_flow.omega import codes
    from shockbench_flow.policies.naive_fq import REPLICATIONS
    from shockbench_flow_agent.scoring import _world

    refs = json.loads((ROOT / "outputs" / "recovery_tune" / f"{task}-e{entropy}-n{episodes}" / "references.json").read_text())
    cache = str(default_cache_dir())
    carried, onsets, ends, short = Counter(), Counter(), Counter(), Counter()
    grid_short_eps, grid_short_weeks = 0, 0
    for r in refs:
        inst, omega, marks, _ = _world(task, entropy, r["episode"], REPLICATIONS, cache)
        a = omega.arrays
        ty, on, dur = np.asarray(a["ev_type"]), np.asarray(a["ev_onset"], dtype=float), np.asarray(a["ev_duration"], dtype=float)
        T = inst.T
        for t, s, d in zip(ty, on, dur):
            name = codes.EVENT_TYPES[int(t)]
            if s <= 0 < s + d:
                carried[name] += 1
            if 0 < s < T:
                onsets[name] += 1
                if d < 2:
                    short[name] += 1
            if 0 < s + d < T and s + d > 0:
                ends[name] += 1
        nominal = np.array([inst.nodes[g].grid.deliverable for g in inst.grids])
        base = np.array([inst.nodes[g].grid.base_load for g in inst.grids])
        G = np.asarray(marks.G_bar)
        below = (G < base[None, :] - 1e-9).any(axis=1)  # a grid cannot serve its base load even with all fuel
        grid_short_eps += bool(below.any())
        grid_short_weeks += int(below.sum())
    n = len(refs)
    print(f"{n} episodes of {T} weeks; per episode:")
    print(f"{'type':22s} {'active at week 1':>17s} {'onsets in episode':>18s} {'of them under 2 wk':>19s} {'ends in episode':>16s}")
    for name in codes.EVENT_TYPES:
        print(f"{name:22s} {carried[name] / n:17.2f} {onsets[name] / n:18.2f} {short[name] / n:19.2f} {ends[name] / n:16.2f}")
    print(f"episodes where some grid's deliverable power falls below its base load: {grid_short_eps} of {n}; such weeks per episode {grid_short_weeks / n:.1f}")


if __name__ == "__main__":
    fire.Fire(main)

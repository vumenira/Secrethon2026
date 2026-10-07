"""Offline checks of Window.warn: equal to recover at a lead of 0; close to the full truth at a lead of the window."""

import sys
from pathlib import Path

import numpy as np


ROOT = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026")
sys.path.insert(0, str(ROOT / "experiments" / "approaches"))
import foresight as F  # noqa: E402


def main():
    from shockbench_flow.evaluation.cache import default_cache_dir
    from shockbench_flow.policies.naive_fq import REPLICATIONS
    from shockbench_flow_agent.scoring import _world

    cache = str(default_cache_dir())
    names = [n for n in F.arrays_of("all") if n != "sigma_scr"]
    same0 = diff0 = cells = cells_diff = more = 0
    for episode in range(8):
        inst, omega, marks, _ = _world("small", 12345, episode, REPLICATIONS, cache)
        tr = F.truth_of(inst, omega, marks)
        w = F.Window(tr, {})
        for t in range(1, tr.T + 1):
            H = min(20, tr.T - t + 1)
            for name in names:
                held = np.broadcast_to(tr.instant[name][t - 1], (H, *tr.instant[name][t - 1].shape)).copy()
                a, b = held.copy(), held.copy()
                w.recover(name, a, t, H)
                w.warn(name, b, t, H, 0)
                if np.array_equal(a, b):
                    same0 += 1
                else:
                    diff0 += 1
                c = held.copy()
                w.warn(name, c, t, H, H)
                truth = tr.avg[name][t - 1 : t - 1 + H]
                cells += truth.size
                cells_diff += int((~F.same(c, truth)).sum())
                c2 = held.copy()
                w.warn(name, c2, t, H, min(2, H))
                more += int((~F.same(c2, a)).sum())  # cells a 2-week warning changes against recovery alone
    print(f"warn at lead 0 against recover: {same0} windows equal, {diff0} different")
    print(f"warn at lead H against the true window: {cells_diff} of {cells} cells differ ({cells_diff / cells:.2e})")
    print(f"cells a 2-week warning changes against recovery alone: {more}")


if __name__ == "__main__":
    main()

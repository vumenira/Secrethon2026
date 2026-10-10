"""War-risk weeks per strait over the 64 used Full episodes (marks only; no play, no solve)."""
import numpy as np
from shockbench_flow.evaluation.cache import default_cache_dir
from shockbench_flow.policies.naive_fq import REPLICATIONS
from shockbench_flow_agent.scoring import _world

NAMES = ["hormuz", "malacca", "suez", "cape", "taiwan", "panama", "turkish"]
tot = np.zeros((7, 3))
eps = np.zeros((7, 3))
for root, N in ((12345, 16), (2026, 16), (777, 32)):
    for n in range(N):
        inst, omega, marks, fallback = _world("full", root, n, REPLICATIONS, str(default_cache_dir()))
        wr = np.asarray(marks.wr_class)
        line = []
        for c in range(7):
            for cls in (1, 2):
                w = int((wr[:, c] == cls).sum())
                tot[c, cls] += w
                eps[c, cls] += w > 0
                if w:
                    first = int(np.argmax(wr[:, c] == cls)) + 1
                    line.append(f"{NAMES[c]} class {cls}: {w} weeks from {first} (open {float(np.asarray(marks.o)[wr[:, c] == cls, c].mean()):.2f})")
        print(root, n, "; ".join(line), flush=True)
print("weeks by strait and class (1 red_sea, 2 hormuz):")
for c in range(7):
    print(f"  {NAMES[c]:8s} class1 {tot[c, 1]:5.0f} weeks in {eps[c, 1]:.0f} episodes; class2 {tot[c, 2]:5.0f} weeks in {eps[c, 2]:.0f} episodes")

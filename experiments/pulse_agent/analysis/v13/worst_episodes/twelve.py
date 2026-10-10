"""Gains of cached variants on the twelve worst episodes and on the other 52 (bn; no play)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SETS = {12345: 16, 2026: 16, 777: 32}
TWELVE = [(777, 26), (777, 8), (777, 30), (12345, 12), (777, 27), (2026, 3), (777, 13), (777, 24), (12345, 0), (777, 15), (12345, 13), (12345, 11)]
names = sys.argv[1:] or ["a13_s8", "a13_s26", "a13_s8_src", "a13_fin"]
cost = {}
for root, N in SETS.items():
    folder = ROOT / "outputs" / "recovery_tune" / f"full-e{root}-n{N}"
    for name in ["v12_all"] + names:
        p = folder / f"{name}.json"
        if p.exists():
            for r in json.loads(p.read_text())["rows"]:
                cost[(name, root, r["episode"])] = r["J_policy_cents"] / 1e11
print("episode      " + " ".join(f"{n:>11s}" for n in names))
tot = {n: [0.0, 0.0, True, True] for n in names}
for root, N in SETS.items():
    for ep in range(N):
        gains = []
        for n in names:
            if (n, root, ep) in cost:
                g = cost[("v12_all", root, ep)] - cost[(n, root, ep)]
                tot[n][0 if (root, ep) in TWELVE else 1] += g
                gains.append(f"{g:+11.1f}")
            else:
                tot[n][2 if (root, ep) in TWELVE else 3] = False
                gains.append(f"{'-':>11s}")
        if (root, ep) in TWELVE:
            print(f"{root:6d}/{ep:<3d}   " + " ".join(gains))
print("twelve       " + " ".join(f"{tot[n][0]:+11.1f}{'' if tot[n][2] else '*'}" for n in names))
print("other 52     " + " ".join(f"{tot[n][1]:+11.1f}{'' if tot[n][3] else '*'}" for n in names) + "    (* not every episode played)")

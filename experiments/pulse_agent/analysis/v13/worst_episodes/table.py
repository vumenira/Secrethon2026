"""Per-episode score and weighted shortfall of v12 on the 64 used Full episodes (cached costs; no play)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SETS = {12345: 16, 2026: 16, 777: 32}
W = {1: 0.50, 2: 0.30, 3: 0.15, 4: 0.05}
rows = []
for root, N in SETS.items():
    folder = ROOT / "outputs" / "recovery_tune" / f"full-e{root}-n{N}"
    refs = json.loads((folder / "references.json").read_text())
    cost = {r["episode"]: r["J_policy_cents"] for r in json.loads((folder / "v12_all.json").read_text())["rows"]}
    for r in refs:
        rows.append((root, r["episode"], r["stratum"], r["J_naive_cents"] / 1e11, r["J_oracle_cents"] / 1e11, cost[r["episode"]] / 1e11))
count = {s: sum(1 for r in rows if r[2] == s) for s in W}
att = sum(W[s] / count[s] * (jn - jo) for _, _, s, jn, jo, _ in rows)
gap = sum(W[s] / count[s] * (j - jo) for _, _, s, jn, jo, j in rows)
print(f"64 episodes by level {count}; pooled score {1 - gap / att:.4f}; weighted attainable {att:.0f} bn; 0.001 = {att / 1000:.2f} bn weighted")
print("per-episode weight (bn of an episode per 0.001 of score):", {s: round(att / 1000 / (W[s] / count[s]), 1) for s in W})
rows.sort(key=lambda r: -W[r[2]] / count[r[2]] * (r[5] - r[4]))
for root, n, s, jn, jo, j in rows:
    w = W[s] / count[s]
    print(f"{root:6d} ep {n:2d} L{s} score {(jn - j) / (jn - jo):.3f} shortfall {j - jo:6.0f} bn  weighted share of pooled shortfall {w * (j - jo) / gap:.3f}  score points {w * (j - jo) / att:.4f}")

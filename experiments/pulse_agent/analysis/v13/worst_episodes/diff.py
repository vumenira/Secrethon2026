"""Per-episode cost difference of a cached play against another, with the pooled score change (no play).

uv run python outputs/w13a/diff.py --a=a13_s8 --b=v12_all --sets="full:12345:16,full:2026:16,full:777:32"
Positive bn: a is cheaper (better). The pooled change counts every set listed as one set, as the score pools it.
"""
import json
from pathlib import Path

import fire
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
W = {1: 0.50, 2: 0.30, 3: 0.15, 4: 0.05}


def main(a, b="v12_all", sets="full:12345:16", detail=True, boot=4000):
    rows = []
    for item in str(sets).split(","):
        task, root, N = item.split(":")
        folder = ROOT / "outputs" / "recovery_tune" / f"{task}-e{root}-n{N}"
        if not (folder / f"{a}.json").exists():
            print(f"{task} {root}: {a} not played")
            continue
        refs = json.loads((folder / "references.json").read_text())
        ca = {r["episode"]: r for r in json.loads((folder / f"{a}.json").read_text())["rows"]}
        cb = {r["episode"]: r for r in json.loads((folder / f"{b}.json").read_text())["rows"]}
        for r in refs:
            n = r["episode"]
            rows.append((f"{task}:{root}", n, r["stratum"], r["J_naive_cents"] / 1e11, r["J_oracle_cents"] / 1e11,
                         ca[n]["J_policy_cents"] / 1e11, cb[n]["J_policy_cents"] / 1e11, ca[n].get("fallback_weeks", 0), ca[n].get("first_error")))
    if not rows:
        return
    strata = np.array([r[2] for r in rows])
    jn, jo, ja, jb = (np.array([r[i] for r in rows]) for i in (3, 4, 5, 6))

    def score(idx, cost):
        num = den = 0.0
        for s, w in W.items():
            m = idx[strata[idx] == s]
            if len(m):
                num += w * np.mean(jn[m] - cost[m])
                den += w * np.mean(jn[m] - jo[m])
        return num / den

    idx = np.arange(len(rows))
    sa, sb = score(idx, ja), score(idx, jb)
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(boot):  # resample episodes within each harm level
        pick = np.concatenate([rng.choice(idx[strata == s], size=int((strata == s).sum())) for s in W if (strata == s).any()])
        draws.append(score(pick, ja) - score(pick, jb))
    lo, hi = np.percentile(draws, [5, 95])
    gain = jb - ja
    print(f"{a} against {b} on {sets}: {len(rows)} episodes; score {sb:.4f} -> {sa:.4f}, {sa - sb:+.4f} (90 % interval {lo:+.4f} to {hi:+.4f});"
          f" better {int((gain > 0.005).sum())} / worse {int((gain < -0.005).sum())} / equal {int((np.abs(gain) <= 0.005).sum())};"
          f" saved {gain.sum():+,.0f} bn, largest gain {gain.max():+,.1f}, largest loss {gain.min():+,.1f};"
          f" fallback weeks {sum(r[7] for r in rows)}, errors {[r[8] for r in rows if r[8]][:2]}")
    if detail:
        for r, g in sorted(zip(rows, gain), key=lambda p: -abs(p[1])):
            if abs(g) > 0.5:
                print(f"   {r[0]:11s} ep {r[1]:2d} L{r[2]}  {g:+8.1f} bn   (score {(r[3] - r[6]) / (r[3] - r[4]):.3f} -> {(r[3] - r[5]) / (r[3] - r[4]):.3f})")


if __name__ == "__main__":
    fire.Fire(main)

"""Cached plays of tune.py against a base, per set and over several sets as one: score, paired difference, interval.

uv run python experiments/pulse_agent/analysis/v13/pooled.py --task=full --sets=12345:16,2026:16,777:32 \
    --base=v12_all --names=f13_def,a13_s26

The score pools as the board's does: the sum over harm levels of weight times mean saving, over the sum of weight
times mean attainable (weights 0.50, 0.30, 0.15, 0.05), with the means taken over the pooled episodes of a level. The
interval is a paired bootstrap over episodes within levels (the 5th and 95th percentile of 4,000 draws).
"""
import json
from pathlib import Path

import fire
import numpy as np

OUT = Path(__file__).resolve().parents[4] / "outputs" / "recovery_tune"
W = {1: 0.50, 2: 0.30, 3: 0.15, 4: 0.05}


def rows(task, root, n, name):
    folder = OUT / f"{task}-e{root}-n{n}"
    refs = {r["episode"]: r for r in json.loads((folder / "references.json").read_text()) if not r.get("excluded")}
    play = {r["episode"]: r for r in json.loads((folder / f"{name}.json").read_text())["rows"]}
    return refs, play


def score(naive, clair, cost, level, idx):
    num = den = 0.0
    for lv, w in W.items():
        m = idx[level[idx] == lv]
        if len(m):
            num += w * (naive[m] - cost[m]).mean()
            den += w * (naive[m] - clair[m]).mean()
    return num / den


def table(naive, clair, base, new, level, draws, seed):
    idx = np.arange(len(naive))
    s0, s1 = score(naive, clair, base, level, idx), score(naive, clair, new, level, idx)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(level == lv) for lv in W if (level == lv).any()]
    d = []
    for _ in range(draws):
        pick = np.concatenate([rng.choice(g, len(g)) for g in groups])
        d.append(score(naive, clair, new, level, pick) - score(naive, clair, base, level, pick))
    lo, hi = np.quantile(d, [0.05, 0.95])
    saved = (base - new) / 1e11  # bn USD
    return s0, s1, lo, hi, int((saved > 1e-6).sum()), int((saved < -1e-6).sum()), float(saved.sum()), float(saved.min())


def main(task="full", sets="12345:16,2026:16,777:32", base="v12_all", names="", draws=4000, seed=0):
    sets = [tuple(int(x) for x in s.split(":")) for s in (sets if isinstance(sets, str) else ",".join(sets)).split(",")]
    names = [x for x in (names if isinstance(names, str) else ",".join(names)).split(",") if x]
    for name in names:
        print(name)
        pooled = [[], [], [], [], []]
        for root, n in sets:
            refs, a = rows(task, root, n, base)
            _refs, b = rows(task, root, n, name)
            eps = [e for e in sorted(refs) if e in a and e in b]
            cols = (
                [refs[e]["J_naive_cents"] for e in eps],
                [refs[e]["J_clairvoyant_cents"] for e in eps],
                [a[e]["J_policy_cents"] for e in eps],
                [b[e]["J_policy_cents"] for e in eps],
                [refs[e]["stratum"] for e in eps],
            )
            errors = [b[e]["first_error"] for e in eps if b[e].get("first_error")]
            for p, c in zip(pooled, cols):
                p.extend(c)
            arr = [np.array(c, dtype=float) for c in cols[:4]] + [np.array(cols[4])]
            s0, s1, lo, hi, up, down, saved, worst = table(*arr, draws, seed)
            print(
                f"  {task} {root} ({len(eps)}): {s0:.4f} -> {s1:.4f}  {s1 - s0:+.4f} ({lo:+.4f} to {hi:+.4f})"
                f"  {up} better / {down} worse  saved {saved:.0f} bn, largest loss {min(worst, 0):.1f} bn"
                + (f"  ERRORS {errors[:2]}" if errors else "")
            )
        if len(sets) > 1:
            arr = [np.array(c, dtype=float) for c in pooled[:4]] + [np.array(pooled[4])]
            s0, s1, lo, hi, up, down, saved, worst = table(*arr, draws, seed)
            print(
                f"  as one ({len(arr[0])}): {s0:.4f} -> {s1:.4f}  {s1 - s0:+.4f} ({lo:+.4f} to {hi:+.4f})"
                f"  {up} better / {down} worse  saved {saved:.0f} bn, largest loss {min(worst, 0):.1f} bn"
            )


if __name__ == "__main__":
    fire.Fire(main)

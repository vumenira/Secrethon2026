"""Pooled paired comparison of cached plays over several sets (no play).

python outputs/s13/pooled.py --a=<name> --b=v12_all [--sets=full:12345:16,full:2026:16,full:777:32]
"""
import json
from pathlib import Path

import fire
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "recovery_tune"
W = {1: 0.50, 2: 0.30, 3: 0.15, 4: 0.05}


def rows(name, sets):
    out = []
    for item in sets.split(","):
        task, entropy, episodes = item.split(":")
        d = OUT / f"{task}-e{entropy}-n{episodes}"
        refs = json.loads((d / "references.json").read_text())
        p = d / f"{name}.json"
        if not p.exists():
            return None
        play = {r["episode"]: r for r in json.loads(p.read_text())["rows"]}
        for ref in refs:
            if ref.get("excluded") is not None:
                continue
            r = play[ref["episode"]]
            out.append((entropy, ref["episode"], ref["stratum"], ref["J_naive_cents"], ref["J_oracle_cents"], r["J_policy_cents"], r.get("first_error"), r.get("fallback_weeks", 0)))
    return out


def score(naive, oracle, J, strata, idx=None):
    num = den = 0.0
    for s, w in W.items():
        m = strata == s if idx is None else None
        if idx is None:
            if not m.any():
                continue
            num += w * (naive[m] - J[m]).mean()
            den += w * (naive[m] - oracle[m]).mean()
    return num / den


def main(a, b="v12_all", sets="full:12345:16,full:2026:16,full:777:32", n_boot=4000, show=6, each=True):
    names = [a] if isinstance(a, str) else list(a)
    for name in names:
        if each and "," in sets:
            for item in sets.split(","):
                one(name, b, item, n_boot, 0)
        one(name, b, sets, n_boot, show)


def one(a, b, sets, n_boot, show):
    ra, rb = rows(a, sets), rows(b, sets)
    if ra is None or rb is None:
        print(f"{a} vs {b} on {sets}: not all played")
        return
    strata = np.array([r[2] for r in ra])
    naive = np.array([r[3] for r in ra], dtype=float)
    oracle = np.array([r[4] for r in ra], dtype=float)
    Ja = np.array([r[5] for r in ra], dtype=float)
    Jb = np.array([r[5] for r in rb], dtype=float)
    sa, sb = score(naive, oracle, Ja, strata), score(naive, oracle, Jb, strata)
    rng = np.random.default_rng(0)
    groups = [(w, np.flatnonzero(strata == s)) for s, w in W.items() if (strata == s).any()]
    picks = [(w, idx[rng.integers(0, idx.size, size=(n_boot, idx.size))]) for w, idx in groups]
    den = sum(w * (naive[p] - oracle[p]).mean(axis=1) for w, p in picks)
    da = sum(w * (naive[p] - Ja[p]).mean(axis=1) for w, p in picks) / den
    db = sum(w * (naive[p] - Jb[p]).mean(axis=1) for w, p in picks) / den
    lo, hi = np.quantile(da - db, [0.05, 0.95])
    delta = (Jb - Ja) / 1e11  # bn saved by a
    errs = [r[6] for r in ra if r[6]]
    print(
        f"{a:28s} vs {b} on {sets}: {sb:.4f} -> {sa:.4f}  diff {sa - sb:+.4f} [{lo:+.4f}, {hi:+.4f}]"
        f"  better/worse {int((delta > 1e-6).sum())}/{int((delta < -1e-6).sum())}  saved {delta.sum():+.0f} bn"
        f"  worst {delta.min():+.1f} best {delta.max():+.1f}  errors {len(errs)} fallback {sum(r[7] for r in ra)}"
    )
    if show:
        order = np.argsort(delta)
        print("   worst:", [(ra[i][0], ra[i][1], ra[i][2], round(float(delta[i]), 1)) for i in order[:show]])
        print("   best: ", [(ra[i][0], ra[i][1], ra[i][2], round(float(delta[i]), 1)) for i in order[::-1][:show]])


if __name__ == "__main__":
    fire.Fire(main)

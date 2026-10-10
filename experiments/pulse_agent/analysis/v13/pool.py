"""usage: pool.py name [name2 ...] --sets 12345:16,2026:16[,777:32]  (prints per-set and pooled diff vs v12_all)"""
import json, sys, numpy as np
from pathlib import Path
R = Path("outputs/recovery_tune"); W = {1: .5, 2: .3, 3: .15, 4: .05}
def load(root, n, name):
    d = R / f"full-e{root}-n{n}"
    refs = {r["episode"]: r for r in json.load(open(d / "references.json"))}
    try: rows = {r["episode"]: r for r in json.load(open(d / f"{name}.json"))["rows"]}
    except FileNotFoundError: return None
    return refs, rows
def arrays(sets, name):
    S, A, L = [], [], []
    for root, n in sets:
        a, b = load(root, n, "v12_all"), load(root, n, name)
        if b is None: return None
        for e, r in a[1].items():
            if e not in b[1] or b[1][e].get("J_policy_cents") is None: return None
            ref = a[0][e]
            if ref.get("excluded"): continue
            att = ref["J_naive_cents"] - ref["J_clairvoyant_cents"]
            S.append(r["J_policy_cents"] - b[1][e]["J_policy_cents"])  # saving of variant minus baseline
            A.append(att); L.append(ref["stratum"])
    return np.array(S, float), np.array(A, float), np.array(L)
def diff(S, A, L, idx=None):
    if idx is None: idx = np.arange(len(S))
    num = den = 0.0
    for l, w in W.items():
        m = idx[L[idx] == l]
        if len(m): num += w * S[m].mean(); den += w * A[m].mean()
    return num / den
def stat(sets, name, B=2000, seed=1):
    t = arrays(sets, name)
    if t is None: return None
    S, A, L = t
    d = diff(S, A, L)
    rng = np.random.default_rng(seed); bs = []
    byl = {l: np.where(L == l)[0] for l in W if (L == l).any()}
    for _ in range(B):
        idx = np.concatenate([rng.choice(v, len(v)) for v in byl.values()])
        bs.append(diff(S, A, L, idx))
    lo, hi = np.percentile(bs, [5, 95])
    return d, lo, hi, int((S > 0).sum()), int((S < 0).sum())
if __name__ == "__main__":
    a = sys.argv[1:]; i = a.index("--sets"); names = a[:i]
    sets = [tuple(map(int, s.split(":"))) for s in a[i + 1].split(",")]
    for nm in names:
        out = [nm]
        for s in sets:
            r = stat([s], nm); out.append("n/a" if r is None else f"{s[0]}: {r[0]:+.4f}")
        r = stat(sets, nm)
        out.append("pooled n/a" if r is None else f"pooled {r[0]:+.4f} ({r[1]:+.4f} to {r[2]:+.4f}) {r[3]}/{r[4]}")
        print(" | ".join(out))

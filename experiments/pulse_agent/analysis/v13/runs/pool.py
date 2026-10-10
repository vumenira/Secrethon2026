"""A variant against v12 on several cached sets, each and pooled: score difference, 90 % paired interval, counts.

uv run python pool.py r13_ends --sets=full:12345:16,full:2026:16,full:777:32 [--base=v12_all]
Score of a set of episodes: sum over harm levels of w * mean saving / sum of w * mean attainable (w = .50 .30 .15 .05),
saving = J_naive - J, attainable = J_naive - J_clairvoyant. The interval resamples episodes within each level.
"""
import json
from pathlib import Path
import fire, numpy as np

OUT = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026\outputs\recovery_tune")
W = {1: 0.50, 2: 0.30, 3: 0.15, 4: 0.05}

def load(task, entropy, n, name, base):
    d = OUT / f"{task}-e{entropy}-n{n}"
    refs = {r["episode"]: r for r in json.loads((d / "references.json").read_text())}
    b = base if (d / f"{base}.json").exists() else "r13_off"
    ja = {r["episode"]: r["J_policy_cents"] for r in json.loads((d / f"{name}.json").read_text())["rows"]}
    jb = {r["episode"]: r["J_policy_cents"] for r in json.loads((d / f"{b}.json").read_text())["rows"]}
    rows = []
    for e, r in refs.items():
        if r.get("excluded") is not None:
            continue
        rows.append((r["stratum"], r["J_naive_cents"] - ja[e], r["J_naive_cents"] - jb[e], r["J_naive_cents"] - r["J_clairvoyant_cents"], f"{task}{entropy}:{e}"))
    return rows, b

def score(rows, col):
    num = den = 0.0
    for s, w in W.items():
        part = [r for r in rows if r[0] == s]
        if part:
            num += w * np.mean([r[col] for r in part])
            den += w * np.mean([r[3] for r in part])
    return num / den

def boot(rows, n_boot=4000, seed=0):
    rng = np.random.default_rng(seed)
    by = {s: [r for r in rows if r[0] == s] for s in W}
    out = []
    for _ in range(n_boot):
        sample = []
        for s, part in by.items():
            if part:
                idx = rng.integers(0, len(part), len(part))
                sample += [part[i] for i in idx]
        out.append(score(sample, 1) - score(sample, 2))
    return np.quantile(out, [0.05, 0.95])

def line(label, rows):
    d = score(rows, 1) - score(rows, 2)
    lo, hi = boot(rows)
    saved = np.array([(r[1] - r[2]) / 1e11 for r in rows])
    worst = min(rows, key=lambda r: r[1] - r[2])
    best = max(rows, key=lambda r: r[1] - r[2])
    print(f"{label:28s} {score(rows, 2):.4f} -> {score(rows, 1):.4f}  {d:+.4f} [{lo:+.4f}, {hi:+.4f}]  better/worse/equal {int((saved > 1e-6).sum())}/{int((saved < -1e-6).sum())}/{int((abs(saved) <= 1e-6).sum())}"
          f"  saved {saved.sum():+.1f} bn  worst {(worst[1] - worst[2]) / 1e11:+.1f} ({worst[4]})  best {(best[1] - best[2]) / 1e11:+.1f} ({best[4]})")

def main(name, sets="full:12345:16,full:2026:16,full:777:32", base="v12_all"):
    sets = sets if isinstance(sets, str) else ",".join(sets)
    pooled = []
    for spec in sets.split(","):
        task, entropy, n = spec.split(":")
        try:
            rows, b = load(task, int(entropy), int(n), name, base)
        except FileNotFoundError as e:
            print(f"{spec}: not played ({Path(str(e.filename)).name})")
            continue
        line(f"{name} {task} {entropy} ({len(rows)}) vs {b}", rows)
        pooled += rows
    if pooled:
        line(f"{name} pooled ({len(pooled)})", pooled)

if __name__ == "__main__":
    fire.Fire(main)

"""A cached variant against a cached base, episode by episode, in pooled score units: who gains, who loses, how concentrated.

The pooled score is linear in the episode costs (sum over harm levels of weight * mean saving, over the same of the
naive-to-clairvoyant gap), so each episode's share of the difference is its cost difference times a fixed weight.

uv run python conc.py --name=u_all --base=v_r1 --task=full --entropy=12345 --episodes=16
uv run python conc.py same --name=u_reg_off --base=v_r1                       # equal to the cent?
"""

import json
from pathlib import Path

import fire

OUT = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026\outputs\recovery_tune")
WEIGHTS = (0.50, 0.30, 0.15, 0.05)


def load(task, entropy, episodes, name):
    d = OUT / f"{task}-e{entropy}-n{episodes}"
    rows = json.loads((d / f"{name}.json").read_text())["rows"]
    return {r["episode"]: r for r in rows}, {r["episode"]: r for r in json.loads((d / "references.json").read_text())}


def weights(refs):
    counts = {s: sum(r["stratum"] == s for r in refs.values()) for s in (1, 2, 3, 4)}
    den = sum(
        WEIGHTS[s - 1] * sum(r["J_naive_cents"] - r["J_oracle_cents"] for r in refs.values() if r["stratum"] == s) / counts[s]
        for s in counts
        if counts[s]
    )
    return {n: WEIGHTS[r["stratum"] - 1] / counts[r["stratum"]] / den for n, r in refs.items()}


def score(rows, refs, w):
    return sum(w[n] * (refs[n]["J_naive_cents"] - rows[n]["J_policy_cents"]) for n in refs)


def main(name="u_all", base="v_r1", task="small", entropy=12345, episodes=64, top=6):
    a, refs = load(task, entropy, episodes, base)
    b, _ = load(task, entropy, episodes, name)
    w = weights(refs)
    d = {n: w[n] * (a[n]["J_policy_cents"] - b[n]["J_policy_cents"]) for n in refs}
    total = sum(d.values())
    gain = sum(v for v in d.values() if v > 0)
    loss = sum(v for v in d.values() if v < 0)
    better = sum(a[n]["J_policy_cents"] - b[n]["J_policy_cents"] > 0 for n in refs)
    worse = sum(a[n]["J_policy_cents"] - b[n]["J_policy_cents"] < 0 for n in refs)
    order = sorted(d, key=lambda n: -d[n])
    errors = [(n, b[n].get("first_error")) for n in refs if b[n].get("first_error")]
    fallback = sum(b[n].get("fallback_weeks", 0) for n in refs)
    print(f"{task} root {entropy}, {episodes} episodes: {name} {score(b, refs, w):.4f} against {base} {score(a, refs, w):.4f}: {total:+.4f}")
    print(f"  episodes better {better}, worse {worse}, equal {len(refs) - better - worse}; gains {gain:+.4f}, losses {loss:+.4f}; fallback weeks {fallback}; errors {errors[:3]}")
    cum = 0.0
    parts = []
    for n in order[:top]:
        cum += d[n]
        parts.append(f"ep {n} (level {refs[n]['stratum']}) {d[n]:+.4f}")
    print(f"  largest gains: {'; '.join(parts)}")
    for k in (1, 2, 3, 4):
        print(f"  the {k} best episodes carry {sum(d[n] for n in order[:k]):+.4f} = {100 * sum(d[n] for n in order[:k]) / total if total else float('nan'):.0f} % of the difference")
    print(f"  without its 2 best episodes the difference is {total - sum(d[n] for n in order[:2]):+.4f}")
    losers = [n for n in reversed(order) if d[n] < 0][:top]
    print("  largest losses: " + "; ".join(f"ep {n} (level {refs[n]['stratum']}) {d[n]:+.4f} ({(a[n]['J_policy_cents'] - b[n]['J_policy_cents']) / 1e11:+.2f} bn USD)" for n in losers))
    by = {}
    for n in refs:
        by.setdefault(refs[n]["stratum"], []).append(d[n])
    print("  by harm level: " + "; ".join(f"level {s}: {sum(v):+.4f} over {len(v)} episodes" for s, v in sorted(by.items())))


def same(name="u_reg_off", base="v_r1", task="small", entropy=12345, episodes=64):
    a, refs = load(task, entropy, episodes, base)
    b, _ = load(task, entropy, episodes, name)
    equal = sum(a[n]["J_policy_cents"] == b[n]["J_policy_cents"] for n in refs)
    worst = max(abs(a[n]["J_policy_cents"] - b[n]["J_policy_cents"]) for n in refs)
    errors = [(n, b[n].get("first_error")) for n in refs if b[n].get("first_error")]
    fallback = sum(b[n].get("fallback_weeks", 0) for n in refs)
    print(f"{task} root {entropy}: {name} equals {base} to the cent in {equal} of {len(refs)} episodes (largest difference {worst} cents); fallback weeks {fallback}; errors {errors[:3]}")


if __name__ == "__main__":
    fire.Fire({"main": main, "same": same})

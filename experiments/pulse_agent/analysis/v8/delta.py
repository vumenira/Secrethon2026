"""Per-episode differences of cached variants against a base on a tuning root, and the exact-cents comparison.

uv run python delta.py show --task=full --names=s_h24,s_h24_g --base=t_v7
uv run python delta.py same --task=small --name=s_reg --base=t_v7
uv run python delta.py spread --task=small --entropy=2026 --name=s_v8 --base=t_v7      # counts only, no episodes
"""

import json
from pathlib import Path

import fire
import numpy as np


ROOT = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026")
OUT = ROOT / "outputs" / "recovery_tune"
TOTAL = {"small": 64, "full": 16}


def load(task, entropy, episodes, name):
    d = OUT / f"{task}-e{entropy}-n{episodes or TOTAL[task]}"
    rows = json.loads((d / f"{name}.json").read_text())["rows"]
    refs = {r["episode"]: r for r in json.loads((d / "references.json").read_text())}
    return {r["episode"]: r for r in rows}, refs


def weights(refs):
    """Each episode's weight in the pooled score's numerator: stratum weight over the stratum's count, over the pooled gap."""
    from shockbench_flow.scoring.rss import STRATUM_WEIGHTS

    strata = [r["stratum"] for r in refs.values()]
    present = sorted(set(strata))
    total = sum(STRATUM_WEIGHTS[s - 1] for s in present)
    w = {n: STRATUM_WEIGHTS[r["stratum"] - 1] / total / strata.count(r["stratum"]) for n, r in refs.items()}
    den = sum(w[n] * (r["J_naive_cents"] - r["J_oracle_cents"]) for n, r in refs.items())
    return {n: w[n] / den for n in refs}


def show(task="small", names="s_h24", base="t_v7", entropy=12345, episodes=None, top=64):
    assert entropy == 12345, "single episodes are looked at on the tuning root only"
    names = names if isinstance(names, (list, tuple)) else str(names).split(",")
    b, refs = load(task, entropy, episodes, base)
    w = weights(refs)
    cols = {}
    for name in names:
        v, _ = load(task, entropy, episodes, name)
        cols[name] = {n: (b[n]["J_policy_cents"] - v[n]["J_policy_cents"]) for n in refs}
    order = sorted(refs, key=lambda n: cols[names[0]][n] * w[n])
    print(f"{task} root {entropy}: per episode, share of the episode's own gap | pooled units (x 1e-4), against {base}")
    print("  ep L   gap bn | " + " | ".join(f"{name:>20s}" for name in names))
    shown = order if len(order) <= top else order[: top // 2] + order[-top // 2 :]
    for n in shown:
        D = refs[n]["J_naive_cents"] - refs[n]["J_oracle_cents"]
        print(f"  {n:2d} {refs[n]['stratum']} {D / 1e11:8.1f} | " + " | ".join(f"{cols[name][n] / D:+9.4f} {1e4 * cols[name][n] * w[n]:+9.2f}" for name in names))
    for name in names:
        x = np.array([cols[name][n] * w[n] for n in refs])
        pos, neg = x[x > 0].sum(), x[x < 0].sum()
        srt = np.sort(x)[::-1]
        print(f"  {name}: pooled {x.sum():+.4f}; gains {pos:+.4f} in {int((x > 0).sum())} episodes, losses {neg:+.4f} in {int((x < 0).sum())}, unchanged {int((x == 0).sum())}; the 3 best carry {srt[:3].sum():+.4f}, the 5 best {srt[:5].sum():+.4f}; without the 3 best {x.sum() - srt[:3].sum():+.4f}")


def same(task="small", name="s_reg", base="t_v7", entropy=12345, episodes=None):
    a, refs = load(task, entropy, episodes, name)
    b, _ = load(task, entropy, episodes, base)
    eq = sum(a[n]["J_policy_cents"] == b[n]["J_policy_cents"] for n in refs)
    worst = max(abs(a[n]["J_policy_cents"] - b[n]["J_policy_cents"]) for n in refs)
    bad = sum((a[n].get("fallback_weeks") or 0) for n in refs)
    errs = [a[n]["first_error"] for n in refs if a[n].get("first_error")]
    print(f"{task} root {entropy}: {name} equals {base} to the cent in {eq} of {len(refs)} episodes (largest difference {worst} cents); fallback weeks {bad}; errors {errs[:2]}")


def spread(task="small", name="s_v8", base="t_v7", entropy=2026, episodes=None):
    """How concentrated a difference is: counts and sums only, no episode is named or shown."""
    a, refs = load(task, entropy, episodes, name)
    b, _ = load(task, entropy, episodes, base)
    w = weights(refs)
    x = np.array([(b[n]["J_policy_cents"] - a[n]["J_policy_cents"]) * w[n] for n in refs])
    srt = np.sort(x)[::-1]
    k = len(x)
    bad = sum((a[n].get("fallback_weeks") or 0) for n in refs)
    errs = sum(1 for n in refs if a[n].get("first_error"))
    print(
        f"{task} root {entropy}, {k} episodes, {name} against {base}: pooled {x.sum():+.4f}; better {int((x > 0).sum())}, worse {int((x < 0).sum())}, the same to the cent {int((x == 0).sum())};"
        f" gains {x[x > 0].sum():+.4f}, losses {x[x < 0].sum():+.4f}; the best episode {srt[0]:+.4f}, the 3 best {srt[:3].sum():+.4f}, without the 3 best {x.sum() - srt[:3].sum():+.4f};"
        f" the worst episode {srt[-1]:+.4f}, the 3 worst {srt[-3:].sum():+.4f}; fallback weeks {bad}, episodes with an error {errs}"
    )


if __name__ == "__main__":
    fire.Fire({"show": show, "same": same, "spread": spread})

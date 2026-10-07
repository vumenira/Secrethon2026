"""Cached variants of one episode set, pair by pair: score, paired difference, its 90 % bootstrap interval (as tune.py
report computes it).

uv run python pairs.py table --rows=y_all,y_grid --bases=ft_q_e20_fd8 --task=full --entropy=12345
uv run python pairs.py same --pairs=y_reg_pulse=ft_q_e20_fd8 --task=full --entropy=12345
uv run python pairs.py add --base=... --a=... --b=... --both=... --task=small --entropy=12345
"""

import json
from pathlib import Path

import fire
import numpy as np


TUNE = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026\outputs\recovery_tune")
COUNT = {"small": 64, "full": 16}


def out_dir(task, entropy):
    return TUNE / f"{task}-e{entropy}-n{COUNT[task]}"


def rows_of(name, task, entropy):
    return json.loads((out_dir(task, entropy) / f"{name}.json").read_text())["rows"]


def costs(name, task, entropy):
    return [r["J_policy_cents"] for r in rows_of(name, task, entropy)]


def episode_set(task, entropy):
    from shockbench_flow_agent import EpisodeSet

    return EpisodeSet.build(task, COUNT[task], entropy=entropy, n_jobs=1, verbose=False)


def paired(es, a, b, n_boot=4000):
    from shockbench_flow_agent.scoring import _boot_stats

    boots = _boot_stats(list(es.references), [a, b], True, n_boot, 0)
    d = boots[0] - boots[1]
    lo, hi = np.quantile(d[np.isfinite(d)], [0.05, 0.95])
    return es.rss(a)["rss"] - es.rss(b)["rss"], lo, hi, float((d > 0).mean())


def names(value):
    return list(value) if isinstance(value, (list, tuple)) else [v for v in str(value).split(",") if v]


def table(rows, bases, task="small", entropy=12345, n_boot=4000):
    es = episode_set(task, entropy)
    refs = list(es.references)
    strata = [r["stratum"] for r in refs]
    print(f"{task}, root {entropy}, {len(refs)} episodes; by harm level {[strata.count(s) for s in (1, 2, 3, 4)]}")
    for n in names(rows):
        J = costs(n, task, entropy)
        t = es.rss(J)
        by = " ".join(f"{t['rss_by_stratum'].get(s, float('nan')):.3f}" for s in (1, 2, 3, 4))
        rs = rows_of(n, task, entropy)
        bad = sum(r["fallback_weeks"] for r in rs), sum(1 for r in rs if r.get("first_error"))
        print(f"{n}: score {t['rss']:.4f}, by harm level {by}; fallback weeks {bad[0]}, episodes with an error {bad[1]},"
              f" slowest episode {max(r['seconds'] for r in rs):.1f} s")
        for base in names(bases):
            if base == n:
                continue
            Jb = costs(base, task, entropy)
            d, lo, hi, p = paired(es, J, Jb, n_boot)
            delta = (np.array(Jb, dtype=float) - np.array(J, dtype=float)) / 1e11
            print(f"   against {base:16s} ({es.rss(Jb)['rss']:.4f}) {d:+.4f} [{lo:+.4f}, {hi:+.4f}]  P(better) {p:.2f}"
                  f"  better/worse {int((delta > 1e-6).sum())}/{int((delta < -1e-6).sum())}"
                  f"  mean saving {delta.mean():+.3f} bn USD")


def same(pairs, task="small", entropy=12345):
    for pair in names(pairs):
        a, b = pair.split("=")
        ra = {r["episode"]: r for r in rows_of(a, task, entropy)}
        rb = {r["episode"]: r for r in rows_of(b, task, entropy)}
        assert sorted(ra) == sorted(rb), "not the same episodes"
        assert all(ra[n]["omega_hash"] == rb[n]["omega_hash"] for n in ra), "not the same scenarios"
        off = [(n, ra[n]["J_policy_cents"] - rb[n]["J_policy_cents"]) for n in sorted(ra)
               if ra[n]["J_policy_cents"] != rb[n]["J_policy_cents"]]
        bad = sum(r["fallback_weeks"] for r in ra.values()), [r["first_error"] for r in ra.values() if r.get("first_error")]
        print(f"{task} root {entropy}: {a} vs {b}: {len(ra) - len(off)} of {len(ra)} episodes equal to the cent; {a}:"
              f" fallback weeks {bad[0]}, errors {bad[1][:2]}")
        if off:
            print("   differing (episode, cents):", off[:10])


def add(base, a, b, both, task="small", entropy=12345):
    """The combination's costs against base + (a - base) + (b - base), per episode."""
    es = episode_set(task, entropy)
    J = {n: np.array(costs(n, task, entropy), dtype=object) for n in (base, a, b, both)}
    summed = [int(x) for x in J[a] + J[b] - J[base]]
    r = {n: es.rss([int(x) for x in J[n]])["rss"] for n in J}
    print(f"{base} {r[base]:.4f}; {a} {r[a]:.4f} ({r[a] - r[base]:+.4f}); {b} {r[b]:.4f} ({r[b] - r[base]:+.4f});"
          f" sum of the two gains {r[a] + r[b] - 2 * r[base]:+.4f}; {both} {r[both]:.4f} ({r[both] - r[base]:+.4f})")
    d, lo, hi, p = paired(es, [int(x) for x in J[both]], summed)
    print(f"   {both} minus the additive costs: {d:+.4f} [{lo:+.4f}, {hi:+.4f}], P(above) {p:.2f}")


def episodes(a, b, task="small", entropy=12345, top=8):
    """The episodes where two variants differ most (bn USD saved by a over b), with their harm level."""
    es = episode_set(task, entropy)
    refs = list(es.references)
    Ja, Jb = costs(a, task, entropy), costs(b, task, entropy)
    delta = [(float(y - x) / 1e11, r["episode"], r["stratum"]) for x, y, r in zip(Ja, Jb, refs)]
    span = [(r["J_naive_cents"] - r["J_oracle_cents"]) / 1e11 for r in refs]
    print(f"{task} root {entropy}: {a} against {b}; total saving {sum(d for d, *_ in delta):+.1f} bn USD")
    order = sorted(range(len(delta)), key=lambda i: delta[i][0])
    for i in order[:top] + order[-top:]:
        d, n, s = delta[i]
        print(f"   episode {n:3d} level {s}: {d:+9.2f} bn USD ({d / span[i]:+.4f} of its naive-to-clairvoyant span)")


if __name__ == "__main__":
    fire.Fire({"table": table, "same": same, "add": add, "episodes": episodes})

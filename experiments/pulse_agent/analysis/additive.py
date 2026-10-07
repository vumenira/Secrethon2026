"""Do the gains of two forecasts add up? The combination's costs against base + (a - base) + (b - base), per episode.

uv run python additive.py --base=mpc_det_h20 --a=t1b_v3_q0 --b=t1a_v3_final --both=m_all_h20
uv run python additive.py pairs --rows=m_all_h20,m_all_h24 --bases=mpc_det_h20,t1b_v3_q0
"""
import json
from pathlib import Path

import fire
import numpy as np

OUT = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026\outputs\recovery_tune\small-e12345-n64")


def costs(name):
    return [r["J_policy_cents"] for r in json.loads((OUT / f"{name}.json").read_text())["rows"]]


def episode_set():
    from shockbench_flow_agent import EpisodeSet

    return EpisodeSet.build("small", 64, entropy=12345, n_jobs=1, verbose=False)


def paired(es, a, b, n_boot=4000):
    """score(a) - score(b), its 90 % paired bootstrap interval, the share of resamples above 0 (as tune.py report)."""
    from shockbench_flow_agent.scoring import _boot_stats

    refs = list(es.references)
    boots = _boot_stats(refs, [a, b], True, n_boot, 0)
    d = boots[0] - boots[1]
    lo, hi = np.quantile(d[np.isfinite(d)], [0.05, 0.95])
    return es.rss(a)["rss"] - es.rss(b)["rss"], lo, hi, float((d > 0).mean())


def add(base="mpc_det_h20", a="t1b_v3_q0", b="t1a_v3_final", both="m_all_h20"):
    es = episode_set()
    J = {n: np.array(costs(n), dtype=object) for n in (base, a, b, both)}
    summed = [int(x) for x in J[a] + J[b] - J[base]]  # the costs if the two savings simply added, episode by episode
    r = {n: es.rss([int(x) for x in J[n]])["rss"] for n in J}
    r_sum = es.rss(summed)["rss"]
    print(f"{base} {r[base]:.4f}; {a} {r[a]:.4f} ({r[a] - r[base]:+.4f}); {b} {r[b]:.4f} ({r[b] - r[base]:+.4f})")
    print(f"sum of the two gains: {r[a] + r[b] - 2 * r[base]:+.4f}; score if the savings added per episode {r_sum:.4f}")
    print(f"{both} {r[both]:.4f} ({r[both] - r[base]:+.4f})")
    d, lo, hi, p = paired(es, [int(x) for x in J[both]], summed)
    print(f"{both} minus the additive costs: {d:+.4f} [{lo:+.4f}, {hi:+.4f}], P(above) {p:.2f}")
    gap = np.array([float(x) for x in (J[both] - np.array(summed, dtype=object))]) / 1e11  # bn USD, + : costs more
    sa = np.array([float(x) for x in J[base] - J[a]]) / 1e11
    sb = np.array([float(x) for x in J[base] - J[b]]) / 1e11
    print(f"episodes where both forecasts change the cost: {int(((np.abs(sa) > 1e-6) & (np.abs(sb) > 1e-6)).sum())} of {len(sa)}"
          f" (a alone changes {int((np.abs(sa) > 1e-6).sum())}, b alone {int((np.abs(sb) > 1e-6).sum())})")
    print(f"episodes where the combination is exactly the sum: {int((np.abs(gap) < 1e-9).sum())};"
          f" costs less than the sum {int((gap < -1e-6).sum())}, more {int((gap > 1e-6).sum())};"
          f" mean gap {gap.mean():+.4f} bn USD, largest {gap.min():+.3f} / {gap.max():+.3f}")
    print(f"mean saving, bn USD per episode: a {sa.mean():.4f}, b {sb.mean():.4f}, both"
          f" {np.mean([float(x) for x in J[base] - J[both]]) / 1e11:.4f}")


def pairs(rows, bases, n_boot=4000):
    es = episode_set()
    rows = rows if isinstance(rows, (list, tuple)) else rows.split(",")
    bases = bases if isinstance(bases, (list, tuple)) else bases.split(",")
    for n in rows:
        table = es.rss(costs(n))
        by = " ".join(f"{table['rss_by_stratum'].get(s, float('nan')):.3f}" for s in (1, 2, 3, 4))
        print(f"{n}: score {table['rss']:.4f}, by harm level {by}")
        for base in bases:
            d, lo, hi, p = paired(es, costs(n), costs(base), n_boot)
            delta = (np.array(costs(base), dtype=float) - np.array(costs(n), dtype=float)) / 1e11
            print(f"   against {base:16s} {d:+.4f} [{lo:+.4f}, {hi:+.4f}]  P(better) {p:.2f}"
                  f"  better/worse episodes {int((delta > 1e-6).sum())}/{int((delta < -1e-6).sum())}")


if __name__ == "__main__":
    fire.Fire({"add": add, "pairs": pairs})

"""The report of experiments/approaches/tune.py for chosen variants, with seconds per episode added (scratch).

uv run python table.py --names=t3a_base,t3a_w2 [--base=mpc_det_h20] [--time_base=t3a_base]
"""

import json
from pathlib import Path

import fire
import numpy as np


OUT = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026\outputs\recovery_tune")


def main(names, base="mpc_det_h20", time_base="t3a_base", task="small", entropy=12345, episodes=64, n_boot=4000):
    from shockbench_flow_agent import EpisodeSet
    from shockbench_flow_agent.scoring import _boot_stats

    names = [n for n in (names.split(",") if isinstance(names, str) else list(names)) if n]
    es = EpisodeSet.build(task, episodes, entropy=entropy, n_jobs=1, verbose=False)
    out = OUT / f"{task}-e{entropy}-n{episodes}"
    refs = list(es.references)
    docs = {n: json.loads((out / f"{n}.json").read_text()) for n in [base, *names]}
    costs = {n: [r["J_policy_cents"] for r in d["rows"]] for n, d in docs.items()}
    secs = {n: np.array([r["seconds"] for r in d["rows"]]) for n, d in docs.items()}
    strata = np.array([r["stratum"] for r in refs])
    print(f"{task}, root {entropy}, {episodes} episodes; by harm level {[int((strata == s).sum()) for s in (1, 2, 3, 4)]}")
    order = [base, *names]
    boots = dict(zip(order, _boot_stats(refs, [costs[n] for n in order], True, n_boot, 0)))
    base_rss = es.rss(costs[base])["rss"]
    tb = secs[time_base].mean() if time_base in secs else float("nan")
    for n in order:
        table = es.rss(costs[n])
        d = boots[n] - boots[base]
        lo, hi = np.quantile(d[np.isfinite(d)], [0.05, 0.95])
        delta = (np.array(costs[base], dtype=float) - np.array(costs[n], dtype=float)) / 1e11
        by = " ".join(f"{table['rss_by_stratum'].get(s, float('nan')):.3f}" for s in (1, 2, 3, 4))
        rows = docs[n]["rows"]
        bad = sum(r["fallback_weeks"] for r in rows)
        errs = sum(1 for r in rows if r.get("first_error"))
        same = int(sum(a == b for a, b in zip(costs[n], costs[base])))
        print(
            f"{n:16s} {table['rss']:.4f} diff {table['rss'] - base_rss:+.4f} [{lo:+.4f}, {hi:+.4f}] P {float((d > 0).mean()):.2f}"
            f" | levels {by} | better/worse {int((delta > 1e-6).sum())}/{int((delta < -1e-6).sum())} same {same}"
            f" | saved bn {delta.sum():+.0f} by level {' '.join(f'{delta[strata == s].sum():+.0f}' for s in (1, 2, 3, 4))}"
            f" | s/ep {secs[n].mean():.2f} x{secs[n].mean() / tb:.2f} max {secs[n].max():.1f}"
            f" | fallback wk {bad} err eps {errs} | {json.dumps(docs[n]['spec'].get('params', {}))}"
        )


if __name__ == "__main__":
    fire.Fire(main)

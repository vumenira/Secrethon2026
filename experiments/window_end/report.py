"""Cached tune.py variants against a base, by harm level: each level's score and its paired difference with a 90 %
interval (episodes resampled within the level).

uv run python experiments/window_end/report.py --base=v8_on --task=small --entropy=12345 --episodes=64
"""

import json
import sys
from pathlib import Path

import fire
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "approaches"))
from tune import episode_set, run_dir  # noqa: E402


def level_rss(refs, costs):
    """One harm level's score: the episodes pooled as the leaderboard pools a level."""
    from shockbench_flow.evaluation.results import episode_rss

    records = [{**r, "J_policy_cents": int(j), "stratum": 1} for r, j in zip(refs, costs)]
    return episode_rss(records, (1.0,))["strata"]["1"]["rss"]


def main(base="v8_on", task="small", entropy=12345, episodes=64, n_boot=2000, names=None):
    es = episode_set(task, entropy, episodes, 1)
    out = run_dir(task, entropy, episodes)
    refs = [r for r in es.references]
    costs = {
        p.stem: [r["J_policy_cents"] for r in json.loads(p.read_text())["rows"]]
        for p in sorted(out.glob("*.json"))
        if p.stem != "references"
    }
    keep = [i for i, r in enumerate(refs) if r["excluded"] is None]
    names = [
        n for n in ((names.split(",") if isinstance(names, str) else list(names)) if names else costs) if n != base
    ]
    rng = np.random.default_rng(0)
    print(f"{task}, root {entropy}, {episodes} episodes; differences against {base}, by harm level (90 % interval)")
    for level in (1, 2, 3, 4):
        idx = [i for i in keep if refs[i]["stratum"] == level]
        if not idx:
            continue
        sub = [refs[i] for i in idx]
        b = [costs[base][i] for i in idx]
        draws = [rng.integers(0, len(idx), len(idx)) for _ in range(n_boot)]
        print(f"level {level}: {len(idx)} episodes, {base} {level_rss(sub, b):.4f}")
        for n in names:
            c = [costs[n][i] for i in idx]
            diff = level_rss(sub, c) - level_rss(sub, b)
            boot = [
                level_rss([sub[j] for j in d], [c[j] for j in d]) - level_rss([sub[j] for j in d], [b[j] for j in d])
                for d in draws
            ]
            lo, hi = np.quantile(boot, [0.05, 0.95])
            print(f"  {n:12s} {level_rss(sub, c):.4f}  diff {diff:+.4f} [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    fire.Fire(main)

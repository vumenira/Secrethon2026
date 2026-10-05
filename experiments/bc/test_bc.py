"""Test the behaviour cloning (BC) agent against the kit's agents, on the same episodes.

    uv run python experiments/bc/test_bc.py                                  # the 20 dev episodes of Tiny
    uv run python experiments/bc/test_bc.py --entropy=999 --episodes=40      # 40 fresh episodes of another root

It scores ``--agent`` (default bc_tiny) and each agent of ``--others`` (heuristic, template) with the scorer's own
computation (as ``sbf evaluate``), then checks with a paired interval (as ``sbf compare``) whether BC beats the first
of them. Score: 0 = the naive rule, 1 = the clairvoyant oracle.

Do not test on the training root (train_bc.py's --entropy, 12345 by default): the network has seen those episodes.

Files: outputs/bc/test_<date_time>/results.json.
"""

import json
import time
from pathlib import Path

import fire

from sbf_starter import scoring


ROOT = Path(__file__).resolve().parents[2]


def fmt(x, digits=3):
    return "-" if x is None else f"{x:.{digits}f}"


def main(
    agent: str = "bc_tiny",
    others: tuple[str, ...] = ("heuristic", "template"),
    task: str = "tiny",
    episodes: str | int = "dev",
    entropy: int = 0,
    n_jobs: int = -1,
    out: str | None = None,
) -> None:
    """Print a table of scores and a verdict; save them in results.json.

    Args:
        agent: the agent to test (a name in agents/ or a path).
        others: agents to compare with; the first one gets the paired check.
        task: tiny, small or full.
        episodes: dev (20 episodes, 5 per harm level) or a count k (episodes 0..k-1 of ``entropy``).
        entropy: 0 for the dev episodes; another integer for episodes of your own.
        n_jobs: workers of a first run's reference computation (-1: all cores).
    """
    if isinstance(others, str):
        others = (others,)
    options = {"entropy": entropy, "n_jobs": n_jobs, "verbose": False}
    names = [agent, *others]
    print(
        f"scoring {', '.join(names)} on {task}, episodes {episodes!r}, root {entropy} "
        "(the first run on a network computes the reference costs: about a minute on Tiny)"
    )
    scores = {}
    for name in names:
        scores[name] = scoring.evaluate(name, task, episodes, **options)
        print(f"  {name}: {fmt(scores[name].rss)}")

    rival = others[0]
    duel = scoring.compare(agent, rival, task, episodes, **options)

    print()
    print(f"{'agent':<14}{'score':>8}{'90 % interval':>20}{'mean cost, M USD':>20}{'naive-played weeks':>21}")
    for name, s in scores.items():
        interval = "-" if s.interval is None else f"{s.interval[0]:.2f} to {s.interval[1]:.2f}"
        print(f"{name:<14}{fmt(s.rss):>8}{interval:>20}{s.cost_usd / 1e6:>20,.1f}{s.fallback_weeks:>21}")
    any_score = next(iter(scores.values()))
    print(f"{'naive':<14}{0:>8.3f}{'':>20}{any_score.naive_cost_usd / 1e6:>20,.1f}")
    if any_score.oracle_cost_usd is not None:
        print(f"{'oracle':<14}{1:>8.3f}{'':>20}{any_score.oracle_cost_usd / 1e6:>20,.1f}")

    print()
    lo, hi = duel.interval if duel.interval is not None else (None, None)
    print(f"{agent} minus {rival}: {fmt(duel.diff)} (90 % interval {fmt(lo, 2)} to {fmt(hi, 2)})")
    if lo is None:
        verdict = "no verdict: the comparison could not be computed"
    elif lo > 0:
        verdict = f"{agent} is better than {rival} on these episodes"
    elif hi < 0:
        verdict = f"{agent} is worse than {rival} on these episodes"
    else:
        verdict = f"these episodes cannot tell {agent} and {rival} apart (the interval holds 0)"
    print(f"verdict: {verdict}")

    out = Path(out or ROOT / f"outputs/bc/test_{time.strftime('%Y-%m-%d_%H-%M-%S')}")
    out.mkdir(parents=True, exist_ok=True)
    results = {
        "task": task,
        "episodes": episodes,
        "entropy": entropy,
        "scores": {
            n: {
                "rss": s.rss,
                "interval": s.interval,
                "rss_by_harm_level": s.rss_by_stratum,
                "cost_usd": s.cost_usd,
                "fallback_weeks": s.fallback_weeks,
            }
            for n, s in scores.items()
        },
        "naive_cost_usd": any_score.naive_cost_usd,
        "oracle_cost_usd": any_score.oracle_cost_usd,
        "comparison": {"a": agent, "b": rival, "diff": duel.diff, "interval": duel.interval, "verdict": verdict},
    }
    (out / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"saved: {out / 'results.json'}")


if __name__ == "__main__":
    fire.Fire(main)

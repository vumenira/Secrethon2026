"""Where the score's money is: naive's and the clairvoyant plan's costs by component, with and without disruptions.

    uv run python experiments/eda/cost_decomp.py                          # Small, dev episodes 0 6 11 26 41 76
    uv run python experiments/eda/cost_decomp.py --episodes='[0,1,2]' --entropy=999

For each episode it plays the naive anchor (score 0) and solves the clairvoyant LP (score 1), on the real scenario
and on its event-free twin (the same demand noise and seasons, every disruption removed). The gap naive - oracle on
the twin is the part of the score that needs no forecasting at all: naive's own inefficiency in calm weather.

One oracle solve is about 15 s on Small. Files: outputs/eda/costs_<task>_<date_time>/costs.json.
"""

import json
import time
from pathlib import Path

import fire
import numpy as np
from shockbench_flow.dynamics.env import rollout
from shockbench_flow.evaluation.cache import fq_quantiles
from shockbench_flow.hosting.tasks import scenario, task_generator
from shockbench_flow.marks import compute_marks
from shockbench_flow.omega.injected import event_free
from shockbench_flow.oracle.lp import build_lp, lp_costs, solve_oracle
from shockbench_flow.policies.naive_fq import anchor_policy, fallback_spec


ROOT = Path(__file__).resolve().parents[2]
CACHE = str(Path.home() / ".cache" / "shockbench-flow")  # the kit's cache: naive's F_Q is computed once there
FQ_REPLICATIONS = 1000  # the leaderboard's
COMPONENTS = ("freight", "war_risk", "tariff", "holding", "queue_holding", "shortage", "disposal", "shed")


def naive_costs(inst, params, om, fallback):
    traj = rollout(inst, anchor_policy(inst, params, FQ_REPLICATIONS), om, "prediction_free", fallback=fallback)
    comp = {c: sum(getattr(r.costs, c) for r in traj.records) for c in COMPONENTS}
    comp["salvage"] = -traj.salvage
    return comp


def oracle_costs(inst, om):
    model = build_lp(inst, compute_marks(inst, om))
    res = solve_oracle(model)
    if res.x is None:
        return None
    weeks, credit = lp_costs(model, res.x)
    comp = {c: sum(getattr(w, c) for w in weeks) for c in COMPONENTS}
    comp["salvage"] = -credit
    return comp


def main(task: str = "small", episodes=(0, 6, 11, 26, 41, 76), entropy: int = 0):
    inst, params = task_generator(task)
    fq_quantiles(inst, params, FQ_REPLICATIONS, cache_dir=CACHE)  # naive's queue table from the kit's disk cache
    fallback = fallback_spec(inst, params, FQ_REPLICATIONS)
    out = []
    for n in episodes:
        t0 = time.time()
        om = scenario(task, n, entropy=entropy)
        row = {"episode": n}
        for name, o in (("real", om), ("event_free", event_free(om, inst))):
            row[name] = {"naive": naive_costs(inst, params, o, fallback), "oracle": oracle_costs(inst, o)}
        out.append(row)
        r, f = row["real"], row["event_free"]
        tot = lambda c: sum(c.values()) / 1e9 if c else float("nan")  # noqa: E731
        print(
            f"ep {n}: real naive {tot(r['naive']):.0f} oracle {tot(r['oracle']):.0f} | event-free naive "
            f"{tot(f['naive']):.0f} oracle {tot(f['oracle']):.0f} bn USD ({time.time() - t0:.0f} s)",
            flush=True,
        )
    keys = COMPONENTS + ("salvage",)
    print(f"\nmean over {len(out)} episodes, bn USD")
    print(
        f"{'component':14s} {'naive':>8s} {'oracle':>8s} {'gap':>8s} | "
        f"{'naive_ef':>8s} {'oracle_ef':>9s} {'gap_ef':>8s}"
    )
    for k in keys + ("total",):
        vals = []
        for sc in ("real", "event_free"):
            for p in ("naive", "oracle"):
                xs = [sum(r[sc][p].values()) if k == "total" else r[sc][p][k] for r in out if r[sc][p]]
                vals.append(np.mean(xs) / 1e9)
        print(
            f"{k:14s} {vals[0]:8.1f} {vals[1]:8.1f} {vals[0] - vals[1]:8.1f} | "
            f"{vals[2]:8.1f} {vals[3]:9.1f} {vals[2] - vals[3]:8.1f}"
        )
    d = ROOT / "outputs" / "eda" / f"costs_{task}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "costs.json").write_text(json.dumps(out, indent=1))
    print(f"written to {d}")


if __name__ == "__main__":
    fire.Fire(main)

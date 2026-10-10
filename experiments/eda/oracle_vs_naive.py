"""Where the clairvoyant plan beats naive: per grid, per market, per fab, per supply node and per route (Small).

    uv run python experiments/eda/oracle_vs_naive.py                     # dev episodes 0 1 2 3 5 6 11 26 41 76

Plays the naive anchor and solves the clairvoyant LP on each episode, then prints the per-episode means: power shed
per grid and unmet demand per market (USD), fab lots started, supply lifted against what was available, and the 30
(route, good) pairs whose flows differ most in value. About 6 s per episode. Prints only.
"""

from collections import defaultdict

import fire
import numpy as np
from cost_decomp import CACHE, FQ_REPLICATIONS  # same folder
from shockbench_flow.dynamics.env import rollout
from shockbench_flow.evaluation.cache import fq_quantiles
from shockbench_flow.hosting.tasks import scenario, task_generator
from shockbench_flow.marks import compute_marks
from shockbench_flow.oracle.lp import build_lp, solve_oracle
from shockbench_flow.policies.naive_fq import anchor_policy, fallback_spec


def main(task: str = "small", episodes=(0, 6, 11, 26, 41, 76, 1, 2, 3, 5), entropy: int = 0):
    inst, params = task_generator(task)
    fq_quantiles(inst, params, FQ_REPLICATIONS, cache_dir=CACHE)
    fb = fallback_spec(inst, params, FQ_REPLICATIONS)
    K = [c.id for c in inst.commodities]
    loss = defaultdict(lambda: [0.0, 0.0])  # (kind, where) -> naive, oracle USD
    weekly = defaultdict(lambda: np.zeros((2, inst.T)))
    starts = defaultdict(lambda: [0.0, 0.0])
    flow = defaultdict(lambda: [0.0, 0.0])
    supply = defaultdict(lambda: [0.0, 0.0, 0.0])  # naive, oracle, available
    for n in episodes:
        om = scenario(task, n, entropy=entropy)
        mk = compute_marks(inst, om)
        tr = rollout(inst, anchor_policy(inst, params, FQ_REPLICATIONS), om, "prediction_free", fallback=fb, marks=mk)
        for r in tr.records:
            t = r.week - 1
            for gi, g in enumerate(inst.grids):
                v = r.shed[gi] * inst.nodes[g].grid.voll
                loss[("shed", inst.nodes[g].id)][0] += v
                weekly["shed"][0, t] += v
            for di, d in enumerate(inst.demands):
                v = (r.lost[di] + r.backlog[di]) * d.pi
                loss[("short", f"{inst.nodes[d.node].id}/{K[d.k]}")][0] += v
                weekly["short"][0, t] += v
            for fi, f in enumerate(inst.fabs):
                starts[inst.nodes[f].id][0] += r.lots_started[fi]
            for (e, k, _lane), q in r.x.items():
                flow[(inst.edges[e].id, K[k])][0] += q
            for s, q in enumerate(r.lift):
                if q:
                    sl = inst.stock_slots[s]
                    supply[(inst.nodes[sl.node].id, K[sl.k])][0] += q
        for s, sl in enumerate(inst.stock_slots):
            if sl.node in inst.supply_nodes:
                supply[(inst.nodes[sl.node].id, K[sl.k])][2] += float(np.asarray(mk.supply)[:, s].sum())
        model = build_lp(inst, mk)
        res = solve_oracle(model)
        for j, key in enumerate(model.keys):
            x = res.x[j]
            if not x:
                continue
            tag, t = key[0], key[1] - 1
            if tag == "ysh":
                g = inst.grids[key[2]]
                v = x * inst.nodes[g].grid.voll
                loss[("shed", inst.nodes[g].id)][1] += v
                weekly["shed"][1, t] += v
            elif tag in ("U", "B"):
                d = inst.demands[key[2]]
                loss[("short", f"{inst.nodes[d.node].id}/{K[d.k]}")][1] += x * d.pi
                weekly["short"][1, t] += x * d.pi
            elif tag == "p":
                starts[inst.nodes[inst.fabs[key[2]]].id][1] += x
            elif tag == "x":
                flow[(inst.edges[key[2]].id, K[key[3]])][1] += x
            elif tag == "lift":
                sl = inst.stock_slots[key[2]]
                supply[(inst.nodes[sl.node].id, K[sl.k])][1] += x
        print(f"episode {n} done", flush=True)
    N = len(episodes)
    print("\nlosses per episode, bn USD: naive, oracle, gap")
    for k, (a, b) in sorted(loss.items(), key=lambda kv: -(kv[1][0] - kv[1][1])):
        print(f"{k[0]:6s} {k[1]:20s} {a / N / 1e9:8.1f} {b / N / 1e9:8.1f} {(a - b) / N / 1e9:8.1f}")
    for c in ("shed", "short"):
        w = weekly[c] / N / 1e9
        q = [(round(w[0][i : i + 13].sum(), 1), round(w[1][i : i + 13].sum(), 1)) for i in range(0, inst.T, 13)]
        print(f"{c} by quarter of the episode (naive, oracle): {q}")
    print("\nfab lots started per episode, k wafers: naive, oracle")
    for k, (a, b) in starts.items():
        print(f"{k:20s} {a / N / 1e3:8.0f} {b / N / 1e3:8.0f}")
    print("\nsupply lifted per episode, k units: naive, oracle, available")
    for k, (a, b, c) in sorted(supply.items()):
        print(f"{k[0]:16s} {k[1]:8s} {a / N / 1e3:9.0f} {b / N / 1e3:9.0f} {c / N / 1e3:9.0f}")
    v = {c.id: c.v for c in inst.commodities}
    print("\nroutes with the largest difference in value shipped, k units per episode")
    for (e, k), (a, b) in sorted(flow.items(), key=lambda kv: -abs(kv[1][1] - kv[1][0]) * v[kv[0][1]])[:30]:
        print(f"{e:45s} {k:12s} naive {a / N / 1e3:9.1f} oracle {b / N / 1e3:9.1f}")


if __name__ == "__main__":
    fire.Fire(main)

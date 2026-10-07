"""Play an agent folder on Full root 12345 and keep, per episode, the weekly cost parts by node, beside the clairvoyant plan's.

uv run python <this> --agent=outputs/recovery_tune/agents/f_e10_w4 --name=f_e10_w4 --episodes=16 --n_jobs=2 [--oracle]
"""

import sys
import time
from pathlib import Path

import fire
import numpy as np


OUT = Path(__file__).resolve().parent / "gap"
TASK, ENTROPY = "full", 12345


def one(n, agent, name, oracle):
    from shockbench_flow.dynamics.env import rollout
    from shockbench_flow.dynamics.state import COST_COMPONENTS
    from shockbench_flow.evaluation.cache import default_cache_dir
    from shockbench_flow.hosting.docker import without_secret_like
    from shockbench_flow.oracle.lp import ORACLE_METHOD, build_lp, lp_costs, solve_oracle
    from shockbench_flow.policies.naive_fq import REPLICATIONS
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _metered_shim, _policy_seed, _world
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    cache = str(default_cache_dir())
    inst, omega, marks, fallback = _world(TASK, ENTROPY, n, REPLICATIONS, cache)
    voll = np.array([inst.nodes[g].grid.voll for g in inst.grids])
    pi = np.array([d.pi for d in inst.demands])
    out = {}
    if agent:
        with without_secret_like():
            factory = load_agent_class(agent, f"submission_{name}")
            shim = _metered_shim(factory, None)
            start = time.perf_counter()
            traj = rollout(inst, shim, omega, "standard", _policy_seed(ENTROPY, n, NO_ZIP_SHA256), marks=marks, fallback=fallback)
            seconds = time.perf_counter() - start
        planner = getattr(shim.agent, "planner", None)
        unload_agent()
        rec = traj.records
        out = {
            "J_cents": np.array(traj.J_cents, dtype=object),
            "salvage": traj.salvage,
            "comps": np.array([[getattr(r.costs, c) for c in COST_COMPONENTS] for r in rec]),
            "shed": np.array([r.shed for r in rec]) * voll,
            "short": np.array([(r.lost + r.backlog) for r in rec]) * pi,
            "stock": np.array([r.stock for r in rec]),
            "disposal": np.array([r.disposal for r in rec]),
            "lots_started": np.array([r.lots_started for r in rec]),
            "served_load": np.array([r.served_load for r in rec]),
            "demand": np.array([r.demand for r in rec]),
            "served": np.array([r.served for r in rec]),
            "seconds": seconds,
            "failed": getattr(planner, "failed", -1),
            "errors": len(shim.errors),
        }
        OUT.mkdir(exist_ok=True)
        np.savez_compressed(OUT / f"{name}_{n}.npz", **out)
    if oracle:
        model = build_lp(inst, marks)
        res = solve_oracle(model, method=ORACLE_METHOD)
        weekly, salvage = lp_costs(model, res.x)
        tmpl, nc, T = model.meta["template"], model.meta["nc"], model.T
        z = res.x.reshape(T, nc)
        S = len(inst.stock_slots)
        stock = np.full((T, S), np.nan)
        disposal = np.zeros((T, S))
        for s in range(S):
            if ("I", s) in tmpl:
                stock[:, s] = z[:, tmpl[("I", s)]]
            if ("O", s) in tmpl:
                disposal[:, s] = z[:, tmpl[("O", s)]]
        short = np.zeros((T, len(inst.demands)))
        served = np.zeros((T, len(inst.demands)))
        for do, d in enumerate(inst.demands):
            short[:, do] = z[:, tmpl[("B" if d.backlog else "U", do)]] * d.pi
            served[:, do] = z[:, tmpl[("D", do)]]
        ora = {
            "J_cents": np.array(res.J_cents, dtype=object),
            "salvage": salvage,
            "comps": np.array([[getattr(w, c) for c in COST_COMPONENTS] for w in weekly]),
            "shed": np.array([z[:, tmpl[("ysh", go)]] for go in range(len(inst.grids))]).T * voll,
            "short": short,
            "stock": stock,
            "disposal": disposal,
            "lots_started": np.array([z[:, tmpl[("p", fo)]] for fo in range(len(inst.fabs))]).T,
            "served": served,
            "status": res.status,
        }
        OUT.mkdir(exist_ok=True)
        np.savez_compressed(OUT / f"oracle_{n}.npz", **ora)
    return n, (int(out["J_cents"]) if out else None), (out.get("seconds") if out else None)


def main(agent="", name="", episodes=16, n_jobs=2, oracle=False, only=None):
    from joblib import Parallel, delayed

    ns = list(range(episodes)) if only is None else [int(x) for x in (only if isinstance(only, (list, tuple)) else [only])]
    agent = str(Path(agent).resolve()) if agent else ""
    rows = Parallel(n_jobs=n_jobs)(delayed(one)(n, agent, name, oracle) for n in ns)
    for row in rows:
        print(row, flush=True)


if __name__ == "__main__":
    fire.Fire(main)

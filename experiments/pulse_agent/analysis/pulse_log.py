"""One grid's weeks under the pulse: fuel at the terminal and the grid, arrivals, what the fabs wanted and got.

uv run python <this> --agent=pulse --episode=3 --grid=grid_jp
"""

from pathlib import Path

import fire
import numpy as np


ROOT = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026")
HERE = Path(__file__).resolve().parent


def folder(name):
    for base in (ROOT / "agents", ROOT / "outputs" / "recovery_tune" / "agents", HERE / "agents"):
        if (base / name / "agent.py").is_file():
            return str(base / name)
    raise ValueError(name)


def main(agent="pulse", episode=3, grid="grid_jp", task="full", entropy=12345, first=1, last=70):
    from shockbench_flow.dynamics.env import rollout
    from shockbench_flow.evaluation.cache import default_cache_dir
    from shockbench_flow.hosting.docker import without_secret_like
    from shockbench_flow.policies.naive_fq import REPLICATIONS
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _metered_shim, _policy_seed, _world
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    cache = str(default_cache_dir())
    inst, omega, marks, fallback = _world(task, entropy, episode, REPLICATIONS, cache)
    N, K, S, E = inst.nodes, inst.commodities, inst.stock_slots, inst.edges
    with without_secret_like():
        factory = load_agent_class(folder(agent), f"submission_{agent}")
        shim = _metered_shim(factory, None)
        traj = rollout(inst, shim, omega, "standard", _policy_seed(entropy, episode, NO_ZIP_SHA256), marks=marks, fallback=fallback)
    log = list(getattr(getattr(shim.agent, "pulse", None), "log", []))
    failed = getattr(shim.agent, "pulse_failed", None)
    unload_agent()
    rec = traj.records
    g = inst.node_index[grid]
    go = inst.grid_ordinal[g]
    ga = N[g].grid
    fabs = inst.grid_fabs[go]
    by_week = {(t, gg): (mode, moved) for t, gg, mode, moved in log}
    print(f"episode {episode} {agent} {grid}: J {traj.J_cents / 1e11:.1f} bn, pulse failures {failed}, weeks the pulse changed a flow {len(log)}")
    feeders = {}
    for k in ga.fuels:
        feeders[k] = [e for e in inst.in_edges[g] if k in E[e].K and E[e].tau == 0]
    head = "week mode  want  got  shed "
    for k in ga.fuels:
        head += f"| {K[k].id}: tank(wk) in(wk) sent(wk) grid(wk) burnt "
    print(head)
    for t in range(first, min(last, inst.T) + 1):
        r = rec[t - 1]
        mode = by_week.get((t, go), ("-", {}))[0]
        want = sum(N[inst.fabs[fo]].fab.e * marks.alpha_bar[t - 1, fo] * N[inst.fabs[fo]].fab.cap0 for fo in fabs)
        got = float(sum(r.energy[fo] for fo in fabs))
        line = f"{t:4d} {mode:4s} {want:5.0f} {got:5.0f} {r.shed[go]:6.0f} "
        for k in ga.fuels:
            burn = ga.shares[k] * ga.deliverable
            tank = sum(r.stock[inst.slot_index[(E[e].tail, k)]] for e in feeders[k]) / burn
            sent = sum(q for (e, kk, _l), q in r.x.items() if kk == k and e in feeders[k]) / burn
            arrive = sum(q for (e, kk, _l), q in rec[t - 1].x.items() if kk == k and E[e].head in {E[f].tail for f in feeders[k]}) / burn
            line += f"| {tank:6.2f} {arrive:5.2f} {sent:5.2f} {r.stock[inst.slot_index[(g, k)]] / burn:5.2f} {r.segment.get((go, k), 0.0) / burn:5.2f} "
        print(line)


if __name__ == "__main__":
    fire.Fire(main)

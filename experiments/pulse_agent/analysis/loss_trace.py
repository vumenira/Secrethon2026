"""One episode, two agents: base load shed per grid, lots per fab, and what the pulse did, to explain a loss.

uv run python <this> --a=ft_s_q_e20_fd8 --b=t3a_e_w4 --task=small --episode=3
"""

from collections import Counter
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


def main(a="ft_s_q_e20_fd8", b="t3a_e_w4", task="small", episode=3, entropy=12345, weekly="", first=1, last=52):
    from shockbench_flow.dynamics.env import rollout
    from shockbench_flow.evaluation.cache import default_cache_dir
    from shockbench_flow.hosting.docker import without_secret_like
    from shockbench_flow.policies.naive_fq import REPLICATIONS
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _metered_shim, _policy_seed, _world
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    cache = str(default_cache_dir())
    inst, omega, marks, fallback = _world(task, entropy, episode, REPLICATIONS, cache)
    N, K, E = inst.nodes, inst.commodities, inst.edges
    out = {}
    for name in (a, b):
        with without_secret_like():
            factory = load_agent_class(folder(name), f"submission_{name}")
            shim = _metered_shim(factory, None)
            traj = rollout(inst, shim, omega, "standard", _policy_seed(entropy, episode, NO_ZIP_SHA256), marks=marks, fallback=fallback)
        log = list(getattr(getattr(shim.agent, "pulse", None), "log", []) or [])
        failed = getattr(shim.agent, "pulse_failed", None)
        unload_agent()
        out[name] = (traj, log, failed)
        print(f"{name}: J {traj.J_cents / 1e11:.1f} bn, errors {len(shim.errors)}, pulse failures {failed}, weeks the pulse changed a flow {len(log)}")
    ta, tb = out[a][0], out[b][0]
    T = inst.T
    print("base load shed (bn USD) and fab energy (GWh), whole episode: a | b")
    for go, g in enumerate(inst.grids):
        voll = N[g].grid.voll
        sa = sum(r.shed[go] for r in ta.records) * voll / 1e9
        sb = sum(r.shed[go] for r in tb.records) * voll / 1e9
        fabs = inst.grid_fabs[go]
        ea = sum(float(sum(r.energy[fo] for fo in fabs)) for r in ta.records)
        eb = sum(float(sum(r.energy[fo] for fo in fabs)) for r in tb.records)
        modes = Counter((mode, tuple(sorted(K[k].id for k in moved))) for t, gg, mode, moved in out[a][1] if gg == go)
        print(f"  {N[g].id:9s} shed {sa:8.1f} | {sb:8.1f} ({sa - sb:+7.1f})   fab energy {ea:8.0f} | {eb:8.0f}   pulse weeks {dict(modes)}")
    sha = sum(r.costs.shortage for r in ta.records) / 1e9
    shb = sum(r.costs.shortage for r in tb.records) / 1e9
    print(f"unserved demand bn: {sha:.1f} | {shb:.1f} ({sha - shb:+.1f}); shed bn: {sum(r.costs.shed for r in ta.records) / 1e9:.1f} | {sum(r.costs.shed for r in tb.records) / 1e9:.1f};"
          f" other bn: {sum(r.costs.total() - r.costs.shortage - r.costs.shed for r in ta.records) / 1e9:.1f} | {sum(r.costs.total() - r.costs.shortage - r.costs.shed for r in tb.records) / 1e9:.1f}")
    if weekly:
        g = inst.node_index[weekly]
        go = inst.grid_ordinal[g]
        ga = N[g].grid
        fabs = inst.grid_fabs[go]
        by_week = {t: (mode, moved) for t, gg, mode, moved in out[a][1] if gg == go}
        feeders = {k: [e for e in inst.in_edges[g] if k in E[e].K and E[e].tau == 0] for k in ga.fuels}
        print(f"{weekly}: week, pulse mode, G/G0 | a: fab energy, shed, then per fuel tank(wk) sent(wk) grid stock(wk) burnt | b: fab energy, shed, per fuel grid stock(wk) burnt")
        for t in range(first, min(last, T) + 1):
            r, q = ta.records[t - 1], tb.records[t - 1]
            line = f"{t:4d} {by_week.get(t, ('-', {}))[0]:4s} {marks.G_bar[t - 1, go] / ga.deliverable:5.2f} | {float(sum(r.energy[fo] for fo in fabs)):5.0f} {r.shed[go]:6.0f} "
            for k in ga.fuels:
                burn = ga.shares[k] * ga.deliverable
                tank = sum(r.stock[inst.slot_index[(E[e].tail, k)]] for e in feeders[k]) / burn
                sent = sum(x for (e, kk, _l), x in r.x.items() if kk == k and e in feeders[k]) / burn
                line += f"[{K[k].id[:3]} {tank:5.2f} {sent:5.2f} {r.stock[inst.slot_index[(g, k)]] / burn:5.2f} {r.segment.get((go, k), 0.0) / burn:4.2f}] "
            line += f"| {float(sum(q.energy[fo] for fo in fabs)):5.0f} {q.shed[go]:6.0f} "
            for k in ga.fuels:
                burn = ga.shares[k] * ga.deliverable
                line += f"[{K[k].id[:3]} {q.stock[inst.slot_index[(g, k)]] / burn:5.2f} {q.segment.get((go, k), 0.0) / burn:4.2f}] "
            print(line)


if __name__ == "__main__":
    fire.Fire(main)

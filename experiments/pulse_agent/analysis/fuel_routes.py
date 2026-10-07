"""Where a fuel goes in an agent's episode: supply, prohibited edges, flow per edge, arrivals per grid against burn.

uv run python <this> --agent=f_e10_w4 --episode=0 --good=crude
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


def main(agent="f_e10_w4", episode=0, good="crude", first=14, last=60, task="full", entropy=12345):
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
    k = inst.commodity_index[good]
    with without_secret_like():
        factory = load_agent_class(folder(agent), f"submission_{agent}")
        shim = _metered_shim(factory, None)
        traj = rollout(inst, shim, omega, "standard", _policy_seed(entropy, episode, NO_ZIP_SHA256), marks=marks, fallback=fallback)
    unload_agent()
    rec = traj.records
    sl = slice(first - 1, last)
    weeks = last - first + 1
    print(f"episode {episode}, {agent}, {good}, weeks {first}..{last}: per week means")
    print("supply slots: nominal, available (mean), lifted (mean), stock at the end")
    for s, st in enumerate(S):
        if st.k == k and st.node in inst.supply_nodes:
            avail = float(np.mean(marks.supply[sl, s]))
            lift = float(np.mean([r.lift[s] for r in rec[sl]]))
            print(f"  {N[st.node].id:22s} nominal {st.supply:9.0f} avail {avail:9.0f} lifted {lift:9.0f} stock {rec[last - 1].stock[s]:9.0f}")
    flow = {}
    for r in rec[sl]:
        for (e, kk, lane), q in r.x.items():
            if kk == k:
                flow[e] = flow.get(e, 0.0) + q / weeks
    print("edges carrying it or prohibited for it: flow per week, capacity (mean), share of weeks prohibited, lead")
    for e, ed in enumerate(E):
        if k not in ed.K:
            continue
        banned = float(np.mean(marks.prohibited[sl, e, k]))
        f = flow.get(e, 0.0)
        if f > 1e-6 or banned > 0:
            cap = float(np.mean(marks.u[sl, e]))
            print(f"  {ed.id:44s} {N[ed.tail].type[:5]:5s}->{N[ed.head].type[:5]:5s} flow {f:9.0f} cap {cap:9.0f} (nominal {ed.u0:9.0f}) banned {banned:.2f} tau {ed.tau}")
    print("chokepoints: open share (mean), throughput tb (mean / nominal)")
    for ci, c in enumerate(inst.chokepoints):
        print(f"  {N[c].id:16s} open {float(np.mean(marks.o[sl, ci])):.2f} kappa_tb {float(np.mean(marks.kappa[sl, ci, 0])):9.0f} nominal mu {N[c].chokepoint.mu[0]:9.0f}"
              f" queue of it at the end {sum(q for (cc, kk, _l), q in rec[last - 1].queue.items() if cc == c and kk == k):9.0f}")
    print("grids: burn at full rate, arrivals per week, burnt per week, stock at the end (weeks)")
    for go, g in enumerate(inst.grids):
        ga = N[g].grid
        if k not in ga.fuels:
            continue
        burn = ga.shares[k] * ga.deliverable
        arr = sum(flow.get(e, 0.0) for e in inst.in_edges[g] if k in E[e].K)
        burnt = float(np.mean([r.segment.get((go, k), 0.0) for r in rec[sl]]))
        s = inst.slot_index[(g, k)]
        print(f"  {N[g].id:9s} burn {burn:8.0f} arrivals {arr:8.0f} burnt {burnt:8.0f} ({burnt / burn:.2f}) stock {rec[last - 1].stock[s] / burn:5.2f} wk")
    print("terminals: stock at the end, storage")
    for s, st in enumerate(S):
        if st.k == k and N[st.node].type == "terminal":
            print(f"  {N[st.node].id:10s} stock {rec[last - 1].stock[s]:9.0f} storage {st.storage:9.0f}")
    pool = inst.commodity_pool[k]
    print("fleet measure of the pool:", inst.params.fleet_measure[pool], "share", inst.params.fleet_share[pool])


if __name__ == "__main__":
    fire.Fire(main)

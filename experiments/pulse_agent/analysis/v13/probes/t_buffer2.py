"""Why the sink buffer adds nothing: its terms at some weeks of a Full episode."""
import sys, time, json
from collections import defaultdict
import numpy as np
import gymnasium as gym
import shockbench_flow_gym  # noqa: F401
from shockbench_flow_gym import agent_config_from_reset
from sbf_starter import env_id
folder, task, episode = sys.argv[1], sys.argv[2], int(sys.argv[3])
look = [int(a) for a in sys.argv[4:]]
sys.path.insert(0, folder)
import agent as A
from sbflow.policies import lp_common as L
A.PARAMS.update({"sink_buffer": True, "strict": True})
env = gym.make(env_id(task)); obs, info = env.reset(options={"episode": episode})
ag = A.Agent(agent_config_from_reset(env, obs, info))
for w in range(1, max(look) + 1):
    a = ag.act(obs)
    if w in look:
        pl, B = ag.planner, ag.buffer
        inst = pl._inst; N, K = inst.nodes, inst.commodities
        wobs = ag.wire(obs); t = int(wobs["week"])
        plan = pl.last[:2]; model, x = plan; tm, nc = model.meta["template"], int(model.meta["nc"])
        fc = wobs["demand_forecast"]; ahead = {(n, k, h): q for n, k, h, q in zip(fc["node"], fc["k"], fc["h"], fc["qty"])}
        stock = {(n, k): q for n, k, q in zip(wobs["stock"]["node"], wobs["stock"]["k"], wobs["stock"]["qty"])}
        now, soon = defaultdict(float), defaultdict(float)
        pp = wobs["pipeline"]
        for e, k, q, week in zip(pp["edge"], pp["k"], pp["qty"], pp["arrival_week"]):
            (now if week == t else soon if week == t + 1 else defaultdict(float))[(inst.edges[e].head, k)] += q
        sent = {s: float(q) for s, q in enumerate(a["flows"])}
        vals = pl._memory.values
        print(f"--- week {t}")
        for node, k, s, share, due, ways in B.sinks:
            f0, f1 = ahead[(node, k, 0)], ahead[(node, k, 1)]
            left = max(stock.get((node, k), 0) + now.get((node, k), 0) - f0, 0)
            there = left + soon.get((node, k), 0) + sum(sent[x] for x in due)
            need = (1 + share) * f1 - there
            do = [i for i, d in enumerate(inst.demands) if (d.node, d.k) == (node, k)][0]
            U2 = x[1 * nc + tm[("U", do)]]
            parts = []
            for c, e, tail, ss in ways:
                spare = vals["u"][e] - sum(sent[y] for y in B.edge_slots[e])
                idle = B.unused(plan, tail, k)
                st = stock.get((tail, k), 0) - sum(sent[y] for y in B.tail_slots[(tail, k)])
                parts.append(f"{N[tail].id}: edge spare {spare/f1:5.2f} idle {idle/f1:5.2f} stock left {st/f1:5.2f}")
            print(f"  {N[node].id:9s} {K[k].id:9s} forecast next {f1:9.0f}; there next week {there/f1:5.2f} of it; need {need/f1:5.2f}; plan loses next week {U2/f1:5.2f} | " + "; ".join(parts[:4]))
    obs, r, term, trunc, info = env.step(a)

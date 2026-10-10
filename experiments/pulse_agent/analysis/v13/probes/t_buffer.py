"""Play some weeks of a Full episode with the sink buffer on and print what it adds."""
import sys, time, json
import numpy as np
import gymnasium as gym
import shockbench_flow_gym  # noqa: F401
from shockbench_flow_gym import agent_config_from_reset
from sbf_starter import env_id
folder, task, episode, weeks = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
sys.path.insert(0, folder)
import agent as A
A.PARAMS.update(json.loads(sys.argv[5]) if len(sys.argv) > 5 else {})
A.PARAMS["strict"] = True
env = gym.make(env_id(task)); obs, info = env.reset(options={"episode": episode})
ag = A.Agent(agent_config_from_reset(env, obs, info))
total = lost = 0.0; cpu = 0.0
for w in range(1, weeks + 1):
    c = time.process_time(); a = ag.act(obs); cpu += time.process_time() - c
    obs, r, term, trunc, info = env.step(a); total += r
    if term or trunc: break
inst = ag.planner._inst
N, K = inst.nodes, inst.commodities
log = ag.buffer.log if ag.buffer is not None else []
print(f"{w} weeks, cost {-total/1e9:.2f} bn, cpu {cpu:.1f} s; buffer top-ups {len(log)} in {len({l[0] for l in log})} weeks")
by = {}
for t, node, k, q in log:
    key = (N[node].id, K[k].id); by[key] = by.get(key, 0.0) + q
dem = {(N[d.node].id, K[d.k].id): d.dbar for d in inst.demands}
for key, q in sorted(by.items(), key=lambda kv: -kv[1]):
    print(f"   {key[0]:10s} {key[1]:9s} added {q:12,.0f} = {q/dem[key]:6.2f} weeks of base demand over the play")
print("   sinks served:", [(N[n].id, K[k].id, round(sh, 3), len(ways)) for n, k, s, sh, due, ways in (ag.buffer.sinks if ag.buffer else [])][:20])

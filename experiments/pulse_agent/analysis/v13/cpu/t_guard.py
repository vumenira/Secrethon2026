"""The guard's path: cut the solve in given weeks and watch what is played; then a real limit that is too short."""
import sys, time, json
import numpy as np
import gymnasium as gym
import shockbench_flow_gym  # noqa: F401
from shockbench_flow_gym import agent_config_from_reset
from sbf_starter import env_id

folder, task, episode, weeks = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
test = json.loads(sys.argv[5]) if len(sys.argv) > 5 else []
budget = float(sys.argv[6]) if len(sys.argv) > 6 else None
sys.path.insert(0, folder)
import agent as A  # noqa: E402
from sbflow.policies import naive as N  # noqa: E402
A.PARAMS["guard_test"] = test
A.PARAMS["guard_budget"] = budget
env = gym.make(env_id(task))
obs, info = env.reset(options={"episode": episode})
ag = A.Agent(agent_config_from_reset(env, obs, info))
print(f"plans in the cache before the first act: {len(N._PLANS)}; budget {ag.budget}")
total = 0.0
for w in range(1, weeks + 1):
    c = time.process_time(); action = ag.act(obs); dt = time.process_time() - c
    tel = ag.planner.telemetry[-1]
    mark = "" if not tel.fallback else f"  <- not solved: {tel.trail}; weeks played from the last plan so far {ag.planner.guard_weeks}"
    print(f"week {w:3d}: cpu {dt:5.2f} s, iterations {tel.iterations:6d}, flows sum {action['flows'].sum():14.0f}, basis {ag.planner._session._col is not None} catching up {ag.planner._session.catching_up}{mark}", flush=True)
    obs, reward, term, trunc, info = env.step(action); total += reward
    if term or trunc: break
print(f"plans in the cache after: {len(N._PLANS)}; cost of these weeks {-total/1e9:.1f} bn; errors: pulse {ag.pulse_failed}, forecast/reserve {ag.planner.failed}")

"""CPU of Agent(config) + the first act, per episode (the plan cache emptied each time, as in a fresh container)."""
import sys, time, ctypes
if sys.platform == "win32":
    k = ctypes.windll.kernel32
    k.SetProcessAffinityMask(k.GetCurrentProcess(), 0b0100)
    k.SetPriorityClass(k.GetCurrentProcess(), 0x80)
import numpy as np
import gymnasium as gym
import shockbench_flow_gym  # noqa: F401
from shockbench_flow_gym import agent_config_from_reset
from sbf_starter import env_id

folder, task = sys.argv[1], sys.argv[2]
episodes = [int(a) for a in sys.argv[3:]]
sys.path.insert(0, folder)
import agent as A  # noqa: E402
from sbflow.policies import naive as N  # noqa: E402
env = gym.make(env_id(task))
rows = []
for n in episodes:
    obs, info = env.reset(options={"episode": n})
    config = agent_config_from_reset(env, obs, info)
    N._PLANS.clear()
    c, w = time.process_time(), time.perf_counter()
    ag = A.Agent(config); out = ag.act(obs)
    dt, dw = time.process_time() - c, time.perf_counter() - w
    tel = ag.planner.telemetry[-1]
    # second week, for the warm solve
    obs2, *_ = env.step(out)
    c = time.process_time(); ag.act(obs2); d2 = time.process_time() - c
    rows.append((n, dt, dw, tel.iterations, d2, ag.planner.telemetry[-1].iterations))
    print(f"episode {n:2d}: week 1 cpu {dt:5.2f} s (wall {dw:5.2f}), solve iterations {tel.iterations:6d}; week 2 cpu {d2:5.2f} s, iterations {rows[-1][5]}", flush=True)
a = np.array([r[1] for r in rows]); it = np.array([r[3] for r in rows])
print(f"week 1 cpu: median {np.median(a):.2f}, max {a.max():.2f}, min {a.min():.2f}; iterations median {np.median(it):.0f}, max {it.max()}")

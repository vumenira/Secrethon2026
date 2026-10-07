"""Process time of ``act`` per week of several agent folders on the same Full episodes, in one process.

uv run python <this> --agents=mpc_det_h20,ft_x,ft_y --episodes=0,5

A name is a folder of agents/ or of outputs/recovery_tune/agents/. The first is the base the others are compared with.
``Agent(config)`` is charged to week 1, as the server's meter does. Agents are played in turn on each episode, so a
busy machine slows them alike (process time counts this process's CPU only, but shared caches still cost).
"""

import json
import time
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


def main(agents, episodes=(0, 5), task="full", entropy=12345, out="cpu.json"):
    from shockbench_flow.dynamics.env import rollout
    from shockbench_flow.evaluation.cache import default_cache_dir
    from shockbench_flow.hosting.docker import without_secret_like
    from shockbench_flow.policies.naive_fq import REPLICATIONS
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _metered_shim, _policy_seed, _world
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    names = list(agents) if isinstance(agents, (list, tuple)) else [a for a in str(agents).split(",") if a]
    ns = [int(n) for n in (episodes if isinstance(episodes, (list, tuple)) else [episodes])]
    cache = str(default_cache_dir())
    times = {name: [] for name in names}
    costs = {name: [] for name in names}
    for n in ns:
        inst, omega, marks, fallback = _world(task, entropy, n, REPLICATIONS, cache)
        for name in names:
            with without_secret_like():
                factory = load_agent_class(folder(name), f"submission_{name}")
                shim = _metered_shim(factory, None)
                weeks, carry = [], [0.0]
                reset, act = shim.reset, shim.act

                def timed_reset(static, obs, seed, reset=reset, carry=carry):
                    start = time.process_time()
                    reset(static, obs, seed)
                    carry[0] = time.process_time() - start

                def timed_act(obs, act=act, weeks=weeks, carry=carry):
                    start = time.process_time()
                    action = act(obs)
                    weeks.append(carry[0] + time.process_time() - start)
                    carry[0] = 0.0
                    return action

                shim.reset, shim.act = timed_reset, timed_act
                traj = rollout(inst, shim, omega, "standard", _policy_seed(entropy, n, NO_ZIP_SHA256), marks=marks, fallback=fallback)
            failed = getattr(getattr(shim.agent, "planner", None), "failed", None)
            unload_agent()
            times[name].append(weeks)
            costs[name].append(traj.J_cents)
            w = np.array(weeks)
            print(f"episode {n} {name:28s} mean {w.mean():.3f} s  median {np.median(w):.3f}  slowest {w.max():.3f} (week {int(w.argmax()) + 1})"
                  f"  week 1 {w[0]:.3f}  errors {len(shim.errors)} failed {failed}  J {traj.J_cents}", flush=True)
    base = np.concatenate([np.array(w) for w in times[names[0]]])
    print("\nall episodes together, relative to", names[0])
    for name in names:
        w = np.concatenate([np.array(x) for x in times[name]])
        print(f"  {name:28s} mean {w.mean():.3f} s ({w.mean() / base.mean():.2f}x)  slowest {w.max():.3f} s ({w.max() / base.max():.2f}x)"
              f"  99th pct {np.quantile(w, 0.99):.3f} ({np.quantile(w, 0.99) / np.quantile(base, 0.99):.2f}x)")
    (HERE / out).write_text(json.dumps({"episodes": ns, "times": times, "costs": costs}))


if __name__ == "__main__":
    fire.Fire(main)

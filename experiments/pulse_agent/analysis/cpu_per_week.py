"""The agent's own CPU per week (process time around act), for variants of the tuning harness, on the tuning root.

uv run python cpu_per_week.py --names=mpc_det_h20,t3b_h24 --episodes=0,2

Each variant's folder is outputs/recovery_tune/agents/<name> (the harness's copy with its params.json), else
agents/<name>. Every variant plays the same episodes of Small, root 12345, one after another in this process, so the
ratios do not depend on the simulator's time; they still move a little with the machine's load.
"""

import sys
import time
from pathlib import Path

import fire
import numpy as np


ROOT = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026")


def main(names, episodes="0,2", task="small", entropy=12345, n_set=64):
    from shockbench_flow.dynamics.env import rollout, took_fallback
    from shockbench_flow.hosting.docker import without_secret_like
    from shockbench_flow_agent import EpisodeSet
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _policy_seed, _world
    from shockbench_flow_agent.shim import AgentShim, load_agent_class, unload_agent

    names = [names] if isinstance(names, str) else list(names)
    eps = [int(e) for e in (episodes if isinstance(episodes, (tuple, list)) else str(episodes).split(","))]
    if n_set:  # the harness's episode set (its references are cached for Small)
        es = EpisodeSet.build(task, n_set, entropy=entropy, n_jobs=1)
        _task, _entropy, regime, fq, cache = es._spec
    else:  # no references needed to time an agent: the world of an episode alone (Full)
        from shockbench_flow.evaluation.cache import default_cache_dir
        from shockbench_flow.policies.naive_fq import REPLICATIONS

        regime, fq, cache = "standard", REPLICATIONS, str(default_cache_dir())

    class Metered(AgentShim):
        def __init__(self, factory):
            super().__init__(factory)
            self.cpu, self.reset_cpu = [], 0.0

        def reset(self, static, obs, policy_seed):
            start = time.process_time()
            super().reset(static, obs, policy_seed)
            self.reset_cpu = time.process_time() - start

        def act(self, obs):
            start = time.process_time()
            action = super().act(obs)
            self.cpu.append(time.process_time() - start)
            return action

    base = None
    print(f"{task}, root {entropy}, episodes {eps}: the agent's CPU seconds per week (process time around act)")
    for name in names:
        root = ROOT / "outputs" / "recovery_tune" / "agents" / name
        if not root.is_dir():
            root = ROOT / "agents" / name
        cpu, solve, iters, sizes, costs, fell, failed, errors = [], [], [], [], [], 0, 0, 0
        for n in eps:
            inst, omega, marks, fallback = _world(task, entropy, n, fq, cache)
            with without_secret_like():
                factory = load_agent_class(str(root), f"submission_{name}")
                shim = Metered(factory)
                traj = rollout(inst, shim, omega, regime, _policy_seed(entropy, n, NO_ZIP_SHA256), marks=marks,
                               fallback=fallback)
            planner = getattr(shim.agent, "planner", None)
            week = np.array(shim.cpu)
            week[0] += shim.reset_cpu  # Agent(config) counts toward week 1
            cpu.extend(week.tolist())
            if planner is not None:
                solve.extend(s.seconds for s in planner.telemetry)
                iters.extend(s.iterations for s in planner.telemetry)
                failed += getattr(planner, "failed", 0) or 0
            fell += sum(took_fallback(r) for r in traj.records)
            errors += len(shim.errors)
            costs.append(traj.J_cents)
            unload_agent()
        cpu, iters = np.array(cpu), np.array(iters if iters else [0])
        if base is None:
            base = cpu
        print(
            f"{name:22s} mean {cpu.mean():.3f}  median {np.median(cpu):.3f}  p95 {np.quantile(cpu, 0.95):.3f}"
            f"  max {cpu.max():.3f}  | ratio to first: mean x{cpu.mean() / base.mean():.2f}"
            f"  max x{cpu.max() / base.max():.2f}  | simplex iterations per week mean {iters.mean():.0f}"
            f" max {iters.max()}  | fallback weeks {fell} reserve failures {failed} errors {errors}"
            f"  | J {[int(c) for c in costs]}",
            flush=True,
        )


if __name__ == "__main__":
    sys.exit(fire.Fire(main))

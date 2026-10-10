"""CPU of the planner's week with and without the strait reserve, on the same episodes, one after the other in one
process (process time, which is what the server meters; the machine's load moves wall time, not this).

uv run python outputs/recovery_tune/tools/with_lock.py uv run python outputs/w13a/cpu.py --eps="12345:0,777:30"
"""
import json
import shutil
import sys
import time
from pathlib import Path

import fire
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CONFIGS = {"off": {}, "s26": {"strait_reserve": True, "strait_weeks": 26.0}}


def play(folder, task, root, n, label):
    from shockbench_flow.dynamics.env import rollout
    from shockbench_flow.evaluation.cache import default_cache_dir
    from shockbench_flow.hosting.docker import without_secret_like
    from shockbench_flow.policies.naive_fq import REPLICATIONS
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _metered_shim, _policy_seed, _world
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    inst, omega, marks, fallback = _world(task, root, n, REPLICATIONS, str(default_cache_dir()))
    cpu, its = [], []
    with without_secret_like():
        factory = load_agent_class(str(folder), f"w13a_cpu_{label}_{root}_{n}")
        mod = sys.modules[factory.__module__]
        plain = mod.Agent.act

        def act(self, observation):
            start = time.process_time()
            out = plain(self, observation)
            cpu.append(time.process_time() - start)
            return out

        mod.Agent.act = act
        shim = _metered_shim(factory, None)
        traj = rollout(inst, shim, omega, "standard", _policy_seed(root, n, NO_ZIP_SHA256), marks=marks, fallback=fallback)
        its = [int(s.iterations) for s in shim.agent.planner.telemetry]
        guard = int(shim.agent.planner.guard_weeks)
    unload_agent()
    cpu = np.array(cpu)
    print(f"{label:4s} {root}/{n}: J {traj.J_cents / 1e11:,.1f} bn; CPU per week: first {cpu[0]:.2f} s, second {cpu[1]:.2f}, median {np.median(cpu):.2f},"
          f" mean {cpu.mean():.2f}, max after week 1 {cpu[1:].max():.2f} (week {int(np.argmax(cpu[1:])) + 2}); simplex iterations: first week {its[0]},"
          f" mean {np.mean(its):.0f}; guard weeks {guard}; errors {len(shim.errors)}", flush=True)


def main(eps, task="full"):
    for label, params in CONFIGS.items():
        folder = HERE / "agents" / f"cpu_{label}"
        if folder.exists():
            shutil.rmtree(folder)
        shutil.copytree(ROOT / "agents" / "mpc_pulse_rec_v13w", folder, ignore=shutil.ignore_patterns("__pycache__", "params.json"))
        (folder / "params.json").write_text(json.dumps({"strict": True} | params))
    for item in str(eps).split(","):
        root, n = (int(v) for v in item.split(":"))
        for label in CONFIGS:
            play(HERE / "agents" / f"cpu_{label}", task, root, n, label)


if __name__ == "__main__":
    sys.exit(fire.Fire(main))

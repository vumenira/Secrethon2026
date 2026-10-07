"""Process time of ``Agent.act`` per week for two agent folders, played one after the other in this process.

Also (``strict``): the folder at its defaults with ``strict`` on and off, cost compared to the cent.

uv run python cpu.py --task=full --only="[0,3]"
uv run python cpu.py strict --task=small --only="[0,25]"
"""

import sys
import time
from pathlib import Path

import fire
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402


def play(task, n, folder, params, entropy=12345):
    from shockbench_flow.dynamics.env import rollout
    from shockbench_flow.hosting.docker import without_secret_like
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _metered_shim, _policy_seed
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    inst, omega, marks, fallback = C.world(task, n, entropy)
    cpu = []
    with without_secret_like():
        factory = load_agent_class(str(C.ROOT / "agents" / folder), "submission_cpu")
        module = sys.modules[factory.__module__]
        module.PARAMS.update(params)
        orig_init, orig_act = module.Agent.__init__, module.Agent.act
        box = {"init": 0.0}

        def init(self, config=None):
            t0 = time.process_time()
            orig_init(self, config)
            box["init"] = time.process_time() - t0

        def act(self, observation):
            t0 = time.process_time()
            out = orig_act(self, observation)
            cpu.append(time.process_time() - t0 + (box.pop("init", 0.0) if "init" in box else 0.0))
            return out

        module.Agent.__init__, module.Agent.act = init, act
        shim = _metered_shim(factory, None)
        traj = rollout(inst, shim, omega, "standard", _policy_seed(entropy, n, NO_ZIP_SHA256), marks=marks, fallback=fallback)
        errors = list(shim.errors)
    agent = shim.agent
    failed = (agent.planner.failed, agent.pulse_failed)
    unload_agent()
    return int(traj.J_cents), np.array(cpu), errors, failed


def main(task="full", only=(0, 3), a="mpc_pulse_rec_v4", b="mpc_pulse_rec_v6", rounds=1):
    ns = [only] if isinstance(only, int) else [int(v) for v in only]
    acc = {a: [], b: []}
    for _ in range(int(rounds)):
        for n in ns:
            for folder in (a, b):
                J, cpu, errors, failed = play(task, n, folder, {})
                acc[folder].append(cpu)
                print(f"{task} episode {n} {folder}: act process time per week mean {cpu.mean():.3f} s, median {np.median(cpu):.3f} s, slowest {cpu.max():.3f} s (week {int(cpu.argmax()) + 1}; week 1 with Agent(config) {cpu[0]:.3f} s); errors {errors[:1]} failures {failed}", flush=True)
    for folder, runs in acc.items():
        allw = np.concatenate(runs)
        print(f"{folder}: {len(allw)} weeks, mean {allw.mean():.3f} s, slowest {allw.max():.3f} s")
    ra, rb = np.concatenate(acc[a]), np.concatenate(acc[b])
    print(f"{b} over {a}: mean {rb.mean() / ra.mean():.3f}, slowest {rb.max() / ra.max():.3f}")


def strict(task="small", only=(0, 25), folder="mpc_pulse_rec_v6"):
    ns = [only] if isinstance(only, int) else [int(v) for v in only]
    for n in ns:
        Ja, _c, ea, fa = play(task, n, folder, {"strict": True})
        Jb, _c, eb, fb = play(task, n, folder, {"strict": False})
        print(f"{task} episode {n}: strict {Ja} cents, strict off {Jb} cents: {'equal' if Ja == Jb else 'DIFFERENT'}; errors {ea[:1]} {eb[:1]}; failures (planner, pulse) {fa} {fb}", flush=True)


if __name__ == "__main__":
    fire.Fire({"main": main, "strict": strict})

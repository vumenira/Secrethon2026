"""CPU per week of the package's mpc_scen at given S (its budgets on the scoring host are documented) and of an agent.

On the scoring host mpc_scen fits 1 s per week on small at S = 3 (p95 of the weekly act CPU, every measured episode)
and not at S = 4 (shockbench_flow/policies/mpc_scen.py, S_BUDGET). The same numbers here bound the host's speed.
"""
import ctypes, sys, time
import numpy as np

if sys.platform == "win32":  # one P-core, high priority: the machine is busy with evaluations
    k = ctypes.windll.kernel32
    k.SetProcessAffinityMask(k.GetCurrentProcess(), 0b0100)
    k.SetPriorityClass(k.GetCurrentProcess(), 0x80)

from shockbench_flow.dynamics.env import rollout
from shockbench_flow.evaluation.cache import default_cache_dir, fq_quantiles
from shockbench_flow.hosting.tasks import get_task, task_generator
from shockbench_flow.policies import registry
from shockbench_flow.policies.mpc_scen import MpcScen, MpcScenParams
from shockbench_flow.policies.naive_fq import REPLICATIONS
from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
from shockbench_flow_agent.scoring import _policy_seed, _world

task, what = sys.argv[1], sys.argv[2]
weeks = int(sys.argv[3])
episodes = [int(a) for a in sys.argv[4:]]
cache = str(default_cache_dir())
inst0, params = task_generator(task)
fq = fq_quantiles(inst0, params, REPLICATIONS, cache_dir=cache).quantiles
ctx = registry.PolicyContext(fq_quantile=fq, generator=registry.GeneratorRef(task, get_task(task).gamma))


class Stop(Exception):
    pass


for n in episodes:
    inst, omega, marks, fallback = _world(task, 0, n, REPLICATIONS, cache)
    if what.startswith("S"):
        policy = MpcScen(MpcScenParams(S=int(what[1:])), ctx)
    else:
        policy = registry.make_policy(what, ctx)
    cpu = []
    act, reset = policy.act, policy.reset

    def timed(obs, act=act, cpu=cpu):
        if len(cpu) >= weeks:
            raise Stop
        c = time.process_time(); a = act(obs); cpu.append(time.process_time() - c)
        return a

    def timed_reset(*args, reset=reset, **kw):
        c = time.process_time(); r = reset(*args, **kw); timed_reset.s = time.process_time() - c
        return r

    policy.act, policy.reset = timed, timed_reset
    try:
        rollout(inst, policy, omega, "standard", _policy_seed(0, n, NO_ZIP_SHA256), marks=marks, fallback=fallback)
    except Stop:
        pass
    a = np.array(cpu)
    print(f"{task} {what} episode {n}: {len(a)} weeks; reset {timed_reset.s:.2f} s; week 1 {a[0]:.2f}; median {np.median(a):.3f}; p95 {np.percentile(a, 95):.3f}; max {a.max():.2f} s CPU", flush=True)

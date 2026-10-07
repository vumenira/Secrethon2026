"""The fallbacks of agents/mpc_pulse_rec, each provoked once in one episode of Small root 12345, strict off and on.

Week 3: the forecast raises. Week 5: the reserve's model raises. Week 7: the planner raises after its solve. Week 9: the
pulse's adjust raises. A second play: the pulse cannot be built. Also: the parameter checks at load.
"""

import importlib
import json
import shutil
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026")
WORK = Path(__file__).parent / "smoke_agents" / "mpc_pulse_rec_v3_fb"
AGENT = "mpc_pulse_rec_v3"
MODULES = ("agent", "recovery", "reserve", "pulse")


def copy(params):
    if WORK.exists():
        shutil.rmtree(WORK)
    shutil.copytree(ROOT / "agents" / AGENT, WORK, ignore=shutil.ignore_patterns("__pycache__", "params.json"))
    if params is not None:
        (WORK / "params.json").write_text(json.dumps(params))


def play(params, n=0, no_build=False):
    from shockbench_flow.dynamics.env import rollout, took_fallback
    from shockbench_flow.hosting.docker import without_secret_like
    from shockbench_flow_agent import EpisodeSet
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _metered_shim, _policy_seed, _world
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    copy(params)
    es = EpisodeSet.build("small", 64, entropy=12345, n_jobs=1, verbose=False)
    task, entropy, regime, fq, cache = es._spec
    inst, omega, marks, fallback = _world(task, entropy, n, fq, cache)
    with without_secret_like():
        factory = load_agent_class(str(WORK), "submission_fb")
        module = sys.modules["submission_fb"]
        recovery, reserve, pulse = sys.modules["recovery"], sys.modules["reserve"], sys.modules["pulse"]
        update0, rolled0, action0 = recovery.Recovery.update, reserve.rolled_lp, module.lp_common.week1_action
        adjust0, init0 = pulse.Pulse.adjust, pulse.Pulse.__init__
        sent = {}

        def update(self, obs, memory):
            if int(obs["week"]) == 3 and not no_build:
                raise RuntimeError("provoked: the forecast")
            return update0(self, obs, memory)

        def rolled(inst, obs, *args, **kwargs):
            if int(obs["week"]) == 5 and not no_build:
                raise RuntimeError("provoked: the reserve")
            return rolled0(inst, obs, *args, **kwargs)

        def action(inst, model, x, obs, *args, **kwargs):
            if int(obs["week"]) == 7 and not no_build:
                raise RuntimeError("provoked: the planner")
            return action0(inst, model, x, obs, *args, **kwargs)

        def adjust(self, obs, *args, **kwargs):
            if int(obs["week"]) == 9:
                raise RuntimeError("provoked: the pulse")
            return adjust0(self, obs, *args, **kwargs)

        def init(self, *args, **kwargs):
            raise RuntimeError("provoked: the pulse cannot be built")

        recovery.Recovery.update, reserve.rolled_lp, module.lp_common.week1_action = update, rolled, action
        pulse.Pulse.adjust = adjust
        if no_build:
            pulse.Pulse.__init__ = init
        send0 = module.Agent.send_max

        def send_max(self, o):
            sent[int(o["week"][0])] = True
            return send0(self, o)

        module.Agent.send_max = send_max
        shim = _metered_shim(factory, None)
        try:
            seed = _policy_seed(entropy, n, NO_ZIP_SHA256)
            traj = rollout(inst, shim, omega, regime, seed, marks=marks, fallback=fallback)
        finally:
            module.lp_common.week1_action = action0
            pulse.Pulse.__init__ = init0
    agent, planner = shim.agent, shim.agent.planner
    print(
        f"strict {module.PARAMS['strict']}, pulse cannot be built {no_build}: forecast failed {planner.forecast_failed},"
        f" reserve failed {planner.reserve_failed}, pulse failed {agent.pulse_failed}; pulse"
        f" {'on' if agent.pulse is not None else 'off'}, weeks it changed a flow"
        f" {len(agent.pulse.log) if agent.pulse is not None else 0}; weeks that sent the maximum {sorted(sent)}; errors"
        f" the runner saw {[(w, str(e)[:45]) for w, e in shim.errors]}; weeks the naive rule played"
        f" {sum(took_fallback(r) for r in traj.records)}; broken {agent.broken}; J {traj.J_cents}"
    )
    unload_agent()


def load(params):
    copy(params)
    for name in MODULES:
        sys.modules.pop(name, None)
    sys.path[:] = [p for p in sys.path if p != str(WORK)]
    sys.path.insert(0, str(WORK))
    importlib.invalidate_caches()
    try:
        return importlib.import_module("agent")
    finally:
        sys.path.remove(str(WORK))
        for name in list(sys.modules):
            if name in MODULES or name.split(".")[0] == "sbflow":
                sys.modules.pop(name)


if __name__ == "__main__":
    play({"strict": False})
    play({"strict": True})
    play({"strict": False}, no_build=True)

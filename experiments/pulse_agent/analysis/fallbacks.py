"""The three fallbacks of agents/mpc_combined, each provoked once in one episode of root 12345, strict off and on.

Week 3: the forecast raises. Week 5: the reserve's model raises. Week 7: the planner raises after its solve.
Also: the parameter checks at load.
"""

import importlib
import json
import shutil
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026")
WORK = Path(__file__).parent / "smoke_agents" / "mpc_combined_fb"


def copy(params):
    if WORK.exists():
        shutil.rmtree(WORK)
    shutil.copytree(ROOT / "agents" / "mpc_combined", WORK, ignore=shutil.ignore_patterns("__pycache__", "params.json"))
    if params is not None:
        (WORK / "params.json").write_text(json.dumps(params))


def play(params, n=0):
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
        module, recovery, reserve = sys.modules["submission_fb"], sys.modules["recovery"], sys.modules["reserve"]
        update0, rolled0, action0 = recovery.Recovery.update, reserve.rolled_lp, module.lp_common.week1_action
        sent = {}

        def update(self, obs, memory):
            if int(obs["week"]) == 3:
                raise RuntimeError("provoked: the forecast")
            return update0(self, obs, memory)

        def rolled(inst, obs, *args, **kwargs):
            if int(obs["week"]) == 5:
                raise RuntimeError("provoked: the reserve")
            return rolled0(inst, obs, *args, **kwargs)

        def action(inst, model, x, obs, *args, **kwargs):
            if int(obs["week"]) == 7:
                raise RuntimeError("provoked: the planner")
            return action0(inst, model, x, obs, *args, **kwargs)

        recovery.Recovery.update, reserve.rolled_lp, module.lp_common.week1_action = update, rolled, action
        send0 = module.Agent.send_max

        def send_max(self, o):
            sent[int(o["week"][0])] = True
            return send0(self, o)

        module.Agent.send_max = send_max
        shim = _metered_shim(factory, None)
        try:
            traj = rollout(inst, shim, omega, regime, _policy_seed(entropy, n, NO_ZIP_SHA256), marks=marks, fallback=fallback)
        finally:
            module.lp_common.week1_action = action0
    planner = shim.agent.planner
    print(
        f"strict {module.PARAMS['strict']}: forecast failed {planner.forecast_failed}, reserve failed"
        f" {planner.reserve_failed}, failed {planner.failed}; weeks that sent the maximum {sorted(sent)}; errors the"
        f" runner saw {[(w, str(e)[:40]) for w, e in shim.errors]}; weeks the naive rule played"
        f" {sum(took_fallback(r) for r in traj.records)}; broken {shim.agent.broken}; J {traj.J_cents}"
    )
    unload_agent()


def load(params):
    copy(params)
    for name in ("agent", "recovery", "reserve"):
        sys.modules.pop(name, None)
    sys.path[:] = [p for p in sys.path if p != str(WORK)]
    sys.path.insert(0, str(WORK))
    importlib.invalidate_caches()
    try:
        return importlib.import_module("agent")
    finally:
        sys.path.remove(str(WORK))
        for name in list(sys.modules):
            if name in ("agent", "recovery", "reserve") or name.split(".")[0] == "sbflow":
                sys.modules.pop(name)


if __name__ == "__main__":
    print("defaults:", json.dumps(load(None).PARAMS))
    good = [
        {"kinds": [], "strict": True},
        {"end_value": 0.0, "reserve_weeks": 0.0, "shed_ramp": 0.0},
        {"h_weeks": 24},
        {"h_weeks": None},
        {"reserve_fuels": "rationed", "reserve_feeders": True, "reserve_weeks": 6},
        {"rules": {"port_stoppage": {"first": "open"}, "port_slowdown": {"mode": "keep"}}},
    ]
    bad = [
        {"demand_scale": 1.1},
        {"throughput_scale": 0.9},
        {"by_event": {}},
        {"reserve_fuels": "lng"},
        {"reserve_weeks": -1},
        {"end_value": "one"},
        {"shed_ramp": None},
        {"h_weeks": 0},
        {"strict": "false"},
        {"rules": {"storm": {"mode": "mean"}}},
        {"kinds": ["closure", "edge"]},
    ]
    for params in good:
        load(params)
        print("accepted:", json.dumps(params))
    for params in bad:
        try:
            load(params)
        except (ValueError, TypeError) as exc:
            print("refused: ", json.dumps(params), "->", type(exc).__name__, str(exc)[:110])
        else:
            raise SystemExit(f"NOT refused: {params}")
    play({"strict": False})
    play({"strict": True})

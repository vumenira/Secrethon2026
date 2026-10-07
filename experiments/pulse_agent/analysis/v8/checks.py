"""Checks of agents/mpc_pulse_rec_v8 that the scored plays do not make (tuning root only).

uv run python checks.py loose --task=small --episodes=0,5,17 --compare=s_name   # strict off equals the strict play
uv run python checks.py fail --task=small --n=17 --week=3                      # a failure inside the new rule
uv run python checks.py cpu --task=full --episodes=0,1,2                        # act's process time, v8 against v7
"""

import json
import sys
import time
from pathlib import Path

import fire
import numpy as np


ROOT = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026")
OUT = ROOT / "outputs" / "recovery_tune"
TOTAL = {"small": 64, "full": 16}
ENTROPY = 12345


def world(task, n):
    from shockbench_flow.evaluation.cache import default_cache_dir
    from shockbench_flow.policies.naive_fq import REPLICATIONS
    from shockbench_flow_agent.scoring import _world

    return _world(task, ENTROPY, n, REPLICATIONS, str(default_cache_dir()))


def cached(task, name):
    rows = json.loads((OUT / f"{task}-e{ENTROPY}-n{TOTAL[task]}" / f"{name}.json").read_text())["rows"]
    return {r["episode"]: r["J_policy_cents"] for r in rows}


def play(task, n, agent, params, patch=None):
    """One episode of an agent folder with ``params`` put into its PARAMS; ``patch(module, box)`` may wrap its code."""
    from shockbench_flow.dynamics.env import rollout, took_fallback
    from shockbench_flow.hosting.docker import without_secret_like
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _metered_shim, _policy_seed
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    inst, omega, marks, fallback = world(task, n)
    box = {"weeks": []}
    with without_secret_like():
        factory = load_agent_class(str(ROOT / "agents" / agent), "submission_v8check")
        module = sys.modules[factory.__module__]
        module.PARAMS.update(params or {})
        act0 = module.Agent.act

        def act(self, observation):
            wall, cpu = time.perf_counter(), time.process_time()
            try:
                return act0(self, observation)
            finally:
                box["weeks"].append((time.perf_counter() - wall, time.process_time() - cpu))

        module.Agent.act = act
        if patch:
            patch(module, box)
        shim = _metered_shim(factory, None)
        traj = rollout(inst, shim, omega, "standard", _policy_seed(ENTROPY, n, NO_ZIP_SHA256), marks=marks, fallback=fallback)
        errors = list(shim.errors)
    agent_obj = shim.agent
    out = {
        "J": int(traj.J_cents),
        "fallback_weeks": sum(took_fallback(r) for r in traj.records),
        "errors": errors,
        "pulse_failed": agent_obj.pulse_failed,
        "planner_failed": getattr(agent_obj.planner, "failed", None),
        "broken": agent_obj.broken,
        "log": list(agent_obj.pulse.log) if agent_obj.pulse is not None else [],
        "weeks": box["weeks"],
        "box": box,
    }
    unload_agent()
    return out


def ints(v):
    return [int(x) for x in (v if isinstance(v, (list, tuple)) else str(v).split(","))]


def loose(task="small", episodes="0,5,17", compare=None, agent="mpc_pulse_rec_v8", params=None):
    """The folder as it is (strict off, its own defaults unless ``params``) against the strict play cached as ``compare``."""
    base = cached(task, compare)
    same = 0
    for n in ints(episodes):
        r = play(task, n, agent, dict(params or {}) | {"strict": False})
        ok = r["J"] == base[n]
        same += ok
        opens = sum(1 for e in r["log"] if e[2] == "open")
        print(
            f"{task} episode {n}: J {r['J']} {'=' if ok else '!='} {compare}; fallback weeks {r['fallback_weeks']}; errors {len(r['errors'])};"
            f" pulse failed {r['pulse_failed']}, planner failed {r['planner_failed']}, broken {r['broken']}; weeks the hold acted {opens}",
            flush=True,
        )
    print(f"{task}: {same} of {len(ints(episodes))} episodes with strict off equal to {compare} to the cent")


def fail(task="small", n=17, week=3, what="plan_short", agent="mpc_pulse_rec_v8", params=None):
    """Make a function of the new rule (``Pulse.plan_short`` or ``Pulse.deliver``) raise in one week, strict off:
    that week must send the planner's flows as they were."""
    params = dict(params or {}) | {"strict": False, "open_hold": True}

    def patch(module, box):
        pulse = sys.modules["pulse"]
        orig, pulsed0 = getattr(pulse.Pulse, what), module.Agent.pulsed
        box["raised"], box["same"], box["changed"] = [], [], []

        def broken(*args, **kwargs):
            if box.get("now") == week:
                box["raised"].append(week)
                raise RuntimeError("provoked: the opening hold fails")
            return orig(*args, **kwargs)

        def pulsed(self, obs, action):
            box["now"] = int(obs["week"])
            before = json.dumps(action, sort_keys=True, default=float)
            out = pulsed0(self, obs, action)
            (box["same"] if out is action and json.dumps(out, sort_keys=True, default=float) == before else box["changed"]).append(box["now"])
            return out

        setattr(pulse.Pulse, what, staticmethod(broken) if what == "deliver" else broken)
        module.Agent.pulsed = pulsed

    r = play(task, n, agent, params, patch)
    box = r["box"]
    print(f"{task} episode {n}: Pulse.{what} raised in weeks {box['raised']}; pulse failed {r['pulse_failed']}; fallback weeks {r['fallback_weeks']}; errors {len(r['errors'])}; broken {r['broken']}; planner failed {r['planner_failed']}")
    print(f"   week {week}: the action sent is the planner's own, unchanged: {week in box['same']}; weeks the pulse changed the action: {len(box['changed'])} of {len(box['same']) + len(box['changed'])}")
    opens = sorted((e[0], e[1]) for e in r["log"] if e[2] == "open")
    print(f"   (week, grid) the hold logged: {opens}; J {r['J']}")


def cpu(task="full", episodes="0,1,2", new="mpc_pulse_rec_v8", old="mpc_pulse_rec_v7", params=None):
    """Process time of ``Agent.act`` per week, the two folders played in turn in this process."""
    rows = {new: [], old: []}
    first = {new: [], old: []}
    for n in ints(episodes):
        for name in (new, old):
            r = play(task, n, name, dict(params or {}) if name == new else {})
            cpus = [c for _w, c in r["weeks"]]
            first[name].append(cpus[0])
            rows[name] += cpus[1:]
            print(f"{task} episode {n} {name}: week 1 {cpus[0]:.2f} s; later weeks mean {np.mean(cpus[1:]):.3f} s, median {np.median(cpus[1:]):.3f} s, slowest {max(cpus[1:]):.2f} s (process time); pulse failed {r['pulse_failed']}", flush=True)
    for name in (new, old):
        x = np.array(rows[name])
        print(f"{name}: {len(x)} weeks after the first: mean {x.mean():.3f} s, median {np.median(x):.3f} s, 99th percentile {np.quantile(x, 0.99):.2f} s, slowest {x.max():.2f} s; week 1 (with the planner's start) slowest {max(first[name]):.2f} s")
    a, b = np.array(rows[new]), np.array(rows[old])
    print(f"ratio of means {a.mean() / b.mean():.3f}; of slowest weeks {a.max() / b.max():.3f}")


if __name__ == "__main__":
    fire.Fire({"loose": loose, "fail": fail, "cpu": cpu})

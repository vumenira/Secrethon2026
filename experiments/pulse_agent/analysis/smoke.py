"""Play a copy of an agent folder with a params.json on a few episodes of root 12345 and show its fallbacks and the
time its weeks take (the whole ``Agent.act``, wall and CPU; week 1 carries the planner's start).

uv run python smoke.py --agent=mpc_pulse_rec_v4 --params="{'strict': True}" --episodes=0,1 --task=small --compare=name
"""

import json
import shutil
import sys
import time
from pathlib import Path

import fire


ROOT = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026")
WORK = Path(__file__).parent / "smoke_agents"
COUNT = {"small": 64, "full": 16}


def main(agent="mpc_combined", params=None, episodes="0,1", tag="a", compare=None, task="small", entropy=12345):
    from shockbench_flow.dynamics.env import rollout, took_fallback
    from shockbench_flow.hosting.docker import without_secret_like
    from shockbench_flow_agent import EpisodeSet
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _metered_shim, _policy_seed, _world
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    assert entropy == 12345, "smoke plays on the tuning root only"
    root = WORK / f"{agent}_{tag}"
    if root.exists():
        shutil.rmtree(root)
    shutil.copytree(ROOT / "agents" / agent, root, ignore=shutil.ignore_patterns("__pycache__", "params.json"))
    if params:
        (root / "params.json").write_text(json.dumps(dict(params)))
    es = EpisodeSet.build(task, COUNT[task], entropy=entropy, n_jobs=1, verbose=False)
    task, entropy, regime, fq, cache = es._spec
    ns = [int(n) for n in (episodes if isinstance(episodes, (list, tuple)) else str(episodes).split(","))]
    cached = {}
    if compare:
        run = ROOT / "outputs" / "recovery_tune" / f"{task}-e{entropy}-n{COUNT[task]}"
        cached = {r["episode"]: r["J_policy_cents"] for r in json.loads((run / f"{compare}.json").read_text())["rows"]}
    equal, worst = 0, (0.0, 0.0, 0.0, 0.0)  # slowest week 1 (wall, CPU), slowest later week (wall, CPU)
    for n in ns:
        inst, omega, marks, fallback = _world(task, entropy, n, fq, cache)
        with without_secret_like():
            factory = load_agent_class(str(root), "submission_smoke")
            module = sys.modules["submission_smoke"]
            if n == ns[0]:
                print("PARAMS:", json.dumps(module.PARAMS))
            act0, weeks = module.Agent.act, []

            def act(self, observation, act0=act0, weeks=weeks):
                wall, cpu = time.perf_counter(), time.process_time()
                try:
                    return act0(self, observation)
                finally:
                    weeks.append((time.perf_counter() - wall, time.process_time() - cpu))

            module.Agent.act = act
            shim = _metered_shim(factory, None)
            start = time.perf_counter()
            seed = _policy_seed(entropy, n, NO_ZIP_SHA256)
            traj = rollout(inst, shim, omega, regime, seed, marks=marks, fallback=fallback)
            seconds = time.perf_counter() - start
        planner = getattr(shim.agent, "planner", None)
        pulse = getattr(shim.agent, "pulse", None)
        first, later = weeks[0], (max(w for w, _c in weeks[1:]), max(c for _w, c in weeks[1:]))
        worst = (max(worst[0], first[0]), max(worst[1], first[1]), max(worst[2], later[0]), max(worst[3], later[1]))
        same = "" if n not in cached else f" {'=' if cached[n] == traj.J_cents else '!='} {compare}"
        equal += n in cached and cached[n] == traj.J_cents
        print(
            f"episode {n}: J {traj.J_cents}{same}; fallback weeks {sum(took_fallback(r) for r in traj.records)};"
            f" errors {len(shim.errors)} {shim.errors[:1]}; forecast failed"
            f" {getattr(planner, 'forecast_failed', None)}, reserve failed {getattr(planner, 'reserve_failed', None)},"
            f" pulse failed {getattr(shim.agent, 'pulse_failed', None)}; pulse weeks"
            f" {len(pulse.log) if pulse is not None else None}; broken {getattr(shim.agent, 'broken', None)};"
            f" {seconds:.1f} s; week 1 {first[0]:.2f} s wall, {first[1]:.2f} s CPU; slowest later week"
            f" {later[0]:.2f} s wall, {later[1]:.2f} s CPU",
            flush=True,
        )
        unload_agent()
    if compare:
        print(f"{task}: {equal} of {len(ns)} episodes equal to {compare} to the cent")
    print(
        f"{task}, {len(ns)} episodes: slowest week 1 {worst[0]:.2f} s wall, {worst[1]:.2f} s CPU; slowest later week"
        f" {worst[2]:.2f} s wall, {worst[3]:.2f} s CPU"
    )


if __name__ == "__main__":
    fire.Fire(main)

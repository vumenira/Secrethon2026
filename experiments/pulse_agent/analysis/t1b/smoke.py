"""Play single episodes of the tuning root with an agent folder in this process, logging the forecast's layers.

uv run python smoke.py <agent folder> <params json or -> <episode> [episode ...]
"""

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path


def main(folder, params, episodes, task="small", entropy=12345):
    from shockbench_flow.dynamics.env import rollout, took_fallback
    from shockbench_flow.evaluation.cache import default_cache_dir
    from shockbench_flow.hosting.docker import without_secret_like
    from shockbench_flow.hosting.tasks import task_generator
    from shockbench_flow.policies.naive_fq import REPLICATIONS
    from shockbench_flow_agent import EpisodeSet
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _metered_shim, _policy_seed, _world
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    es = EpisodeSet.build(task, 64, entropy=entropy, n_jobs=1, verbose=False)
    spec = es._spec
    task, entropy, regime, fq, cache = spec
    work = Path(tempfile.mkdtemp(prefix="t1b_smoke_"))
    root = work / "agent"
    shutil.copytree(folder, root, ignore=shutil.ignore_patterns("__pycache__", "params.json"))
    if params != "-":
        (root / "params.json").write_text(params)
    inst0, _ = task_generator(task)
    names = {"closure": [inst0.nodes[c].id for c in inst0.chokepoints], "grid": [inst0.nodes[g].id for g in inst0.grids]}
    for n in episodes:
        inst, omega, marks, fallback = _world(task, entropy, n, fq, cache)
        log = []
        with without_secret_like():
            factory = load_agent_class(str(root), "submission_smoke")
            rec = sys.modules["recovery"].Recovery
            plain = rec.update

            def update(self, obs, memory, plain=plain, log=log):
                plain(self, obs, memory)
                state = {f"{k[0]} {names[k[0]][k[1]]}": [(round(f, 3), e, first, seen) for f, e, first, seen in v] for k, v in getattr(self, "layers", {}).items()}
                if not log or log[-1][1] != state:
                    log.append((int(obs["week"]), state))

            rec.update = update
            shim = _metered_shim(factory, None)
            start = time.perf_counter()
            seed = _policy_seed(entropy, n, NO_ZIP_SHA256)
            traj = rollout(inst, shim, omega, regime, seed, marks=marks, fallback=fallback)
            seconds = time.perf_counter() - start
        planner = getattr(shim.agent, "planner", None)
        failed = getattr(planner, "failed", None)
        unload_agent()
        print(
            f"episode {n}: J {traj.J_cents} fallback weeks {sum(took_fallback(r) for r in traj.records)} "
            f"errors {shim.errors[:2]} forecast failures {failed} seconds {seconds:.1f}"
        )
        for week, state in log:
            print("   week", week, state)
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3:]])

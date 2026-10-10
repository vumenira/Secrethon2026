"""Play an agent folder on chosen episodes and keep the simulator's records, in the census pickles' layout plus the
flows on every edge and the costs: for reading one episode of a variant against v12's.

uv run python outputs/recovery_tune/tools/with_lock.py uv run python outputs/w13a/play.py --name=a13_s8 \
    --eps="777:30,12345:7" [--params='{"strait_reserve": true}']
The folder played is outputs/recovery_tune/agents/<name> if it exists (tune.py's copy of a variant), else a copy of
agents/mpc_pulse_rec_v13w with --params (strict always) made there. Output: outputs/w13a/data/full-e<root>/<name>_ep<n>.pkl
"""
import json
import pickle
import shutil
import sys
import time
from pathlib import Path

import fire
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def one(folder, task, root, n, name):
    from shockbench_flow.dynamics.env import rollout
    from shockbench_flow.evaluation.cache import default_cache_dir
    from shockbench_flow.hosting.docker import without_secret_like
    from shockbench_flow.policies.naive_fq import REPLICATIONS
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _metered_shim, _policy_seed, _world
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    out = HERE / "data" / f"{task}-e{root}"
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"{name}_ep{n}.pkl"
    if target.exists():
        return
    inst, omega, marks, fallback = _world(task, root, n, REPLICATIONS, str(default_cache_dir()))
    start = time.perf_counter()
    with without_secret_like():
        shim = _metered_shim(load_agent_class(str(folder), f"w13a_{name}_{root}_{n}"), None)
        traj = rollout(inst, shim, omega, "standard", _policy_seed(root, n, NO_ZIP_SHA256), marks=marks, fallback=fallback)
    recs = traj.records
    pulse = getattr(shim.agent, "pulse", None)
    doc = {
        "episode": n,
        "J_cents": traj.J_cents,
        "errors": [(int(w), str(e)) for w, e in shim.errors],
        "log": list(getattr(pulse, "log", [])),
        "energy": np.array([r.energy for r in recs]),
        "lots": np.array([r.lots_started for r in recs]),
        "shed": np.array([r.shed for r in recs]),
        "segment": [dict(r.segment) for r in recs],
        "stock": np.array([r.stock for r in recs], dtype=np.float32),
        "lost": np.array([r.lost for r in recs]),
        "demand": np.array([r.demand for r in recs]),
        "disposal": np.array([r.disposal for r in recs], dtype=np.float32),
        "lift": np.array([r.lift for r in recs], dtype=np.float32),
        "executed": [dict(r.executed) for r in recs],
        "x": [dict(r.x) for r in recs],
        "costs": [r.costs.as_dict() for r in recs],
        "salvage": traj.salvage,
    }
    unload_agent()
    with open(target, "wb") as f:
        pickle.dump(doc, f, protocol=4)
    print(f"{name} {root}/{n}: J {traj.J_cents / 1e11:,.1f} bn, errors {doc['errors'][:1]} ({time.perf_counter() - start:.0f} s)", flush=True)


def main(name, eps, params=None, task="full", n_jobs=2):
    from joblib import Parallel, delayed

    folder = ROOT / "outputs" / "recovery_tune" / "agents" / name
    if params is not None or not folder.exists():
        given = json.loads(params) if isinstance(params, str) else dict(params or {})
        folder = HERE / "agents" / name
        if folder.exists():
            shutil.rmtree(folder)
        shutil.copytree(ROOT / "agents" / "mpc_pulse_rec_v13w", folder, ignore=shutil.ignore_patterns("__pycache__", "params.json"))
        (folder / "params.json").write_text(json.dumps({"strict": True} | given))
    pairs = [tuple(int(v) for v in item.split(":")) for item in str(eps).split(",")]
    Parallel(n_jobs=n_jobs)(delayed(one)(folder, task, root, n, name) for root, n in pairs)


if __name__ == "__main__":
    sys.exit(fire.Fire(main))

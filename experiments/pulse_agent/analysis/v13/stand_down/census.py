"""Play an agent folder with the pulse's census on and keep, per episode, the census and the simulator's records.

uv run python outputs/recovery_tune/tools/with_lock.py uv run python outputs/s13/census.py run --task=full \
    --entropy=12345 --episodes=16 --n_jobs=2 --name=s13_census

The folder is a copy of ``base`` (agents/mpc_pulse_rec_v13s) with ``params`` as its params.json (strict always), as
experiments/approaches/tune.py makes it. Each episode is written to outputs/s13/<name>/<task>-e<root>/ep<n>.pkl and
the costs to outputs/recovery_tune/<task>-e<root>-n<N>/<name>.json in tune.py's format, so ``same.py`` and ``tune.py
report`` read them like any other play.
"""

import json
import pickle
import sys
import time
from pathlib import Path

import fire
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "approaches"))
HERE = Path(__file__).resolve().parent


def play(root, spec, episodes, out):
    from shockbench_flow.dynamics.env import rollout, took_fallback
    from shockbench_flow.hosting.docker import without_secret_like
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _metered_shim, _policy_seed, _world
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    task, entropy, regime, fq, cache = spec
    rows = []
    for n in episodes:
        inst, omega, marks, fallback = _world(task, entropy, n, fq, cache)
        with without_secret_like():
            factory = load_agent_class(root, "submission_census")
            plain = factory.new_pulse

            def new_pulse(self, inst_, plain=plain):
                p = plain(self, inst_)
                if p is not None:
                    p.census = []
                return p

            factory.new_pulse = new_pulse
            shim = _metered_shim(factory, None)
            start = time.perf_counter()
            traj = rollout(inst, shim, omega, regime, _policy_seed(entropy, n, NO_ZIP_SHA256), marks=marks, fallback=fallback)
            seconds = time.perf_counter() - start
        pulse = shim.agent.pulse
        recs = traj.records
        slots = sorted({s for r in recs for s in r.executed})
        doc = {
            "episode": n,
            "J_cents": traj.J_cents,
            "census": pulse.census,
            "log": pulse.log,
            "gain": dict(pulse.gain),
            "last": dict(pulse.last),
            "full_draw": dict(pulse.full_draw),
            "energy": np.array([r.energy for r in recs]),
            "lots": np.array([r.lots_started for r in recs]),
            "shed": np.array([r.shed for r in recs]),
            "served_load": np.array([r.served_load for r in recs]),
            "segment": [dict(r.segment) for r in recs],
            "stock": np.array([r.stock for r in recs], dtype=np.float32),
            "lost": np.array([r.lost for r in recs]),
            "demand": np.array([r.demand for r in recs]),
            "backlog": np.array([r.backlog for r in recs]),
            "disposal": np.array([r.disposal for r in recs], dtype=np.float32),
            "executed": [dict(r.executed) for r in recs],
            "costs": [r.costs.as_dict() for r in recs],
            "salvage": traj.salvage,
        }
        unload_agent()
        Path(out).mkdir(parents=True, exist_ok=True)
        with open(Path(out) / f"ep{n}.pkl", "wb") as f:
            pickle.dump(doc, f, protocol=4)
        rows.append(
            {
                "episode": n,
                "J_policy_cents": traj.J_cents,
                "fallback_weeks": sum(took_fallback(r) for r in recs),
                "first_error": f"week {shim.errors[0][0]}: {shim.errors[0][1]}" if shim.errors else None,
                "seconds": round(seconds, 3),
                "omega_hash": omega.hash,
            }
        )
    return rows


def run(task="full", entropy=12345, episodes=16, n_jobs=2, name="s13_census", base="mpc_pulse_rec_v13s", params=None, redo=False):
    import tune
    from joblib import Parallel, delayed

    target = tune.run_dir(task, entropy, episodes) / f"{name}.json"
    out = HERE / name / f"{task}-e{entropy}"
    if target.exists() and not redo:
        print(f"{name} on {task} root {entropy}: cached", flush=True)
        return
    es = tune.episode_set(task, entropy, episodes, n_jobs)
    spec = {"base": base, "params": {"strict": True} | dict(params or {})}
    root = tune.folder(name, spec)
    tune.wait_for_memory()
    start = time.perf_counter()
    ns = list(es.episodes)
    chunks = [c for c in np.array_split(np.array(ns), min(len(ns), n_jobs)) if c.size]
    parts = Parallel(n_jobs=n_jobs)(delayed(play)(str(root), es._spec, [int(n) for n in c], str(out)) for c in chunks)
    rows = [r for part in parts for r in part]
    target.write_text(json.dumps({"spec": spec, "rows": rows}, indent=1))
    table = es.rss([r["J_policy_cents"] for r in rows])
    errors = [r["first_error"] for r in rows if r.get("first_error")]
    print(
        f"{name}: score {table['rss']:.4f} fallback weeks {sum(r['fallback_weeks'] for r in rows)} errors {errors[:2]}"
        f" ({time.perf_counter() - start:.0f} s)",
        flush=True,
    )


def many(sets, n_jobs=2, name="s13_census", base="mpc_pulse_rec_v13s", params=None):
    """``run`` on several sets under one lock: ``sets`` is "task:root:episodes,task:root:episodes"."""
    for item in sets.split(","):
        task, entropy, episodes = item.split(":")
        run(task, int(entropy), int(episodes), n_jobs, name, base, params)


def some(eps, task="full", name="s13_trace", base="mpc_pulse_rec_v13s", params=None, n_jobs=2):
    """Record single episodes: ``eps`` is "root:episode,root:episode"; the set sizes are those of the used sets."""
    import tune
    from joblib import Parallel, delayed

    sizes = {12345: 16, 2026: 16, 777: 32} if task == "full" else {12345: 64, 2026: 64}
    spec = {"base": base, "params": {"strict": True} | dict(params or {})}
    root = tune.folder(name, spec)
    jobs = []
    for item in (eps.split(",") if isinstance(eps, str) else eps):
        entropy, n = (int(x) for x in str(item).split(":"))
        es = tune.episode_set(task, entropy, sizes[entropy], 1)
        jobs.append((es._spec, n, str(HERE / name / f"{task}-e{entropy}")))
    tune.wait_for_memory()
    rows = Parallel(n_jobs=n_jobs)(delayed(play)(str(root), sp, [n], out) for sp, n, out in jobs)
    for (sp, n, _out), r in zip(jobs, rows):
        print(sp[1], n, "J bn", r[0]["J_policy_cents"] / 1e11, "error", r[0]["first_error"], flush=True)


if __name__ == "__main__":
    fire.Fire({"run": run, "many": many, "some": some})

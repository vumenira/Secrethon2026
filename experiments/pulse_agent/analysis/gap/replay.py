"""Replay agents/mpc_pulse_rec_v4 (folder defaults) with its trajectory recorded; one dump per episode in gap/traj.

The folder is loaded as the scorer loads it and never edited; ``PARAMS["strict"]`` is set in the loaded module (as the
cached row ``v_r1`` was played) and ``Agent.pulsed`` is wrapped in memory to keep the planner's flows before the pulse.
Each episode's cost is compared with the cached row to the cent, and its trajectory is run through the package's
replay test (53) against the clairvoyant LP.

uv run python replay.py --task=small --n_jobs=2
"""

import pickle
import sys
import time

import fire
import numpy as np

import common as C


def one(task, n, plans=False):
    from shockbench_flow.dynamics.env import rollout
    from shockbench_flow.hosting.docker import without_secret_like
    from shockbench_flow.oracle.lp import build_lp
    from shockbench_flow.oracle.replay import replay, trajectory_vector
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _metered_shim, _policy_seed
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    inst, omega, marks, fallback = C.world(task, n)
    start = time.perf_counter()
    before = {}
    kept, template = {}, {}
    with without_secret_like():
        factory = load_agent_class(str(C.AGENT), "submission_gap")
        module = sys.modules[factory.__module__]
        module.PARAMS.update({"strict": True})
        orig = module.Agent.pulsed

        def pulsed(self, obs, action):
            out = orig(self, obs, action)
            flows = action.get("flows") or {"slot": [], "qty": []}
            before[int(obs["week"])] = dict(zip([int(s) for s in flows["slot"]], [float(q) for q in flows["qty"]]))
            return out

        module.Agent.pulsed = pulsed
        if plans:
            box = [0]
            orig_act = module.Planner.act

            def act(self, obs):
                box[0] = int(obs["week"])
                return orig_act(self, obs)

            def record(self, model, res):
                if res.ok:
                    template.update(model.meta["template"])
                    kept[box[0]] = np.asarray(res.x, dtype=np.float32).reshape(model.T, model.meta["nc"]).copy()

            module.Planner.act = act
            module.Planner._record = record
        shim = _metered_shim(factory, None)
        traj = rollout(inst, shim, omega, "standard", _policy_seed(12345, n, NO_ZIP_SHA256), marks=marks, fallback=fallback)
        errors = list(shim.errors)
    agent = shim.agent
    log = list(agent.pulse.log) if agent.pulse is not None else []
    failed = (getattr(agent.planner, "forecast_failed", None), getattr(agent.planner, "reserve_failed", None), agent.pulse_failed)
    unload_agent()
    seconds = time.perf_counter() - start
    model = build_lp(inst, marks)
    rep = replay(model, traj)
    zpi = trajectory_vector(model, traj)
    S = len(inst.action_slots)
    T = inst.T
    requested, executed, planned = np.zeros((T, S)), np.zeros((T, S)), np.zeros((T, S))
    for ti, r in enumerate(traj.records):
        for s, q in r.requested.items():
            requested[ti, int(s)] = q
        for s, q in r.executed.items():
            executed[ti, int(s)] = q
        for s, q in before.get(ti + 1, {}).items():
            planned[ti, s] = q
    dump = {
        "episode": n,
        "task": task,
        "J_cents": int(traj.J_cents),
        "salvage": float(traj.salvage),
        "zpi": zpi,
        "requested": requested,
        "executed": executed,
        "planned": planned,
        "override_requested": [dict(r.override_requested) for r in traj.records],
        "override_executed": [dict(r.override_executed) for r in traj.records],
        "scrapped": np.array([r.scrapped for r in traj.records]),
        "costs": [r.costs.as_dict() for r in traj.records],
        "pulse_log": log,
        "errors": errors,
        "failed": failed,
        "replay": {"feasible": bool(rep.feasible), "worst": float(rep.worst_residual), "row": tuple(rep.worst_row), "cost_equal": bool(rep.cost_equal), "J_lp_cents": int(rep.J_lp_cents)},
        "seconds": seconds,
    }
    if plans:
        dump["plans"], dump["plan_template"] = kept, dict(template)
    path = C.traj_path(task, n) if not plans else C.HERE / "traj" / f"{task}_{n}_plans.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(dump))
    return n, int(traj.J_cents), errors[:1], failed, dump["replay"], round(seconds, 1)


def main(task="small", only=None, n_jobs=2, plans=False, redo=False):
    from joblib import Parallel, delayed

    cached = C.cached(task)
    refs = C.references(task)
    ns = sorted(refs) if only is None else ([only] if isinstance(only, int) else [int(v) for v in only])
    todo = [n for n in ns if redo or plans or not C.traj_path(task, n).exists()]
    start = time.perf_counter()
    results = Parallel(n_jobs=n_jobs)(delayed(one)(task, n, plans) for n in todo) if todo else []
    same = 0
    for n, J, errors, failed, rep, seconds in results:
        ok = J == cached[n]
        same += ok
        r = refs[n]
        score = (r["J_naive_cents"] - J) / (r["J_naive_cents"] - r["J_oracle_cents"])
        print(
            f"episode {n} level {r['stratum']}: {score:.4f} {'same as' if ok else 'DIFFERS from'} the cached row (by {J - cached[n]} cents);"
            f" replay feasible {rep['feasible']} (worst {rep['worst']:.1e} at {rep['row'][:3]}), cost equal {rep['cost_equal']}; failures {failed}"
            f"{' ERRORS ' + str(errors) if errors else ''} ({seconds} s)",
            flush=True,
        )
    print(f"{same} of {len(results)} episodes cost the same as {C.CACHED[task]} to the cent; {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    fire.Fire(main)

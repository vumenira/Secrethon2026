"""Play an agent folder that is not under agents/ on root 12345 and cache its costs where tune.py's report reads them.

uv run python <this> --agent=<folder> --name=ft_x --task=full --episodes=16 --n_jobs=2

The same routine as the harness (the scorer's rollout with the episode's marks and naive fallback); the policy seed is
the one of an agent class rather than of the folder's zip, which this planner does not use.
"""

import json
import time
from pathlib import Path

import fire


ROOT = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026")
OUT = ROOT / "outputs" / "recovery_tune"


def one(n, agent, name, task, entropy):
    from shockbench_flow.dynamics.env import rollout, took_fallback
    from shockbench_flow.evaluation.cache import default_cache_dir
    from shockbench_flow.hosting.docker import without_secret_like
    from shockbench_flow.policies.naive_fq import REPLICATIONS
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _metered_shim, _policy_seed, _world
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    inst, omega, marks, fallback = _world(task, entropy, n, REPLICATIONS, str(default_cache_dir()))
    with without_secret_like():
        factory = load_agent_class(agent, f"submission_{name}")
        shim = _metered_shim(factory, None)
        start = time.perf_counter()
        traj = rollout(inst, shim, omega, "standard", _policy_seed(entropy, n, NO_ZIP_SHA256), marks=marks, fallback=fallback)
        seconds = time.perf_counter() - start
    planner = getattr(shim.agent, "planner", None)
    row = {
        "episode": n,
        "J_policy_cents": traj.J_cents,
        "fallback_weeks": sum(took_fallback(r) for r in traj.records),
        "cpu_weeks": 0,
        "invalid_entries": sum(len(r.invalid) for r in traj.records if not took_fallback(r)),
        "first_error": f"week {shim.errors[0][0]}: {shim.errors[0][1]}" if shim.errors else None,
        "forecast_failed": getattr(planner, "failed", None),
        "pulse_failed": getattr(shim.agent, "pulse_failed", None),
        "pulse_weeks": len(getattr(getattr(shim.agent, "pulse", None), "log", []) or []),
        "weeks": inst.T,
        "seconds": round(seconds, 3),
        "omega_hash": omega.hash,
    }
    unload_agent()
    return row


def main(agent, name, task="full", entropy=12345, episodes=16, n_jobs=2, redo=False):
    from joblib import Parallel, delayed

    agent = str(Path(agent).resolve())
    target = OUT / f"{task}-e{entropy}-n{episodes}" / f"{name}.json"
    if target.exists() and not redo:
        raise SystemExit(f"{target} exists")
    refs = json.loads((target.parent / "references.json").read_text())
    rows = Parallel(n_jobs=n_jobs)(delayed(one)(n, agent, name, task, entropy) for n in range(episodes))
    for ref, row in zip(refs, rows):
        if ref["omega_hash"] != row["omega_hash"]:
            raise RuntimeError(f"episode {row['episode']}: another scenario than the cached reference")
    params = json.loads((Path(agent) / "params.json").read_text()) if (Path(agent) / "params.json").is_file() else {}
    spec = {"folder": agent, "params": params, "played_by": "scratchpad/play_cache.py (not tune.py run)"}
    target.write_text(json.dumps({"spec": spec, "rows": rows}, indent=1))
    print(f"{name}: fallback weeks {sum(r['fallback_weeks'] for r in rows)}, planner failures {sum(r['forecast_failed'] or 0 for r in rows)},"
          f" pulse failures {sum(r['pulse_failed'] or 0 for r in rows)}, errors {[r['first_error'] for r in rows if r['first_error']][:2]},"
          f" weeks the pulse changed a flow {sum(r['pulse_weeks'] for r in rows)}, slowest episode {max(r['seconds'] for r in rows):.0f} s")
    print("J:", [r["J_policy_cents"] for r in rows])


if __name__ == "__main__":
    fire.Fire(main)

"""Play episodes of the tuning root with an agent folder and params, in this process; print costs and fuel stocks.

uv run python diag_play.py --base=buf_lp --params="{...}" --episodes=0,1,2 --detail
Scratch only: reads the repo, writes a copy of the agent folder under the scratchpad.
"""

import json
import shutil
import sys
import time
from pathlib import Path

import fire
import numpy as np


REPO = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026")
WORK = Path(__file__).resolve().parent / "agents"


def dump_first_model(module, inst, at_week):
    """Print what the reserve builder added to the model of week ``at_week`` (once per episode)."""
    original = module.Builder.finish
    seen = {"done": False}

    def finish(self):
        base_nc = len(self.cols.keys)
        base_ub, base_eq = len(self.ub.names), len(self.eq.names)
        ruled = dict(self.priority)
        model = original(self)
        if self.settings.week == at_week and not seen["done"]:
            seen["done"] = True
            N, K = inst.nodes, inst.commodities
            nc = model.meta["nc"]
            pr = model.meta["priority"].reshape(model.T, nc)
            print(f"   [dump week {at_week}] H {model.T} columns/week {base_nc} -> {nc}, ub rows {base_ub} -> {len(model.ub_rows)}, eq rows {base_eq} -> {len(model.eq_rows)}, nnz {model.A_ub.nnz + model.A_eq.nnz}")
            b = model.b_ub.reshape(model.T, len(model.ub_rows))
            for r, name in enumerate(model.ub_rows):
                if name[0].startswith("reserve"):
                    j = model.meta["template"][name]
                    s = None
                    label = name
                    if name[0] == "reserve":
                        label = (N[inst.grids[name[1]]].id, K[name[2]].id)
                    else:
                        label = ("feeder", N[inst.stock_slots[name[1]].node].id, K[inst.stock_slots[name[1]].k].id)
                    print(f"      {label}: target wk1 {-b[0, r]:.1f} last {-b[-1, r]:.1f} price {pr[0, j]:.1f}")
            for go, g in enumerate(inst.grids):
                j = model.meta["template"][("ysh", go)]
                print(f"      ysh {N[g].id}: price wk1 {pr[0, j]:.4g} wk2 {pr[1, j]:.4g} last {pr[-1, j]:.4g} (rules alone wk1 {np.asarray(ruled.get(j, 0.0)).ravel()[0]:.4g})")
            changed = {}
            for j, key in enumerate(model.columns):
                d = pr[:, j] - np.asarray(ruled.get(j, 0.0))
                if key[0] in ("I", "x", "p", "xi") and d.any():
                    changed.setdefault(key[0], []).append((key, float(d.min()), int((d != 0).sum())))
            for tag, items in changed.items():
                print(f"      end value on {len(items)} '{tag}' columns, e.g. {items[:3]}")
        return model

    module.Builder.finish = finish


def main(base="buf_lp", params=None, episodes="0", task="small", entropy=12345, detail=False, tag="x", every=4, dump=0, longfuel=False):
    from shockbench_flow.dynamics.env import rollout, took_fallback
    from shockbench_flow.evaluation.cache import default_cache_dir
    from shockbench_flow.hosting.docker import without_secret_like
    from shockbench_flow.policies.naive_fq import REPLICATIONS
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _metered_shim, _policy_seed, _world
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    if isinstance(params, str):
        params = json.loads(Path(params).read_text() if params.endswith(".json") else params)
    params = dict(params or {})
    root = WORK / f"{base}-{tag}"
    if root.exists():
        shutil.rmtree(root)
    shutil.copytree(REPO / "agents" / base, root, ignore=shutil.ignore_patterns("__pycache__", "params.json"))
    if params:
        (root / "params.json").write_text(json.dumps(params))
    ns = [int(n) for n in str(episodes).strip("()[]").split(",")] if not isinstance(episodes, (tuple, list)) else episodes
    cache = str(default_cache_dir())
    out = []
    for n in ns:
        inst, omega, marks, fallback = _world(task, entropy, n, REPLICATIONS, cache)
        with without_secret_like():
            factory = load_agent_class(root, f"submission_diag_{tag}")
            if dump:
                dump_first_model(sys.modules["reserve"], inst, int(dump))
            shim = _metered_shim(factory, None)
            start = time.perf_counter()
            traj = rollout(inst, shim, omega, "standard", _policy_seed(entropy, n, NO_ZIP_SHA256), marks=marks, fallback=fallback)
            seconds = time.perf_counter() - start
        planner = getattr(shim.agent, "planner", None)
        failed = getattr(planner, "failed", None)
        tel = getattr(planner, "telemetry", [])
        iters = sum(s.iterations for s in tel)
        solve = sum(s.seconds for s in tel)
        worst = max((s.seconds for s in tel), default=0.0)
        unload_agent()
        comp = {}
        for r in traj.records:
            for k, v in r.costs.as_dict().items():
                comp[k] = comp.get(k, 0.0) + v
        row = {
            "episode": n,
            "J_cents": traj.J_cents,
            "fallback_weeks": sum(took_fallback(r) for r in traj.records),
            "errors": len(shim.errors),
            "first_error": shim.errors[0] if shim.errors else None,
            "reserve_failed": failed,
            "broken": getattr(shim.agent, "broken", None),
            "seconds": round(seconds, 2),
            "planner_seconds": round(solve, 2),
            "slowest_week": round(worst, 3),
            "iterations": iters,
            "max_week_iterations": max((s.iterations for s in tel), default=0),
            "rungs_beyond_first": sum(len(s.trail) > 1 for s in tel),
        }
        out.append(row)
        print(json.dumps(row), flush=True)
        print("   bn USD:", {k: round(v / 1e9, 2) for k, v in comp.items() if v}, "salvage", round((traj.salvage or 0) / 1e9, 3), flush=True)
        if detail:
            N, K, S = inst.nodes, inst.commodities, inst.stock_slots
            cols = []
            for go, g in enumerate(inst.grids):
                ga = N[g].grid
                for k in ga.fuels:
                    burn = ga.shares[k] * ga.deliverable
                    if (S[inst.slot_index[(g, k)]].storage / burn > 50) != bool(longfuel):
                        continue
                    feeders = [
                        inst.slot_index[(inst.edges[e].tail, k)]
                        for e in inst.in_edges[g]
                        if k in inst.edges[e].K and inst.edges[e].tau == 0 and (inst.edges[e].tail, k) in inst.slot_index
                    ]
                    cols.append((f"{N[g].id[5:]}.{K[k].id[:3]}", inst.slot_index[(g, k)], feeders, burn))
            print("   week | " + " ".join(f"{c[0]:>11s}" for c in cols) + " | shed GWh by grid | unserved bn", flush=True)
            for r in traj.records:
                if r.week % every and r.week != 1 and not r.shed.any():
                    continue
                cells = " ".join(
                    f"{r.stock[s] / burn:5.2f}+{sum(r.stock[f] for f in fs) / burn:5.2f}" for _n, s, fs, burn in cols
                )
                print(
                    f"   {r.week:4d} | {cells} | {' '.join(f'{v:7.0f}' for v in r.shed)} | {r.costs.shortage / 1e9:6.2f}",
                    flush=True,
                )
    return None


if __name__ == "__main__":
    fire.Fire(main)

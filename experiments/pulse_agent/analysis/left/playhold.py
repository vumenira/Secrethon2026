"""A play test of holding the starting gas: agents/mpc_pulse_rec_v6 with no gas delivered from the terminal to chosen
grids in weeks ``first``..``last``; before and after it the agent plays alone (its pulse then finds a grid without gas
and a filled terminal, and starts its own run of whole weeks).

A grid is chosen in week 1 from what the agent itself knows then: its planner's own first plan (the window LP of
week 1, read through the hook the folder already has) burns less gas at that grid in window weeks 4..12 than
``below`` of the segment's full rate. ``below`` above 1 chooses every grid named in ``grids``. The agent folder is
loaded as the scorer loads it and never edited.

uv run python playhold.py --task=small --first=2 --last=4 --below=0.9 --n_jobs=2
"""

import json
import sys
import time

import fire
import numpy as np

import common as C


def one(task, n, first, last, below, grids):
    from shockbench_flow.dynamics.env import rollout
    from shockbench_flow.hosting.docker import without_secret_like
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _metered_shim, _policy_seed
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    inst, omega, marks, fallback = C.world(task, n)
    box = {"week": 0, "ratio": {}, "slots": {}, "chosen": None}
    with without_secret_like():
        factory = load_agent_class(str(C.AGENT), "submission_hold")
        module = sys.modules[factory.__module__]
        module.PARAMS.update({"strict": True})
        orig_act, orig_plan = module.Agent.act, module.Planner.act
        orig_record = module.Planner._record

        def plan_act(self, obs):
            box["week"] = int(obs["week"])
            return orig_plan(self, obs)

        def record(self, model, res):
            if box["week"] == 1 and res.ok and not box["ratio"]:
                pinst = self._inst
                tmpl, nc = model.meta["template"], model.meta["nc"]
                x = np.asarray(res.x, dtype=float).reshape(model.T, nc)
                for go, g in enumerate(pinst.grids):
                    ga = pinst.nodes[g].grid
                    k = ga.rationed
                    if k is None or pinst.nodes[g].id not in grids:
                        continue
                    j = tmpl.get(("G", go, k))
                    if j is None:
                        continue
                    full = ga.shares[k] * ga.deliverable
                    box["ratio"][pinst.nodes[g].id] = float(x[3:12, j].mean() / full)
                    box["slots"][pinst.nodes[g].id] = [s for s, (e, kk, _lane) in enumerate(pinst.action_slots) if kk == k and pinst.edges[e].head == g and pinst.edges[e].tau == 0]
                box["chosen"] = [name for name, r in box["ratio"].items() if r < below]
            return orig_record(self, model, res)

        def act(self, observation):
            out = orig_act(self, observation)
            t = int(observation["week"][0])
            if box["chosen"] and first <= t <= last:
                flows = np.array(out["flows"], dtype=float, copy=True)
                for name in box["chosen"]:
                    flows[box["slots"][name]] = 0.0
                out = dict(out, flows=flows)
            return out

        module.Planner.act = plan_act
        module.Planner._record = record
        module.Agent.act = act
        shim = _metered_shim(factory, None)
        traj = rollout(inst, shim, omega, "standard", _policy_seed(12345, n, NO_ZIP_SHA256), marks=marks, fallback=fallback)
        errors = list(shim.errors)
    unload_agent()
    return n, int(traj.J_cents), box["chosen"], box["ratio"], (errors[:1] or None)


def main(task="small", first=2, last=4, below=0.9, grids="grid_tw,grid_kr,grid_jp", n_jobs=2, only=None):
    from joblib import Parallel, delayed
    from shockbench_flow.scoring.rss import STRATUM_WEIGHTS

    refs = C.references(task)
    cached = C.cached(task)
    grids = tuple(grids) if isinstance(grids, (list, tuple)) else tuple(v for v in str(grids).split(",") if v)
    ns = sorted(refs) if only is None else ([only] if isinstance(only, int) else [int(v) for v in only])
    start = time.perf_counter()
    results = Parallel(n_jobs=n_jobs)(delayed(one)(task, n, int(first), int(last), float(below), grids) for n in ns)
    J = {n: j for n, j, _c, _r, _e in results}
    chosen = {n: c for n, _j, c, _r, _e in results}
    ratio = {n: r for n, _j, _c, r, _e in results}
    errs = [(n, e) for n, _j, _c, _r, e in results if e]
    name = f"playhold_{task}_{first}_{last}_{below}_{'-'.join(g.replace('grid_', '') for g in grids)}"
    (C.HERE / f"{name}.json").write_text(json.dumps({"J": J, "chosen": chosen, "ratio": ratio, "errors": [(n, str(e)) for n, e in errs]}, indent=1))
    sub = sorted(J)
    base = {n: cached[n] for n in sub}
    count = {g: sum(g in (chosen[n] or []) for n in sub) for g in grids}
    touched = [n for n in sub if chosen[n]]
    if len(sub) == len(refs):
        a, _ = C.pooled(refs, base)
        b, lev = C.pooled(refs, J)
        rng = np.random.default_rng(12345)
        by = {}
        for n in sub:
            by.setdefault(refs[n]["stratum"], []).append(n)
        diffs = []
        for _ in range(2000):
            num_a = num_b = den = 0.0
            for s, members in by.items():
                pick = rng.choice(members, size=len(members), replace=True)
                wgt = STRATUM_WEIGHTS[s - 1] / len(pick)
                for m in pick:
                    num_a += wgt * (refs[m]["J_naive_cents"] - base[m])
                    num_b += wgt * (refs[m]["J_naive_cents"] - J[m])
                    den += wgt * (refs[m]["J_naive_cents"] - refs[m]["J_oracle_cents"])
            diffs.append((num_b - num_a) / den)
        lo, hi = np.percentile(diffs, [5, 95])
        print(f"{name}: {b:.4f} against {a:.4f} for the agent alone: {b - a:+.4f} (90 % paired interval {lo:+.4f} to {hi:+.4f}); {sum(J[n] < base[n] for n in sub)} episodes better, {sum(J[n] > base[n] for n in sub)} worse, {sum(J[n] == base[n] for n in sub)} the same to the cent")
        print(f"  grids held, episodes: {count}; episodes with a grid held {len(touched)}; by level {' '.join(f'{s}: {v:.4f}' for s, v in lev.items())}")
    d = sorted(((refs[n]["J_naive_cents"] - J[n]) / (refs[n]["J_naive_cents"] - refs[n]["J_oracle_cents"]) - (refs[n]["J_naive_cents"] - base[n]) / (refs[n]["J_naive_cents"] - refs[n]["J_oracle_cents"]), n) for n in sub)
    print(f"  largest losses {[(n, round(v, 4)) for v, n in d[:5]]}; largest gains {[(n, round(v, 4)) for v, n in d[-5:]]}")
    if errs:
        print(f"  errors: {errs[:4]}")
    print(f"{time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    fire.Fire(main)

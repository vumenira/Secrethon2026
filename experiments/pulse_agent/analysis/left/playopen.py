"""A play test of the opening: agents/mpc_pulse_rec_v6 with the flows of its first weeks replaced by a plan's.

The agent folder is loaded as the scorer loads it and never edited. Each week the agent acts as usual (so its planner
and its pulse keep their memory); in weeks 1..``weeks`` the flows it sends are then replaced by those of a plan of
q1.py (``true``: the clairvoyant optimum under our calendar, closest to our play; ``V``, ``P``, ``N``: the optimum of a
world that knows less), with releases at straits left to the agent. From week ``weeks`` + 1 the agent plays alone.

uv run python playopen.py --task=small --plan=true --weeks=4 --n_jobs=2
"""

import json
import pickle
import sys
import time

import fire
import numpy as np

import common as C


def one(task, n, plan, weeks):
    from shockbench_flow.dynamics.env import rollout
    from shockbench_flow.hosting.docker import without_secret_like
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _metered_shim, _policy_seed
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    inst, omega, marks, fallback = C.world(task, n)
    script = None
    if weeks > 0:
        if plan == "requested":  # our own recorded requests: the machinery's check
            script = C.load_traj(task, n)["requested"][:weeks]
        else:
            x = pickle.loads((C.HERE / "q1" / f"{task}_{n}.pkl").read_bytes()).get(plan)
            if x is None:  # no plan of this kind for the episode: the agent alone (its cached cost)
                return n, C.cached(task)[n], None
            keys = pickle.loads((C.HERE / "res" / f"{task}_{n}_sol.pkl").read_bytes())["keys"]
            col = {key: j for j, key in enumerate(keys)}
            S = len(inst.action_slots)
            script = np.zeros((weeks, S))
            for s, (e, k, lane) in enumerate(inst.action_slots):
                j = col.get(("x", e, k, lane))
                if j is not None:
                    script[:, s] = x[:weeks, j]
            script[script < 1e-9] = 0.0
    with without_secret_like():
        factory = load_agent_class(str(C.AGENT), "submission_open")
        module = sys.modules[factory.__module__]
        module.PARAMS.update({"strict": True})
        orig = module.Agent.act

        def act(self, observation):
            out = orig(self, observation)
            t = int(observation["week"][0])
            # a blackout week shows no mask: the agent's own flows stand that week
            if script is not None and t <= weeks and observation["action_mask.observed"][0] != 0:
                flows = np.where(np.asarray(observation["action_mask"]) > 0, script[t - 1], 0.0)
                out = dict(out, flows=flows)
            return out

        module.Agent.act = act
        shim = _metered_shim(factory, None)
        traj = rollout(inst, shim, omega, "standard", _policy_seed(12345, n, NO_ZIP_SHA256), marks=marks, fallback=fallback)
        errors = list(shim.errors)
    unload_agent()
    return n, int(traj.J_cents), (errors[:1] or None)


def main(task="small", plan="true", weeks=4, n_jobs=2, only=None):
    from joblib import Parallel, delayed

    refs = C.references(task)
    cached = C.cached(task)
    ns = sorted(refs) if only is None else ([only] if isinstance(only, int) else [int(v) for v in only])
    start = time.perf_counter()
    results = Parallel(n_jobs=n_jobs)(delayed(one)(task, n, plan, int(weeks)) for n in ns)
    J = {n: j for n, j, _e in results if j is not None}
    errs = [(n, e) for n, _j, e in results if e]
    name = f"playopen_{task}_{plan}_{weeks}"
    (C.HERE / f"{name}.json").write_text(json.dumps({"J": J, "errors": [(n, str(e)) for n, e in errs]}, indent=1))
    sub = sorted(J)
    base = {n: cached[n] for n in sub}
    same = sum(J[n] == cached[n] for n in sub)
    if len(sub) == len(refs):
        a, _ = C.pooled(refs, base)
        b, lev = C.pooled(refs, J)
        # a paired bootstrap of the pooled difference, resampling episodes inside each harm level
        rng = np.random.default_rng(12345)
        by = {}
        for n in sub:
            by.setdefault(refs[n]["stratum"], []).append(n)
        diffs = []
        for _ in range(2000):
            pick = []
            for s, members in by.items():
                pick += list(rng.choice(members, size=len(members), replace=True))
            # weights of the resample: the pooled score is a ratio of weighted means, formed here by hand
            from shockbench_flow.scoring.rss import STRATUM_WEIGHTS

            num_a = num_b = den = 0.0
            for s, members in by.items():
                chosen = [m for m in pick if refs[m]["stratum"] == s]
                wgt = STRATUM_WEIGHTS[s - 1] / len(chosen)
                for m in chosen:
                    num_a += wgt * (refs[m]["J_naive_cents"] - base[m])
                    num_b += wgt * (refs[m]["J_naive_cents"] - J[m])
                    den += wgt * (refs[m]["J_naive_cents"] - refs[m]["J_oracle_cents"])
            diffs.append((num_b - num_a) / den)
        lo, hi = np.percentile(diffs, [5, 95])
        better = sum(J[n] < base[n] for n in sub)
        worse = sum(J[n] > base[n] for n in sub)
        print(f"{name}: {b:.4f} against {a:.4f} for the agent alone: {b - a:+.4f} (90 % paired interval {lo:+.4f} to {hi:+.4f}); {better} episodes better, {worse} worse, {same} the same to the cent; by level {' '.join(f'{s}: {v:.4f}' for s, v in lev.items())}")
    else:
        d = [(refs[n]["J_naive_cents"] - J[n]) / (refs[n]["J_naive_cents"] - refs[n]["J_oracle_cents"]) - (refs[n]["J_naive_cents"] - base[n]) / (refs[n]["J_naive_cents"] - refs[n]["J_oracle_cents"]) for n in sub]
        print(f"{name}: {len(sub)} episodes; per-episode score differences {[round(v, 4) for v in d]}; the same to the cent {same}")
    if errs:
        print(f"  errors: {errs[:4]}")
    print(f"{time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    fire.Fire(main)

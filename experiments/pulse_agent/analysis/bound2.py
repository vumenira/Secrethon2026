"""The clairvoyant plan's own cost under the planning rules, per episode: one whole-episode LP at week 1.

Per episode, on the true marks and the true demand, the week-1 window of the whole episode with the planning rules:
(a) the package's model (base load before fabs priced in the first week only), (b) that price in every week.
Solved with linprog's interior point (the oracle's solver); the plan's cost is (23) without the prices.

uv run python bound2.py --task=full --episodes=16
"""

import json
import sys
import time
from pathlib import Path

import fire
import numpy as np


ROOT = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026")
sys.path.insert(0, str(ROOT / "experiments" / "approaches"))
import foresight as F  # noqa: E402


def one(spec, n):
    from shockbench_flow.dynamics.env import Env
    from shockbench_flow.hosting.docker import without_secret_like
    from shockbench_flow.policies.base import reset_policy
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _metered_shim, _policy_seed, _world
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    task, entropy, regime, fq, cache = spec
    inst, omega, marks, fallback = _world(task, entropy, n, fq, cache)
    truth = F.truth_of(inst, omega, marks)
    variant = {"fields": "all", "demand": True}
    got = {}
    with without_secret_like():
        factory = load_agent_class(str(F.PROBE), F.MODULE)
        mod = sys.modules[factory.__module__]
        lp = sys.modules["sbflow.oracle.lp"]
        L = mod.L
        P = mod.Planner
        P.h_weeks, P.foresight = 10**6, F.Window(truth, variant)

        def act(self, obs):
            inst_v, t = self._inst, int(obs["week"])
            self._memory.update(inst_v, obs)
            H = L.window_length(self._H, t, inst_v.T)
            arrays = self._window_arrays(inst_v, obs, H)
            assert self.failed == 0
            for flag in (False, True):
                lp.BASE_FIRST_EVERY_WEEK = flag
                start = time.perf_counter()
                model = L.rolled_lp(inst_v, obs, arrays, H, fab_hits=self.hits, planning_rules=True)
                res = lp.solve_oracle(model)
                got[flag] = (res.J_usd, res.status, round(time.perf_counter() - start, 1), model.A_eq.shape, H)
                weekly, salvage = lp.lp_costs(model, res.x)
                nc = model.meta["nc"]
                x = res.x.reshape(H, nc)
                lots = [0.0] * len(inst_v.fabs)
                shed = [0.0] * len(inst_v.grids)
                for j, key in enumerate(model.columns):
                    if key[0] == "p":
                        lots[key[1]] = float(x[:, j].sum())
                    elif key[0] == "ysh":
                        shed[key[1]] = float(x[:, j].sum())
                parts = {k: float(sum(getattr(w, k) for w in weekly)) for k in weekly[0].as_dict()}
                # how the plan uses a grid's energy: fab energy drawn in weeks the same grid sheds base load, and the
                # largest of shed share + fab energy share over grid-weeks (at most 1: a mix of whole and dark weeks)
                col = {key: j for j, key in enumerate(model.columns)}
                E_all = E_shed = 0.0
                hull_max, hull_over, grid_weeks = 0.0, 0, 0
                for go, g in enumerate(inst_v.grids):
                    fabs = [fo for fo in inst_v.grid_fabs[go] if inst_v.nodes[inst_v.fabs[fo]].fab.e > 0]
                    if not fabs or ("ysh", go) not in col:
                        continue
                    ysh = x[:, col[("ysh", go)]]
                    E = sum(x[:, col[("E", fo)]] for fo in fabs)
                    cap = sum(
                        inst_v.nodes[inst_v.fabs[fo]].fab.e * arrays["alpha_bar"][:, fo] * inst_v.nodes[inst_v.fabs[fo]].fab.cap0
                        for fo in fabs
                    )
                    ratio = ysh / arrays["y_bar"][:, go] + E / cap
                    tol = 1e-6 * arrays["y_bar"][:, go]
                    E_all += float(E.sum())
                    E_shed += float(E[ysh > tol].sum())
                    hull_max = max(hull_max, float(ratio.max()))
                    hull_over += int((ratio > 1 + 1e-6).sum())
                    grid_weeks += int(((ysh > tol) & (E > 1e-9)).sum())
                got[("detail", flag)] = {
                    "parts": parts,
                    "salvage": float(salvage),
                    "lots": lots,
                    "shed": shed,
                    "fab_energy": E_all,
                    "fab_energy_while_shedding": E_shed,
                    "hull_max": hull_max,
                    "hull_over": hull_over,
                    "grid_weeks_both": grid_weeks,
                }
            lp.BASE_FIRST_EVERY_WEEK = False
            return {"week": t, "flows": {"slot": [], "qty": []}, "overrides": None, "hold": None}

        P.act = act
        shim = _metered_shim(factory, None)
        env = Env(fallback=fallback)
        seed = _policy_seed(entropy, n, NO_ZIP_SHA256)
        obs, info = env.reset(inst, regime, omega, seed, marks=marks, policy_name="submission")
        reset_policy(shim, info["static"], obs, seed, info.get("omega"))
        shim.act(obs)
        errors = list(shim.errors)
    unload_agent()
    return n, got, errors


def main(task="full", entropy=12345, episodes=16, n_jobs=2, limit=None):
    from joblib import Parallel, delayed

    es = F.tune.episode_set(task, entropy, episodes, 1)
    refs = list(es.references)
    ns = list(es.episodes)[: limit or None]
    start = time.perf_counter()
    results = Parallel(n_jobs=n_jobs)(delayed(one)(es._spec, int(n)) for n in ns)
    plans = {False: [], True: []}
    details = {}
    for (n, got, errors), r in zip(results, refs):
        details[int(n)] = {("bf" if flag else "free"): got.get(("detail", flag)) for flag in (False, True)}
        gap = (r["J_naive_cents"] - r["J_oracle_cents"]) / 100
        naive = r["J_naive_cents"] / 100
        if errors or not got:
            print(f"episode {n}: FAILED {errors}")
            continue
        for flag in (False, True):
            plans[flag].append(got[flag][0])
        d = got[("detail", False)]
        share = d["fab_energy_while_shedding"] / d["fab_energy"] if d["fab_energy"] else float("nan")
        print(
            f"   clairvoyant plan: {share:.1%} of fab energy drawn while its grid sheds base load ({d['grid_weeks_both']}"
            f" grid-weeks); largest shed share + fab share {d['hull_max']:.3f}, above 1 in {d['hull_over']} grid-weeks"
        )
        print(
            f"episode {n} level {r['stratum']}: package's rules {(naive - got[False][0]) / gap:.4f} (status {got[False][1]},"
            f" {got[False][2]} s), base load first every week {(naive - got[True][0]) / gap:.4f} (status {got[True][1]},"
            f" {got[True][2]} s); rows {got[False][3]}, {got[False][4]} weeks",
            flush=True,
        )
    print(f"{time.perf_counter() - start:.0f} s")
    if len(plans[False]) == len(refs):
        for flag, label in ((False, "package's rules"), (True, "base load first every week")):
            table = es.rss([int(round(j * 100)) for j in plans[flag]])
            print(f"week-1 plan, {label}: score of the plan's cost {table['rss']:.4f} by level {table['rss_by_stratum']}")
        out = Path(__file__).parent / f"bound2_{task}_{episodes}.json"
        out.write_text(json.dumps({"episodes": ns, "free": plans[False], "bf": plans[True], "details": details}))


if __name__ == "__main__":
    fire.Fire(main)

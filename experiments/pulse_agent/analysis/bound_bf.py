"""The clairvoyant plan's cost when it may not shed base load to power a fab: one 52-week LP per episode at week 1.

Per episode: the week-1 window of the whole episode on the true marks and the true demand, with the planning rules
(first week) and (a) the package's model, (b) base load before fabs priced in every week. The plan's true cost
(23), without the prices, against the naive rule and the clairvoyant plan, pooled as the score is.
"""

import json
import sys
from pathlib import Path

import fire
import numpy as np


ROOT = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026")
sys.path.insert(0, str(ROOT / "experiments" / "approaches"))
import foresight as F  # noqa: E402


def main(task="small", entropy=12345, episodes=64, limit=None):
    from shockbench_flow.dynamics.env import Env
    from shockbench_flow.hosting.docker import without_secret_like
    from shockbench_flow.policies.base import reset_policy
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _metered_shim, _policy_seed, _world
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    es = F.tune.episode_set(task, entropy, episodes, 1)
    refs = list(es.references)
    spec = es._spec
    variant = {"fields": "all", "demand": True, "h": 52}
    plans = {False: [], True: []}
    ns = list(es.episodes)[: limit or None]
    for n in ns:
        inst, omega, marks, fallback = _world(task, entropy, n, spec[3], spec[4])
        truth = F.truth_of(inst, omega, marks)
        for flag in (False, True):
            got = {}
            with without_secret_like():
                factory = load_agent_class(str(F.PROBE), F.MODULE)
                mod = sys.modules[factory.__module__]
                lp = sys.modules["sbflow.oracle.lp"]
                lp.BASE_FIRST_EVERY_WEEK = flag
                P = mod.Planner
                P.h_weeks, P.foresight = 52, F.Window(truth, variant)

                def record(self, model, res, got=got, lp=lp):
                    got["J"] = lp.lp_usd(model, res.x)[0]

                P._record = record
                shim = _metered_shim(factory, None)
                env = Env(fallback=fallback)
                seed = _policy_seed(entropy, n, NO_ZIP_SHA256)
                obs, info = env.reset(inst, "standard", omega, seed, marks=marks, policy_name="submission")
                reset_policy(shim, info["static"], obs, seed, info.get("omega"))
                shim.act(obs)
            unload_agent()
            plans[flag].append(got["J"])
        r = refs[ns.index(n)]
        gap = (r["J_naive_cents"] - r["J_oracle_cents"]) / 100
        print(
            f"episode {n} level {r['stratum']}: plan {(r['J_naive_cents'] / 100 - plans[False][-1]) / gap:.4f},"
            f" base load first every week {(r['J_naive_cents'] / 100 - plans[True][-1]) / gap:.4f}",
            flush=True,
        )
    if len(ns) == len(refs):
        out = F.tune.run_dir(task, entropy, episodes)
        for flag, label in ((False, "week-1 plan, package's rules"), (True, "week-1 plan, base load first every week")):
            table = es.rss([int(round(j * 100)) for j in plans[flag]])
            print(f"{label}: score of the plan's cost {table['rss']:.4f} by level {table['rss_by_stratum']}")
        for name in ("fs_all_dem_h52", "fs_bf_all_dem_h52", "mpc_det_h20"):
            if (out / f"{name}.json").exists():
                print(f"{name} realised: {es.rss(F.costs_of(out, name))['rss']:.4f}")
        (Path(__file__).parent / "bound_bf.json").write_text(json.dumps({"free": plans[False], "bf": plans[True]}))


if __name__ == "__main__":
    fire.Fire(main)

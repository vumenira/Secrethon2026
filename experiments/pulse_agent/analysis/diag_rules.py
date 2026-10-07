"""Which of the plan's later weeks the simulator does not deliver: the plan's week 2 against the next plan's week 1."""

import sys
from pathlib import Path

import fire
import numpy as np


ROOT = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026")
sys.path.insert(0, str(ROOT / "experiments" / "approaches"))
import foresight as F  # noqa: E402


def main(episode=0, name="fs_all_dem_h52", task="small", entropy=12345, first=7, last=12, tags="p,E,y,ysh,G,lam,rho,short,xi,D,U,B"):
    from shockbench_flow.dynamics.env import rollout
    from shockbench_flow.evaluation.cache import default_cache_dir
    from shockbench_flow.hosting.docker import without_secret_like
    from shockbench_flow.policies.naive_fq import REPLICATIONS
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _metered_shim, _policy_seed, _world
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    cache = str(default_cache_dir())
    inst, omega, marks, fallback = _world(task, entropy, episode, REPLICATIONS, cache)
    truth = F.truth_of(inst, omega, marks)
    variant = F.VARIANTS[name]
    want = set(tags.split(",")) if isinstance(tags, str) else set(tags)
    plans = {}
    with without_secret_like():
        factory = load_agent_class(str(F.PROBE), F.MODULE)
        mod = sys.modules[factory.__module__]
        P = mod.Planner
        P.h_weeks = variant.get("h", 20)
        P.foresight = F.Window(truth, variant)
        lp = sys.modules["sbflow.oracle.lp"]

        def record(self, model, res):
            t = len(self.telemetry) + 1
            if res.x is None or not first <= t <= last + 1:
                return
            nc = model.meta["nc"]
            weekly, _ = lp.lp_costs(model, res.x)
            out = {}
            for w in (0, 1):
                if w >= model.T:
                    continue
                vals = {key: float(res.x[w * nc + j]) for j, key in enumerate(model.columns) if key[0] in want}
                out[w] = (vals, weekly[w].as_dict())
            plans[t] = out

        P._record = record
        shim = _metered_shim(factory, None)
        rollout(inst, shim, omega, "standard", _policy_seed(entropy, episode, NO_ZIP_SHA256), marks=marks, fallback=fallback)
        vinst = shim.agent.planner._inst
    unload_agent()
    names = {
        "p": lambda k: f"lots fab {vinst.nodes[vinst.fabs[k[1]]].id}",
        "E": lambda k: f"energy fab {vinst.nodes[vinst.fabs[k[1]]].id}",
        "y": lambda k: f"base load served {vinst.nodes[vinst.grids[k[1]]].id}",
        "ysh": lambda k: f"base load shed {vinst.nodes[vinst.grids[k[1]]].id}",
        "G": lambda k: f"output {vinst.nodes[vinst.grids[k[1]]].id} fuel {None if k[2] is None else vinst.commodities[k[2]].id}",
        "lam": lambda k: f"load factor {vinst.nodes[vinst.grids[k[1]]].id}",
        "rho": lambda k: f"lot ratio {vinst.nodes[vinst.grids[k[1]]].id}",
        "short": lambda k: f"fuel short {vinst.nodes[vinst.grids[k[1]]].id} {vinst.commodities[k[2]].id}",
        "xi": lambda k: f"osat starts {vinst.nodes[vinst.osats[k[1]]].id} {vinst.commodities[k[2]].id}",
        "D": lambda k: f"served {vinst.nodes[vinst.demands[k[1]].node].id} {vinst.commodities[vinst.demands[k[1]].k].id}",
        "U": lambda k: f"lost {vinst.nodes[vinst.demands[k[1]].node].id} {vinst.commodities[vinst.demands[k[1]].k].id}",
        "B": lambda k: f"backlog {vinst.nodes[vinst.demands[k[1]].node].id} {vinst.commodities[vinst.demands[k[1]].k].id}",
    }
    for t in range(first, last + 1):
        if t not in plans or t + 1 not in plans or 1 not in plans[t]:
            continue
        planned, pc = plans[t][1]
        done, dc = plans[t + 1][0]
        print(f"--- week {t + 1}: planned at week {t} against done (the week-{t + 1} plan's first week)")
        print("   cost  " + "  ".join(f"{k} {pc[k] / 1e9:.2f}->{dc[k] / 1e9:.2f}" for k in pc if abs(pc[k] - dc[k]) > 1e7))
        for key in planned:
            a, b = planned[key], done[key]
            if abs(a - b) > 1e-6 * max(1.0, abs(a), abs(b)):
                print(f"   {names[key[0]](key):45s} planned {a:14.4f} done {b:14.4f}")


if __name__ == "__main__":
    fire.Fire(main)

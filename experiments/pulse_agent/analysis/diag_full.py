"""Why full foresight to the episode's end is not 1.0: plan against execution, week by week (one episode)."""

import json
import sys
from pathlib import Path

import fire
import numpy as np


ROOT = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026")
sys.path.insert(0, str(ROOT / "experiments" / "approaches"))
import foresight as F  # noqa: E402


def main(episode=0, name="fs_all_dem_h52", task="small", entropy=12345, rules=True, weeks=12):
    from shockbench_flow.dynamics.env import rollout
    from shockbench_flow.evaluation.cache import default_cache_dir
    from shockbench_flow.hosting.docker import without_secret_like
    from shockbench_flow.policies.naive_fq import REPLICATIONS
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _metered_shim, _policy_seed, _world, _reference_row
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    cache = str(default_cache_dir())
    inst, omega, marks, fallback = _world(task, entropy, episode, REPLICATIONS, cache)
    refs = json.loads((ROOT / "outputs" / "recovery_tune" / f"{task}-e{entropy}-n64" / "references.json").read_text())
    ref = next(r for r in refs if r["episode"] == episode)
    assert ref["omega_hash"] == omega.hash
    naive, oracle = ref["J_naive_cents"] / 100, ref["J_oracle_cents"] / 100
    print(f"episode {episode}: naive {naive / 1e9:.2f} bn, clairvoyant {oracle / 1e9:.2f} bn, gap {(naive - oracle) / 1e9:.2f} bn")
    truth = F.truth_of(inst, omega, marks)
    variant = F.VARIANTS[name]
    trace = []
    with without_secret_like():
        factory = load_agent_class(str(F.PROBE), F.MODULE)
        mod = sys.modules[factory.__module__]
        P = mod.Planner
        P.h_weeks = variant.get("h", 20)
        P.foresight = F.Window(truth, variant)
        lp = sys.modules["sbflow.oracle.lp"]
        lp.BASE_FIRST_EVERY_WEEK = bool(variant.get("base_first"))
        L = mod.L
        if not rules:
            orig_init = P.__init__

            def init(self, params=None, context=None):
                orig_init(self, mod.MpcDetParams(planning_rules=False), context)

            P.__init__ = init

        def record(self, model, res):
            if res.x is None:
                trace.append(None)
                return
            J, _ = lp.lp_usd(model, res.x)
            weekly, salvage = lp.lp_costs(model, res.x)
            trace.append({"t": len(trace) + 1, "plan_J": J, "obj": res.objective, "week1": weekly[0].as_dict(), "H": model.T})

        P._record = record
        orig_act = P.act
        first = {}

        def act(self, obs):
            if int(obs["week"]) == 1:  # the week-1 window without the planning rules: the clairvoyant LP itself
                self._memory.update(self._inst, obs)
                H = L.window_length(self._H, 1, self._inst.T)
                arrays = self._window_arrays(self._inst, obs, H)
                m = L.rolled_lp(self._inst, obs, arrays, H, fab_hits=self.hits, planning_rules=False)
                r = L.LPSession().solve(L.to_highs_lp(m), L.WindowShape.of(m))
                first["free"] = lp.lp_usd(m, r.x)[0]
                first["H"] = H
            return orig_act(self, obs)

        P.act = act
        shim = _metered_shim(factory, None)
        traj = rollout(inst, shim, omega, "standard", _policy_seed(entropy, episode, NO_ZIP_SHA256), marks=marks, fallback=fallback)
    planner = shim.agent.planner
    print("hook failures", planner.failed, "planner errors", shim.agent.act_errors, "lp fallbacks", sum(s.fallback for s in planner.telemetry))
    unload_agent()
    J = traj.J_cents / 100
    print(f"realised J {J / 1e9:.2f} bn -> share of the gap {(naive - J) / (naive - oracle):.4f}")
    print(f"week-1 window of {first['H']} weeks without planning rules: J {first['free'] / 1e9:.2f} bn (clairvoyant {oracle / 1e9:.2f})")
    print(f"week-1 window with planning rules ({rules}): plan J {trace[0]['plan_J'] / 1e9:.2f} bn")
    done = 0.0
    print(" t   spent   plan-to-go  projected J  step   | realised week (shortage shed holding freight) vs planned week")
    prev = None
    for rec, tr in zip(traj.records, trace):
        proj = done + tr["plan_J"]
        c = rec.costs
        p = tr["week1"]
        step = 0.0 if prev is None else proj - prev
        if rec.week <= weeks or abs(step) > 0.5e9:
            print(
                f"{rec.week:2d} {done / 1e9:8.2f} {tr['plan_J'] / 1e9:10.2f} {proj / 1e9:10.2f} {step / 1e9:+7.2f}  |"
                f" real sh {c.shortage / 1e9:6.2f} shed {c.shed / 1e9:6.2f} hold {(c.holding + c.queue_holding) / 1e9:5.2f} fr {c.freight / 1e9:5.2f}"
                f" | plan sh {p['shortage'] / 1e9:6.2f} shed {p['shed'] / 1e9:6.2f} hold {(p['holding'] + p['queue_holding']) / 1e9:5.2f} fr {p['freight'] / 1e9:5.2f}"
            )
        prev = proj
        done += c.total()
    print(f"final: spent {done / 1e9:.2f} bn, salvage {traj.salvage / 1e9:.2f} bn, J {(done - traj.salvage) / 1e9:.2f} bn")


if __name__ == "__main__":
    fire.Fire(main)

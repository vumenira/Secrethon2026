"""The strait reserve's targets in the first window of an episode: the model is built, nothing is solved or played.

uv run python outputs/w13a/dry_strait.py --task=full --root=12345 --n=0
"""
import sys
from pathlib import Path

import fire
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


class Stop(Exception):
    pass


def main(task="full", root=12345, n=0, weeks=8.0, sources=False, strait=True, agent="agents/mpc_pulse_rec_v13w"):
    from shockbench_flow.dynamics.env import Env
    from shockbench_flow.evaluation.cache import default_cache_dir
    from shockbench_flow.policies.base import reset_policy
    from shockbench_flow.policies.naive_fq import REPLICATIONS
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _metered_shim, _policy_seed, _world
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    inst, omega, marks, fallback = _world(task, root, n, REPLICATIONS, str(default_cache_dir()))
    factory = load_agent_class(str((ROOT / agent).resolve()), "w13a_dry")
    mod = sys.modules[factory.__module__]
    mod.PARAMS.update(strait_reserve=bool(strait), strait_weeks=float(weeks), reserve_sources=bool(sources), strict=True)
    reserve = sys.modules["reserve"]
    plain = reserve.build_lp

    def build_lp(inst_r, marks_r, settings, *, planning_rules):
        b = reserve.Builder(inst_r.at_digest(marks_r.instance_digest), marks_r, planning_rules, settings)
        model = b.build()
        N, K = inst_r.nodes, inst_r.commodities
        print(f"settings: strait_weeks {settings.strait_weeks}, price {settings.price}; columns per week {model.meta['nc']}")
        for c, k, target, price in b.strait_targets():
            print(f"  {N[c].id:12s} {K[k].id:6s} target week 1 {target[0]:9.0f}, last window week {target[-1]:9.0f}; price {price:,.0f} per unit-week")
        for name, slot, target, price in b.reserve_targets():
            if name[0] == "reserve_feeder":
                sl = inst_r.stock_slots[slot]
                print(f"  feeder {N[sl.node].id:14s} {K[sl.k].id:8s} target week 1 {min(target[0], sl.storage):9.0f} of storage {sl.storage:9.0f}")
        names = [key for key in model.columns if key[0] == "reserve_strait"]
        print(f"  {len(names)} reserve_strait columns; {sum(1 for r in model.ub_rows if r[0] == 'reserve_strait')} rows")
        raise Stop

    reserve.build_lp = build_lp
    shim = _metered_shim(factory, None)
    env = Env(fallback=fallback)
    seed = _policy_seed(root, n, NO_ZIP_SHA256)
    obs, info = env.reset(inst, "standard", omega, seed, marks=marks, policy_name="submission")
    reset_policy(shim, info["static"], obs, seed, info.get("omega"))
    try:
        shim.act(obs)
    except Stop:
        pass
    print("errors:", [str(e)[:200] for e in shim.errors][:2])
    unload_agent()


if __name__ == "__main__":
    fire.Fire(main)

"""Week 1 on Full: build and cold solve of shorter windows, and how far their first-week flows are from the 20-week plan's."""
import sys, time
import numpy as np
import gymnasium as gym
import shockbench_flow_gym  # noqa: F401
from shockbench_flow_gym import agent_config_from_reset
from sbf_starter import env_id
folder, task = sys.argv[1], sys.argv[2]
episodes = [int(a) for a in sys.argv[3:]]
sys.path.insert(0, folder)
import agent as A
from sbflow.policies import lp_common as L
import reserve
env = gym.make(env_id(task)); hs = L._highspy()
for n in episodes:
    obs, info = env.reset(options={"episode": n})
    ag = A.Agent(agent_config_from_reset(env, obs, info)); w = ag.wire(obs)
    pl = A.Planner(A.MpcDetParams(), A.PolicyContext()); pl.reset(ag.static, w, ag.seed)
    inst = pl._inst; pl._memory.update(inst, w)
    H = L.window_length(pl._H, 1, inst.T); arrays = pl._window_arrays(inst, w, H)
    out = {}
    for h in (20, 12, 10, 8, 6, 4):
        c = time.process_time()
        small = {k: (a[:h] if getattr(a, "ndim", 0) >= 1 and a.shape[0] == H else a) for k, a in arrays.items()}
        model = reserve.rolled_lp(inst, w, small, h, A.PARAMS, planning_rules=pl.params.planning_rules)
        lp = L.to_highs_lp(model); tb = time.process_time() - c
        hh = hs.Highs(); L._set_options(hh, dict(L.HIGHS_OPTIONS)); hh.passModel(lp)
        c = time.process_time(); hh.run(); ts = time.process_time() - c
        x = np.array(hh.getSolution().col_value); cols = L.action_columns(inst, model)
        out[h] = (tb, ts, x[cols], hh.modelStatusToString(hh.getModelStatus()), L._iterations(hh.getInfo()))
    ref = out[20][2]
    print(f"episode {n}: " + "; ".join(f"H={h}: build {tb:.2f} solve {ts:.2f} s ({it} it, {st}), flows differing from H=20 by >1%: {np.mean(np.abs(a - ref) > 0.01 * np.maximum(np.abs(ref), 1.0)):.2f}, sum |diff|/sum {np.abs(a - ref).sum() / max(np.abs(ref).sum(), 1):.2f}" for h, (tb, ts, a, st, it) in out.items()), flush=True)

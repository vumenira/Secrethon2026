"""Week 1 on Full in two stages: a short window solved cold, then the 20-week window started from its basis."""
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

def run(lp, opts=None, basis=None):
    h = hs.Highs(); L._set_options(h, {**L.HIGHS_OPTIONS, **(opts or {})}); h.passModel(lp)
    if basis is not None: h.setBasis(basis)
    c = time.process_time(); h.run(); dt = time.process_time() - c
    b = h.getBasis()
    return dt, L._iterations(h.getInfo()), h.modelStatusToString(h.getModelStatus()), np.array(h.getSolution().col_value), (L._from_basis(b) if b.valid else None), h.getInfo().objective_function_value

for n in episodes:
    obs, info = env.reset(options={"episode": n})
    ag = A.Agent(agent_config_from_reset(env, obs, info)); w = ag.wire(obs)
    pl = A.Planner(A.MpcDetParams(), A.PolicyContext()); pl.reset(ag.static, w, ag.seed)
    inst = pl._inst; pl._memory.update(inst, w)
    H = L.window_length(pl._H, 1, inst.T); arrays = pl._window_arrays(inst, w, H)
    def build(h):
        c = time.process_time()
        small = {k: a[:h].copy() for k, a in arrays.items()}
        model = reserve.rolled_lp(inst, w, L.read_only(small), h, A.PARAMS, planning_rules=pl.params.planning_rules)
        return model, L.to_highs_lp(model), L.WindowShape.of(model), time.process_time() - c
    m20, lp20, s20, tb20 = build(20)
    d0, i0, st0, x0, b0, o0 = run(lp20)
    line = [f"ep {n}: cold H=20 {tb20:.2f}+{d0:.2f} s ({i0} it)"]
    for h in (8, 12, 16):
        m, lp, s, tb = build(h)
        d1, i1, st1, x1, b1, o1 = run(lp)
        col, row = L._shift_arrays(b1[0], b1[1], s, s20, 0)
        d2, i2, st2, x2, b2, o2 = run(lp20, basis=L._to_basis(col, row, alien=True))
        same = abs(o2 - o0) <= 1e-9 * abs(o0)
        line.append(f"H={h}: {tb:.2f}+{d1:.2f} s then H=20 from its basis {d2:.2f} s ({i2} it, {st2}, same optimum {same}) -> total {tb + d1 + tb20 + d2:.2f} s")
    print(" | ".join(line), flush=True)

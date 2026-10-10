"""Week-1 solve on Full: cold, against a start from another episode's week-1 optimal basis."""
import sys, time
import numpy as np
import gymnasium as gym
import shockbench_flow_gym  # noqa: F401
from shockbench_flow_gym import agent_config_from_reset
from sbf_starter import env_id

folder, task = sys.argv[1], sys.argv[2]
ref, others = int(sys.argv[3]), [int(a) for a in sys.argv[4:]]
sys.path.insert(0, folder)
import agent as A  # noqa: E402
from sbflow.policies import lp_common as L  # noqa: E402
import reserve  # noqa: E402
env = gym.make(env_id(task))
hs = L._highspy()

def week1(n):
    obs, info = env.reset(options={"episode": n})
    ag = A.Agent(agent_config_from_reset(env, obs, info))
    w = ag.wire(obs)
    pl = A.Planner(A.MpcDetParams(), A.PolicyContext()); pl.reset(ag.static, w, ag.seed)
    inst = pl._inst; pl._memory.update(inst, w)
    H = L.window_length(pl._H, 1, inst.T)
    arrays = pl._window_arrays(inst, w, H)
    model = reserve.rolled_lp(inst, w, arrays, H, A.PARAMS, planning_rules=pl.params.planning_rules)
    return inst, model, L.to_highs_lp(model)

def run(lp, opts=None, basis=None):
    h = hs.Highs(); L._set_options(h, {**L.HIGHS_OPTIONS, **(opts or {})}); h.passModel(lp)
    if basis is not None: h.setBasis(basis)
    c = time.process_time(); h.run(); dt = time.process_time() - c
    st = h.modelStatusToString(h.getModelStatus()); inf = h.getInfo()
    return dt, L._iterations(inf), st, np.array(h.getSolution().col_value), L._from_basis(h.getBasis()), inf.objective_function_value

inst, model, lp = week1(ref)
dt, it, st, x, (bc, br), obj = run(lp)
print(f"reference episode {ref}: cold {dt:.2f} s, {it} iterations")
for n in others:
    inst, model, lp = week1(n)
    d0, i0, s0, x0, _b, o0 = run(lp)
    d1, i1, s1, x1, _b, o1 = run(lp, basis=L._to_basis(bc, br, alien=True))
    d2, i2, s2, x2, _b, o2 = run(lp, basis=L._to_basis(bc, br, alien=False))
    cols = L.action_columns(inst, model)
    print(f"episode {n:2d}: cold {d0:.2f} s {i0:6d} it | from the reference basis (alien) {d1:.2f} s {i1:6d} it {s1} | (as is) {d2:.2f} s {i2:6d} it {s2}"
          f" | objective equal {abs(o1 - o0) <= 1e-9 * abs(o0)} / {abs(o2 - o0) <= 1e-9 * abs(o0)}; action columns max |diff| {np.abs(x1[cols] - x0[cols]).max():.2e} / {np.abs(x2[cols] - x0[cols]).max():.2e}", flush=True)

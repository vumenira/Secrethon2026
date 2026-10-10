"""Where week 1's CPU goes on Full, and how HiGHS behaves when a cold solve is cut short."""
import sys, time
import numpy as np
import gymnasium as gym
import shockbench_flow_gym  # noqa: F401
from shockbench_flow_gym import agent_config_from_reset
from sbf_starter import env_id

folder, task, episode = sys.argv[1], sys.argv[2], int(sys.argv[3])
sys.path.insert(0, folder)
c0 = time.process_time()
import agent as A  # noqa: E402
from sbflow.policies import lp_common as L  # noqa: E402
import reserve  # noqa: E402
print(f"import agent: {time.process_time() - c0:.3f} s cpu")
env = gym.make(env_id(task))
obs, info = env.reset(options={"episode": episode})
c = time.process_time(); config = agent_config_from_reset(env, obs, info); print(f"agent_config: {time.process_time() - c:.3f}")
c = time.process_time(); ag = A.Agent(config); print(f"Agent(config): {time.process_time() - c:.3f}")
c = time.process_time(); w = ag.wire(obs); print(f"wire: {time.process_time() - c:.3f}")
c = time.process_time(); pl = A.Planner(A.MpcDetParams(), A.PolicyContext()); pl.reset(ag.static, w, ag.seed); print(f"Planner.reset: {time.process_time() - c:.3f}")
c = time.process_time(); pu = ag.new_pulse(pl._inst); print(f"Pulse: {time.process_time() - c:.3f}")
inst, t = pl._inst, int(w["week"])
c = time.process_time(); pl._memory.update(inst, w); H = L.window_length(pl._H, t, inst.T); arrays = pl._window_arrays(inst, w, H); print(f"memory + window arrays: {time.process_time() - c:.3f}")
c = time.process_time(); model = reserve.rolled_lp(inst, w, arrays, H, A.PARAMS, planning_rules=pl.params.planning_rules); print(f"reserve.rolled_lp: {time.process_time() - c:.3f}")
c = time.process_time(); lp = L.to_highs_lp(model); shape = L.WindowShape.of(model); print(f"to_highs_lp: {time.process_time() - c:.3f}  cols {lp.num_col_} rows {lp.num_row_}")
hs = L._highspy()

def run(opts, basis=None, label=""):
    h = hs.Highs(); L._set_options(h, {**L.HIGHS_OPTIONS, **opts}); h.passModel(lp)
    if basis is not None: h.setBasis(basis)
    c, wl = time.process_time(), time.perf_counter(); h.run(); dt, dw = time.process_time() - c, time.perf_counter() - wl
    st = h.modelStatusToString(h.getModelStatus()); inf = h.getInfo(); b = h.getBasis()
    print(f"  {label:34s} {st:22s} cpu {dt:6.3f} wall {dw:6.3f} iters {L._iterations(inf):6d} obj {inf.objective_function_value:.6e} basis valid {b.valid}")
    return h, b, st

print("cold solves:")
h0, b0, _ = run({}, label="dual, presolve on (v12)")
x0 = np.array(h0.getSolution().col_value)
run({"presolve": "off"}, label="dual, presolve off")
run({"solver": "ipm", "run_crossover": "on"}, label="ipm + crossover")
run({"simplex_strategy": 4}, label="primal simplex")
print("interrupted cold solves, then continued from what they leave:")
for lim in (0.1, 0.25, 0.5):
    for pre in ("on", "off"):
        h1, b1, st = run({"time_limit": lim, "presolve": pre}, label=f"limit {lim} presolve {pre}")
        if b1.valid:
            h2, b2, st2 = run({}, basis=b1, label="   continued from its basis")
            x2 = np.array(h2.getSolution().col_value)
            cols = L.action_columns(inst, model)
            print(f"      week-1 action columns: max |diff| to the cold optimum {np.abs(x2[cols] - x0[cols]).max():.3e}")
print("same model again from its own optimal basis (a perfect warm start):")
run({}, basis=b0, label="from optimal basis")
c = time.process_time(); act = L.week1_action(inst, model, x0, w, L.prohibited_now(pl._memory, t)); print(f"week1_action: {time.process_time() - c:.3f}")

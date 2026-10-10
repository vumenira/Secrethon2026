"""Cold solve of the week-1 window on Full under different HiGHS options: CPU seconds and iterations."""
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
OPTS = {
    "v12 (dual, presolve)": {},
    "dantzig": {"simplex_dual_edge_weight_strategy": 0},
    "devex": {"simplex_dual_edge_weight_strategy": 1},
    "steepest": {"simplex_dual_edge_weight_strategy": 2},
    "scale 0": {"simplex_scale_strategy": 0},
    "scale 2": {"simplex_scale_strategy": 2},
    "scale 4": {"simplex_scale_strategy": 4},
    "crash 1": {"simplex_crash_strategy": 1},
    "crash 2": {"simplex_crash_strategy": 2},
    "no perturb": {"dual_simplex_cost_perturbation_multiplier": 0.0},
    "price 0": {"simplex_price_strategy": 0},
    "price 1": {"simplex_price_strategy": 1},
    "choose solver": {"solver": "choose"},
}
tot = {k: [] for k in OPTS}
for n in episodes:
    obs, info = env.reset(options={"episode": n})
    ag = A.Agent(agent_config_from_reset(env, obs, info)); w = ag.wire(obs)
    pl = A.Planner(A.MpcDetParams(), A.PolicyContext()); pl.reset(ag.static, w, ag.seed)
    inst = pl._inst; pl._memory.update(inst, w)
    H = L.window_length(pl._H, 1, inst.T); arrays = pl._window_arrays(inst, w, H)
    model = reserve.rolled_lp(inst, w, arrays, H, A.PARAMS, planning_rules=pl.params.planning_rules)
    lp = L.to_highs_lp(model)
    row = []
    for name, o in OPTS.items():
        h = hs.Highs()
        try:
            L._set_options(h, {**L.HIGHS_OPTIONS, **o})
        except RuntimeError as err:
            row.append(f"{name}: refused"); continue
        h.passModel(lp); c = time.process_time(); h.run(); dt = time.process_time() - c
        st = h.modelStatusToString(h.getModelStatus())
        tot[name].append(dt)
        row.append(f"{name}: {dt:.2f}s {L._iterations(h.getInfo())}it{'' if st == 'Optimal' else ' ' + st}")
    print(f"ep {n}: " + " | ".join(row), flush=True)
print("mean / max seconds:", {k: (round(float(np.mean(v)), 2), round(float(np.max(v)), 2)) for k, v in tot.items() if v})

"""The sink buffer in one window LP: what the plan ships to sinks with and without it (Full, week 1 and a later week)."""
import sys, time, json
import numpy as np
import gymnasium as gym
import shockbench_flow_gym  # noqa: F401
from shockbench_flow_gym import agent_config_from_reset
from sbf_starter import env_id
folder, task, episode, upto = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
sys.path.insert(0, folder)
import agent as A
from sbflow.policies import lp_common as L
import reserve
env = gym.make(env_id(task)); obs, info = env.reset(options={"episode": episode})
ag = A.Agent(agent_config_from_reset(env, obs, info))
for w in range(1, upto):
    obs, *_ = env.step(ag.act(obs))
wobs = ag.wire(obs); pl = ag.planner; inst = pl._inst; t = int(wobs["week"])
pl._memory.update(inst, wobs); H = L.window_length(pl._H, t, inst.T); arrays = pl._window_arrays(inst, wobs, H)
hs = L._highspy()
def plan(params):
    model = reserve.rolled_lp(inst, wobs, arrays, H, params, planning_rules=pl.params.planning_rules)
    lp = L.to_highs_lp(model); h = hs.Highs(); L._set_options(h, dict(L.HIGHS_OPTIONS)); h.passModel(lp)
    c = time.process_time(); h.run(); dt = time.process_time() - c
    x = np.array(h.getSolution().col_value); nc = int(model.meta["nc"]); tm = model.meta["template"]
    return model, x.reshape(int(model.T), nc), tm, dt, h.getInfo().objective_function_value, lp.num_col_, lp.num_row_
base = plan(A.PARAMS)
on = plan({**A.PARAMS, "sink_weeks": float(sys.argv[5]), "sink_price": float(sys.argv[6])})
print(f"week {t}: plain {base[5]} cols {base[6]} rows, {base[3]:.2f} s; with the buffer {on[5]} cols {on[6]} rows, {on[3]:.2f} s")
N, K = inst.nodes, inst.commodities
print("sink            good      pi      demand(w2)  planned end stock w2..w4: plain | buffer        lost in plan w2..w6: plain | buffer")
for do, d in enumerate(inst.demands):
    s = inst.slot_index[(d.node, d.k)]
    row = []
    for (model, X, tm, *_r) in (base, on):
        I = X[:, tm[("I", s)]]; U = X[:, tm[("U", do)]] if ("U", do) in tm else np.zeros(len(X))
        row.append((I[1:4], U[1:6].sum()))
    dem = base[0].meta.get("demand") if False else None
    print(f"{N[d.node].id:14s} {K[d.k].id:9s} {d.pi:7.0f}   {' '.join(f'{v:9.0f}' for v in row[0][0])} | {' '.join(f'{v:9.0f}' for v in row[1][0])}     {row[0][1]:10.0f} | {row[1][1]:10.0f}")
print()
for name in sys.argv[7:]:
    node, good = name.split(":")
    do = [i for i, d in enumerate(inst.demands) if N[d.node].id == node and K[d.k].id == good][0]
    d = inst.demands[do]; s = inst.slot_index[(d.node, d.k)]
    for label, (model, X, tm, *_r) in (("plain", base), ("buffer", on)):
        I = X[:, tm[("I", s)]]; U = X[:, tm[("U", do)]]; D = X[:, tm[("D", do)]]
        ins = [e for e in inst.in_edges[d.node] if d.k in inst.edges[e].K]
        arr = np.zeros(len(X))
        for e in ins:
            for key, j in tm.items():
                if key[0] == "x" and key[1] == e and key[2] == d.k:
                    tau = inst.edges[e].tau
                    arr[tau:] += X[: len(X) - tau, j]
        print(f"{node} {good} {label:6s} week:   " + " ".join(f"{w:7d}" for w in range(1, 13)))
        print(f"   served D            " + " ".join(f"{v:7.0f}" for v in D[:12]))
        print(f"   lost U              " + " ".join(f"{v:7.0f}" for v in U[:12]))
        print(f"   end stock I         " + " ".join(f"{v:7.0f}" for v in I[:12]))
        print(f"   arrivals planned    " + " ".join(f"{v:7.0f}" for v in arr[:12]))

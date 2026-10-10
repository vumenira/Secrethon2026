import sys, time, json
import gymnasium as gym
import shockbench_flow_gym  # noqa: F401
from shockbench_flow_gym import agent_config_from_reset
from sbf_starter import env_id
sys.path.insert(0, sys.argv[1])
import agent as A
from sbflow.policies import naive as N
from sbflow.policies import lp_common as L
from sbflow.instance.io import load_instance
sha256_hex = N.sha256_hex
env = gym.make(env_id("full")); obs, info = env.reset(options={"episode": 0})
ag = A.Agent(agent_config_from_reset(env, obs, info)); w = ag.wire(obs)
static = ag.static
def t(label, f):
    c = time.process_time(); r = f(); print(f"{label:40s} {time.process_time() - c:.3f}"); return r
for rnd in ("cold cache", "warm cache"):
    print(rnd)
    t("sha256_hex(static instance)", lambda: sha256_hex(static["instance"]))
    t("cached_plan", lambda: N._cached_plan(static, None))
    t("cached_plan again", lambda: N._cached_plan(static, None))
    t("internal_fallback", lambda: L.internal_fallback(A.PolicyContext(), static, w, 0))
    t("Planner.reset", lambda: (lambda p: p.reset(static, w, 0))(A.Planner(A.MpcDetParams(), A.PolicyContext())))
N._PLANS.clear()
t("Planner.reset, cache emptied", lambda: (lambda p: p.reset(static, w, 0))(A.Planner(A.MpcDetParams(), A.PolicyContext())))
t("Planner.reset, cache warm", lambda: (lambda p: p.reset(static, w, 0))(A.Planner(A.MpcDetParams(), A.PolicyContext())))

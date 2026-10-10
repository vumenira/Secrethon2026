"""Run the pulse of an agent folder over the recorded observations of a played episode: a smoke test, not a play.

uv run python outputs/s13/replay_smoke.py --params="{'mild_start': True}"

No simulator and no LP: the planner's ``act`` is replaced by the recorded action of the week (the flows the recorded
agent sent, after its own pulse), so each week the pulse sees a state that agent's play really reached and rules on
it. What it sends changes nothing in the weeks after. It shows that a new rule's code runs (every failure raises)
and where it would act; what the rule is worth takes a play.
"""
import copy
import json
import sys
from collections import Counter
from pathlib import Path

import fire
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RECORDS = ROOT / "outputs" / "view" / "mpc_pulse_rec_v12_full_e12345"


def main(agent="mpc_pulse_rec_v13s", params=None, records=str(RECORDS), show="start", folder=None, detail=""):
    from shockbench_flow.information.flat import FlatLayout
    from shockbench_flow_agent.convert import agent_config
    from shockbench_flow_agent.shim import load_agent_class

    factory = load_agent_class(folder or str(ROOT / "agents" / agent), "submission_smoke")
    mod = sys.modules["submission_smoke"]
    mod.PARAMS |= {"strict": True} | dict(params or {})

    def act(self, obs):  # the planner: no solve, the recorded flows
        self._memory.update(self._inst, obs)
        self.window = None
        return self.recorded

    mod.Planner.act = act
    for path in sorted(Path(records).glob("record_ep*.npz")):
        z = np.load(path, allow_pickle=True)
        static = json.loads(str(z["static"]))
        layout = FlatLayout.from_static(static, None)
        keys = [k for k in z.files if k.startswith("obs.")]
        weeks = z["action.flows"].shape[0]

        def observation(w, z=z, keys=keys):
            return {k[4:]: z[k][w] for k in keys}

        ag = factory(agent_config(copy.deepcopy(static), 0, layout, observation(0)))
        codes = Counter()
        for w in range(weeks):
            obs = observation(w)
            flows = z["action.flows"][w]
            live = np.flatnonzero(obs["action_mask"])
            if ag.planner is not None:
                ag.planner.recorded = {"flows": {"slot": live.tolist(), "qty": flows[live].tolist()}}
            else:  # the planner starts inside decide: hand the first week's flows to the class
                mod.Planner.recorded = {"flows": {"slot": live.tolist(), "qty": flows[live].tolist()}}
            if ag.pulse is not None and ag.pulse.census is None:
                ag.pulse.census = []
            ag.decide(obs)
            if ag.pulse.census is None:
                ag.pulse.census = []
        for t, go, code, _row in ag.pulse.census:
            codes[code] += 1
        rules = Counter(rule for _t, _go, rule, _moved in ag.pulse.log)
        print(path.name, "census", dict(codes), "\n   log", dict(rules))
        per = Counter((go, code) for t, go, code, _row in ag.pulse.census if t <= ag.pulse.last[go])
        for go in sorted({g for g, _c in per}):
            print("    grid", go, {c: n for (g, c), n in sorted(per.items()) if g == go})
        for t, go, code, row in ag.pulse.census if detail else ():
            if code in (detail.split(",") if isinstance(detail, str) else detail) and t <= ag.pulse.last[go]:
                fu = {
                    k: tuple(round(f[x], 2 if x == "ration" else None) for x in ("burn", "ration", "on_hand", "tank", "inflow", "feed"))
                    for k, f in row["fuels"].items()
                }
                m, got = round(row["power"] - row["y"]), round(row["able"] - row["y"])
                print("     ", t, go, code, "m", m, "want", round(row["want"]), "able-y", got, row.get("blocked"), fu)
        for t, go, rule, moved in ag.pulse.log:
            if rule == show:
                print("   ", t, go, rule, {k: round(v / 1e9, 2) for k, v in moved.items()})


if __name__ == "__main__":
    fire.Fire(main)

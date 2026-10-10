"""From the recorded observations of a played episode: per grid, the whole weeks made against what the fuel allowed.

uv run python outputs/s13/replay_duty.py

No simulator and no LP (the planner's act is replaced by the recorded flows, as in replay_smoke.py). Per grid and
fuel the pulse ever counted short: the weeks it was short or held, what reached its terminal and its pipes over the
sellable weeks as a share of the burn (the duty the fuel allows), and the weeks the pulse called whole.
"""
import copy
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import fire
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RECORDS = ROOT / "outputs" / "view" / "mpc_pulse_rec_v12_full_e12345"
GRID = ["tw", "kr", "jp", "cn", "us", "eu", "sea", "in"]
FUEL = ["lng", "crude", "nuc"]


def main(agent="mpc_pulse_rec_v13s", records=str(RECORDS)):
    from shockbench_flow.information.flat import FlatLayout
    from shockbench_flow_agent.convert import agent_config
    from shockbench_flow_agent.shim import load_agent_class

    factory = load_agent_class(str(ROOT / "agents" / agent), "submission_smoke")
    mod = sys.modules["submission_smoke"]
    mod.PARAMS |= {"strict": True}

    def act(self, obs):
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
        ag = factory(agent_config(copy.deepcopy(static), 0, layout, {k[4:]: z[k][0] for k in keys}))
        for w in range(weeks):
            obs = {k[4:]: z[k][w] for k in keys}
            live = np.flatnonzero(obs["action_mask"])
            rec = {"flows": {"slot": live.tolist(), "qty": z["action.flows"][w][live].tolist()}}
            if ag.planner is not None:
                ag.planner.recorded = rec
            else:
                mod.Planner.recorded = rec
            if ag.pulse is not None and ag.pulse.census is None:
                ag.pulse.census = []
            ag.decide(obs)
            if ag.pulse.census is None:
                ag.pulse.census = []
        p = ag.pulse
        codes = defaultdict(Counter)
        shorted = defaultdict(int)
        inflow = defaultdict(float)
        burn = {}
        first = {}
        for t, go, code, row in p.census:
            if t > p.last[go]:
                continue
            codes[go][code] += 1
            marked = set(row.get("short", ())) | set(row.get("also", ())) | set(row.get("blocked", ()))
            for k, f in row["fuels"].items():
                if f["small"]:
                    continue
                inflow[(go, k)] += f["inflow"] / max(f["burn"], 1e-9)
                burn[(go, k)] = f["burn"]
                if k in marked:
                    shorted[(go, k)] += 1
                    first.setdefault((go, k), t)
        print(path.name)
        for go in sorted(codes):
            n = sum(codes[go].values())
            line = f"  {GRID[go]:4s} weeks {n} {dict(codes[go])}"
            for (g2, k), s in sorted(shorted.items()):
                if g2 == go and s >= 3:
                    line += f" | {FUEL[k]} short {s} wk from {first[(go, k)]}, terminal inflow {inflow[(go, k)] / n:.2f} of burn"
            print(line)


if __name__ == "__main__":
    fire.Fire(main)

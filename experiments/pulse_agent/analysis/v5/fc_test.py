"""How well ``Pulse.forecast`` predicts the calendar: its forecast of every week against what the play then did.

uv run python <this> --task=small --episodes=25,12,50,60 [--show=grid_kr --episode_show=25]

The play is v4's (no calendar told); the forecast is computed every week beside it and compared, by weeks ahead, with
the share of the fabs' full draw each grid then had. Two reference forecasts: every week ON, and this week's share
held for every later week.
"""

import sys
from pathlib import Path

import fire
import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lab  # noqa: E402


def collect(task, n, agent="ft_buffer_v5", params=None, H=20):
    w = lab.world(task, n)
    inst, marks = w[0], w[2]
    seen = {}

    def patch(mods):
        P = mods["pulse"].Pulse
        orig = P.adjust

        def adjust(self, obs, values, prohibited, action):
            t = int(obs["week"])
            seen[t] = self.forecast(obs, values, prohibited, min(H, inst.T - t + 1))
            return orig(self, obs, values, prohibited, action)

        P.adjust = adjust

    res = lab.play(task, n, agent, dict(lab.V4) | dict(params or {}), patch=patch, world_=w, tag="fc")
    share, _room, _draw = lab.calendar_of(inst, marks, res["traj"])
    return inst, marks, res, seen, share


def main(episodes, task="small", show="", episode_show=None, H=20):
    ns = [int(n) for n in (episodes if isinstance(episodes, (list, tuple)) else [episodes])]
    leads = list(range(1, H))
    err = {name: {w: [] for w in leads} for name in ("forecast", "all on", "held")}
    agree = {name: {w: [0, 0] for w in leads} for name in ("forecast", "all on", "held")}
    for n in ns:
        inst, marks, res, seen, share = collect(task, n, H=H)
        grids = lab.fab_grids(inst)
        for t, fc in seen.items():
            for go in grids:
                series = fc.get(go)
                if series is None:
                    continue
                for w in leads:
                    if w >= len(series) or t - 1 + w >= inst.T:
                        continue
                    truth = share[t - 1 + w, go]
                    now = share[t - 1, go]
                    for name, value in (("forecast", series[w]), ("all on", 1.0), ("held", now)):
                        err[name][w].append(abs(value - truth))
                        if truth < 0.999 or value < 0.999:  # a grid-week some forecast or the truth calls short
                            pass
                        agree[name][w][0] += int((value >= 0.5) == (truth >= 0.5))
                        agree[name][w][1] += 1
        if show and (episode_show is None or int(episode_show) == n):
            go = inst.grid_ordinal[inst.node_index[show]]
            print(f"episode {n} {show}: week | truth of the next 12 weeks | forecast of them (x: 1, .: 0, digit: tenths)")

            def glyph(v):
                return "x" if v >= 0.999 else "." if v <= 0.001 else str(int(v * 10))

            for t in sorted(seen):
                fc = seen[t].get(go)
                if fc is None:
                    continue
                truth = "".join(glyph(share[t - 1 + w, go]) for w in range(min(13, inst.T - t + 1)))
                guess = "".join(glyph(v) for v in fc[:13])
                print(f"   {t:3d} {truth:13s} {guess}")
    print(f"{task}, episodes {ns}: mean absolute error of the share, and the share of grid-weeks called right (ON at 0.5), by weeks ahead")
    print("   ahead | " + " | ".join(f"{name:>17s}" for name in err))
    for w in leads:
        if not err["forecast"][w]:
            continue
        print(f"   {w:5d} | " + " | ".join(f"{np.mean(err[name][w]):7.3f}  {agree[name][w][0] / max(agree[name][w][1], 1):7.3f}" for name in err))


if __name__ == "__main__":
    fire.Fire(main)

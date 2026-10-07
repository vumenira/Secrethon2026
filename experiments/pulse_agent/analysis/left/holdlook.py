"""The gas-holding play test by episode: which grids were held, what the planner's first plan expected there, what our
own play did there afterwards, and the change in score. Also the gain by whether the held grids did run short.

uv run python holdlook.py --name=playhold_small_2_4_0.9_tw-kr-jp
"""

import json
import pickle

import fire
import numpy as np

import common as C
from report import weights


def main(task="small", name="playhold_small_2_4_0.9_tw-kr-jp", show=6):
    refs = C.references(task)
    cached = C.cached(task)
    w = weights(task, sorted(refs))
    r = json.loads((C.HERE / f"{name}.json").read_text())
    J = {int(k): v for k, v in r["J"].items()}
    chosen = {int(k): (v or []) for k, v in r["chosen"].items()}
    ratio = {int(k): v for k, v in r["ratio"].items()}
    grids = ["grid_tw", "grid_kr", "grid_jp", "grid_eu"]
    cal = pickle.loads((C.HERE / f"opening_{task}_cal.pkl").read_bytes())  # our own calendar of weeks 1..16
    rows = []
    for n in sorted(J):
        D = refs[n]["J_naive_cents"] - refs[n]["J_oracle_cents"]
        d = (cached[n] - J[n]) / D
        z = cal[n]["z"]
        info = []
        short = 0
        for g in chosen[n]:
            gi = grids.index(g)
            off = int((z[gi, 4:12] == 0).sum())  # our own play: weeks 5..12 that shed at this grid
            short += off >= 2
            info.append(f"{g[5:]} plan {ratio[n][g]:.2f}, ours shed {off} of weeks 5-12")
        rows.append((d, n, info, len(chosen[n]), short, w[n] * (cached[n] - J[n]) / 100))
    rows.sort()
    print(f"{name}: per-episode change in score (share of the episode's gap)")
    for d, n, info, _k, _s, _p in rows[:show] + rows[-show:]:
        print(f"  episode {n:2d} level {refs[n]['stratum']}: {d:+.4f}   " + "; ".join(info))
    # pooled gain by how many of the held grids did run short in our own play
    for label, pick in (("every held grid ran short in our play (2 or more of weeks 5-12 shed)", lambda r: r[3] > 0 and r[4] == r[3]), ("some held grid did not", lambda r: r[3] > 0 and r[4] < r[3]), ("no grid held", lambda r: r[3] == 0)):
        sub = [r for r in rows if pick(r)]
        print(f"  {label}: {len(sub)} episodes, pooled {sum(r[5] for r in sub):+.4f}; better {sum(r[0] > 0 for r in sub)}, worse {sum(r[0] < 0 for r in sub)}")
    # by grid: held grids that ran short or not
    for g in grids[:3]:
        held = [n for n in J if g in chosen[n]]
        gi = grids.index(g)
        ran = [n for n in held if int((cal[n]["z"][gi, 4:12] == 0).sum()) >= 2]
        print(f"  {g}: held in {len(held)} episodes; of them it ran short in our own play in {len(ran)}; not held but ran short: {sum(1 for n in J if g not in chosen[n] and int((cal[n]['z'][gi, 4:12] == 0).sum()) >= 2)}")


if __name__ == "__main__":
    fire.Fire(main)
